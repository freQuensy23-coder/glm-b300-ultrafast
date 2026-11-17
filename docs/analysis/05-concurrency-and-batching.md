# Concurrency and batching

Question: does this engine serve only one request at a time, or does it batch multiple concurrent
requests into one forward step?

Answer up front: **one request at a time on the GPU**. `m` is never "number of user requests"; it is
the number of *rows of a single sequence* that one forward pass advances. There is no continuous
batching, no chunked-prefill interleaving, and no multi-sequence KV cache to interpolate into.
Throughput across clients comes from replicas + RDMA KV transfer, plus a "prepare next jobs" pipeline
that runs off-GPU.

## 1. What `m` means

### 1.1 `Meta` — one contiguous position range of one sequence

```rust
// crates/engine/src/model.rs:29-35
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Meta {
    pub pos: i32,
    pub m: i32,
    pub tokens: [i32; 16],
}
```

`Meta` carries **one** `pos` for the whole block, one row count `m`, and a fixed 16-entry token array.
There is no per-row array of positions, no per-row request/sequence id, no attention mask for
row-to-row isolation. Every kernel that takes `Meta` (`model.rs:1769`, `:1782`, `:1789`, `:1853`,
`:1963`, `:2039`, `:2066`) therefore assumes: rows are `pos .. pos + m` of *one* sequence, and row `i`
attends to all earlier positions of that same sequence plus its own prefix.

`set_meta` makes the single-sequence contract explicit:

```rust
// crates/engine/src/model.rs:2371-2377
pub fn set_meta(&mut self, meta: &Meta) -> Result<()> {
    ensure!(meta.m >= 1 && meta.m as usize <= MAX_M);
    ensure!((meta.pos + meta.m) as usize <= self.max_seq, "sequence exceeds max_seq {}", self.max_seq);
    self.meta = self.meta_step;
    self.host_pos_end = (meta.pos + meta.m) as usize;
    self.cur_m = meta.m as usize;
```

`host_pos_end = pos + m` (`model.rs:2375`) — the "sequence so far" counter is a single integer for the
whole engine, not a per-request table.

The chunked-prefill path proves the same thing from the other direction: a long prompt is split into
`Meta` records that are *consecutive slices of one prompt*, each 16 rows apart.

```rust
// crates/engine/src/model.rs:2104-2109
let mut all = vec![Meta { pos: pos as i32, m: n as i32, tokens: [0; 16] }];
for (k, ch) in tokens.chunks(MAX_M).enumerate() {
    let mut mt = Meta { pos: (pos + k * MAX_M) as i32, m: ch.len() as i32, tokens: [0; 16] };
```

A second request would need a second `pos` base and a disjoint KV region; the struct has nowhere to
put either.

### 1.2 `Rank::forward(m)`

```rust
// crates/engine/src/model.rs:2444-2447
pub fn forward(&mut self, m: usize, n_layers: usize, mut trace: Option<&mut Vec<Vec<u16>>>) -> Result<()> {
    self.moe_rows_ok(m)?;
    self.forward_impl(m, 0, n_layers, true, true, trace.as_deref_mut(), None)
}
```

`m` is the row count of *this step*. Where it comes from:

| caller | `m` | meaning |
|---|---|---|
| prefill chunk loop `model.rs:2330-2340` | `(n - pos).min(chunk)`, `chunk <= pmax` (`:2307`) | that many tokens of *the current prompt* |
| prefill verify/replay `model.rs:2266` | `n - s`, last 128 rows | decoder replay rows of the same prompt |
| spec draft `model.rs:3967`, `:4019` | `cfg.dspark_block_size` | draft-block rows of the same sequence |
| spec verify `model.rs:3971`, `:3994` | `blk`/`l+1` | draft+1 rows to accept |
| plain decode | 1 | next token |

For `m > MAX_M` the pass is sliced internally into 16-row sub-steps that share one buffer set:

```rust
// crates/engine/src/model.rs:2350-2359
fn sub16<F: FnMut(&mut Self, usize, usize) -> Result<()>>(&mut self, m: usize, mut f: F) -> Result<()> {
    if m <= MAX_M || self.meta != self.metas {
        return f(self, 0, m);
    }
    ...
    for k in 0..m.div_ceil(MAX_M) {
        let rows = MAX_M.min(m - k * MAX_M);
        self.meta = self.metas + ((1 + k) * std::mem::size_of::<Meta>()) as u64;
        self.cur_m = rows;
```

So "8K-row prefill" is 512 consecutive `Meta` slices of one prompt, not 512 requests.

### 1.3 The KV cache is one sequence's worth of memory

```rust
// crates/engine/src/model.rs:342-349
pub struct LayerCache {
    pub win: DevPtr,
    pub comp: Option<DevPtr>,
    pub idx_k: Option<DevPtr>,
    ...
}

// crates/engine/src/model.rs:686-693  (allocated once, in Rank::new)
for l in 0..c.n_layers + c.n_mtp_layers {
    let ratio = c.compress_ratios[l];
    let src = c.is_kv_source(l);
    caches.push(LayerCache {
        win: a.zeroed(max_seq * hd * bf)?,
        comp: if src { Some(a.zeroed(max_seq / ratio * hd * bf)?) } else { None },
```

`max_seq * head_dim * 2` bytes per layer, addressed by absolute position — a ring/linear buffer for
**one** sequence. There is no block table, no slot table, no per-sequence base offset. The encoder tail
is likewise a single window: `TailBufs`/`EncoderTail` at `model.rs:594-609`, `host_pos_end` at `:545`.

## 2. Are two requests ever resident on the GPU in one step?

No. Three independent confirmations:

1. **One job slot in the whole engine.** `Shared.job: Mutex<Option<Job>>` (`crates/server/src/engine.rs:55`);
   rank 0 pops one job and publishes it there (`engine.rs:67-88`), every rank reads *the same* job
   after a barrier (`engine.rs:87-88`), and the slot is cleared only after the job finishes
   (`engine.rs:90-93`).
2. **The per-rank main loop is strictly sequential.** `engine.rs:207-211`:
   ```rust
   while let Some(job) = shared.next_job(r, admission)? {
       w.job(&job)?;
       drop(job);
       shared.retired(r);
   }
   ```
   `Worker::job` (`engine.rs:257-377`) runs prefill (`:298`), the whole speculative decode loop
   (`:319-358`) and completion (`:376`) before returning. There is no yield point inside it, so no
   other job can be injected mid-prefill or mid-decode.
3. **No API for it.** A prompt array is rejected rather than split:
   ```
   crates/server/src/api.rs:543  "... (batches are not supported)"
   crates/server/src/api.rs:326  if s.n.is_some_and(|n| n != 1) { return Err(bad("only n = 1 is supported")); }
   ```

## 3. How `crates/server` schedules jobs

Flow: HTTP handler → `Job` → process-wide std channel → forwarder thread → admission `Queue` → rank
threads. Each artifact is a strict one-at-a-time admission with a bounded *preparation* pipeline in
front of it.

### 3.1 Ingress: unbounded, no admission control

```rust
// crates/server/src/api.rs:348-351
let (tx, rx) = mpsc::unbounded_channel();
let cancel = Arc::new(AtomicBool::new(false));
let job = Job { trace: trace.clone(), session: session.clone(), ... cancel: cancel.clone(), events: tx };
app.jobs.send(job).map_err(|_| ApiError(StatusCode::SERVICE_UNAVAILABLE, "engine stopped".into()))?;
```

One engine thread per process, fed by one channel (`crates/server/src/main.rs:69-72`). Queue depth is
unbounded, so excess clients simply wait; nothing rejects or sheds load.

### 3.2 The admission queue: FIFO, and the "6 prepared" window

```rust
// crates/server/src/request_queue.rs:6
pub const MAX_PREPARED_REQUESTS: usize = 6;

// crates/server/src/request_queue.rs:99-105  (Queue::prepare)
if !s.preparing {
    if s.ready.len() + s.retiring < MAX_PREPARED_REQUESTS
        && let Some(job) = s.pending.pop_front()
    {
        s.preparing = true;
        return Some(job);
    }
```

`pop()` (`request_queue.rs:126-146`) takes from the front of `ready` — pure FIFO, no priority, no
preemption. The intent is pinned by a test: `fifo_admission_does_not_preempt_active`
(`crates/server/src/request_queue_tests.rs:39-53`).

The 6-slot window is a **KV-prefetch** window, not a batching window: a separate thread drains it into
peer lookups while the GPU is busy.

```rust
// crates/server/src/engine.rs:142-159
if fetch_start.recv().is_err() { return; }
while let Some(mut job) = admission.prepare() {
    let candidate = if job.cancel.load(Ordering::Relaxed) { None } else {
        let result = shared.runtime.block_on(shared.peer().lookup(&job.session, job.prompt()));
        ...
    };
    let candidate = candidate.map(|c| align_job(&mut job, c, max_seq));
    admission.prepared(job, candidate);
}
```

`host_budget` sizes host RAM for exactly this depth (`crates/server/src/host_budget.rs:51-53`:
`let queued = MAX_PREPARED_REQUESTS as u64;`).

### 3.3 What is *not* there

- **No continuous batching.** Nothing appends a new request's rows to an in-flight step: `Meta` holds
  one `pos` (§1.1), and the prefill loop consumes only the current job's tokens
  (`model.rs:2330-2344`, driven from `engine.rs:258` `let toks = &job.tokens[..];`).
- **No chunked-prefill interleaving.** `--chunk` (`main.rs:29`, `model.rs:2307`) chunks *one* prompt;
  the loop runs to completion inside `Worker::job` before decode starts, and there is no scheduler that
  can suspend a prefill half-way and run another request's decode.
- **No slot/paged KV.** §1.3: one linear KV region per layer, addressed by absolute position.

### 3.4 What *does* overlap (intra-process pipelining, not concurrency)

1. **KV prefetch for up to 6 future jobs** — §3.2.
2. **Async KV snapshot export.** Capture is issued on the compute stream into a device image
   (`crates/engine/src/kvstore.rs:80-101`), then a dedicated `kv-publisher` thread converts and
   publishes it (`crates/server/src/publisher.rs:155-168`); submission is non-blocking
   (`publisher.rs:213-219`, `try_send` on a `sync_channel(2)`), so the GPU moves on to the next job
   while the previous job's KV is still being shipped.
3. **Same-session fencing.** Before admitting a job, rank 0 waits for that session's outstanding
   publication (`engine.rs:72-74`, `Pending::wait` with a 2 s bound at `engine.rs:61`; only the same
   session blocks — `publisher.rs:101-105`, test at `:241-247`). This is what keeps "one KV region"
   safe for multi-turn: a new turn cannot start until the previous turn's snapshot left the GPU — and,
   invisibly, the next job's prefill overwrites KV from position `common` onward (`engine.rs:277`,
   `:298`; `common` is 0 when no prefix is restored).
4. **Speculative decode pre-launch.** Inside one request, the captured graph is launched one iteration
   ahead of the read (`engine.rs:333-341`: `launch`, `spec_snapshot(launched % 2)`, then
   `spec_snapshot_read(launched % 2)`), so the GPU never idles on the host loop.

## 4. Where concurrency actually comes from

**Multiple server processes, not multiple requests per process.**

- Each `serve` process owns one engine, one admission queue, one job slot (§2). Replicas are separate
  processes, typically one per node/GPU set (`deploy/README.md:18-33`).
- They coordinate through Redis (ownership directory + pub/sub) and share KV only when the model,
  tokenizer, `--max-seq`, GPU count and decode numerics match (`docs/SERVER.md:9-19`).
- A conversation's next turn can land on a different replica and pull the KV from the owner over RDMA,
  host-to-host, instead of re-running the prompt (`README.md:71-82`).
- **Many conversations stay cached off-GPU**: the host-side `PeerCache` keeps an LRU of snapshots,
  `MAX_ENTRIES = 1024` (`crates/kv-transfer/src/lib.rs:33`, eviction loop at `:181-193`), with up to
  8 concurrent leases (`MAX_LEASES = 8`, `:32`, checked at `:226`). A job's prefix is copied back into
  the single GPU cache at admission via `RestoreSlot::import_prefix`
  (`crates/engine/src/kvstore.rs:192-198`, called at `engine.rs:277`).
- Net effect: a process multiplexes *across time* (one sequence at a time, KV swapped in/out) and
  *across processes* (replicas + transfer), never *within a step*.

## 5. Row limits per step

| limit | value | where | what it constrains |
|---|---|---|---|
| `MAX_M` | 16 | `crates/model/src/limits.rs:1` | `Meta.tokens[16]`, logits buffer, comm/NVLS staging row capacity |
| decode rows | **8** | `model.rs:2449-2462` | draft block (`dspark_block_size`) + 1 verify rows |
| draft block | ≤ 7 | `model.rs:3958`, `:4007` (`blk + 1 <= MAX_M`), error text `:2455` | spec verify must stay within 8 rows |
| tensor-core MoE | ≤ 8 | `ops.rs:968`; `model.rs:2454` (`experts.rm`) | tcgen05 FP4 MoE / decode GEMVs |
| `tc_gemv` | ≤ 8 | `crates/engine/src/ops.rs:1119` | tcgen05 decode GEMV |
| `tc_gemv2_aa` | 1..=8 | `crates/engine/src/ops.rs:1322` | tcgen05 decode GEMV (wqkv/wq_b/wo/gate) |
| `ffn_sgl` | ≤ 8 | `crates/engine/src/model.rs:1148-1151` | SGLang low-batch MoE chain |
| sampling verify head | ≤ 8 | `model.rs:2565` | 8-row sampling/verify |
| head / logits | ≤ 16 | `model.rs:2550-2551` | `head: {m} rows (the logits buffer holds MAX_M)` |
| generic GEMV | 1..=16 | `crates/engine/src/ops.rs:342` | `gemv supports 1..=16 rows` |
| prefill forward | ≤ `PMAX` = 8192 | `model.rs:13`, `:17-19`, `:2307` | `--chunk` upper bound; sliced into 16-row `Meta`s |

The gate that decides "is this a legal decode step":

```rust
// crates/engine/src/model.rs:2449-2462
fn moe_rows_ok(&self, rows: usize) -> Result<()> {
    if rows <= 8 {
        return Ok(());
    }
    ensure!(
        rows > MAX_M || !self.w.layers.iter().any(|l| l.experts.rm),
        "a {rows}-row decode step (the verify of a {}-token draft block) is not supported: the tensor-core MoE takes at most 8 rows. Use a draft block <= 7",
        rows - 1
    );
    ensure!(
        rows > MAX_M || self.tg2.is_none(),
        "a {rows}-row decode step (the verify of a {}-token draft block) is not supported: the tcgen05 decode GEMVs (wqkv, wq_b, wo_a/wo_b, gate) serve steps of at most 8 rows, so this step would take the mma.sync GEMVs while plain decode takes the tcgen05 ones (different numerics: spec != plain). Use a draft block <= 7",
        rows - 1
    );
    Ok(())
}
```

Two things follow from `rows > MAX_M || ...`:

- **Decode is capped at 8 rows** on this model. `experts.rm` selects the row-major FP4 expert layout
  that the tensor-core MoE reads (`crates/engine/src/weights.rs:82-91`; tensor maps are built only for
  `rm` layers at `model.rs:1090`, and the MoE gather switches to `moe_gather_quant_nat` at `:3123`),
  and `tg2` is the tcgen05 decode GEMV set. So the window **9..=16 rows is explicitly rejected** — it
  is an illegal decode width, not a legal small batch.
- **Prefill is exempt only because `rows > MAX_M`** — a 17..=8192-row pass is by definition a prefill
  slice, not a decode step. This is the structural reason the 8-row cap is not a general row cap: the
  code's own classification is "≤ 8 rows = decode width; 9..=16 = impossible decode width; > 16 =
  prefill", and `MAX_M` is where that line sits.

Note the gate is only in `forward` (`:2445`): the prefill path calls `forward_impl` directly
(`model.rs:2340`) and never runs `moe_rows_ok`, which is how a 9..=16-row prompt tail still works. Any
batched decode would have to survive both this gate and the prefill-only kernel paths.

`MAX_M = 16` is therefore a *buffer* bound (16 tokens in `Meta`, 16 rows of logits, `NVLS_CAP =
MAX_M * 6400 + 1024` at `model.rs:116`, `cap_b = MAX_M * per_tok + 1024` at `:716`), not a batch
target, and the effective decode width is 8.

## 6. What a single-B300 port must add or drop

To serve multiple clients from one process, all of the following are prerequisites — none is a flag:

1. **Multi-sequence KV cache.** Replace the single `LayerCache` region (`model.rs:342-349`, `:686-693`)
   with a slot/paged allocation and a per-row sequence→slot mapping. Today `Meta` has one `pos`
   (`:32`), so attention has no way to know which cache a row belongs to.
2. **Per-row step metadata.** `Meta.tokens: [i32; 16]` (`:34`) and `cur_m` (`:547`) must become a
   per-row `(position, slot, request)` triple, and every kernel that reads `meta->pos/m`
   (`:1769`, `:1963`, `:2039`, `:2066`, `:2952` attention, `:3345` KV write, `:3463` index top-k) must
   switch to per-row addressing. This is the largest single change: it touches all row-parameterised
   kernels.
3. **A scheduler that can admit and evict mid-flight.** Replace FIFO `Queue::pop`/`Worker::job`
   (`request_queue.rs:126-146`, `engine.rs:207-211`) with a per-step mix of prefills and decodes, and
   add a KV allocator with eviction to host (the `PeerCache` LRU at `crates/kv-transfer/src/lib.rs:181-193` is
   the natural backing store).
4. **Per-shape CUDA graphs.** The speculative graph is captured once for a fixed row count
   (`engine.rs:243-250` → `capture_spec`/`capture_spec_vadapt`, `model.rs:3936-4045`), with the verify
   arity baked into a conditional switch (`model.rs:4013`, `:4030`). Batching needs either a graph per
   batch width or a graph-free fallback path.
5. **Drop nothing — the numerics constraint stays.** The 8-row decode cap is a numerics invariant, not
   a perf choice: mixing a 9+ row step into decode switches to `mma.sync` GEMVs, i.e.
   "different numerics: spec != plain" (`model.rs:2460`). A batched decode must therefore either stay
   ≤ 8 rows *total across requests* or accept re-baselining every KV snapshot fingerprint
   (`docs/SERVER.md:9-10`).

Cheapest interim path (what the architecture already assumes): keep one request per step and scale with
replicas + KV transfer + prepared-request pipelining (§3.4, §4).

## Verdict

1. **Batch of independent requests in one step? No.** `m` is rows of one sequence (`Meta` has a single
   `pos`: `model.rs:32`); one `Shared::job` slot (`engine.rs:55`) and one FIFO admission (`request_queue.rs:126-146`).
2. **Max concurrent sequences resident: 1 on GPU.** Off-GPU: 6 prepared/pending jobs
   (`request_queue.rs:6`), up to 1024 host-cached conversations with 8 leases
   (`crates/kv-transfer/src/lib.rs:32-33`). Max rows in a decode step: 8 (`model.rs:2449-2462`); 8192 for a
   chunked prefill (`model.rs:13`, `:17-19`).
3. **A single-B300 port must add** a paged multi-sequence KV cache + per-row step metadata + a
   prefill/decode-mixing scheduler + per-shape graphs, and **must keep** the ≤ 8-row decode numerics
   bound; **or drop** batching entirely and serve clients by serializing per process and scaling out
   with replicas and RDMA KV transfer, as `docs/SERVER.md:9-19` and `README.md:71-82` do.
