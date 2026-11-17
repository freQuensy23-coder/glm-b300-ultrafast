# Multi-GPU / tensor-parallel inventory

Question: which mechanisms in this codebase exist **only** to support tensor-parallel / multi-server
execution, and what exactly breaks or must be disabled for a single-GPU (TP=1) deployment?

## 0. Answer up front

- **TP is a hard, structural assumption, not a flag.** Every parameter block carries `[_; MAX_TP]`
  peer arrays (`crates/engine/src/model.rs:15`), kernel entry points are instantiated per width
  (`comm_push2_tp2/4/8`), and several kernels hard-code the rank count as a template parameter
  (`variant_zcomm_dec.cu:3` `TPC = 4`, `variant_zzzzzzzztp8.cu:4-8` `TPC = 8`).
- **There is no NVSHMEM and no multicast memory.** The "NVLS" naming is a channel, not a fabric
  primitive: `setup_nvls` allocates a per-rank staging buffer and exchanges raw device pointers
  between the rank threads of one process; the all-reduce is plain `st.relaxed.sys.global` stores
  into peer UVA addresses. No `nvshmem_*`, no `cuMemMap`/`cuMemMulticast`. Verified by grep over
  `crates/**` (`nvshmem|cuMemMap|multicast` → zero hits outside the `NVLS_*` identifiers).
- **`Rank::new` currently cannot construct a TP=1 rank for this model.** The comm channel width is
  derived as `5*dim/(2*tp)`; at `tp=1` that is `2.5*dim`, which trips the sizing assertion at
  `crates/engine/src/model.rs:704`. This is the first thing any TP=1 build must change.
- **Everything else degrades rather than fails**: peer access loop is empty, NVLS channel is skipped,
  all `tp==4||tp==8` fast paths fall back, weight shards collapse to full tensors — but the
  all-reduce call sites stay live and execute a self-round-trip through the comm buffer, which is
  *required*, because the HC epilogue is fused into the same kernel.
- The RDMA KV path + Redis ownership directory are **not** TP machinery; they are replica machinery.
  They are, however, **unconditionally started** at boot (`crates/server/src/engine.rs:195-205`), so a
  TP=1 build that has no Redis/NIC still has to skip that block.

## A. NVLS channel / multicast setup

### A1. `setup_nvls` — the cross-rank channel allocator
`crates/engine/src/model.rs:133-157`; callers `crates/server/src/engine.rs:189`,
`crates/engine/src/bin/run.rs:118`.

Returns immediately unless `profile::sm103() && tp >= 2` (`:134-136`) — i.e. B300 at TP4/TP8 only.
Otherwise it asserts the GPU count equals `tp` (`:137`), allocates `3*tp*NVLS_CAP` floats plus
`NVLS_FLAGS*16` sentinel words (`:139-141`), publishes its own base pointer into a process-global
`Mutex<Vec<u64>>` (`:125`, `:143-148`) synchronized by the caller's `std::sync::Barrier`, then
collects every rank's pointer into `NvlsBufs::recv[0..tp]` (`:149-154`). "Multicast" here means
"every rank writes into every other rank's copy of the same slot"; there is no hardware multicast
window and no NVSHMEM heap.

`TP-only: yes | needed at TP=1: no | action: keep (the `tp < 2` early return already disables it)`

### A2. Channel sizing constants and the pointer-exchange global
`NVLS_CAP`/`NVLS_FLAGS` at `crates/engine/src/model.rs:116-117`; `NVLS_UC` at `:125`.

`NVLS_CAP = MAX_M*6400+1024` is the per-rank slot width for a decode step (`MAX_M = 16` rows of at most
6400 values each); `NVLS_FLAGS = 4096` is the per-CTA Lamport-flag region that follows the data.
`NVLS_UC` is only correct because all ranks are threads of one process — it is a pointer registry, not
a network bootstrap.

`TP-only: yes | needed at TP=1: no | action: keep`

### A3. `NvlsBufs` and its `Rank` slots
`crates/engine/src/model.rs:119-123` (struct), `:523-524` (`nvls`, `nvls_hold`), `:990-992`
(initialized to `None`).

Holds the staging pointer plus the per-rank `recv` array; `nvls_hold` keeps the `cu::Buffer` alive for
the lifetime of the rank. `None` is the normal state for any non-sm103 or single-GPU run.

`TP-only: yes | needed at TP=1: no | action: keep`

### A4. The NVLS branch of `comm_ex`
`crates/engine/src/model.rs:1992-2041`.

The second half of every all-reduce: if a channel exists and the mode/geometry fits, it rewrites the
generic `CommParams` into an `NvlsParams` (`:1994-2005`) and dispatches one of the fused kernels
(`comm_dec_*`, `comm_nvls_ucpush`, `comm_lp_*`) instead of `comm_push`. The whole block is gated on
`self.nvls`, so at TP=1 it is dead code and control falls through to `:2042-2050`. The three `ensure!`
guards at `:2042-2044` exist only because an armed producer (`aa`/`lp`/`sgl`) may only be consumed by
the fused NVLS kernel.

`TP-only: yes | needed at TP=1: no | action: keep`

### A5. `aa_arm` — the fused "GEMM + all-reduce push" arm
`crates/engine/src/model.rs:1720-1736`; produced at `:1698-1699` (`tg2_wo`), consumed at `:2011-2025`.

Returns an `AaArm` only when `tp == 4 || tp == 8` and the residual geometry fits
(`:1723-1728`); `tp == 8` selects the 8-pointer payload, everything else the 4-pointer one
(`:1732-1735`). The consumer then replaces its `comm_dec_*` entry with `variant_zzzzzzaa/comm_dec*_attn_pp`
and the producer's `wo_b` GEMV with `tc_gemv3yp*_part`, i.e. the GEMM itself stores partials straight
into peer buffers so that the all-reduce needs no separate write pass.

`TP-only: yes | needed at TP=1: no | action: keep (returns `None` at tp=1)`

## B. `comm` / `comm_ex` all-reduce + all-gather machinery

### B1. `CommRank` — the per-rank comm buffer layout
`crates/engine/src/model.rs:37-60` (offsets/`samp_bytes`), `:722-734` (allocation), `:716-721` (layout
computation).

One arena carved into regions whose sizes are all linear in `tp`: reduction/scatter (`rs_off`,
`3*tp*cap*4`), all-gather (`ag_off`, `6*tp*cap*4`), engram directory (`dir_off`), an optional
"short" (`COMM_SHORT_*`) pair of buffers for continue-prefill appends, and a sampling exchange region
(`samp_off`). Every region is indexed by peer, so the layout only has meaning when `tp > 1`.

`TP-only: yes | needed at TP=1: no | action: keep (sizes collapse to one slot at tp=1)`

### B2. `comm_ex` prologue and payload sizing
`crates/engine/src/model.rs:1909-1915`.

Validates that the payload fits the channel (`:1910`), consumes any pending `aa`/`lp`/`sgl` arm
(`:1911-1912`), and rejects the 16-bit two-shot source mode below TP=2 (`:1914`). Note the assertion
at `:1910` is the *runtime* twin of the construction-time assertion at `:704`.

`TP-only: yes | needed at TP=1: yes (as a validity check) | action: keep`

### B3. Two-shot (`Comm2Params`) prefill all-reduce
`crates/engine/src/model.rs:1915-1963`.

For prefill (`cur_m > MAX_M`) and `tp > 1`, reduction and gather are split into two passes over
separate `rs`/`ag` regions and dispatched by a kernel name built from `tp` at runtime
(`:1957-1960`: `comm_push2_k1_tp{tp}` for the short append case, `comm_push2_tp{tp}` otherwise). Only
`2/4/8` exist (`crates/kernels/cu/comm.cu:533-543`, plus `comm_push2_k1_tp8` at
`crates/kernels/cu/variant_zzzzzzzztp8.cu:18`), so the built name would miss `Kernels::get` on
`tp == 1` — unreachable only because of the `tp > 1` precondition at `:1915`.

`TP-only: yes | needed at TP=1: no | action: keep (guarded)`

### B4. `comm_push` — the generic fallback, and the only path left at TP=1
`crates/kernels/cu/comm.cu:134-280`; dispatched at `crates/engine/src/model.rs:2045-2050`.

Each CTA writes its 4-float slice to **every** rank's slot (`:184-189`), publishes Lamport bounds
through `state[b*4]`, then polls every rank's copy until the sentinel is gone and sums the `p.tp`
parts (`:203-260`). At `tp == 1` the peer list is `[self]`, so this becomes a write-then-read-back
identity reduction: correct, but one full residual round-trip per all-reduce, twice per layer.

`TP-only: yes (peer scatter/poll) | needed at TP=1: yes (it is still called) | action: skip-by-config — either stub it out together with the HC epilogue, or leave it as the identity round-trip`

### B5. Fusion constraint: the all-reduce carries the HC epilogue
`crates/kernels/cu/comm.cu:89-131` (`sum_out`, `hc_epi_thread`, `hc_epi_reduce`); modes
`CM_SUM`/`CM_GATHER`/`CM_SUM_HCPOST` at `crates/engine/src/model.rs:332-334`.

`CM_SUM_HCPOST` is the common case for the residual stream: after summing parts the kernel applies
`post`/`comb` mixing and the hyper-connection epilogue. Therefore "disable the all-reduce at TP=1" is
**not** a no-op at the call sites `:1237`, `:2991`, `:3171`, `:3610`, `:3811` — the same kernel also
performs the HC epilogue. A TP=1 change must either keep `comm_push` with `tp=1` or add an
epilogue-only kernel.

`TP-only: partly (reduce is TP, epilogue is not) | needed at TP=1: yes | action: keep`

### B6. `comm` wrapper and the live call sites
`crates/engine/src/model.rs:1738-1740` (wrapper); call sites: `:1237` (sgl MoE GEMM2),
`:2909` (indexer top-k merge), `:2991` (attention output), `:3171` (prefill MoE), `:3210`/`:3214`
(engram sum + gather), `:3610` (decode MoE), `:3686` (`prep_gather`, MTP main projection).

None of these call sites is inside a `tp > 1` branch, so all of them execute at TP=1.

`TP-only: yes | needed at TP=1: yes (as currently written) | action: keep`

### B7. Channel initialization kernels
`crates/kernels/cu/comm.cu:281-284` (`comm_init`, fills `-0.f`), `:544-546` (`comm_init2`, fills
`SENT2 = 0x80008000`), `:712-714` (`comm_fill_u32`). Launches: `comm_init`/`comm_init2` from
`crates/engine/src/model.rs:741-753`; `comm_fill_u32` fetched at `:138` and launched at `:141`, i.e.
only inside `setup_nvls`.

Sets the "not yet arrived" sentinel that the polling loops test (`arrived`, `arr_bf2`). Also TP-only in
the sense that the sentinel protocol only matters when a peer writes the slot.

`TP-only: yes | needed at TP=1: no (harmless) | action: keep`

### B8. `hw().comm_ctas_per_sm`
`crates/engine/src/ops.rs:199`, `:210-218`; used at `crates/engine/src/model.rs:1951`.

A per-SM CTA budget (18 on sm10x) that only the two-shot all-reduce uses to cap its grid:
`blocks = min(virtual_blocks, comm_ctas_per_sm * sms)`.

`TP-only: yes | needed at TP=1: no | action: keep`

## C. Kernel variants that exist only for TP

### C1. `comm.cu` payload structs and `MAX_TP`
`crates/kernels/cu/comm.cu:10` (`#define MAX_TP 8`), `:20-41` (`CommParams` with `recv[MAX_TP]`),
`:287-310` (`Comm2Params` with `rs`/`ag` peer arrays).

Every parameter block carries one pointer per rank. The Rust mirrors are
`crates/engine/src/model.rs:76-100` and `:195-219`.

`TP-only: yes | needed at TP=1: no | action: keep`

### C2. `comm_push2_tp{2,4,8}` / `comm_push2_k1_tp{2,4}`
`crates/kernels/cu/comm.cu:533-543`; tp8 short variant `crates/kernels/cu/variant_zzzzzzzztp8.cu:18`.

Template instantiations with the rank count as a compile-time constant (`comm_push2_body<TP>`), which
lets the compiler unroll the peer loop. Selected by string construction at
`crates/engine/src/model.rs:1957-1960`.

`TP-only: yes | needed at TP=1: no | action: keep`

### C3. `variant_zcomm_dec.cu` — fused decode all-reduce, hard-wired to 4 ranks
`crates/kernels/cu/variant_zcomm_dec.cu:3` (`TPC = 4`), `:33` (`slot = TPC*cap`), `:170-183` (entry
points `comm_dec_x16`, `comm_dec_f32`, `comm_dec_attn_{1..8}`, `comm_dec_ffn_{1..8}`).

Consumes the NVLS channel directly: reads its own contribution, stores 16-bit partials into the peer
slots, then polls and sums `TPC` parts with the HC epilogue inlined. The `TPC = 4` default is not a
default in practice — it fixes the slot stride, so it is only valid for a 4-rank channel; the 8-rank
case is a separate file. Selected at `crates/engine/src/model.rs:2009-2019`.

`TP-only: yes | needed at TP=1: no | action: keep`

### C4. `variant_zzzzzzzztp8.cu` — the 8-rank re-instantiation
`crates/kernels/cu/variant_zzzzzzzztp8.cu:1-18`.

`#define COMM_DEC_NO_ENTRIES` suppresses the 4-rank entries and re-instantiates `comm_dec_body` with
`TPC = 8` (`comm_dec8_*`), plus the 8-rank short two-shot kernel. Its existence is the clearest
evidence that the rank count is baked into the machine code, not passed at runtime.

`TP-only: yes | needed at TP=1: no | action: keep`

### C5. `variant_zzzzzzaa.cu` — GEMM-side all-reduce push
`crates/kernels/cu/variant_zzzzzzaa.cu:4-17` (`AaPush` `recv[4]`, `AaPush8` `recv[8]`, `TPC` constants),
`:244-249` (`tc_gemv3yp_part`, `tc_gemv3yp8_part`), `:372-373` (`comm_dec_attn_pp`, `comm_dec8_attn_pp`).

Rust mirrors: `crates/engine/src/ops.rs:1292-1318` (`AaPush`/`AaPush8`/`AaArm`, with `size_of`
assertions of 56 and 88 bytes) and selection at `crates/engine/src/ops.rs:1352-1367`.

`TP-only: yes | needed at TP=1: no | action: keep`

### C6. `variant_zzzzzzzzzlp.cu` + `ffn_lp_kernel` — Lamport-armed FFN all-reduce
`crates/engine/src/model.rs:127-131` (`ffn_lp_kernel` name/geometry table), `:3616-3630`
(`ffn_lp_ok`, requires `tp == 4` and a live NVLS channel), `:2026-2029` (dispatch).

The MoE rows are written into the Lamport buffer while the all-reduce for the previous stage is still
in flight; only 3- and 6-expert, ≤8-row decode steps with `tp == 4` qualify.

`TP-only: yes | needed at TP=1: no | action: keep`

### C7. `variant_zzzzzzzzzzsgl.cu` — fused shared-expert MoE with in-channel reduce
`crates/engine/src/model.rs:3036-3040` (`sgl_chain_ok` requires `tp == 4 || tp == 8` and
`self.nvls.is_some()`), `:2026-2037` (kernel selection), `:1113-1134` (`sgl_init`);
weight-side gate at `crates/engine/src/weights.rs:425-426` (`(tp, inter)` must be `(4,576)` or `(8,288)`).

`TP-only: yes | needed at TP=1: no | action: keep`

### C8. Build gating: variants are compiled only for `sm_103a`
`crates/kernels/build.rs:33`.

`if ... (variant(&src) && arch != "-arch=sm_103a") { continue; }` — on any other arch the
`variant_*` cubins are not embedded at all, so `Kernels::get("variant_…")` would fail rather than fall
back. This is an arch gate that happens to coincide with the TP4/TP8 deployment; it is not TP-specific
by itself.

`TP-only: no (arch-only) | needed at TP=1: no | action: keep`

## D. Row / head / vocab / expert sharding

### D1. `shard_rows`
`crates/engine/src/model.rs:337-340`; uses at `:1380-1384` (`idx_rows`), `:2115-2118` (`set_prefill`),
`:4147` (test).

Splits a row range `[0, m)` into per-rank quarters on a 4-row boundary:
`b(r) = (r*m/tp) & !3`. At `tp == 1` it returns `(0, m)`, i.e. a no-op — but the callers still take the
sharded code path, so `set_prefill` writes a rank-local `Meta` and `idx_rows` bails out explicitly at
`tp == 1` (`:1377`).

`TP-only: yes | needed at TP=1: no | action: keep (identity at tp=1)`

### D2. Head / group / expert-width division by `tp`
`crates/engine/src/model.rs:755` (`heads = n_heads/tp`), `:758` (`inter = moe_inter_dim/tp`),
`:776` (`o_groups/tp`), `:830` (`5*dim/tp`), `:836`; mirrors in `crates/engine/src/weights.rs:386-395`
(`heads`, `groups`, `in_b`), `:422` (`inter`), `:480-486` (shared `w2` columns), `:540`
(`main_proj` rows).

This is the actual TP decomposition: attention heads (and their `wq_b`/`wo_a`/`wo_b` slices and
`attn_sink` rows, `weights.rs:392-397`), the o-projection LoRA group block, the shared-expert
intermediate width, and the MTP `main_proj` output rows are all cut per rank. At `tp == 1` each
division yields the full tensor, which is the behaviour a single-GPU build wants — but it also means
per-GPU weight memory scales as `1/tp` (`inter = moe_inter_dim/tp` at `weights.rs:422` is the dominant
term, since every rank holds all routed experts but only `1/tp` of each expert's rows).

`TP-only: yes | needed at TP=1: no (collapses correctly) | action: keep`

### D3. Vocab / head sharding
`crates/engine/src/weights.rs:563-566` (`v = vocab_size/tp`, `head` rows `[rank*v, (rank+1)*v)`,
`head_fp8_host` likewise); consumers `crates/engine/src/model.rs:2555-2558` (`vloc`, `vpad`),
`:2562-2571` (`pick_head` offset `rank*vloc`), `:2416-2425` (`read_logits`/`read_last_logits`).

Each rank computes only its vocab slice of the logits; the slices are recombined by the sampler, not by
a gather (see F1). At `tp == 1` the slice is the whole vocabulary. Note the implicit constraint
`vocab_size % tp == 0`.

`TP-only: yes | needed at TP=1: no | action: keep`

### D4. `markov_head` vocab shard
`crates/engine/src/weights.rs:544-548`.

Same `v = vocab_size/tp` split for the draft-head projection.

`TP-only: yes | needed at TP=1: no | action: keep`

### D5. Routed-expert weight packing
`crates/engine/src/weights.rs:421-444`; packing primitives `crates/model/src/packing.rs:54-62`
(`expert_shard`, `expert_shard_rowmajor`, `expert_shard_sgl`); tp=1 covered by the unit test at
`crates/model/src/packing.rs:261-272`.

Experts are **not** distributed across ranks: each rank reads all `n_exp` experts and keeps rows
`[r*inter, (r+1)*inter)` of each. So TP is intra-expert width splitting, not expert parallelism; the
padding constant differs per width (`(4,576)`, `(8,288)` at `weights.rs:425`).

`TP-only: yes (the split) | needed at TP=1: no | action: keep`

### D6. Shared-expert shard (FP8 path)
`crates/engine/src/weights.rs:459-487`.

Rows `[r*inter, (r+1)*inter)` of `w1`/`w3` interleaved in 32-row blocks, and columns
`[r*inter, (r+1)*inter)` of `w2`.

`TP-only: yes | needed at TP=1: no | action: keep`

### D7. Engram table shard + cross-rank gather
Shard: `crates/engine/src/weights.rs:489-508` (`shard = rows_total.div_ceil(tp)`, table/scales sliced
to `[r*shard, (r+1)*shard)`, `wkv` rows `nk/tp`, `Engram::shard_rows` at `:74-80`).
Gather: `crates/engine/src/model.rs:3210` (`CM_SUM` over `ecols`), `:3214` (`CM_GATHER` into `b_ekv`
with `ld_out = 5*dim/2`). Offset arithmetic: `:3204-3205`, buffer `:830`
(`ekv_part = m*5*dim/tp`).

This is the widest cross-rank payload in the engine and the origin of the `5*dim/(2*tp)` term in
`per_tok` (see H2). At TP=1 the table is whole and both comm calls are self-identity.

`TP-only: yes | needed at TP=1: no (identity) | action: keep`

### D8. Engram peer directory
`crates/engine/src/model.rs:64-73` (`EgPeers` with `table/scales/part/flag` per rank),
`:735-740` (each rank publishes its four pointers into `comm.recv + dir_off`),
`:3217-3236` (`eng_load_peers` downloads every peer's four pointers),
`:3279-3306` (`engram_pre_peer` + the `engram_apply_peer` launch, which reads remote shards directly).

A hand-rolled device-side directory so one rank can read another rank's engram shard without a gather.
Requires `peers.len() == tp` (`:3218`) and non-zero pointers on every rank (`:3229`).

`TP-only: yes | needed at TP=1: no | action: keep`

### D9. Indexer top-k row split
`crates/engine/src/model.rs:1376-1385` (`idx_rows`, explicitly `tp == 1 → None`), `:2899-2911`
(per-rank `topk_mc` over `[r0,r1)` then `rows_i2f` → `comm_ex(CM_SUM)` → `rows_f2i`),
scratch sizing `:914-916` via `score_rows` (`:1372-1374`).

Long-prefill indexer selection is partitioned by query rows and merged through the comm channel. At
TP=1 `idx_rows` returns `None`, so the merge is skipped — this is the one sharded path that is already
TP-safe by construction.

`TP-only: yes | needed at TP=1: no | action: keep (already guarded)`

### D10. `score_rows` scratch sizing
`crates/engine/src/model.rs:1372-1374`; uses `:782-783` (`scores`, `bscores`), `:914-916`, `:1428`;
test `:4140-4154`.

`tp == 1` gives `pmax` rows (full), otherwise `(pmax/tp + 8).max(4*tp + MAX_M)`. Correct at TP=1 but
allocates the *largest* variant of every score buffer — a single-GPU memory cost, not a correctness
issue.

`TP-only: yes | needed at TP=1: no | action: keep`

## E. Peer buffers and peer access

### E1. `Rank::peers` — the peer comm-pointer table
`crates/engine/src/model.rs:522` (field), `:990` (init `Vec::new()`), `:1946-1949`, `:1989-1991`
(copied into every param block), `:2594-2596` (sampler).

Filled from a shared `Mutex<Vec<u64>>` after all ranks allocate their comm buffer:
`crates/server/src/engine.rs:126`/`:186-188`, `crates/engine/src/bin/run.rs:87`/`:115-117`.

`TP-only: yes | needed at TP=1: no | action: keep (single element at tp=1)`

### E2. `cuCtxEnablePeerAccess` between every rank pair
`crates/server/src/engine.rs:171-178`, `crates/engine/src/bin/run.rs:100-107`, wrapper
`crates/cuda/src/lib.rs:119-126` (`Device::enable_peer`), also
`crates/engine/src/bin/test_block_verify.rs:391-398`.

Enables P2P so that the plain `st.global`/`ld.global` in `comm.cu` can reach another GPU's memory
through UVA. The loop body never executes at `tp == 1`. Note the implied topology: `cuCtxEnablePeerAccess`
needs the peer's `CUcontext` in the same process, and no Cargo.toml in the workspace depends on
NCCL/gloo/MPI (verified), so the only supported arrangement is one process with `tp` primary contexts.

`TP-only: yes | needed at TP=1: no | action: keep`

### E3. Peer access is a hard prerequisite for the channel, not a fallback
`crates/kernels/cu/comm.cu:43-45` (`st_remote_v4`), `:184-189`, `:628-633`.

The kernels address peer memory as if it were local; there is no PCIe/NVLink capability check and no
host-staged fallback. Consequence for TP=1: nothing to check; consequence for multi-GPU on a
non-P2P-capable topology: silent wrong results / faults.

`TP-only: yes | needed at TP=1: no | action: keep`

## F. Sampling all-gather

### F1. `samp_stats` cross-rank partial exchange
Buffer: `crates/engine/src/model.rs:56-59` (`SAMP_SO_S2/FR/BV` offsets, `samp_bytes`), `:708-709`
(`samp`, `samp_rec`), `:716` (`off_b`, `cap_b`, a second full channel for the sampling pass).
Dispatch: `:2550-2560` (`head`, `srec` with `vloc` and `rank*vloc`), `:2562-2596` (`pick_head`,
`q.recv[r] = peers[r] + off`).
Kernels: `crates/kernels/cu/sample.cu:150-165`, `:236-244`, `:298-317`, `:645-662`, `:756-790`
(loops over `p.tp` peers), `:821` (`samp_stats`), `:33` (`spart`).

Grammar-constrained / rejection sampling needs the full vocabulary distribution, which is split across
ranks (D3). Instead of gathering logits, each rank writes its per-shard statistics into every peer's
`samp` region and reads back all `tp` parts. At TP=1 the loops degenerate to a single self-entry, so
the path stays correct but the whole exchange mechanism is dead weight.

`TP-only: yes | needed at TP=1: no | action: keep`

## G. RDMA KV transfer and the Redis ownership directory

### G1. `PeerCache` startup is unconditional (and not TP-related)
`crates/kv-transfer/src/lib.rs:289-298` (struct), `:304-330` (`start`, requires non-empty
fingerprint, positive budgets, routable bind IP, then a live Redis and a data plane);
called from `crates/server/src/engine.rs:195-205` (rank 0 only) with options from
`crates/server/src/peer.rs:20-71`; fatal exit on RDMA-specific errors at
`crates/server/src/engine.rs:149-153`.

This is *replica*-level machinery: it shares prompt KV between independent `serve` processes. It has
no notion of `tp`, no rank argument, and no sharding (`crates/kv-transfer/src/*` contains no `tp` or
`rank` symbol at all). It is nevertheless mandatory at boot for the server binary, so a single-GPU
deployment with no Redis and no RDMA NIC must delete or bypass the `r == 0` startup block.

`TP-only: no | needed at TP=1: no (unless cross-replica reuse is wanted) | action: skip-by-config`

### G2. RDMA-only data plane
`crates/kv-transfer/src/rdma.rs:9-22` (`RdmaRequired` marker, `is_rdma_required`),
`:29-73` (`validate_env`: `VELO_RDMA_RENDEZVOUS_DISABLE` and software RMA are hard errors; `UCX_TLS`
must be `rc_*`/`ud_*`), `:162-189` (`DataPlane::start`, bidirectional GET probe, or
`"this binary has no Linux Velo UCX support"` when built without the feature); feature wiring
`crates/kv-transfer/Cargo.toml:5-7` (`default = ["rdma"]`).

`TP-only: no | needed at TP=1: no | action: skip-by-config (drop the `rdma` default feature and the publisher)`

### G3. Redis ownership directory
`crates/kv-transfer/src/directory.rs:10-13` (`DEFAULT_REDIS_URL`, `DEFAULT_DIRECTORY_PREFIX`,
`RECORD_TTL` = 24h), `:43-92` (Lua scripts: epoch/sequence allocation, `LOOKUP`, `CLAIM` with
`PUBLISH`, `CHECK`), `:109-126` (`Directory`, key shapes `{prefix}:owner:{scope}:{key}` and
`{prefix}:onboarded:{scope}`); client side `crates/kv-transfer/src/lib.rs:477-493` (owner lookup and
remote/local decision), `:739-770` (claim/onboard).
Configuration: `crates/server/src/peer.rs:33-44` (`DSUF_REDIS_URL`, default `redis://127.0.0.1:6379`).

A distributed lease directory: exactly one replica owns a given prompt's KV image, and ownership
changes are broadcast on a Redis pub/sub channel. Nothing in it is per-GPU.

`TP-only: no | needed at TP=1: no | action: skip-by-config`

### G4. The KV fingerprint embeds the TP degree
`crates/server/src/peer.rs:24-30` (`…\0tp={}` from `engine.gpus.len()`), `crates/kv-transfer/src/lib.rs:329-333`
(`scope = digest("rdma-v4\0{fingerprint}\0snapshot-v{VERSION}")`), test `crates/server/src/peer.rs:76-82`.

The only place where TP leaks into the KV path: a TP4 and a TP8 replica compute different scopes, so
they never exchange snapshots. Since the payload itself is not TP-sharded (G5), this is a
compatibility guard, not a layout requirement.

`TP-only: no (a guard on it) | needed at TP=1: yes if KV transfer is kept | action: keep`

### G5. Snapshot payload is not head-sharded
`crates/engine/src/kvstore.rs:306-360` (`encoded_layout` over `rk.caches`/`rk.mtp_caches`),
`crates/engine/src/kvstore.rs:47-49` (`fits`), and `crates/server/src/publisher.rs:184-202`
(`Publisher::capture` takes one `Rank`; the publisher itself is constructed only for `r == 0` at
`crates/server/src/engine.rs:200`).

The per-layer window/compressor/index caches hold the MLA latent (`max_seq * head_dim`), which every
rank computes in full — `wkv`/compressor/indexer weights are loaded unsharded
(`crates/engine/src/weights.rs:399-419`, `:514`). So rank 0's image is a complete snapshot, and a TP=1
replica with the same fingerprint would restore a valid one.

`TP-only: no | needed at TP=1: n/a | action: keep`

## H. Hard gates on the TP degree

### H1. `MAX_TP = 8`
`crates/engine/src/model.rs:15`; all `[_; MAX_TP]` fields (`:65-68`, `:78`, `:122`, `:149`, `:176`,
`:198-199`, `:2576`, `:3873`), kernel-side `crates/kernels/cu/comm.cu:10`, `crates/kernels/cu/sample.cu:25`.

Caps the width; there is no `tp > 8` path at all.

`TP-only: yes | needed at TP=1: no | action: keep`

### H2. `per_tok` assertion — **the actual TP=1 blocker**
`crates/engine/src/model.rs:703-705`.

```rust
let per_tok = c.dim.max(5 * c.dim / (2 * tp)).max(ecols).next_multiple_of(256);
ensure!(per_tok <= COMM_MAX_PER_TOKEN, "comm payload per token {per_tok} exceeds COMM_MAX_PER_TOKEN");
```

`COMM_MAX_PER_TOKEN = 6400` (`:24`). The middle term is the engram `ekv` gather payload
`nloc/2` with `nloc = nk/tp` and `nk = dim*(hc_mult+1)` (`crates/engine/src/weights.rs:501`,
`:3214`) — the hard-coded `5` implies `hc_mult = 4`. It is therefore written as a **per-rank** width,
and at `tp == 1` it becomes `2.5*dim`; for this checkpoint's `dim = 5120` (assumed by
`crates/engine/src/model.rs:1118`, `crates/engine/src/weights.rs:425`) that is `12800 > 6400`, so
`Rank::new` returns an error and the server exits during weight load. Any `dim > 2560` trips it.

`TP-only: yes | needed at TP=1: yes (as a bound) | action: keep the check, raise the constant (see delta)`

### H3. `short_enabled`
`crates/engine/src/model.rs:720-721`, buffers `:733`, launches `:748-751`.

Enables the "short" reduction channels used for continue-prefill appends ≤512 rows. The predicate is
`tp == 2 || tp == 4 || (tp == 8 && variant_cc())` (`:720`) — no `tp == 1` case, so the extra regions
are not even allocated at TP=1.

`TP-only: yes | needed at TP=1: no | action: keep`

### H4. `dec_ok` / decode-kernel selection
`crates/engine/src/model.rs:2009-2019`.

`comm_dec*` variants require `tp == 4 || tp == 8`; `tp == 8` switches the name prefix to the
`variant_zzzzzzzztp8` module. Below TP=4 the fallback is `comm_nvls_ucpush` (still needs an NVLS
channel, i.e. TP≥2 and sm103) or plain `comm_push`.

`TP-only: yes | needed at TP=1: no | action: keep`

### H5. `sgl_chain_ok`
`crates/engine/src/model.rs:3036-3040`. Requires `tp == 4 || tp == 8` and a live NVLS channel; see C7.

`TP-only: yes | needed at TP=1: no | action: keep`

### H6. `tc_cluster_for` — `tp == 4` only
`crates/engine/src/model.rs:1523-1530` (`let on = self.tp == 4;`).

Enables the two-CTA cluster tensor-core GEMV for decode `wo_b`/`qb` shapes. Not a correctness gate:
at `tp != 4` it returns `None` and the generic `gemv` path runs. Cost of TP=1: the cluster GEMV
optimisation is unavailable (it is also arch-gated through `self.w.tc`).

`TP-only: yes (performance) | needed at TP=1: no | action: keep`

### H7. `profile::Sm103Tp4` / `Sm103Tp8` — the master gate
`crates/cuda/src/profile.rs:5-22` (`Profile` enum, `init` maps `((10,3), 4) → Sm103Tp4`,
`((10,3), 8) → Sm103Tp8`, everything else `Base`; `assert!(!READ_EARLY…)` at `:15`);
consumers: `crates/server/src/main.rs:55`, `crates/engine/src/bin/run.rs:77`,
`crates/engine/src/model.rs:951-959` (`mk_kmax`, `lp_mixf`, `ffn_lp_buf`), `:1230`, `:1668`,
`crates/engine/src/ops.rs:1168` (`tcgemv2_on`), `crates/engine/src/weights.rs:566`
(`head_fp8_host`).

One enum carries both the arch and the TP degree, so *every* B300-specific fast path is also a
TP4/TP8-only path. A TP=1 run on B300 lands in `Profile::Base`, which disables tcgen05 GEMV (`tg2`),
the SGL MoE chain, the Lamport FFN arm, the FP8 head copy and the `mk_kmax` decoder. This is the single
largest cluster of TP-coupled code in the tree.

`TP-only: partly (arch ∧ tp) | needed at TP=1: no | action: skip-by-config (profile resolves to Base)`

### H8. `--tp` harness assertion
`crates/engine/src/bin/test_block_verify.rs:365` (`ensure!(tp == 1 || tp == 4 || tp == 8, "--tp must be 1, 4 or 8")`),
peer setup `:391-398`.

The only place in the tree that explicitly accepts `tp == 1`, and only for the sampler test.

`TP-only: no (test harness) | needed at TP=1: n/a | action: keep`

### H9. Server CLI defaults and GPU-count parsing
`crates/server/src/main.rs:11` (usage), `:27` (`gpus: (0..8).collect()`), `:38` (`--gpus`),
`:52` (must be distinct, non-empty), `:55` (`profile::init` from `gpus[0]`'s arch and `gpus.len()`).

`tp` is not a parameter — it *is* `--gpus` length. A TP=1 deployment is `--gpus 0`.

`TP-only: no | needed at TP=1: yes | action: keep`

### H10. Rank threads, barrier and lockstep job dispatch
`crates/server/src/engine.rs:120-131` (`tp = gpus.len()`, `Barrier::new(tp)`), `:162-166`
(one scoped thread per rank), `:170-199` (peer access, weight load, `setup_nvls`, per-rank warmup),
`:206-213` (job loop), `:67-93` (`next_job`/`retired`: rank 0 pulls the job, all ranks meet at the
barrier).

The server is a lockstep `tp`-thread process by construction. At TP=1 every barrier degenerates to a
no-op and rank 0 does all the work; no code change is needed, but the "replica = one process" model
means a single-GPU deployment is exactly the `tp == 1` degenerate case.

`TP-only: yes | needed at TP=1: yes (degenerate) | action: keep`

## I. Not TP-only — do not delete these

| Mechanism | Citation | Why it is not TP |
|---|---|---|
| `hc_*` fork/join, `hc_mix`, `hc_epi` | `crates/engine/src/model.rs:1878-1888`, `crates/kernels/cu/hcpipe.cu:15`, `crates/kernels/cu/hcepi.cuh` | hyper-connections (model architecture: `hc_mult`, `hc_sinkhorn_iters` at `crates/model/src/config.rs:43-45`), not cross-GPU |
| `par_fork`/`par_join`, `side`/`par` streams | `crates/engine/src/model.rs:1890-1898`, `:552-561` | intra-GPU stream overlap |
| `gemv_slices` | `crates/engine/src/model.rs:1900-1906` | row-slicing for `m > MAX_M` |
| cluster launches (`Launch.cluster`) | `crates/engine/src/model.rs:1416`, `:1422`, `:2958`, `:2963` | thread-block clusters (sm90+), single GPU |
| `MAX_M` chunking of `Meta` | `crates/engine/src/model.rs:2104-2118` | batch granularity, uses `shard_rows` only incidentally |

## J. Minimal TP=1 delta

Concrete sites that must change for a working single-GPU build (`--gpus 0`), in the order they fail:

- `crates/engine/src/model.rs:24` + `:703-705` — the `COMM_MAX_PER_TOKEN = 6400` bound. `per_tok` is
  *correct* at `tp == 1` (the engram `ekv` gather payload `nloc/2` at `:3214` really is the full
  `2.5*dim`), but the constant is a tp≥2 bound derived from `5*dim/(2*tp) <= 6400`. Fix: raise the
  constant (or derive it from `cfg.dim`), keep the `:704` assertion. Without this, `Rank::new` fails
  during weight load.
- `crates/engine/src/model.rs:1909-2050` (`comm_ex`) — decide the TP=1 semantics of the residual
  reduction. As written it self-round-trips through `comm_push` (`:2045-2050`) with `p.tp == 1` and
  `recv[0] = self`. Either keep it (correct, costs one write+poll+read per all-reduce) or add a
  local-only kernel that still performs the `CM_SUM_HCPOST` epilogue (`crates/kernels/cu/comm.cu:89-131`)
  for the five call sites `:1237`, `:2991`, `:3171`, `:3610`, `:3686`.
- `crates/engine/src/model.rs:716-733` (`CommRank` allocation) — region sizes are `3*tp*cap*4`; with
  `cap = pmax*per_tok` and `per_tok = 2.5*dim` that is `3*8192*12800*4 B ≈ 1.26 GB` of arena for the
  channel at `dim = 5120`, `pmax = 8192`. Same total as TP4 (the `tp` factor cancels), so this is not
  a new cost — but it is now one device's memory instead of four.
- `crates/engine/src/model.rs:782-783`, `:914-916` (`score_rows`) — `tp == 1` allocates the full
  `pmax`-row score and top-k scratch (`:1372-1374`); this is the largest avoidable single-GPU
  allocation and the place to reintroduce a cap if memory is tight.
- `crates/engine/src/model.rs:1720-1736` (`aa_arm`), `:1523-1530` (`tc_cluster_for`),
  `:3036-3040` (`sgl_chain_ok`), `:3616-3630` (`ffn_lp_ok`), `:2009-2019` (`dec_ok`) — all return
  `None`/`false` at `tp == 1`; no edit needed, but they must not be replaced by unconditional
  variants, and `crates/engine/src/model.rs:1957-1960` must stay behind its `tp > 1` guard (there is
  no `comm_push2_tp1` in `crates/kernels/cu/comm.cu:533-543`).
- `crates/engine/src/model.rs:1698-1704` / `crates/engine/src/ops.rs:1352-1367` — the `AaArm`
  plumbing must keep receiving `None` at TP=1, otherwise the `tc_gemv3yp*_part` entry points (which
  write into peer buffers) would be selected for a rank with no peers.
- `crates/cuda/src/profile.rs:16-20` — add nothing; TP=1 must resolve to `Profile::Base`, which is
  what disables `tg2`/`sgl`/`lp`/`head_fp8` (`crates/engine/src/model.rs:951-959`, `:1230`,
  `crates/engine/src/weights.rs:566`). Verify that `Base` still has a working decode GEMV path
  (`crates/engine/src/model.rs:1532-1542` → `gemv_slices`).
- `crates/engine/src/model.rs:1113-1134` (`sgl_init`) and `crates/engine/src/weights.rs:425-426` —
  the SGL branch is keyed on `(tp, inter) ∈ {(4,576),(8,288)}`; at `tp == 1` `inter = moe_inter_dim`,
  so the branch is skipped and expert packing falls through to `expert_shard`/`expert_shard_tc`
  (`crates/engine/src/weights.rs:438`). No edit, but the FP4 layout test at
  `crates/model/src/packing.rs:261-272` is the only coverage for the `tp == 1` packing.
- `crates/server/src/engine.rs:171-178`, `crates/engine/src/bin/run.rs:100-107` — peer-access loops are
  already empty at `tp == 1`; keep them (they are the guard that makes `comm.cu`'s plain
  `st.global`/`ld.global` legal).
- `crates/server/src/engine.rs:195-205` — the `r == 0` block unconditionally starts `PeerCache`
  (Redis + RDMA). A single-GPU build that does not want cross-replica KV must skip this block;
  `PeerCache::start` fails hard otherwise (`crates/kv-transfer/src/lib.rs:305-330`,
  `crates/kv-transfer/src/rdma.rs:162-189`).
- `crates/server/src/peer.rs:24-30` — if `PeerCache` is kept, the fingerprint already carries
  `tp = gpus.len()`, so a TP=1 replica will not collide with TP4/TP8 snapshots; nothing to change.
- `crates/server/src/main.rs:27` — default `--gpus` is `0..8`; a TP=1 default must be `vec![0]`
  (or leave it and always pass `--gpus 0`).
- `crates/engine/src/weights.rs:563-566`, `:545-546` — vocab shards collapse to the full head; no edit,
  but confirm `vocab_size % tp` style constraints are not asserted anywhere else (they are not).
- `crates/engine/src/weights.rs:422` — `inter = moe_inter_dim/tp` is the dominant weight-memory term:
  TP=1 means each GPU carries the full routed-expert population at full width, i.e. `tp`× the TP4/TP8
  per-GPU footprint. The repo contains no evidence that this fits one device; treat single-GPU as a
  functional target, not a deployment target.
