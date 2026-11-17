# The single-B300 GLM port

Two new crates, nothing in the original tree was modified except the workspace `members` list:
`crates/glm-kernels` (CUDA, ~880 lines of `.cu` plus a launch registry) and `crates/glm`
(~2 900 lines of Rust). The existing DeepSeek path is untouched and still builds.

The port exists because **upstream cannot run on one GPU**: `per_tok` assertions, the
`Sm103Tp4/Tp8` profile gate, `MAX_TP`, and the `--tp` harness assertion all reject `tp = 1`, and
the kernels that would be needed are only instantiated for 4 and 8 ranks (`03-multi-gpu-inventory.md`
§H). Serving GLM on a single B300 is therefore a new engine that reuses the upstream *ideas*,
not the upstream *engine*.

## What runs where

| file | contents |
| --- | --- |
| `crates/glm-kernels/cu/gemv.cu` | block-FP8 GEMV (split-K, 32×32 shared staging) and tiled GEMM with a scale companion |
| `cu/attn.cu` | split-KV flash decode (one CTA per head × row × split), combine, tiled causal prefill |
| `cu/misc.cu` | RMSNorm, per-head QK norm, partial RoPE, swiglu, residual add, cast, concat, KV append, bias add, row copy |
| `cu/moe.cu` | sigmoid router with group top-k and correction bias, expert bucketing, packed expert FFN |
| `cu/sample.cu` | on-device sampling and speculative accept/commit |
| `crates/glm/src/engine.rs` | prefill, block verification, adaptive draft length, MTP drafting, the round bookkeeping |
| `crates/glm/src/weights.rs` | checkpoint → device: FP8 adoption or on-load quantization, every tensor padded to 128 |
| `crates/glm/src/engine.rs`, `server.rs`, `main.rs` | OpenAI-compatible HTTP surface, `--plan`, one-shot generation |

Everything compiles to `cubin` for `sm_103a` at build time (`build.rs`, `GLM_SM` overrides the
arch); the five translation units build with `-O3 -std=c++17` and no errors
(`attn` 59 KB, `gemv` 68 KB, `misc` 68 KB, `moe` 104 KB, `sample` 103 KB of cubin).

## The decode round

State: `pos` = committed positions, one unprocessed token (the *anchor*) at `pos`, both KV
caches truncated to `pos`.

1. **Draft.** The MTP layer is run `blk` times, autoregressively, over
   `[anchor, d0, … d_{blk-2}]`; each step is `rmsnorm(embed(tok))` ⊕ `rmsnorm(h_prev)` →
   `eh_proj` → one decoder layer → norm → head → sample. Its own KV slot is layer `n_layers`.
2. **Verify.** One target forward over `m = blk + 1` rows starting at `pos`.
3. **Accept.** `glm_spec_accept` compares each draft against the target row on device (greedy) or
   does the `min(1, p/q)` rejection test (stochastic), writes the emitted tokens, and commits the
   prefix. `pos += committed`, the anchor becomes the last emitted token, `h_prev` becomes the
   hidden state of row `committed - 1`.
4. **Settle.** If every draft was accepted the target advanced one row past the drafter, so the
   drafter replays that single row (`Engine::mtp_settle`).

The invariant that makes the rollback cheap: the anchor is *always* the only unprocessed token,
so no cache entry is ever stale and nothing needs to be re-prefilled. Upstream gets the same
property by re-prefilling the tail; here it falls out of the round structure.

Draft length comes from `spec::SpecPolicy`, which maximises
`expected_tokens(b) / cost(b)` with `expected_tokens(b) = 1 + Σ a^i` over an EWMA of the
observed per-token acceptance rate `a`. That is the same control law as the confidence head in
upstream, with the head replaced by a statistic the engine can measure itself — item 4 of
`02-transfer-matrix.md`.

## The captured round

`--graph <blk>` replaces the per-token launch sequence with one `CUgraph`:

1. captured on the top-level graph: `embed(anchor)` → the MTP body for `blk` rows. Row `i+1`
   reads its token id straight from the device slot row `i` sampled into, so no id ever
   comes back to the host mid-round;
2. `glm_spec_pick` smooths the observed acceptance rate and calls
   `cudaGraphSetConditional(handle, L - 1)`;
3. `cuGraphAddNode` with type `CU_GRAPH_NODE_TYPE_CONDITIONAL` and `CU_GRAPH_COND_TYPE_SWITCH`
   gets `blk` body graphs back; body `i` is captured with `build_rows(i+1)`, `forward_ids(i+2)`
   and `accept_launch(i+1)` — a different verification width each.

Every kernel argument is a buffer pointer. The only immediates are shapes and the draft
width baked per body, so the exec replays at any position: the step offset and the token ids
arrive through `tok`/`tob`, which the host rewrites before each launch. A round therefore
costs one graph launch plus a single 8-byte read of `(emitted, committed)`.

The sampler and the accept kernel take the RNG step from a device counter that they advance
with `atomicAdd`, so temperature sampling stays stochastic inside the graph instead of
repeating one baked-in draw.

Two things this cannot do, both structural: a captured body has a fixed row count (hence one
body per width), and the drafter cache is advanced by the graph itself, which is why
`mtp_pos` simply follows `pos` in this path.

## What was dropped, and why it does not cost anything

Every multi-GPU mechanism: NVLS channels, peer access, `comm`/`comm_ex`, the fused all-reduce
arms, the TP-only kernel variants, expert/head/vocab sharding, the RDMA KV store with its Redis
directory, the KV prefetch window, and the snapshot path. With one rank there is no peer to
reduce into and no ownership directory to consult; `03-multi-gpu-inventory.md` lists each one
with the call site that would have to exist for it to matter.

Also dropped: CED (items 12–14) and the Markov drafter head (item 3). Both need trained
DeepSeek components that a GLM checkpoint does not contain.

## Batching

One request at a time, exactly like upstream, and for the same structural reason: the kernels
address rows as `pos .. pos + m` of a single sequence, the KV cache is one contiguous buffer per
layer with one position counter, and attention has no cross-row mask. `m` is the verification
width (≤ 16), never a batch of users.

That is not a regression — it is what upstream does — but it is the honest answer to "does this
engine batch": no. A second request waits on the engine mutex. Throughput across clients comes
from replicas, which on one GPU means one process per device. Adding continuous batching would
mean a per-row position array, a paged or per-sequence KV layout, and an attention mask, i.e. a
different engine; `05-concurrency-and-batching.md` documents the same conclusion for upstream.

## Running it

```
glm-serve --ckpt /models/GLM-5.3-Flash --plan          # resolves every tensor, no GPU needed
glm-serve --ckpt /models/GLM-5.3-Flash --prompt "hi"   # one-shot, prints tok/s
glm-serve --ckpt /models/GLM-5.3-Flash --port 8000     # /v1/completions, /v1/chat/completions
```

`--plan` is the pre-flight to run before renting a B300: it reports resolved slots, checkpoint
payload, KV bytes per token and the KV footprint at 32 K / 128 K / 512 K / 1 M context. KV size
at load time is chosen from free memory after the weights are resident.

## Verification that exists, and what does not

`cargo test` → 36 tests, all passing, none requiring a device:

- synthetic GLM checkpoint (two layers, one dense one MoE, an MTP block, one tensor shipped as
  block-FP8) written to a temp dir as real `safetensors`, then driven through the loader:
  every slot resolves, a truncated download is reported with the tensor named, bf16 becomes
  FP8 within the e4m3 bound, FP8 is adopted with its scales, expert packing has the expected
  byte layout and row strides;
- quantizer padding and round-trip error, config parsing and validation of both GLM and
  DeepSeek-style key names, tensor-naming alternatives, KV capacity planning, the sampler
  mirrored against the CUDA splitmix, acceptance sampling, and the draft-length policy.

No GPU was available, so **nothing has been executed on device**. The kernels are compile-clean
for `sm_103a` and the launch signatures match their `__global__` declarations by construction,
but first-run correctness is unverified: expect to shake out scale indexing and the rope
convention against the real checkpoint. The three gaps to close first are CUDA-graph capture of
the verify round (items 5–6), the MTP/verify overlap on a second stream (item 11), and real
tokenizer/chat-template handling from `tokenizer_config.json`.

## Tooling state

- `cargo check --all-targets -p deepseek-ultrafast-glm -p glm-kernels` is clean: no errors, no
  warnings. `cargo test` on the same two packages: 35 tests, all passing.
- `cargo clippy` on the port reports 19 style lints (`is_multiple_of`, `chunks_exact`,
  `len`/`is_empty`) that auto-fix refuses to apply because the same run also lints the
  upstream `deepseek-ultrafast-cuda`, which has a pre-existing deny-level lint (a public fn
  dereferencing a raw pointer without `unsafe`). Run clippy with `--no-deps` on the port alone
  if you want them silent.
- A workspace-wide `cargo test` cannot build: the upstream `kv-transfer` crate depends on
  `ucx-rs`, which needs UCX installed on the host. Test the port by package name.
