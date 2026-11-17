# Transfer matrix: what moves to GLM 5.3 Flash, what does not

Three verdicts per row:

- **generic** — the idea is model-agnostic and was ported to `crates/glm`.
- **adapt** — the idea survives but its implementation is bound to a DeepSeek artifact; the port
  replaces that artifact with something the GLM checkpoint actually provides.
- **deepseek** — it depends on trained DeepSeek-V4.1 components or on its exact architecture and
  has no meaning for another model.

"Ported" refers to code in this tree, not to a benchmark on hardware.

## A. Speculative decoding and generation

| # | optimization | verdict | in the port |
| --- | --- | --- | --- |
| 1 | multi-token draft, verify as a block | generic | `Engine::step`, `glm_spec_accept` |
| 2 | DSpark MTP drafter (extra decoder layers) | adapt | `Weights::mtp`, `Engine::draft` reads `num_nextn_predict_layers` from the checkpoint; absent → plain decode |
| 3 | low-rank Markov correction of the draft distribution | deepseek | not ported (trained head, `draft_markov.cu`) |
| 4 | confidence-head adaptive draft length | adapt | two layers: host `spec::SpecPolicy` (EWMA of acceptance, drives the uncaptured path) and `glm_spec_pick` on device (drives the captured path) |
| 5 | CUDA conditional graph selecting verify width on device | generic | ported: `Engine::capture_spec_graph` + `glm_spec_pick` calling `cudaGraphSetConditional`; switch node with one body per width |
| 6 | drafting + verification + acceptance inside one graph | generic | ported: draft, pick, all verify bodies and the accept kernel are captured into one `GraphExec`; a round is one launch + one 8-byte copy. Token ids and positions reach it through staged device buffers, so the same exec replays at any offset |
| 7 | GPU-side rejection/acceptance sampling | generic | `glm_spec_accept` (greedy and stochastic) |
| 8 | vocabulary sampling without collecting logits on the host | adapt | `glm_sample`; at one GPU there is nothing to all-gather, only the host round-trip to remove |
| 9 | generation counters and token state on device | generic | `out`/`spec` device buffers, one `cudaMemcpyAsync` per round |
| 10 | double-buffered CUDA-event snapshots | deepseek | obsoleted by multi-GPU streaming; nothing to snapshot with no peer |
| 11 | precompute MTP prep while the target forward runs | deepseek | not applicable: upstream's drafter produces a whole block in one pass, so layer *l+1*'s prep overlaps layer *l*'s forward. A sampled, autoregressive drafter (mine, and GLM's MTP) cannot: row *i+1*'s input is row *i*'s sampled token. The part that does transfer — keeping the host out of the critical path — is satisfied by the captured round |

## B. Attention, prefill and architecture

| # | optimization | verdict | in the port |
| --- | --- | --- | --- |
| 12 | CED: encoder over the full input, no decoder prefill | deepseek | not ported — needs a trained context encoder |
| 13 | rebuild the decoder's 128-token sliding window by bounded replay | deepseek | not ported (consequence of 12); GLM does use sliding-window layers, and `glm_attn_*` take a `window` argument for them |
| 14 | keep encoder-tail activations across turns | deepseek | not ported |
| 15 | GQA with per-head QK RMSNorm and partial rotary | n/a (GLM-specific) | `glm_rmsnorm_heads`, `glm_rope_qk` with `rotary_dim < head_dim` |
| 16 | split-KV flash decode + combine | generic | `glm_attn_decode_split`, `glm_attn_combine`, `cache::decode_splits` |
| 17 | tiled causal prefill | generic | `glm_attn_prefill` |
| 18 | fused q/k/v projection with one bias tensor | generic | `Linear::gemv` + `glm_add_bias` |

## C. MoE

| # | optimization | verdict | in the port |
| --- | --- | --- | --- |
| 19 | sigmoid router with group-limited top-k and correction bias | adapt | `glm_router` (sigmoid/softmax, `n_group`, `topk_group`, bias, `routed_scaling_factor`) |
| 20 | shared expert fused with routed experts | adapt | `ffn_moe`: routed reduce into the residual, then the shared expert adds into it |
| 21 | expert weights packed per layer (`[n][2*inter, dim]`) | generic | `Experts` in `weights.rs` |
| 22 | low-batch SGLang-style chain (`ffn_sgl`) | generic | not ported; the general path covers it, the chain is a latency micro-optimization for another day |
| 23 | sharded experts across ranks | deepseek (TP-only) | dropped — one GPU holds every expert |
| 24 | in-channel all-reduce inside the shared-expert kernel | deepseek (TP-only) | dropped |

## D. Quantization, weights, memory

| # | optimization | verdict | in the port |
| --- | --- | --- | --- |
| 25 | block-scaled FP8 (128×128, E4M3) GEMV/GEMM with a scale companion | generic | `gemv.cu`, `quant.rs`; adopted as-is when the checkpoint is already FP8, otherwise quantized at load |
| 26 | padding layer widths to multiples of 128 | generic | `GlmConfig::npad`, `quant::quantize` |
| 27 | `--plan` style pre-flight of the checkpoint | generic | `glm-serve --plan` resolves every slot without touching the GPU |
| 28 | KV cache as `[head, pos, dim]` bf16 per layer | generic | `cache::KvCache` |
| 29 | RDMA KV transfer + Redis ownership directory | deepseek (multi-GPU/multi-node) | dropped |
| 30 | KV prefetch window | deepseek (needs a peer) | dropped |

## E. Serving and scheduling

| # | optimization | verdict | in the port |
| --- | --- | --- | --- |
| 31 | one job on the GPU at a time, replicas for throughput | generic | kept deliberately: one `Engine` behind one mutex |
| 32 | device-side sampling so the host only reads tokens | generic | kept |
| 33 | double-buffered streaming to the HTTP layer | generic | unbounded channel per request, engine thread never blocks on the socket |
| 34 | continuous batching | absent upstream | still absent; see `06-port.md` §"batching" |

## What is DeepSeek-only, in one sentence

Items 3, 10, 12, 13, 14, 23, 24, 29 and 30 exist because DeepSeek-V4.1-Flash has a trained
context encoder, a trained Markov drafter head, and is served on a multi-rank NVLS cluster.
None of them has a GLM counterpart: GLM 5.3 Flash is a plain decoder with an optional MTP block,
no context encoder, and it must fit on one B300, so there is no rank to talk to.
