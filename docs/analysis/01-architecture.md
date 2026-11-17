# How `deepseek-ultrafast` works

Read this first: it is the map the rest of `docs/analysis/` hangs off. Everything below was
read out of the source tree, not from the README.

## Shape of the thing

A Cargo workspace of five crates plus a set of hand-written CUDA translation units that are
compiled to `cubin` at build time (`crates/kernels/build.rs`) and launched through a thin
`cuLaunchKernel` wrapper (`crates/cuda`). There is no PyTorch, no Triton, no NCCL: the entire
runtime is CUDA driver API plus Redis for cross-process coordination.

| crate | role |
| --- | --- |
| `cuda` | device/buffer/stream/launch plumbing over `libcuda` |
| `kernels` | the `.cu` corpus: GEMM/GEMV, attention, MoE, sampling, comm, plus TP-only variants |
| `model` | checkpoint formats (`safetensors`), tensor names, config, quantized layouts |
| `engine` | the model itself: `model.rs` (forward), `decode_cache.rs` (KV), `kvstore.rs` (RDMA), `weights.rs` |
| `server` | request queue, engine loop, HTTP API, snapshots |

## One sequence, one block of rows

The unit of work is `Meta { pos, m, tokens[16] }`: a contiguous range of `pos .. pos + m`
positions of *one* sequence, with at most 16 rows in flight. `m` is therefore never a batch of
users: it is a verification block during decode, or a chunk of prompt during prefill. The
consequences run through the whole codebase — one KV cache per process, one position counter,
attention with no cross-row masking, and a hard ceiling of 16 on speculative width. Doc `05`
pins this down call site by call site.

## The forward pass

Per layer: RMSNorm → fused QKV GEMV → per-head QK norm → RoPE → KV append → attention (decode
flash attention or a tiled causal prefill) → output GEMV → residual → RMSNorm → MoE → residual.
Weights are block-scaled FP8 (128×128 blocks, E4M3 with an E8M0/FP32 scale), which is why every
GEMV has a scale-index companion kernel and why the layer widths are padded to multiples of 128.

Two paths diverge after the QKV projection:

- **decode** (`m ≤ 16`): split-KV flash attention, one CTA per (query head, row, split), then a
  combine kernel. This is the latency-critical path.
- **prefill** (`m > 16`, up to `MAX_M`): tiled causal attention, GEMM instead of GEMV.

The MoE has three implementations: a general routed-expert chain, an SGLang-style low-batch
chain (`ffn_sgl`, ≤ 8 rows), and the sharded TP variants. Route selection, the shared expert and
the residual are fused into the same kernel where the numerics allow it.

## Speculation

`model.rs` carries four pieces that together form the speculative stack: the MTP drafter (extra
decoder layers stored in the checkpoint, `DSpark`), a low-rank Markov correction over the draft
distribution (`draft_markov.cu`), a confidence head that picks the verify length, and a
CUDA-conditional graph whose branches are the different verify widths. Acceptance sampling runs
on device; the acceptance result also decides which branch of the graph executes next, so the
host is out of the loop for an entire speculative round. Doc `05` shows why a trained
confidence head is what makes that legal, and doc `02` records which of these transplant to
another model family.

## Tensor parallelism

`tp ∈ {4, 8}` with NVLS multicast channels, peer access between every rank pair, and a fused
"GEMM + all-reduce push" arm (`aa_arm`). Row/head/expert/vocab sharding lives in `shard_rows`,
`D2`–`D9` of doc `03`. The important discovery for this port is that **TP=1 is not a supported
configuration**: `per_tok` assertions, the `Sm103Tp4/Tp8` profile gate and the `--tp` harness
assertion all reject it, and several kernels exist only in 4- and 8-rank instantiations. Doc
`03` §H lists every gate. That is the reason this port could not simply pass `--tp 1`.

## Serving

One job on the GPU at a time. Throughput across clients comes from replicas, RDMA KV transfer
between them, and a six-slot KV prefetch window drained by a separate thread. Snapshots are
CUDA-event snapshots of the cache plus the generation counters, double buffered so that the
streaming thread and the decode loop do not stall each other.
