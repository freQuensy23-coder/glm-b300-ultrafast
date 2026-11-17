# Coverage of `basetenlabs/deepseek-ultrafast` for single-GPU GLM serving

Scope of the port: one B300 (sm_103a), batch size 1, GLM-5.3-Flash text path. Anything below is
judged against that target only — multi-GPU, multimodal and DeepSeek-specific mechanisms are
listed with the reason they are out of scope rather than silently dropped.

Kernel inventory was taken from the compiled cubins (`strings` over each `.cubin`), not from
source greps, so `#ifdef`-gated variants are counted.

## Adopted as-is

| Upstream | Used for |
|---|---|
| `tcgemv.cu` | first-generation tensor-core decode GEMV (`tc_gemv_fp8`, `tc_gemv_bf16`) |
| `tcgemv2.cu` | gen-2 decode GEMV, now the default path (`GLM_TC2=0` pins gen 1) |
| `tcgemm.cu` | prefill GEMM |
| `tcmoe.cu` via `variant_zzzzzzzzzzmoe.cu` | routed MoE; that variant *is* `TCMOE_RM + TCMOE_NAT`, so `tcmoe_rm.cu` is a subset of what is already built |
| `gemv.cu` | `mma.sync` fp8/fp4/bf16 GEMV family, compiled and registered as the non-tc fallback |
| `tc.cuh`, `rope.cuh`, `quantblocks.cuh`, `route.cuh`, `sample.cuh`, `common.cuh` | descriptors, rope, quantisation and reduction helpers |
| `kv_fp4.cu` | packing helpers reused as reference for the fp4 KV layout |

## Reimplemented for GLM (same mechanism, GLM shapes)

| Upstream | Port |
|---|---|
| `attn.cu` | `glm_attn_decode_split` / `glm_attn_prefill` with GQA, QK-norm, partial RoPE, split-KV, tiled causal prefill, `combine` |
| `sample.cu` | `glm_argmax`, `glm_sample`, `glm_spec_accept`, `glm_spec_pick`, `glm_build_rows`, position staging |
| `ops.cu` | `glm_rmsnorm(_heads)`, `glm_rope_qk`, `glm_swiglu`, `glm_embed`, `glm_kv_append(_q8/_q4)`, `glm_moe_*`, residual/bias/cat/cast |
| `gemv.cu` extras | own fp8/fp4/bf16 FMA GEMV with `cp.async` staging, row slicing at 16 |
| `variant_zzzzzzzzzzsgl.cu` | `glm_moe_sgl`: same <=8-row fused expert chain, without the hard-coded `HID=5120/SLICE=576` template |

## Out of scope, with reason

| Upstream | Reason |
|---|---|
| `comm.cu`, `variant_zcomm_dec.cu`, `variant_zzzzzzzzixc/ixs/ixt.cu`, `variant_zzzzzzzzzlp.cu`, `variant_zzzzzzzzzztp8.cu` | NVLS / tensor-parallel communication between ranks; single GPU has no ranks |
| `hcpipe.cu`, `variant_zzhc.cu`, `variant_zzzzzzzzhcm.cu`, `ops.cu::embed_hc/hc_mean` | DeepSeek hyper-connections; GLM has plain residual streams |
| `vision.cu` | multimodal vision tower; the GLM-5.3-Flash text checkpoint has no vision weights |
| `draft_head8.cu` | persistent drafter that reads eight draft heads from the checkpoint; GLM ships one MTP layer |
| `draft_markov.cu` | Markov-table drafter; no such tensors in a GLM checkpoint |
| `ops.cu::engram_*` | retrieval-memory hooks tied to the DeepSeek serving stack |
| `variant_topk_cand.cu`, `variant_zzzzzzzsamp`/`sds`/`aud`/`aa`/`y3` | serving-side sampling/audit variants of the upstream harness; the port samples on device inside its own round |

## Still open

- `gemv_xb.cu` — FMA GEMV reading a pre-quantised activation buffer. The tensor-core path already
  consumes a pre-quantised image, so this only helps the non-tc fallback; not ported.
- `kv_fp4.cu` packing kernels (16/32 groups with a validity mask) — the port has its own per-32
  e8m0 fp4 KV packing; the mask only matters for partially filled prestage buffers.
- `ops.cu::rope_quant` — fused rope + quantise for prefill; the port ropes and quantises in two
  passes.
