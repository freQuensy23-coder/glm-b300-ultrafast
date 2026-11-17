# 04 — Kernel inventory (`crates/kernels/cu/`)

Scope: all 33 `.cu` files plus the 7 `.cuh` headers. Built by `crates/kernels/build.rs`, which runs `nvcc -cubin -O3 -arch=sm_100a` (override `DSUF_SM`) and `include_bytes!` the cubins (`crates/kernels/build.rs:13-19`, `:47-66`). Files named `variant_*` are compiled **only** for `sm_103a` (`build.rs:44`); `tcmoe_rm.cu`, `gemv_xb.cu` etc. are `#include` wrappers that re-instantiate a base file with a macro flag.

Verdict legend — exactly one per file:
- **A** = architecture-agnostic (reusable for a GQA/MoE model such as GLM)
- **D** = DeepSeek-V4.1-specific (MLA / DSA indexer / engram / hyper-connections / dspark)

All numerics are bf16 activations + fp32 accumulate unless stated. `Meta` (`pos`, `m`, `tokens[16]`) is the per-launch metadata struct re-declared in most files (`attn.cu:3-7`).

---

## 1. File-by-file inventory

### Core GEMM/GEMV family

**`gemv.cu`** (1907 l) — persistent-ring-TMA GEMV + fused epilogue + non-tensor-core fused MoE. Weights are pre-tiled on the host; one CTA owns 64 output columns (`quad = blockIdx.x`, `gemv.cu:156`), K may be split across a cluster with DSMEM all-reduce (`gemv.cu:458-467`, `:196-217`).
Exports: `gemv_{fp8,fp4,bf16,fp8b}_n{1,2}_p{1,2,4}` (24, macro `gemv.cu:533-540`, invoked `:696-699`), `gemv_bf16_persist_p4` (`:694`), `moe_fp4_swiglu_n{1,2}_p4`, `moe_fp4_store_n{1,2}_p{2,4}` (`:1687-1692`), `moe_fp4_fused_n1` (`:1693`), `moe_fp4_fused_gate_n1` (`:1696`), `moe_w2_n1`, `moe_w2_n2` (`:1904-1905`).
Verdict: **A** — the GEMV/GEMV+MoE machinery is generic (K-tiled fp8/fp4/bf16 matvec, `M ≤ 16`, `ops.rs:311`); only the optional epilogue branches are DS-shaped (indexer-q fp4 requant `gemv.cu:493-509`, MLA KV rope on a donor CTA `:135-143`).

**`gemv_xb.cu`** (10 l) — `#define GEMV_XB 1` + `gemv.cu`; adds an extra bf16 activation buffer to `GemvParams` and exports `gemv_bf16_xb_n1_p{1,2,4}` (`gemv_xb.cu:3-10`).
Verdict: **A**.

**`draft_head8.cu`** (173 l) — `#define GEMV_XB` + copy of the persistent fp8-weights/bf16-act GEMV tuned for the MTP draft head.
Exports: `gemv_fp8b_persist_p4` (`draft_head8.cu:173`).
Verdict: **D** (dspark draft head).

**`tcgemv.cu`** (338 l) — tcgen05 (`sm_100a` tensor-core) decode GEMV: 128-column tile per CTA, weights streamed through a TMA tensor map, activations quantized to fp8 in-kernel, `tcgen05.mma.kind::mxf8f6f4.block_scale`, TMEM→register epilogue, cluster DSMEM push-reduce.
Exports: `tc_gemv_fp8` (`tcgemv.cu:33`), `tc_gemv_bf16` (`:244`).
Verdict: **A** — generic fp8/bf16 matvec; the only DS part is the fused MLA-KV rope branch (`tcgemv.cu:221-230`).

**`tcgemv2.cu`** (547 l) — second-generation tcgen05 GEMV with per-layer fused epilogues selected at compile time by `TgF<KIND,VAR>` (`tcgemv2.cu:44-63`): fused prenorm+gamma (`:95-112`), fp8 requant of the *input* (`:115-147`), fused rope, fp4 requant of the indexer q, sqrt-softplus gate, split output (`n_split`/`out2`), partial rows for cross-CTA reduce.
Exports: `tc_gemv2_fp8` (`:541`), `tc_gemv2_bf16` (`:542`), `tc_gemv2_qkv` (`:543`), `tc_gemv2_qb` (`:544`), `tc_gemv2_part` (`:545`), `tc_gemv2_oa` (`:546`), `tc_gemv2_gate` (`:547`). Constraints in `ops.rs:1322-1332`: `1 ≤ m ≤ 8`, `n % 128 == 0`, `k % (cs*128) == 0` (fp8).
Verdict: **A** — reusable core; `V_QB`/`V_OA`/`V_QKV` variants carry MLA/indexer specifics (`iq_head`/`iq_off` re-quant at `:82-98`).

**`tcgemm.cu`** (839 l) — prefill tensor-core GEMM plus dense quant/norm helpers and the DSA indexer on tensor cores.
Exports: `tc_gemm_fp8` (`:20`), `tc_gemm_fp8p` (persistent variant, `:141`), `tc_quant_rows` (`:300`), `tc_row_sumsq` (`:342`), `tc_gemm_bf16` (`:371`), `tc_norm_rows` (`:455`), `idx_score_tc4b` (`:540`, DSA indexer), `idx_score_tc_dec` (`:696`, DSA indexer).
Verdict: **A** for the GEMM/quant/norm half; `idx_score_tc*` are DSA-specific. File verdict: **A** with the two indexer entries carved out.

**`tc.cuh`** (111 l) — tcgen05/TMA/`ldmatrix` primitives: `mbarrier` init/wait, `cp.async.bulk.tensor.{2,3}d`, `tcgen05.alloc/cp.32x128b.warpx4/mma.kind::mxf8f6f4.block_scale/mma.kind::f16/ld/wait`, SMEM descriptors `sdesc_t`, instruction descriptors `idesc_e4m3`/`idesc_e2m1_e4m3`/`idesc_bf16` (`tc.cuh:4-111`).
Verdict: **A** (hardware primitives; requires `sm_100a`+).

**`common.cuh`** (110 l) — dtype conversions (`e4m3x2↔f16/bf16`, `e2m1x2`), `round_scale_pow2` (`:80`), `warp_sum/max`, `mma.sync m16n8k16` fp16/bf16 wrappers (`:97-110`), `sqrt_softplus` (`:17`), `norm_part_sum` (`:26`), PDL `griddepcontrol` (`:14-15`).
Verdict: **A**.

**`quantblocks.cuh`** (58 l) — `quant_blocks8<Load8>`: per-32-element block max, pow2 e8m0 scale (`:42`), e4m3 encode with the ldmatrix-friendly destination swizzle `dst = xs + m*LDS + kb*32 + 2*(c&1) + 4*(c>>1)` (`:44`), scale written to `sa[kb*MT + m]` (`:54`).
Verdict: **A**.

**`rope.cuh`** (132 l) — `RopeParams` (`:16`) and `rope_body` (`:35`): optional RMS-norm with gamma (`:56-71`), interleaved-pair rope over `[rope_off, n)` with `cos/sin[pos*32 + f]` (`:72-83`), optional quantize (fp8 / fp4-e8m0 / fp4-e4m3) of the written row (`:84-124`), and compaction by `ratio` for pooled/compressed rows (`:43-47`).
Verdict: **A** for norm+rope+fp8 quant; the fp4 KV packing shape (`q4 + r4*64 + e0/2`, `sf4[(r4/128)*512 + (r4%32)*16 + ((r4%128)/32)*4 + e0/32]`, `:118-120`) is the MLA-compressed-KV layout.

**`route.cuh`** (57 l) — `route_token` (`:4`): warp-parallel top-k router (12 experts/lane × 32 lanes = 384), optional `sqrt_softplus` of summed-but-unsorted scores (`:15`), bias add, argmax loop with `__reduce_max/min` tie-break, normalized weights × `route_scale` (`:48-54`).
Verdict: **A**.

### Sampling / speculation

**`sample.cu`** (1188 l) — single fused sampler kernel: temperature, block max/log-sum-exp, top-p nucleus (`:940`, `:1104-1131`), philox counter RNG (`sample.cuh:15-29`), vocab sharded over TP ranks (`p.vloc/vpad/offset`, `SampIo:17-39`), optional cluster-wide exchange (`SAMP_CL`, `:824-833`), and speculative multi-draft acceptance against `vq` (per-draft proposal probabilities) with remainder resampling (`RowDraw`, `:361`; `samp_row_draw`).
Exports: `samp_stats` (`:821`).
Verdict: **A** — the draft/verify path is generic speculative decoding, not dspark-specific.

**`sample.cuh`** (32 l) — `SampParams{temp, top_p, seed, step, pad}` (`:4-11`), `philox4x32_10` (`:15`), `rng_bits(sp, purpose, a, b)` with `RNG_DRAFT/ACCEPT/FINAL` purposes (`:13`, `:26`).
Verdict: **A**.

**`draft_markov.cu`** (269 l) — Markov-chain draft over the vocabulary: per-draft low-rank transition dot (`cx·cw`, `:72-79`), `MK_K=256`-wide chain step (`:43`), top-k of 64-bit (value,index) keys (`:46-55`), confidence threshold `theta` (`:30`) feeding the sampler.
Exports: `markov_rounds_ld` (`:57`).
Verdict: **D** (dspark).

**`ops.cu`** (638 l) — misc dense kernels: embedding + HC broadcast (`embed_hc:25`), `hc_mean:39`, rope variants (`rope_quant:53`, `rope_quant4:54`, `rope_quant2:57`), `scale_bf16:62`, engram memory (`engram_gather:77`, `engram_gather_peer:125`, `engram_signal:159`, `engram_apply_peer:168`, `engram_apply:244`), `markov_rounds:318`, speculative meta/accept (`spec_draft_meta:572`, `spec_verify_meta:582`, `spec_accept:591`), `moe_route_tokens:615`, dtype conversions (`rows_i2f:625`, `rows_f2i:633`).
Verdict: **D** as a file (engram + dspark markov/spec dominate), but `rope_quant*`, `scale_bf16`, `moe_route_tokens`, `rows_i2f` are **A**.

### Attention (MLA + DSA)

**`attn.cu`** (1463 l) — hybrid sliding-window + sparse-latent MLA attention. Head geometry `AH=16` heads per CTA, `AD=512` latent dim, `ACH=64` rope dim (`attn.cu:28`). KV slots = 128 window rows then `ktop` selected compressed rows (`:385`, `:399-410`). DSA lightning indexer and top-k selectors live here too.
Exports: `idx_score` (`:44`), `block_max` (`:130`), `topk_select` (`:241`), `compress_pool` (`:254`), `store_hist_f32` (`:322`), `attn_fused` (`:580`), `attn_fused_hg` (`:581`), `attn_prefill_ks16` (`:743`), `topk_kth` (`:807`), `topk_part` (`:834`), `topk_merge` (`:879`), `topk_fast` (`:969`), `topk_cl_{C}_{T}` family (macro `:1460`).
Verdict: **D** (MLA 512-latent + DSA sparse top-k).

**`variant_zzzzzaud.cu`** (327 l) — alternative `attn_fused` body with SMEM aliasing for the cluster receive buffer; same `AttnFParams`.
Exports: `attn_aud_mqx` (`:327`).
Verdict: **D**.

**`variant_zzzzzzzzzzsds.cu`** (366 l) — split-KV flash-decoding attention (`attn_sds_body`, `:129`) with per-split cluster reduction of (m, l) and O.
Exports: `attn_sds_8` (`:366`).
Verdict: **D** (same MLA geometry `AD=512`).

**`variant_topk_cand.cu`** (481 l) — DSA top-k over an explicit candidate list (`topk_clc_body`, `:403`; `topk_old`, `topk_old_8_1024`).
Exports: `topk_clc2_8_1024` (`:472`), `topk_old_8_1024` (`:477`).
Verdict: **D**.

**`variant_zzzzzzzzixc.cu`** (5 l) — `topk_clc2_2_1024` (`:3`): 2-candidate DSA top-k re-instantiation.
Verdict: **D**.

**`variant_zzzzzzzzixs.cu`** (73 l) — `idx_merge` (`:17`): merges split indexer score blocks into the final top-k with 64-bit (score,index) keys.
Verdict: **D** (DSA indexer).

**`variant_zzzzzzzzixt.cu`** (2 l) — `#define TOPK_OLD 1` + `variant_topk_cand.cu` (legacy selector).
Verdict: **D** (DSA top-k).

### MoE

**`tcmoe.cu`** (588 l) — tensor-core MoE for fp4 experts: `moe_perm` builds the expert-sorted row permutation + 128-row tile list (`:19-69`), `moe_gather_quant` gathers those rows, applies prenorm+gamma and writes an fp8 image `xq` `[M*topk][K]` + block-scale image `sf` (`:88-133`), `tc_moe_body` then streams fp4 expert weights through int4 tile descriptors and does w13→SwiGLU (`TM_SWIGLU`) or w2 accumulate (`TM_STORE`), requantizing the SwiGLU output to fp8 (`aq`/`asf`, `:524-529`) for the second GEMM, with a work-stealing split of the tile range across CTAs (`:198-251`).
Exports: `moe_perm` (`:19`), `moe_gather_quant` (`:88`) / `moe_gather_quant_nat` (`:86`, `TCMOE_RM`), `tc_moe_w13` (`:585`), `tc_moe_w2` (`:586`).
Verdict: **A** — routing/permute/grouped-GEMM structure is generic GLM-style MoE; only the expert weight packing is DeepSeek's fp4 layout (`packing.rs:20-52`).

**`tcmoe_rm.cu`** (2 l) — `#define TCMOE_RM 1` + `tcmoe.cu` (row-major expert weight path).
Verdict: **A**.

**`variant_zzzzzzzzzzmoe.cu`** (731 l) — `TCMOE_RM`+`TCMOE_NAT` re-instantiation (`tc_moe_w13_rm2:7`, `tc_moe_w2_rm:8`) plus `tmoe_fused` (`:217`), which runs gate → w13 → SwiGLU → w2 in one launch with `GateParams`/`TmoeParams`.
Verdict: **A** (fused-MoE schedule is portable; the gate MoE-dot layout `MIX=24, HC=4` is HC-aware).

**`variant_zzzzzzzzzzsgl.cu`** (777 l) — SGLang-style fp4 expert GEMM (`sgm_body`, `:170`) with router fused (`router_body`, `:533`); hardcoded hidden size `HID=5120`, `SLICE=576/640`, TP8 slices `320/384` (`:4`).
Exports: `sgl_gemm1` (`:447`), `sgl_gemm2` (`:448`), `sgl_gemm1_tp8` (`:449`), `sgl_gemm2_tp8` (`:450`), `sgl_router_{M}` / `sgl_router_d_{M}` (`:759`, `:769`).
Verdict: **A** with the caveat that `HID=5120` and the 576-wide slice are DeepSeek constants (`variant_zzzzzzzzzzsgl.cu:4`) — parameterize before reuse.

### Communication / collectives

**`comm.cu`** (714 l) — TP all-reduce/gather over NVLink+DSMEM: one-shot push into peer `recv[]` buffers with a 3-stage epoch protocol (`:13-18`), templated TP specializations, and an NVLS multicast variant; modes `CM_SUM`/`CM_GATHER`/`CM_SUM_HCPOST` (`:16`).
Exports: `comm_push` (`:134`), `comm_init` (`:281`), `comm_push2_tp{2,4,8}`, `comm_push2_k1_tp{2,4,8}` (`:534-544`), `comm_init2` (`:544`), `comm_nvls_ucpush` (`:710`), `comm_fill_u32` (`:712`).
Verdict: **A** for the collective; `CM_SUM_HCPOST` post-mix (`:38-40`) is the hyper-connection epilogue and is **D**.

**`variant_zcomm_dec.cu`** (183 l) — decomposed all-reduce (`comm_dec_body`, `:4`): x16-packed bf16 or f32 payloads, `NS`-way splitting, optional fused FFN-side stream.
Exports: `comm_dec_x16` (`:170`), `comm_dec_f32` (`:171`), `comm_dec_attn_{NS}`, `comm_dec_ffn_{NS}` (`:173-174`).
Verdict: **A**.

**`variant_zzzzzzzztp8.cu`** (18 l) — TP8 instantiations of the above (`comm_dec8_*`, `comm_push2_k1_tp8`).
Verdict: **A**.

**`variant_zzzzzzzzzlp.cu`** (433 l) — low-latency all-reduce polling (`comm_lp_body`, `:18`) with inline `ld.relaxed.gpu.v4` arrive-detection.
Exports: `comm_lp_sgl_6_2` (`:429`), `comm_lp_sgl8_{6,3}_2` (`:430-431`), `comm_lp_ffn_{6,3}_2zp` (`:432-433`).
Verdict: **A**.

**`variant_zzzzzzaa.cu`** (373 l) — GEMV3 partial-projection push (`tc_gemv3yp_part:244`, `tc_gemv3yp8_part:247`) and attention-side prefetch pipeline (`comm_dec_attn_pp:372`, `comm_dec8_attn_pp:373`) — compute/collective overlap.
Verdict: **A**.

### Hyper-connections (HC)

**`hcepi.cuh`** (227 l) — HC math: `HC=4` residual streams, `MIX=24` mixing matrices, `NP=26` per-row partial slots (`:4`), per-thread epilogue `hc_epi_thread` (`:15`) folding `pre[HC]` into `x_raw` and accumulating 24 mix dots + 2 sums into `acc[NP]`, block reduce writing `part[(m*nslices+slice)*NP + j]` (`:51-63`), TMA issue of the `[MIX][HC][SLICE]` weights (`:66-96`), `struct HcMix` (`:100`).
Verdict: **D** (hyper-connections).

**`hcpipe.cu`** (244 l) — HC pipeline kernels: `hc_prep` (`:15`), `hc_mix` (`:37`), `hc_side_dots` (`:50`), `hc_side_dots_p` (`:106`), `hc_fn_split` (`:181`), `hc_side_dots_tc` (`:196`).
Verdict: **D**.

**`variant_zzhc.cu`** (120 l) — HC variants: `hc_prep_pd:82`, `hc_prep_d:88`, `embed_prep:94` (fused embed + HC prep).
Verdict: **D**.

**`variant_zzzzzzzzhcm.cu`** (2 l) — `HC_MIX16` re-instantiation of `hcpipe.cu`.
Verdict: **D**.

### Quantization codec / vision

**`kv_fp4.cu`** (72 l) — KV-cache fp4 codec: e2m1 magnitude with the sign folded into code bit 3 (`:32`), per-`G` scales — e4m3 for `G=16` with round-up clamp (`:13-23`) or e8m0 for `G=32` (`:24-28`) — plus a `valid` word cleared by `atomicAnd` when any element would overflow (`:36-38`).
Exports: `kv_fp4_pack16` (`:61`), `kv_fp4_pack32` (`:64`), `kv_fp4_unpack16` (`:67`), `kv_fp4_unpack32` (`:70`).
Verdict: **A** (generic lossless-sign fp4 codec; suits GQA KV too).

**`vision.cu`** (75 l) — vision-tower pieces: RMS-norm with f32 weight (`:4`), linear+residual+gelu finish (`:19`), SwiGLU (`:27`), 2-D rope from a precomputed cos/sin table (`:35`), patch unfold (`:46`), patch pack with start/newline/end embeddings (`:53`), embedding scatter (`:61`), MoE route for image tokens (`:67`).
Exports: `vision_norm` (`:4`), `vision_linear_finish` (`:19`), `vision_swiglu` (`:27`), `vision_rope` (`:35`), `vision_unfold` (`:46`), `vision_pack` (`:53`), `vision_embed` (`:61`), `vision_route` (`:67`).
Verdict: **A**.

**`variant_zzzzzzzsamp.cu`** (2 l) — `SAMP_CL` re-instantiation of `sample.cu`.
Verdict: **A**.

### GEMM3 variants (weight-box pipelining)

**`variant_zzzztg3.cu`** (549 l) — `tg3_body` (`:114`): tcgen05 GEMV with up-to-`kt` weight boxes per CTA, uniform/chain MMA issue, `TG3_*` flags (`:3`), relative-TMEM allocation.
Verdict: **A**.

**`variant_zzzzzy3.cu`** (6 l) — one extra entry `tc_gemv3yr_qb` (`:5`).
Verdict: **A** (entry only; the `V_QB` epilogue carries MLA quant).

---

## 2. Reusable building blocks for a single-GPU GLM port

Everything below is `sm_100a`+ (tcgen05) or plain `mma.sync`, and none of it reads MLA / DSA / engram / HC state. Weight layouts are produced by the Rust prep helpers cited alongside — port those first, they define the contract.

### 2.1 Weight-tiled GEMV (`mma.sync`, no tensor-core-map needed)

| kernel | file:line | signature |
|---|---|---|
| `gemv_fp8_n{1,2}_p{1,2,4}` | `gemv.cu:534-540`, entries `:696` | `extern "C" __global__ void __launch_bounds__((4*PH+1)*32, 2) name(const __grid_constant__ GemvParams p)` |
| `gemv_fp4_n{1,2}_p{1,2,4}` | `gemv.cu:697` | same |
| `gemv_bf16_n{1,2}_p{1,2,4}` | `gemv.cu:698` | same |
| `gemv_fp8b_n{1,2}_p{1,2,4}` (fp8 weights dequantized to bf16 in-kernel) | `gemv.cu:699`, dequant `:397-414` | same |
| `gemv_bf16_persist_p4` | `gemv.cu:694` | `(const __grid_constant__ GemvParams p)`, 17 warps |

`GemvParams` (`gemv.cu:19-71`): `w, ws, x, out, M, N, K, ldx, ldo, out_f32, alpha, stage_steps, plan, w_stride, mode, swiglu_limit, prenorm_w/eps, norm_part, norm_nslices, ksplit_slots, xq/xsq, oq/osq, ...`. Invoked from `ops.rs:331` (`gemv`); host-side guard `M ≤ 16` via `mt = 8/16` (`ops.rs:311`).

Input layout — **fp8 weights** (`Fmt<FMT_FP8>::TK=32`, `STEP=2048`, `gemv.cu:75`):
- Weights are *not* row-major. `tile_fp8` (`crates/model/src/formats.rs:96-118`) emits, per 64-column block `q`, `steps = k/32` steps of 2048 B; step offset `((rg/4)*steps + ks)*2048 + (rg%4)*512` (`formats.rs:92-94`), i.e. one step = 32(K)×64(N) bytes, split into four 512-B sub-blocks for rows 0-15/16-31/32-47/48-63.
- Within a 512-B sub-block, lane `L` owns 16 B at `base + lane*16`: rows `g = L/4` and `g+8` (`h`), K positions `{2t,2t+1} ∪ {8+2t,9+2t} ∪ {16+2t,17+2t} ∪ {24+2t,25+2t}` with `t = L%4` (`formats.rs:105-112`) — the `ldmatrix.m8n8.x4` A-fragment order consumed directly as MMA A operand (`gemv.cu:415-423`).
- Scales are a **separate** tensor: one e8m0 byte per (32-row group, 32-K block), indexed `ws[(quad*2+sr)*(K/32) + k0/32 + kb]` (`gemv.cu:191-193`) → `[N/32][K/32]` u8.

Input layout — **fp4 weights** (`TK=64`, `STEP=2048+128`, `gemv.cu:76`; `FP4_STEP` `formats.rs:145`):
- One step = 64(K)×64(N) nibbles = 2048 B data + 128 B scales (`tile_fp4`, `formats.rs:147-180`); data sub-block `base + lane*16 + j*4` with the same lane permutation (`:156-167`).
- Scales: 32 B per 16-row group at `+2048 + (rg%4)*32`, each byte-group of 4 = `[r0,k0],[r0,k1],[r1,k0],[r1,k1]` (`formats.rs:168-176`), read as `sg[2]`(row `g`) / `sh[2]`(row `g+8`) at `gemv.cu:371-373`.

Input layout — **bf16 weights**: `tile_bf16`, `formats.rs:182-206`, same 2048-B step/512-B sub-block scheme with `TK=16`.
Activations: bf16 `[M][ldx]` row-major (`gemv.cu:192-200`), copied to SMEM with stride `LDS = KS+32` (fp8/fp4) or `KS+8` (bf16) (`gemv.cu:167-168`).
Epilogues available: `MODE_STORE/SWIGLU/ATOMIC/SSP/PARTIAL` (`gemv.cu:7`), fused rope (`:510-517`), fused prenorm+gamma (`:589-607`).

### 2.2 tcgen05 decode GEMV (tensor-core, TMA)

| kernel | file:line | signature |
|---|---|---|
| `tc_gemv_fp8` | `tcgemv.cu:33` | `void tc_gemv_fp8(const __grid_constant__ TvParams p)` — 128 thr, 1 CTA/SM |
| `tc_gemv_bf16` | `tcgemv.cu:244` | `void tc_gemv_bf16(const __grid_constant__ TvParams p)` |
| `tc_gemv2_fp8` | `tcgemv2.cu:541` | `void tc_gemv2_fp8(const __grid_constant__ Tg2Params p)` |
| `tc_gemv2_bf16` | `tcgemv2.cu:542` | `void tc_gemv2_bf16(const __grid_constant__ Tg2Params p)` |
| `tc_gemv2_qkv` / `_oa` / `_gate` / `_part` | `tcgemv2.cu:543-547` | same signature, different fused epilogue |

`Tg2Params` is size-asserted at 352 B (`tcgemv2.cu:39`), mirrored in `ops.rs`. Guards: `1 ≤ m ≤ 8`, `n % 128 == 0`, `k % (cs*128) == 0` (fp8) / `(cs*64)` (bf16), `cs ≤ 16`, `kt ≤ 8/10` (`ops.rs:1322-1325`).

Input layout:
- **Weights stay row-major**: `[K rows][N cols]` with row stride `kp` (K padded to 128), no pre-swizzle. A TMA descriptor with 128-B swizzle is built by `tensor_map_u8_sw128(wp, kp, n, kp, box_rows)` (`crates/cuda/src/lib.rs:741`; used `weights.rs:204`), box = `128 K × nt N` with `nt ∈ {64,128,256}` (`weights.rs:185`). BF16 uses `tensor_map_bf16_sw128` with box `64 K × nt N` (`lib.rs:787`, `weights.rs:222`).
- **Scale factors**: `tile_sf_blocks` (`formats.rs:216-228`) → `[n/128][k/128][512]` u8, index `(t*kti_blocks + kti)*512 + (rr%32)*16 + (rr/32)*4 + kc`; the kernel streams 512 B per 128×128 tile (`tcgemv.cu:85`, `:157-159`) and feeds them to `tcgen05.cp...32x128b.warpx4` → exactly the mxf8 block-scale TMEM layout. Pad blocks are 127 (`weights.rs:188`).
- **Activations** are bf16 `[M][ldx]`; the kernel RMS-norms and quantizes them per 32-K block with a pow2 e8m0 scale, writing the e4m3 image in tcgen05 B-operand order `sB + t*1024 + m*16 + (kc*2+h2)*128` + `sSFB[t*512 + m*16 + kc]` (`tcgemv.cu:115-147`). Pre-quantized inputs are accepted via `xq/xsq` (`tcgemv2.cu:31-32`, `:165`).
- K may be split across a cluster (`K/(cs*128)` tiles per rank) with an f32 DSMEM push-reduce (`tcgemv.cu:196-217`), or emitted as partial rows (`partial`, `:183-193`).

### 2.3 tcgen05 prefill GEMM

| kernel | file:line | signature |
|---|---|---|
| `tc_gemm_fp8` | `tcgemm.cu:20` | `void tc_gemm_fp8(const __grid_constant__ TgParams p)`, 128 thr, CTA tile `128 M × nt N` |
| `tc_gemm_fp8p` | `tcgemm.cu:141` | persistent variant, same params |
| `tc_gemm_bf16` | `tcgemm.cu:371` | `void tc_gemm_bf16(const __grid_constant__ TbParams p)` (`kind::f16`, `idesc_bf16` `tc.cuh:104`) |

`TgParams` (`tcgemm.cu:3-16`): `ta, tb, sfa, sfb, out, ldo, out_f32, M, N, K, nt, alpha, out2, ldo2, n_split, swiglu, swiglu_limit`. SFA/SFB are the same `tile_sf_blocks` layout; A (activations) is quantized by `tc_quant_rows` (`:300`) or `tc_row_sumsq`+`tc_norm_rows` when a prenorm is wanted.

### 2.4 Norm / quant helpers

| kernel | file:line | signature | layout |
|---|---|---|---|
| `tc_row_sumsq` | `tcgemm.cu:342` | `(const bf16* x, int ldx, int M, int K, float* part)` | writes `part[row*26 + 25]` — 26-f32 slots per row, sum in slot 25 |
| `tc_norm_rows` | `tcgemm.cu:455` | `(const bf16* x, int ldx, int M, int K, const bf16* pw, const float* part, int nslices, float eps, bf16* y)` | reads `part[row*nslices*26 + i*26 + 25]` (`common.cuh:26-37`); `y` packed `[M][K]` |
| `tc_quant_rows` | `tcgemm.cu:300` | `void tc_quant_rows(const __grid_constant__ TqParams p)` | e4m3 rows + e8m0 scales |
| `quant_blocks8<Load8>` | `quantblocks.cuh:8` | `quant_blocks8(warp, nwarps, lane, MT, n, nkb, uint16_t* xs, int LDS, float* sa, Load8 load8, float dqmul)` | dest swizzle `xs + m*LDS + kb*32 + 2*(c&1) + 4*(c>>1)` (`:44`); scale `sa[kb*MT + m]` (`:54`) — pairs with the MMA B-operand reads at `gemv.cu:405-408` |
| `cnorm` in `rope_body` | `rope.cuh:56-71` | in-kernel RMS-norm before rope, gamma `norm_w` | needs `rows_per_token` rows per token |

### 2.5 RoPE

| kernel | file:line | signature |
|---|---|---|
| `rope_body` | `rope.cuh:35` | `rope_body(const int* meta, const RopeParams& p, int row_, int lt, int grp)` — 64 thr per row-group, 8 elems/lane |
| `rope_quant` | `ops.cu:53` | `void rope_quant(const Meta* meta, const RopeParams p)` (64 thr) |
| `rope_quant4` | `ops.cu:54` | same, 4 row-groups, 256 thr |
| `rope_quant2` | `ops.cu:57` | two independent `RopeParams` in one launch (q and kv) |

Layout: `x` is bf16 `[rows][ldx]` with `row = token*rows_per_token + head` (`rope.cuh:39`); cos/sin are fp32 tables indexed `[pos*32 + f]` where `f` is the pair index, so max rotary dim per row is 64 and `rope_off` sets the first rotated element (`rope.cuh:74-79`). Fused quant modes `Q_FP8/Q_FP4_E8M0/Q_FP4_E4M3_16` (`rope.cuh:7`, `:84-124`); `ratio` compacts pooled rows (`:43-47`).

### 2.6 MoE stack

| kernel | file:line | signature |
|---|---|---|
| `moe_perm` | `tcmoe.cu:19` | `void moe_perm(const __grid_constant__ MpParams p)` — 1024 thr; emits `row_pair`, `pair_row`, `row_w`, `row_tile = (tile<<8)|row_in_tile` |
| `moe_gather_quant` | `tcmoe.cu:88` | `void moe_gather_quant(const __grid_constant__ MqParams p)` — gather + prenorm + fp8 quant per 32-K block |
| `tc_moe_w13` | `tcmoe.cu:585` | `void tc_moe_w13(const __grid_constant__ TmParams p)` — w1/w3 + SwiGLU |
| `tc_moe_w2` | `tcmoe.cu:586` | `void tc_moe_w2(const __grid_constant__ TmParams p)` — down-proj accumulate |
| `tc_moe_w13_rm2` / `tc_moe_w2_rm` | `variant_zzzzzzzzzzmoe.cu:7-8` | row-major-weight variants |
| `tmoe_fused` | `variant_zzzzzzzzzzmoe.cu:217` | gate + w13 + SwiGLU + w2 in one launch |
| `moe_fp4_swiglu_n1_p4` … | `gemv.cu:1687-1692` | `mma.sync` fused MoE (no tcgen05) |
| `moe_w2_n1` / `_n2` | `gemv.cu:1904-1905` | down-proj with `MoePlan` (`gemv.cu:10-17`, ≤97 slots × ≤16 rows) |
| `route_token` | `route.cuh:4` | `route_token(m, lane, scores, bias, n_exp, topk, route_scale, sel, selw, route_ids, route_w, nsum)` |
| `moe_route_tokens` | `ops.cu:615` | block-wide variant: `(M, scores, bias, n_exp, topk, route_scale, ids, w)` |
| `sgl_gemm1/gemm2` | `variant_zzzzzzzzzzsgl.cu:447-450` | fp4 experts in the SGLang layout (`packing.rs:85-131`) |

MoE input layout: activation rows are bf16 `[M][ldx]`. `moe_gather_quant` permutes them into `xq` = e4m3 `[M*topk][K]` (row stride `K`, one 32-K block per thread) with a block-scale image `sf[tile][K/128][512]` indexed `(n%32)*16 + (n/32)*4 + kb%4` (`tcmoe.cu:127-131`); `tc_moe_body` then produces the *intermediate* fp8 image `aq`/`asf` in the same 512-B layout for the w2 stage (`:524-529`, `:563-567`). Expert weights are the fp4 pack of `packing.rs:20-52` (64-B `pair` interleave, 4 scale bytes per 8-row group) plus a per-128-row tile list `Tiles[t] = int4(expert, row_offset_in_expert, row_count, 0)` written by `moe_perm` (`tcmoe.cu:58`). `MoePlan` for the `gemv.cu` path is built by `route_plan` inside the kernel (`gemv.cu:715`, called `:1210`).

### 2.7 Sampling

| kernel | file:line | signature |
|---|---|---|
| `samp_stats` | `sample.cu:821` | `void samp_stats(const Meta* meta, const __grid_constant__ SampIo p)` — 1024 thr, one CTA per row (or per row per cluster rank) |

`SampIo` (`sample.cu:17-39`): `logits` fp32 `[m][vpad]`, `vloc` (valid vocab on this rank), `vpad` (row stride), `offset` (global vocab offset of this shard), `drafts`, `vq` (`[m][vpad]` draft probabilities), `sp` (`SampParams`), `recv[8]`, `tp`, `rank`, `seq` (epoch), `spart` (optional precomputed per-quad statistics, stride `nquads*20`, `:862`), `qn`/`qstat`. Semantics: temperature → top-p cut (`:940`) → nucleus CDF (`:1104-1131`) → philox draw per (row, purpose) (`sample.cuh:26`), with speculative accept against `vq` and remainder resampling. Reusable as-is for GLM with `tp = 1`.

### 2.8 Primitives worth copying verbatim

- `mma.sync m16n8k16` bf16/f16 wrappers — `common.cuh:97-110`.
- e4m3/e2m1 conversions, `round_scale_pow2`, `e8m0_to_f32`, `f16x2_dequant` — `common.cuh:39-84`.
- `sqrt_softplus` activation used by both the MoE gate and the router scale — `common.cuh:17-24` (matches `route.cuh:15`).
- SwiGLU with clamp: `gemv.cu:474-483` (`swiglu_limit`); `tc_moe_body<TM_SWIGLU>` (`tcmoe.cu:468`); `vision_swiglu` (`vision.cu:27`).
- tcgen05/TMA wrappers — `tc.cuh:4-111` (mbarrier, `tma2d/3d`, `tmalloc`, `tcmma*`, `tccpsf`, `tcld32`, `u32 sdesc_t`).
- fp4 KV codec — `kv_fp4.cu:61-71` with the sign-in-bit-3 trick (`:32`, `:58`).
- philox — `sample.cuh:15-29`.
- PDL `griddepcontrol.wait/launch_dependents` — `common.cuh:14-15`; every kernel starts with `pdl_wait(); pdl_launch();` and host launches set `.pdl(true)` (e.g. `ops.rs:894`).

### 2.9 Not reusable for GLM

- `attn.cu`, `variant_zzzzzaud.cu`, `variant_zzzzzzzzzzsds.cu`, `variant_zzzzzaud`'s smem aliasing: hardcoded `AD=512` / `ACH=64` / `AH=16` (`attn.cu:28`) = MLA latent + 64-dim rope, plus the DSA slot layout `128 window + ktop compressed` (`attn.cu:385`).
- All `idx_score*` / `topk_*` / `compress_pool` / `store_hist_f32` (DSA indexer) — `attn.cu:44-330`, `tcgemm.cu:540/696`, `variant_topk_cand.cu`, `variant_zzzzzzzzix*`.
- `hcepi.cuh`, `hcpipe.cu`, `variant_zzhc.cu`, `variant_zzzzzzzzhcm.cu`, `CM_SUM_HCPOST` in `comm.cu:16` — hyper-connections (`HC=4`, `MIX=24`).
- `engram_*` in `ops.cu:77-317`, `markov_rounds*` (`ops.cu:318`, `draft_markov.cu`), `spec_*` (`ops.cu:572-614`), `gemv.cu:1693/1696` fused gate+draft — engram/dspark.
- The `iq_*` (indexer-q fp4 requant) and `kvr` (MLA KV rope on a separate CTA) epilogue branches — `gemv.cu:493-509`, `tcgemv.cu:34-42`, `tcgemv2.cu:82-98`.

### 2.10 Port checklist

1. Re-implement `tile_fp8` / `tile_fp4` / `tile_bf16` / `tile_sf_blocks` (`formats.rs:96-228`) or ship weights pre-tiled — these define every layout above.
2. Build TMA descriptors with `tensor_map_u8_sw128` / `tensor_map_bf16_sw128` (`cuda/src/lib.rs:741`, `:787`), boxes 128(K)×nt(N) fp8 and 64(K)×nt(N) bf16.
3. Keep activations bf16 `[M][ldx]`, `M ≤ 8` for tcgen05 decode paths (`ops.rs:1119`, `:1322`), `≤ 16` for the `gemv.cu` `mma.sync` paths.
4. Write a new GQA/flash-decode attention — nothing in `attn.cu` survives the geometry change; the reusable attention-adjacent pieces are only `rope_body` and the fp4 KV codec.
5. Replace the DeepSeek router bias semantics if needed: `route_token` applies `scores + bias` then normalizes top-k (`route.cuh:22-50`), and `sqrt_softplus` is applied to summed-but-unsorted expert scores only when `nsum > 1` (`route.cuh:12-15`).
