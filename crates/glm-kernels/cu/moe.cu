// MoE: router (sigmoid + group-limited top-k + bias), routing lists, expert FFN.
//
// The routed experts are evaluated in two deterministic passes:
//   glm_moe_gate_up -> h[(token,slot), inter] = silu(gate) * up
//   glm_moe_down    -> part[(token,slot), dim]
//   glm_moe_reduce  -> y[token, dim] = residual + sum_slot w * part
// This avoids float atomics, so a rerun reproduces the same logits.
#include "common.cuh"

// One CTA per token row. Scores are sigmoid (GLM/DeepSeek MoE) or softmax.
extern "C" __global__ void glm_router(const bf16* __restrict__ x, const uint8_t* __restrict__ gw,
                                      const float* __restrict__ gsf, const float* __restrict__ bias,
                                      int rows, int dim, int ne, int topk, int n_group, int topk_group, float scale,
                                      int* __restrict__ ids, float* __restrict__ wts, float* __restrict__ gscore,
                                      int sigmoid) {
    extern __shared__ char smem[];
    float* s = (float*)smem;   // ne scores
    float* red = s + ne;       // 8 reduction slots
    int* sel = (int*)(red + 8); // topk selected experts
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nkb = dim >> 7;
    const bf16* xr = x + (size_t)row * dim;

    // scores
    for (int e = tid; e < ne; e += blockDim.x) {
        const uint8_t* wr = gw + (size_t)e * dim;
        float acc = 0.f;
        for (int kb = 0; kb < nkb; ++kb) {
            const float sc = gsf[(size_t)(e >> 7) * nkb + kb];
            float part = 0.f;
            for (int j = 0; j < 128; ++j) part = fmaf(fp8f(wr[(kb << 7) + j]), b2f(xr[(kb << 7) + j]), part);
            acc = fmaf(sc, part, acc);
        }
        s[e] = sigmoid ? (1.f / (1.f + __expf(-acc))) : acc;
    }
    __syncthreads();
    if (!sigmoid) {
        float mx = -INFINITY;
        for (int e = tid; e < ne; e += blockDim.x) mx = fmaxf(mx, s[e]);
        mx = block_max(mx, red);
        float sum = 0.f;
        for (int e = tid; e < ne; e += blockDim.x) {
            s[e] = __expf(s[e] - mx);
            sum += s[e];
        }
        sum = block_sum(sum, red);
        for (int e = tid; e < ne; e += blockDim.x) s[e] /= sum;
        __syncthreads();
    }
    if (tid != 0) return;
    // bias only steers selection, it is not part of the mixing weight
    const int gsz = ne / n_group;
    // group scores: sum of the two largest biased scores inside each group
    for (int g = 0; g < n_group; ++g) {
        float a = -INFINITY, b = -INFINITY;
        for (int e = g * gsz; e < (g + 1) * gsz; ++e) {
            float v = s[e] + (bias ? bias[e] : 0.f);
            if (v > a) {
                b = a;
                a = v;
            } else if (v > b) {
                b = v;
            }
        }
        gscore[g] = a + b;
    }
    for (int i = 0; i < topk; ++i) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < ne; ++e) {
            const int g = e / gsz;
            if (n_group > 1 && topk_group < n_group) {
                // group filter: keep only the topk_group highest groups
                int rank = 0;
                for (int q = 0; q < n_group; ++q)
                    if (gscore[q] > gscore[g]) ++rank;
                if (rank >= topk_group) continue;
            }
            bool used = false;
            for (int j = 0; j < i; ++j)
                if (sel[j] == e) {
                    used = true;
                    break;
                }
            if (used) continue;
            // the balancing bias ranks experts; the mixing weight below uses the unbiased score
            float sv = s[e] + (bias ? bias[e] : 0.f);
            if (sv < 0.f) sv = -INFINITY;
            if (sv > bv) {
                bv = sv;
                best = e;
            }
        }
        if (best < 0) best = 0;
        sel[i] = best;
    }
    float sum = 0.f;
    for (int i = 0; i < topk; ++i) sum += s[sel[i]];
    for (int i = 0; i < topk; ++i) {
        ids[row * topk + i] = sel[i];
        wts[row * topk + i] = (sum > 0.f ? s[sel[i]] / sum : 0.f) * scale;
    }
}

// Build per-expert token lists: offsets[ne+1] and idx[rows*topk] holding row*topk+slot.
extern "C" __global__ void glm_route_lists(const int* __restrict__ ids, int rows, int topk, int ne,
                                           int* __restrict__ offsets, int* __restrict__ idx,
                                           int* __restrict__ cursor) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int e = 0; e <= ne; ++e) offsets[e] = 0;
    for (int i = 0; i < rows * topk; ++i) offsets[ids[i] + 1]++;
    for (int e = 1; e <= ne; ++e) offsets[e] += offsets[e - 1];
    for (int e = 0; e <= ne; ++e) cursor[e] = offsets[e];
    for (int i = 0; i < rows * topk; ++i) {
        const int e = ids[i];
        idx[cursor[e]++] = i;
    }
}

// h[p, inter] = silu(gate) * up for every token p assigned to expert blockIdx.x.
// Dynamic shared memory holds one activation row (dim floats).
extern "C" __global__ void glm_moe_gate_up(bf16* __restrict__ h, const bf16* __restrict__ x,
                                           const uint8_t* __restrict__ w13, const float* __restrict__ sf13,
                                           const int* __restrict__ idx, const int* __restrict__ off, int topk,
                                           int dim, int inter) {
    extern __shared__ char smem[];
    float* sx = (float*)smem;
    const int e = blockIdx.x;
    const int t0 = blockIdx.y * blockDim.x;
    const int row = t0 + threadIdx.x; // grid.y * blockDim covers exactly `inter`
    const int nkb = dim >> 7;
    const int nexp_row = inter; // expert weight rows: [2*inter, dim]
    // experts are packed back to back: [n_experts][2*inter, dim]
    const uint8_t* w13e = w13 + (size_t)e * 2 * inter * dim;
    const float* sf13e = sf13 + (size_t)e * (2 * inter / 128) * nkb;
    for (int p = off[e]; p < off[e + 1]; ++p) {
        const int tok = idx[p] / topk;
        const bf16* xr = x + (size_t)tok * dim;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) sx[i] = b2f(xr[i]);
        __syncthreads();
        if (row >= inter) {
            __syncthreads();
            continue;
        }
        const uint8_t* wg = w13e + (size_t)row * dim;
        const uint8_t* wu = w13e + (size_t)(nexp_row + row) * dim;
        float ag = 0.f, au = 0.f;
        for (int kb = 0; kb < nkb; ++kb) {
            const float sg = sf13e[(size_t)(row >> 7) * nkb + kb];
            const float su = sf13e[(size_t)((nexp_row + row) >> 7) * nkb + kb];
            float pg = 0.f, pu = 0.f;
            for (int j = 0; j < 128; ++j) {
                const float a = sx[(kb << 7) + j];
                pg = fmaf(fp8f(wg[(kb << 7) + j]), a, pg);
                pu = fmaf(fp8f(wu[(kb << 7) + j]), a, pu);
            }
            ag = fmaf(sg, pg, ag);
            au = fmaf(su, pu, au);
        }
        h[(size_t)p * inter + row] = f2b(silu(ag) * au);
        __syncthreads();
    }
}

// part[p, dim] = down_proj(h[p, :]) for the expert's token list.
extern "C" __global__ void glm_moe_down(float* __restrict__ part, const bf16* __restrict__ h,
                                        const uint8_t* __restrict__ wd, const float* __restrict__ sfd,
                                        const int* __restrict__ idx, const int* __restrict__ off, int inter,
                                        int dim) {
    const int e = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x; // output dim row
    if (row >= dim) return;
    const int nkb = inter >> 7;
    // experts are packed back to back: [n_experts][dim, inter]
    const uint8_t* wr = wd + (size_t)e * dim * inter + (size_t)row * inter;
    const float* sfde = sfd + (size_t)e * (dim / 128) * nkb;
    for (int p = off[e]; p < off[e + 1]; ++p) {
        const bf16* hp = h + (size_t)p * inter;
        float acc = 0.f;
        for (int kb = 0; kb < nkb; ++kb) {
            const float sc = sfde[(size_t)(row >> 7) * nkb + kb];
            float dot = 0.f;
            for (int j = 0; j < 128; ++j) dot = fmaf(fp8f(wr[(kb << 7) + j]), b2f(hp[(kb << 7) + j]), dot);
            acc = fmaf(sc, dot, acc);
        }
        part[(size_t)p * dim + row] = acc;
    }
}

// y[token] = residual[token] + sum_slot wts * part[(token,slot)]
extern "C" __global__ void glm_moe_reduce(bf16* __restrict__ y, const float* __restrict__ part,
                                          const float* __restrict__ wts, const bf16* __restrict__ residual, int rows,
                                          int topk, int dim) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * dim; i += gridDim.x * blockDim.x) {
        const int r = i / dim, d = i % dim;
        float acc = 0.f;
        for (int s = 0; s < topk; ++s) acc = fmaf(wts[r * topk + s], part[(size_t)(r * topk + s) * dim + d], acc);
        y[i] = f2b(b2f(residual[i]) + acc);
    }
}

// Epilogue for the tcgen05 MoE path: that path writes one row per (token, slot) pair in the
// gathered order, so the weighted sum runs over `pair_row` instead of the per-expert buckets.
// `out` and `resid` are bf16 rows (the residual stream); ypair is f32 per (token, slot).
extern "C" __global__ void glm_moe_reduce_pair(bf16* out, const float* ypair, const int* pair_row,
                                               const float* w, const bf16* resid, int m, int topk, int dim) {
    const int i = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= m || c >= dim) return;
    float acc = 0.0f;
    for (int k = 0; k < topk; ++k) {
        const int slot = i * topk + k;
        const int r = pair_row[slot];
        acc += ypair[(size_t)r * dim + c] * w[slot];
    }
    out[(size_t)i * dim + c] = f2b(b2f(resid[(size_t)i * dim + c]) + acc);
}

// Short-chain MoE for a handful of rows: one CTA per (token, slot) computes both projections of
// its expert end to end. No gather image, no tile list, no bucketing — at <= 8 rows the launch
// count is what costs, not the arithmetic, so the whole expert fits in one block.
extern "C" __global__ void glm_moe_sgl(float* __restrict__ out, float* __restrict__ scratch,
                                       const bf16* __restrict__ x, const uint8_t* __restrict__ w13,
                                       const float* __restrict__ sf13, const uint8_t* __restrict__ w2,
                                       const float* __restrict__ sf2, const int* __restrict__ ids,
                                       const float* __restrict__ wts, int m, int topk, int dim, int inter) {
    const int r = blockIdx.x;
    if (r >= m * topk) return;
    const int tok = r / topk, slot = r - tok * topk;
    const int e = ids[r];
    const int tid = threadIdx.x;

    extern __shared__ char smem[];
    float* sacc = (float*)smem;              // 2*inter: gate rows then up rows
    float* sact = sacc + 2 * inter;          // inter: swiglu output
    float* sx = sact + inter;                // dim: the token row

    const int nkb = dim >> 7;
    for (int i = tid; i < dim; i += blockDim.x) sx[i] = b2f(x[(size_t)tok * dim + i]);
    __syncthreads();

    const size_t w13e = (size_t)e * 2 * inter * dim;
    const size_t sf13e = (size_t)e * 2 * (inter / 128) * nkb;
    for (int j = tid; j < 2 * inter; j += blockDim.x) {
        const uint8_t* wr = w13 + w13e + (size_t)j * dim;
        const float* sr = sf13 + sf13e + (size_t)(j / 128) * nkb;
        float acc = 0.f;
        for (int kb = 0; kb < nkb; ++kb) {
            float a = 0.f;
            const uint4* v = (const uint4*)(wr + (kb << 7));
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const uint4 q = v[c];
                const uint8_t* b = (const uint8_t*)&q;
                for (int kk = 0; kk < 16; ++kk) {
                    a = fmaf(sx[(kb << 7) + c * 16 + kk], __half2float(__nv_cvt_fp8_to_halfraw(b[kk], __NV_E4M3)), a);
                }
            }
            acc = fmaf(a, sr[kb], acc);
        }
        sacc[j] = acc;
    }
    __syncthreads();
    for (int j = tid; j < inter; j += blockDim.x) {
        const float g = sacc[j], u = sacc[inter + j];
        sact[j] = (g / (1.f + __expf(-g))) * u;
    }
    __syncthreads();

    const size_t w2e = (size_t)e * dim * inter;
    const size_t sf2e = (size_t)e * (dim / 128) * (inter / 128);
    const float w = wts[r];
    for (int n = tid; n < dim; n += blockDim.x) {
        const uint8_t* wr = w2 + w2e + (size_t)n * inter;
        const float* sr = sf2 + sf2e + (size_t)(n / 128) * (inter / 128);
        float acc = 0.f;
        for (int kb = 0; kb < inter / 128; ++kb) {
            float a = 0.f;
            const uint4* v = (const uint4*)(wr + (kb << 7));
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const uint4 q = v[c];
                const uint8_t* b = (const uint8_t*)&q;
                for (int kk = 0; kk < 16; ++kk) {
                    a = fmaf(sact[(kb << 7) + c * 16 + kk], __half2float(__nv_cvt_fp8_to_halfraw(b[kk], __NV_E4M3)), a);
                }
            }
            acc = fmaf(a, sr[kb], acc);
        }
        // Deterministic: one row per (token, slot) lands in scratch and the weighted sum runs
        // afterwards. Float atomics would make a rerun produce different logits.
        scratch[(size_t)r * dim + n] = acc;
    }
    (void)out;
    (void)w;
}
