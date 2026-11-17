// GQA attention for a single GPU: split-KV flash decode + tiled causal prefill.
//
// KV cache layout is [nkv][max_seq][hd]. Positions [0, limit) of a query row are
// visible; `window` > 0 restricts attention to the last `window` positions.
// Dynamic shared memory is used so the tile sizes fit Blackwell's 227 KB/SM.
#include "common.cuh"

#define TILE 64  // KV positions staged per iteration
#define TQ 16    // prefill query rows per CTA

// KV loads go through one place so the cache can be bf16 or fp8 without duplicating either
// kernel: fp8 stores a power-of-two e8m0 scale per 32 values, and the exponent is applied on
// read. A `_q8` instantiation is generated next to each bf16 kernel.
static DEV float e8m0f(uint8_t b) { return __uint_as_float((uint32_t)b << 23); }
static DEV float e4m3f(uint8_t b) { return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)); }

static __device__ const float E2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static DEV float e2m1f(uint8_t nib) {
    const float m = E2M1[nib & 7];
    return (nib & 8) ? -m : m;
}

// MODE: 0 = bf16, 1 = e4m3, 2 = e2m1 nibbles. Both narrow modes carry one e8m0 scale per
// 32 values, so the exponent is applied while staging the tile into shared memory.
template <int MODE>
static DEV void stage_kv(float* sK, float* sV, const uint8_t* kp, const uint8_t* vp,
                         const uint8_t* ks, const uint8_t* vs, int n, int hd) {
    for (int i = threadIdx.x; i < n * hd; i += blockDim.x) {
        if constexpr (MODE == 0) {
            sK[i] = b2f(((const bf16*)kp)[i]);
            sV[i] = b2f(((const bf16*)vp)[i]);
        } else {
            const int r = i / hd, c = i % hd;
            const float sk = e8m0f(ks[r * (hd / 32) + c / 32]);
            const float sv = e8m0f(vs[r * (hd / 32) + c / 32]);
            if constexpr (MODE == 1) {
                sK[i] = e4m3f(kp[i]) * sk;
                sV[i] = e4m3f(vp[i]) * sv;
            } else {
                const uint8_t bk = kp[i / 2], bv = vp[i / 2];
                sK[i] = e2m1f((c & 1) ? (bk >> 4) : (bk & 0xf)) * sk;
                sV[i] = e2m1f((c & 1) ? (bv >> 4) : (bv & 0xf)) * sv;
            }
        }
    }
}


static DEV float dotf(const float* a, const float* b, int n) {
    float s = 0.f;
#pragma unroll 4
    for (int i = 0; i < n; ++i) s = fmaf(a[i], b[i], s);
    return s;
}

// One CTA per (query head, query row, KV split). Partials land in `part`, laid
// out as [row][nq][split] -> { m, l, o[hd] }.
template <int MODE>
static __device__ void decode_body(const bf16* __restrict__ q, const uint8_t* __restrict__ kc,
                                   const uint8_t* __restrict__ vc, const uint8_t* __restrict__ ks,
                                   const uint8_t* __restrict__ vs, float* __restrict__ part,
                                   const int* __restrict__ pos, int rows, int max_seq, int nq,
                                   int nkv, int hd, float scale, int window, int split,
                                   int nsplits, int gqa) {
    extern __shared__ char smem[];
    float* sq = (float*)smem;                 // hd
    float* sK = sq + hd;                      // TILE*hd
    float* sV = sK + TILE * hd;               // TILE*hd
    float* ssc = sV + TILE * hd;              // TILE
    float* so = ssc + TILE;                   // hd
    float* scred = so + hd;                   // 8 reduction slots

    const int tid = threadIdx.x;
    const int h = blockIdx.x, row = blockIdx.y, sp = blockIdx.z;
    const int kvh = h / gqa;
    const int limit = pos[0] + row + 1;
    const int lo_w = window > 0 ? max(0, limit - window) : 0;
    const int span = limit - lo_w;
    const int per = (span + nsplits - 1) / nsplits;
    const int lo = lo_w + sp * per;
    const int hi = min(limit, lo + per);

    for (int i = tid; i < hd; i += blockDim.x) sq[i] = b2f(q[((size_t)row * nq + h) * hd + i]);
    for (int i = tid; i < hd; i += blockDim.x) so[i] = 0.f;
    __syncthreads();
    float m_run = -INFINITY, l_run = 0.f;

    // bytes of K payload per position: 2*hd bf16, hd e4m3, hd/2 e2m1 nibbles
    const size_t prow = MODE == 0 ? (size_t)2 * hd : (MODE == 1 ? (size_t)hd : (size_t)hd / 2);
    const size_t rowe = (size_t)kvh * max_seq + lo;
    const uint8_t* kbase = kc + rowe * prow;
    const uint8_t* vbase = vc + rowe * prow;
    const uint8_t* ksbase = ks + rowe * (hd / 32);
    const uint8_t* vsbase = vs + rowe * (hd / 32);

    for (int p0 = lo; p0 < hi; p0 += TILE) {
        const int n = min(TILE, hi - p0);
        const uint8_t* kp = kbase + (size_t)(p0 - lo) * prow;
        const uint8_t* vp = vbase + (size_t)(p0 - lo) * prow;
        const uint8_t* kp_s = ksbase + (size_t)(p0 - lo) * (hd / 32);
        const uint8_t* vp_s = vsbase + (size_t)(p0 - lo) * (hd / 32);
        stage_kv<MODE>(sK, sV, kp, vp, kp_s, vp_s, n, hd);
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) ssc[i] = dotf(sq, sK + (size_t)i * hd, hd) * scale;
        __syncthreads();
        float m_tile = -INFINITY;
        for (int i = tid; i < n; i += blockDim.x) m_tile = fmaxf(m_tile, ssc[i]);
        m_tile = block_max(m_tile, scred);
        const float m_new = fmaxf(m_run, m_tile);
        const float alpha = __expf(m_run - m_new);
        // Every tile is weighted against the *running* max; mixing m_tile and m_new here
        // inflates a later, lower-max tile relative to the accumulated mass.
        float lsum = 0.f;
        for (int i = tid; i < n; i += blockDim.x) lsum += __expf(ssc[i] - m_new);
        lsum = block_sum(lsum, scred);
        for (int d = tid; d < hd; d += blockDim.x) {
            float o = so[d] * alpha;
            for (int i = 0; i < n; ++i) o = fmaf(__expf(ssc[i] - m_new), sV[(size_t)i * hd + d], o);
            so[d] = o;
        }
        l_run = l_run * alpha + lsum;
        m_run = m_new;
        __syncthreads();
    }

    float* dst = part + (((size_t)row * nq + h) * nsplits + sp) * (hd + 2);
    if (tid == 0) {
        dst[0] = m_run;
        dst[1] = l_run;
    }
    for (int d = tid; d < hd; d += blockDim.x) dst[2 + d] = so[d];
}

// Merge the split partials and write bf16 attention output.
extern "C" __global__ void glm_attn_combine(const float* __restrict__ part, bf16* __restrict__ out, int rows, int nq,
                                            int hd, int nsplits) {
    pdl_wait();
    const int h = blockIdx.x, row = blockIdx.y;
    const float* base = part + ((size_t)row * nq + h) * nsplits * (hd + 2);
    __shared__ float sm, sl;
    __shared__ float sv[16];
    if (threadIdx.x == 0) {
        float m = -INFINITY, l = 0.f;
        for (int s = 0; s < nsplits; ++s) {
            if (base[s * (hd + 2) + 1] <= 0.f) continue;
            m = fmaxf(m, base[s * (hd + 2)]);
        }
        for (int s = 0; s < nsplits; ++s) {
            if (base[s * (hd + 2) + 1] <= 0.f) continue;
            l += base[s * (hd + 2) + 1] * __expf(base[s * (hd + 2)] - m);
        }
        sm = m;
        sl = l > 0.f ? l : 1.f;
    }
    __syncthreads();
    bf16* o = out + ((size_t)row * nq + h) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < nsplits; ++s) {
            if (base[s * (hd + 2) + 1] <= 0.f) continue;
            acc += base[s * (hd + 2) + 2 + d] * __expf(base[s * (hd + 2)] - sm);
        }
        o[d] = f2b(acc / sl);
    }
    pdl_launch();
}

extern "C" __global__ void glm_attn_decode_split(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                const bf16* __restrict__ vc, float* __restrict__ part,
                                                const int* __restrict__ pos, int rows, int max_seq, int nq,
                                                int nkv, int hd, float scale, int window, int split,
                                                int nsplits, int gqa) {
    pdl_wait();
    decode_body<0>(q, (const uint8_t*)kc, (const uint8_t*)vc, nullptr, nullptr, part, pos, rows,
                       max_seq, nq, nkv, hd, scale, window, split, nsplits, gqa);
    pdl_launch();
}

// fp8 KV: payload is e4m3 with one e8m0 scale per 32 values, so the cache costs half the
// bytes and the same kernel body runs with dequantizing loads.
extern "C" __global__ void glm_attn_decode_split_q8(const bf16* __restrict__ q, const uint8_t* __restrict__ kc,
                                                   const uint8_t* __restrict__ vc, const uint8_t* __restrict__ ks,
                                                   const uint8_t* __restrict__ vs, float* __restrict__ part,
                                                   const int* __restrict__ pos, int rows, int max_seq, int nq,
                                                   int nkv, int hd, float scale, int window, int split,
                                                   int nsplits, int gqa) {
    pdl_wait();
    decode_body<1>(q, kc, vc, ks, vs, part, pos, rows, max_seq, nq, nkv, hd, scale, window, split,
                      nsplits, gqa);
    pdl_launch();
}

// Causal prefill: one CTA per (head, TQ query rows). Q, K, V tiles live in shared
// memory; rows beyond `limit` of a tile are masked out.
template <int MODE>
static __device__ void prefill_body(const bf16* __restrict__ q, const uint8_t* __restrict__ kc,
                                    const uint8_t* __restrict__ vc, const uint8_t* __restrict__ ks,
                                    const uint8_t* __restrict__ vs, bf16* __restrict__ out,
                                    const int* __restrict__ pos, int rows, int max_seq, int nq, int nkv,
                                    int hd, float scale, int window, int gqa) {
    extern __shared__ char smem[];
    float* sq = (float*)smem;                  // TQ*hd
    float* sK = sq + TQ * hd;                  // TILE*hd
    float* sV = sK + TILE * hd;                // TILE*hd
    float* sS = sV + TILE * hd;                // TQ*TILE
    float* so = sS + TQ * TILE;                // TQ*hd
    float* sm = so + TQ * hd;                  // TQ
    float* sl = sm + TQ;                       // TQ

    const int tid = threadIdx.x;
    const int h = blockIdx.x;
    const int t0 = blockIdx.y * TQ;
    const int start = pos[0];
    const int nt = min(TQ, rows - t0);
    const int kvh = h / gqa;
    if (nt <= 0) return;

    for (int i = tid; i < nt * hd; i += blockDim.x) sq[i] = b2f(q[((size_t)(t0 + i / hd) * nq + h) * hd + i % hd]);
    for (int i = tid; i < nt * hd; i += blockDim.x) so[i] = 0.f;
    for (int i = tid; i < nt; i += blockDim.x) {
        sm[i] = -INFINITY;
        sl[i] = 0.f;
    }
    __syncthreads();

    const int end = start + t0 + nt; // exclusive bound for the whole tile
    // bytes of K payload per position: 2*hd bf16, hd e4m3, hd/2 e2m1 nibbles
    const size_t prow = MODE == 0 ? (size_t)2 * hd : (MODE == 1 ? (size_t)hd : (size_t)hd / 2);
    for (int p0 = 0; p0 < end; p0 += TILE) {
        const int n = min(TILE, end - p0);
        const size_t rowe = (size_t)kvh * max_seq + p0;
        const uint8_t* kp = kc + rowe * prow;
        const uint8_t* vp = vc + rowe * prow;
        stage_kv<MODE>(sK, sV, kp, vp, ks + rowe * (hd / 32), vs + rowe * (hd / 32), n, hd);
        __syncthreads();
        for (int i = tid; i < nt * n; i += blockDim.x) {
            int t = i / n, p = i % n;
            sS[i] = dotf(sq + (size_t)t * hd, sK + (size_t)p * hd, hd) * scale;
        }
        __syncthreads();
        // causal + sliding-window mask, then per-row softmax update
        for (int t = tid; t < nt; t += blockDim.x) {
            const int qpos = start + t0 + t;
            const int lo = window > 0 ? max(0, qpos + 1 - window) : 0;
            float mx = -INFINITY;
            for (int p = 0; p < n; ++p) {
                const int abs = p0 + p;
                if (abs > qpos || abs < lo) {
                    sS[(size_t)t * n + p] = -INFINITY;
                } else {
                    mx = fmaxf(mx, sS[(size_t)t * n + p]);
                }
            }
            const float m_new = fmaxf(sm[t], mx);
            const float alpha = __expf(sm[t] - m_new);
            float ls = 0.f;
            for (int p = 0; p < n; ++p) {
                float e = __expf(sS[(size_t)t * n + p] - m_new);
                sS[(size_t)t * n + p] = e;
                ls += e;
            }
            sm[t] = m_new;
            sl[t] = sl[t] * alpha + ls;
            for (int d = 0; d < hd; ++d) so[(size_t)t * hd + d] *= alpha;
        }
        __syncthreads();
        for (int i = tid; i < nt * hd; i += blockDim.x) {
            int t = i / hd, d = i % hd;
            float acc = 0.f;
            for (int p = 0; p < n; ++p) acc = fmaf(sS[(size_t)t * n + p], sV[(size_t)p * hd + d], acc);
            so[(size_t)t * hd + d] += acc;
        }
        __syncthreads();
    }

    for (int i = tid; i < nt * hd; i += blockDim.x) {
        int t = i / hd, d = i % hd;
        out[((size_t)(t0 + t) * nq + h) * hd + d] = f2b(so[(size_t)t * hd + d] / sl[t]);
    }
}

extern "C" __global__ void glm_attn_prefill(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                           const bf16* __restrict__ vc, bf16* __restrict__ out,
                                           const int* __restrict__ pos, int rows, int max_seq, int nq, int nkv,
                                           int hd, float scale, int window, int gqa) {
    pdl_wait();
    prefill_body<0>(q, (const uint8_t*)kc, (const uint8_t*)vc, nullptr, nullptr, out, pos, rows,
                        max_seq, nq, nkv, hd, scale, window, gqa);
    pdl_launch();
}

extern "C" __global__ void glm_attn_prefill_q8(const bf16* __restrict__ q, const uint8_t* __restrict__ kc,
                                               const uint8_t* __restrict__ vc, const uint8_t* __restrict__ ks,
                                               const uint8_t* __restrict__ vs, bf16* __restrict__ out,
                                               const int* __restrict__ pos, int rows, int max_seq, int nq,
                                               int nkv, int hd, float scale, int window, int gqa) {
    pdl_wait();
    prefill_body<1>(q, kc, vc, ks, vs, out, pos, rows, max_seq, nq, nkv, hd, scale, window, gqa);
    pdl_launch();
}

extern "C" __global__ void glm_attn_decode_split_q4(const bf16* __restrict__ q, const uint8_t* __restrict__ kc,
                                                   const uint8_t* __restrict__ vc, const uint8_t* __restrict__ ks,
                                                   const uint8_t* __restrict__ vs, float* __restrict__ part,
                                                   const int* __restrict__ pos, int rows, int max_seq, int nq,
                                                   int nkv, int hd, float scale, int window, int split,
                                                   int nsplits, int gqa) {
    pdl_wait();
    decode_body<2>(q, kc, vc, ks, vs, part, pos, rows, max_seq, nq, nkv, hd, scale, window, split,
                   nsplits, gqa);
    pdl_launch();
}

extern "C" __global__ void glm_attn_prefill_q4(const bf16* __restrict__ q, const uint8_t* __restrict__ kc,
                                               const uint8_t* __restrict__ vc, const uint8_t* __restrict__ ks,
                                               const uint8_t* __restrict__ vs, bf16* __restrict__ out,
                                               const int* __restrict__ pos, int rows, int max_seq, int nq,
                                               int nkv, int hd, float scale, int window, int gqa) {
    pdl_wait();
    prefill_body<2>(q, kc, vc, ks, vs, out, pos, rows, max_seq, nq, nkv, hd, scale, window, gqa);
    pdl_launch();
}
