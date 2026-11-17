// GQA attention for a single GPU: split-KV flash decode + tiled causal prefill.
//
// KV cache layout is [nkv][max_seq][hd]. Positions [0, limit) of a query row are
// visible; `window` > 0 restricts attention to the last `window` positions.
// Dynamic shared memory is used so the tile sizes fit Blackwell's 227 KB/SM.
#include "common.cuh"

#define TILE 64  // KV positions staged per iteration
#define TQ 16    // prefill query rows per CTA

static DEV float dotf(const float* a, const float* b, int n) {
    float s = 0.f;
#pragma unroll 4
    for (int i = 0; i < n; ++i) s = fmaf(a[i], b[i], s);
    return s;
}

// One CTA per (query head, query row, KV split). Partials land in `part`, laid
// out as [row][nq][split] -> { m, l, o[hd] }.
extern "C" __global__ void glm_attn_decode_split(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                 const bf16* __restrict__ vc, float* __restrict__ part,
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

    const bf16* kbase = kc + ((size_t)kvh * max_seq + lo) * hd;
    const bf16* vbase = vc + ((size_t)kvh * max_seq + lo) * hd;

    for (int p0 = lo; p0 < hi; p0 += TILE) {
        const int n = min(TILE, hi - p0);
        const bf16* kp = kbase + (size_t)(p0 - lo) * hd;
        const bf16* vp = vbase + (size_t)(p0 - lo) * hd;
        for (int i = tid; i < n * hd; i += blockDim.x) {
            sK[i] = b2f(kp[i]);
            sV[i] = b2f(vp[i]);
        }
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) ssc[i] = dotf(sq, sK + (size_t)i * hd, hd) * scale;
        __syncthreads();
        float m_tile = -INFINITY;
        for (int i = tid; i < n; i += blockDim.x) m_tile = fmaxf(m_tile, ssc[i]);
        m_tile = block_max(m_tile, scred);
        const float m_new = fmaxf(m_run, m_tile);
        const float alpha = __expf(m_run - m_new);
        float lsum = 0.f;
        for (int i = tid; i < n; i += blockDim.x) lsum += __expf(ssc[i] - m_tile);
        lsum = block_sum(lsum, scred);
        for (int d = tid; d < hd; d += blockDim.x) {
            float o = so[d] * alpha;
            for (int i = 0; i < n; ++i) o = fmaf(__expf(ssc[i] - m_tile), sV[(size_t)i * hd + d], o);
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
}

// Causal prefill: one CTA per (head, TQ query rows). Q, K, V tiles live in shared
// memory; rows beyond `limit` of a tile are masked out.
extern "C" __global__ void glm_attn_prefill(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                            const bf16* __restrict__ vc, bf16* __restrict__ out,
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
    for (int p0 = 0; p0 < end; p0 += TILE) {
        const int n = min(TILE, end - p0);
        const bf16* kp = kc + ((size_t)kvh * max_seq + p0) * hd;
        const bf16* vp = vc + ((size_t)kvh * max_seq + p0) * hd;
        for (int i = tid; i < n * hd; i += blockDim.x) {
            sK[i] = b2f(kp[i]);
            sV[i] = b2f(vp[i]);
        }
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
