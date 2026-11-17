// Dense linear layers for the GLM decode and prefill paths.
//
// All three kernels read a weight matrix stored row-major [n_out, k_in] where
// n_out is the output (feature) dimension. `glm_gemv_*` are weight-stationary:
// one CTA owns 64 output rows and streams them exactly once per step, which is
// the property that matters when a decode step is bandwidth bound on weights.
#include "common.cuh"

// ---------------------------------------------------------------- decode GEMV
// y[t, r] = sum_k x[t, k] * W[r, k]      (bf16 activation, fp8 weight)
// grid = n/64, block = 256, k % 128 == 0, n % 64 == 0, m <= GMAX
extern "C" __global__ void glm_gemv_fp8(const uint8_t* __restrict__ w, const float* __restrict__ sf,
                                        const bf16* __restrict__ x, float* __restrict__ y, int n, int k,
                                        int m, int ld_x, int ld_y) {
    __shared__ float shx[GMAX][128];
    __shared__ float red[4][64][GMAX];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2); // output row
    const int q = tid & 3;                      // k quarter within a 128 block
    const int nkb = k >> 7;
    const uint8_t* __restrict__ wr = w + (size_t)r * k;
    const int rb = r >> 7;

    float acc[GMAX];
#pragma unroll
    for (int t = 0; t < GMAX; ++t) acc[t] = 0.f;

    for (int kb = 0; kb < nkb; ++kb) {
        // stage one 128-wide slice of every row of x
        for (int i = tid; i < m * 128; i += 256) shx[i >> 7][i & 127] = b2f(x[(size_t)(i >> 7) * ld_x + (kb << 7) + (i & 127)]);
        __syncthreads();

        const uint8_t* wp = wr + (kb << 7) + (q << 5);
        float wv[32];
        const uint4* v = (const uint4*)wp;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            uint4 u = v[c];
            const uint8_t* b = (const uint8_t*)&u;
#pragma unroll
            for (int j = 0; j < 16; ++j) wv[(c << 4) + j] = fp8f(b[j]);
        }
        const float s = sf[(size_t)rb * nkb + kb];
#pragma unroll
        for (int t = 0; t < GMAX; ++t) {
            if (t >= m) break;
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 32; ++j) d = fmaf(wv[j], shx[t][(q << 5) + j], d);
            acc[t] = fmaf(s, d, acc[t]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < GMAX; ++t) {
        if (t < m) red[q][tid >> 2][t] = acc[t];
    }
    __syncthreads();
    if (q == 0) {
        const int rp = tid >> 2;
#pragma unroll
        for (int t = 0; t < GMAX; ++t) {
            if (t >= m) break;
            float v = red[0][rp][t] + red[1][rp][t] + red[2][rp][t] + red[3][rp][t];
            y[(size_t)t * ld_y + blockIdx.x * 64 + rp] = v;
        }
    }
}

// Same shape, bf16 weights (used for the lm_head and any layer left unquantized).
extern "C" __global__ void glm_gemv_bf16(const bf16* __restrict__ w, const bf16* __restrict__ x, float* __restrict__ y,
                                         int n, int k, int m, int ld_x, int ld_y) {
    __shared__ float shx[GMAX][128];
    __shared__ float red[4][64][GMAX];
    const int tid = threadIdx.x;
    const int r = blockIdx.x * 64 + (tid >> 2);
    const int q = tid & 3;
    const int nkb = k >> 7;
    const bf16* __restrict__ wr = w + (size_t)r * k;

    float acc[GMAX];
#pragma unroll
    for (int t = 0; t < GMAX; ++t) acc[t] = 0.f;

    for (int kb = 0; kb < nkb; ++kb) {
        for (int i = tid; i < m * 128; i += 256) shx[i >> 7][i & 127] = b2f(x[(size_t)(i >> 7) * ld_x + (kb << 7) + (i & 127)]);
        __syncthreads();
        const bf16* wp = wr + (kb << 7) + (q << 5);
        float wv[32];
        for (int j = 0; j < 32; j += 2) {
            __nv_bfloat162 p = *(const __nv_bfloat162*)(wp + j);
            float2 f = __bfloat1622float2(p);
            wv[j] = f.x;
            wv[j + 1] = f.y;
        }
#pragma unroll
        for (int t = 0; t < GMAX; ++t) {
            if (t >= m) break;
            float d = 0.f;
#pragma unroll
            for (int j = 0; j < 32; ++j) d = fmaf(wv[j], shx[t][(q << 5) + j], d);
            acc[t] += d;
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < GMAX; ++t) {
        if (t < m) red[q][tid >> 2][t] = acc[t];
    }
    __syncthreads();
    if (q == 0) {
        const int rp = tid >> 2;
#pragma unroll
        for (int t = 0; t < GMAX; ++t) {
            if (t >= m) break;
            y[(size_t)t * ld_y + blockIdx.x * 64 + rp] = red[0][rp][t] + red[1][rp][t] + red[2][rp][t] + red[3][rp][t];
        }
    }
}

// --------------------------------------------------------------- prefill GEMM
// C(m=32 tokens, n=64 rows) tile, k streamed in 128-wide scaled blocks.
// grid = (n/64, ceil(m/32)), block = 256.
extern "C" __global__ void glm_gemm_fp8(const uint8_t* __restrict__ w, const float* __restrict__ sf,
                                        const bf16* __restrict__ x, float* __restrict__ y, int n, int k, int m,
                                        int ld_x, int ld_y) {
    __shared__ float sha[32][32]; // token x k-chunk
    __shared__ float shb[64][32]; // row   x k-chunk
    const int tid = threadIdx.x;
    const int r0 = blockIdx.x * 64;
    const int t0 = blockIdx.y * 32;
    const int nt = min(32, m - t0);
    const int nkb = k >> 7;
    const int tok = tid >> 3;        // 0..31
    const int rgrp = (tid & 7) << 3; // 8 rows per thread

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.f;

    for (int kb = 0; kb < nkb; ++kb) {
        float part[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) part[i] = 0.f;
        for (int c = 0; c < 128; c += 32) {
            // stage 32 tokens x 32 k and 64 rows x 32 k
            for (int i = tid; i < 32 * 32; i += 256) {
                int t = i >> 5, j = i & 31;
                sha[t][j] = (t < nt) ? b2f(x[(size_t)(t0 + t) * ld_x + (kb << 7) + c + j]) : 0.f;
            }
            for (int i = tid; i < 64 * 32; i += 256) {
                int r = i >> 5, j = i & 31;
                shb[r][j] = fp8f(w[(size_t)(r0 + r) * k + (kb << 7) + c + j]);
            }
            __syncthreads();
            if (tok < nt) {
#pragma unroll
                for (int j = 0; j < 32; ++j) {
                    float a = sha[tok][j];
#pragma unroll
                    for (int i = 0; i < 8; ++i) part[i] = fmaf(a, shb[rgrp + i][j], part[i]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(sf[(size_t)((r0 + rgrp + i) >> 7) * nkb + kb], part[i], acc[i]);
    }

    if (tok < nt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) y[(size_t)(t0 + tok) * ld_y + r0 + rgrp + i] = acc[i];
    }
}
