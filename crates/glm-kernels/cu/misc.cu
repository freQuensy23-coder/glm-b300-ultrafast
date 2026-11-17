// Norms, rotary, activations, KV append and small glue kernels.
#include "common.cuh"

// out[r, :] = rmsnorm(x[r, :]) * w[:]   (bf16 in/out, fp32 accumulation)
extern "C" __global__ void glm_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out,
                                       int rows, int dim, float eps, int ld_x, int ld_out) {
    __shared__ float sv[8];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* xr = x + (size_t)r * ld_x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = b2f(xr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, sv);
    __shared__ float scale;
    if (threadIdx.x == 0) scale = rsqrtf(ss / dim + eps);
    __syncthreads();
    bf16* o = out + (size_t)r * ld_out;
    const float s = scale;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = f2b(b2f(xr[i]) * s * b2f(w[i]));
}

// Per-head RMSNorm used on q and k (GLM applies qk-norm over head_dim).
extern "C" __global__ void glm_rmsnorm_heads(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                             bf16* __restrict__ out, int heads, int hd, float eps) {
    __shared__ float sv[8];
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const bf16* xr = x + ((size_t)row * heads + h) * hd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = b2f(xr[i]);
        ss = fmaf(v, v, ss);
    }
    ss = block_sum(ss, sv);
    __shared__ float scale;
    if (threadIdx.x == 0) scale = rsqrtf(ss / hd + eps);
    __syncthreads();
    bf16* o = out + ((size_t)row * heads + h) * hd;
    const float s = scale;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) o[i] = f2b(b2f(xr[i]) * s * b2f(w[i]));
}

// Half-split RoPE over the first `rot` lanes of each head; positions come from `pos`.
extern "C" __global__ void glm_rope_qk(bf16* __restrict__ q, bf16* __restrict__ k, const int* __restrict__ pos,
                                       int rows, int nq, int nkv, int hd, int rot, float theta) {
    const int half = rot >> 1;
    const int total = rows * (nq + nkv);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total * half; idx += gridDim.x * blockDim.x) {
        int e = idx % half;
        int t = idx / half;
        const bool isq = t < rows * nq;
        int row, head;
        bf16* base;
        if (isq) {
            row = t / nq;
            head = t % nq;
            base = q + ((size_t)row * nq + head) * hd;
        } else {
            int u = t - rows * nq;
            row = u / nkv;
            head = u % nkv;
            base = k + ((size_t)row * nkv + head) * hd;
        }
        const float p = (float)pos[row];
        const float freq = p / powf(theta, (float)(2 * e) / (float)rot);
        const float c = cosf(freq), s = sinf(freq);
        float a = b2f(base[e]), b = b2f(base[e + half]);
        base[e] = f2b(a * c - b * s);
        base[e + half] = f2b(b * c + a * s);
    }
}

// out = silu(gate) * up
extern "C" __global__ void glm_swiglu_mul(bf16* __restrict__ out, const float* __restrict__ gate,
                                          const float* __restrict__ up, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = f2b(silu(gate[i]) * up[i]);
}

// h += y  (h is the bf16 residual stream, y an fp32 projection result)
extern "C" __global__ void glm_add_residual(bf16* __restrict__ h, const float* __restrict__ y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) h[i] = f2b(b2f(h[i]) + y[i]);
}

// dst[t, :] = src[t, :] (fp32 -> bf16), used to feed projections into norm/rope
extern "C" __global__ void glm_cast_f32_bf16(bf16* __restrict__ dst, const float* __restrict__ src, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = f2b(src[i]);
}

// dst = a || b along the last axis (MTP fuses embedding and hidden state)
extern "C" __global__ void glm_cat2(bf16* __restrict__ dst, const bf16* __restrict__ a, const bf16* __restrict__ b,
                                    int rows, int da, int db) {
    const int d = da + db;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * d; idx += gridDim.x * blockDim.x) {
        int r = idx / d, i = idx % d;
        dst[idx] = (i < da) ? a[(size_t)r * da + i] : b[(size_t)r * db + i - da];
    }
}

// Embedding lookup: out[t, :] = table[id[t], :]
extern "C" __global__ void glm_embed(bf16* __restrict__ out, const bf16* __restrict__ table,
                                     const int* __restrict__ ids, int rows, int dim) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* src = table + (size_t)ids[r] * dim;
    bf16* dst = out + (size_t)r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

// KV cache layout: kcache[nkv][max_seq][hd], vcache[nkv][max_seq][hd].
extern "C" __global__ void glm_kv_append(bf16* __restrict__ kc, bf16* __restrict__ vc, const bf16* __restrict__ k,
                                         const bf16* __restrict__ v, const int* __restrict__ pos, int rows,
                                         int nkv, int hd, int max_seq) {
    // `pos` is a device array of absolute positions, so a captured graph replays at any
    // offset without its kernel arguments changing.
    const int p0 = pos[0];
    const int total = rows * nkv * hd / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        int e = i * 8;
        int lane = e % hd;
        int t = (e / hd) % nkv;
        int r = e / (hd * nkv);
        size_t src = ((size_t)r * nkv + t) * hd + lane;
        size_t dst = ((size_t)t * max_seq + p0 + r) * hd + lane;
        *(uint4*)(kc + dst) = *(const uint4*)(k + src);
        *(uint4*)(vc + dst) = *(const uint4*)(v + src);
    }
}

extern "C" __global__ void glm_zero_f32(float* __restrict__ p, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = 0.f;
}

// dst[t, :len] += bias[:len] for `rows` rows laid out with stride `ld`.
extern "C" __global__ void glm_add_bias(bf16* __restrict__ dst, const bf16* __restrict__ bias, int rows, int ld,
                                        int len) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    bf16* d = dst + (size_t)r * ld;
    for (int i = threadIdx.x; i < len; i += blockDim.x) d[i] = f2b(b2f(d[i]) + b2f(bias[i]));
}

extern "C" __global__ void glm_copy_bf16(bf16* __restrict__ dst, const bf16* __restrict__ src, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}
