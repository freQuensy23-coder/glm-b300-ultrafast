// Norms, rotary, activations, KV append and small glue kernels.
#include "common.cuh"

// out[r, :] = rmsnorm(x[r, :]) * w[:]   (bf16 in/out, fp32 accumulation)
extern "C" __global__ void glm_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out,
                                       int rows, int dim, float eps, int ld_x, int ld_out) {
    pdl_wait();
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
    pdl_launch();
}

// Per-head RMSNorm used on q and k (GLM applies qk-norm over head_dim).
extern "C" __global__ void glm_rmsnorm_heads(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                             bf16* __restrict__ out, int heads, int hd, float eps) {
    pdl_wait();
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
    pdl_launch();
}

// Half-split RoPE over the first `rot` lanes of each head; positions come from `pos`.
extern "C" __global__ void glm_rope_qk(bf16* __restrict__ q, bf16* __restrict__ k, const int* __restrict__ pos,
                                       int rows, int nq, int nkv, int hd, int rot, float theta) {
    pdl_wait();
    const int half = rot >> 1;
    const float lgt = log2f(theta);
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
        const float freq = p * exp2f(-(float)(2 * e) / (float)rot * lgt);
        float c, s;
        __sincosf(freq, &s, &c);
        float a = b2f(base[e]), b = b2f(base[e + half]);
        base[e] = f2b(a * c - b * s);
        base[e + half] = f2b(b * c + a * s);
    }
    pdl_launch();
}

// out = silu(gate) * up
extern "C" __global__ void glm_swiglu_mul(bf16* __restrict__ out, const float* __restrict__ gate,
                                          const float* __restrict__ up, int n) {
    pdl_wait();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = f2b(silu(gate[i]) * up[i]);
    pdl_launch();
}

// h += y  (h is the bf16 residual stream, y an fp32 projection result)
extern "C" __global__ void glm_add_residual(bf16* __restrict__ h, const float* __restrict__ y, int n) {
    pdl_wait();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) h[i] = f2b(b2f(h[i]) + y[i]);
    pdl_launch();
}

// dst[t, :] = src[t, :] (fp32 -> bf16), used to feed projections into norm/rope
extern "C" __global__ void glm_cast_f32_bf16(bf16* __restrict__ dst, const float* __restrict__ src, int n) {
    pdl_wait();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = f2b(src[i]);
    pdl_launch();
}

// dst = a || b along the last axis (MTP fuses embedding and hidden state)
extern "C" __global__ void glm_cat2(bf16* __restrict__ dst, const bf16* __restrict__ a, const bf16* __restrict__ b,
                                    int rows, int da, int db) {
    pdl_wait();
    const int d = da + db;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * d; idx += gridDim.x * blockDim.x) {
        int r = idx / d, i = idx % d;
        dst[idx] = (i < da) ? a[(size_t)r * da + i] : b[(size_t)r * db + i - da];
    }
    pdl_launch();
}

// Embedding lookup: out[t, :] = table[id[t], :]
extern "C" __global__ void glm_embed(bf16* __restrict__ out, const bf16* __restrict__ table,
                                     const int* __restrict__ ids, int rows, int dim) {
    pdl_wait();
    const int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* src = table + (size_t)ids[r] * dim;
    bf16* dst = out + (size_t)r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
    pdl_launch();
}

// KV cache layout: kcache[nkv][max_seq][hd], vcache[nkv][max_seq][hd].

// Fused interleaved-rope + KV append: rotates Q in place, rotates K on the fly and writes both
// K and V into the bf16 cache. Replaces glm_rope_qk + glm_kv_append for the bf16 KV mode, so a
// layer issues one launch and one round-trip instead of three.
extern "C" __global__ void glm_rope_kv(bf16* __restrict__ q, const bf16* __restrict__ k,
                                      const bf16* __restrict__ v, bf16* __restrict__ kc,
                                      bf16* __restrict__ vc, const int* __restrict__ pos,
                                      int rows, int nq, int nkv, int hd, int rot, float theta,
                                      int max_seq) {
    pdl_wait();
    const int half = rot >> 1;
    const float lgt = log2f(theta);
    const int qtotal = rows * nq * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < qtotal; idx += gridDim.x * blockDim.x) {
        const int e = idx % half;
        const int t = idx / half;
        const int r = t / nq, head = t % nq;
        bf16* base = q + ((size_t)r * nq + head) * hd;
        const float ang = (float)pos[r] * exp2f(-(float)(2 * e) / (float)rot * lgt);
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = b2f(base[e]), b = b2f(base[e + half]);
        base[e] = f2b(a * c - b * s);
        base[e + half] = f2b(b * c + a * s);
    }
    // one thread per element of the K/V rows, coalesced across the head dimension
    const int kvt = rows * nkv * hd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kvt; i += gridDim.x * blockDim.x) {
        const int lane = i % hd;
        const int t = (i / hd) % nkv;
        const int r = i / (hd * nkv);
        const size_t src = ((size_t)r * nkv + t) * hd + lane;
        const size_t dst = ((size_t)t * max_seq + pos[0] + r) * hd + lane;
        float kv = b2f(k[src]);
        if (lane < rot) {
            const int e = lane < half ? lane : lane - half;
            bf16* pair = (bf16*)(k + src - lane) + e;
            const float ang = (float)pos[r] * exp2f(-(float)(2 * e) / (float)rot * lgt);
            float c, s;
            __sincosf(ang, &s, &c);
            const float a = b2f(pair[0]), b = b2f(pair[half]);
            kv = lane < half ? a * c - b * s : b * c + a * s;
        }
        kc[dst] = f2b(kv);
        vc[dst] = v[src];
    }
    pdl_launch();
}

extern "C" __global__ void glm_kv_append(bf16* __restrict__ kc, bf16* __restrict__ vc, const bf16* __restrict__ k,
                                         const bf16* __restrict__ v, const int* __restrict__ pos, int rows,
                                         int nkv, int hd, int max_seq) {
    pdl_wait();
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
    pdl_launch();
}

extern "C" __global__ void glm_zero_f32(float* __restrict__ p, int n) {
    pdl_wait();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) p[i] = 0.f;
    pdl_launch();
}

// dst[t, :len] += bias[:len] for `rows` rows laid out with stride `ld`.
extern "C" __global__ void glm_add_bias(bf16* __restrict__ dst, const bf16* __restrict__ bias, int rows, int ld,
                                        int len) {
    pdl_wait();
    const int r = blockIdx.x;
    if (r >= rows) return;
    bf16* d = dst + (size_t)r * ld;
    for (int i = threadIdx.x; i < len; i += blockDim.x) d[i] = f2b(b2f(d[i]) + b2f(bias[i]));
    pdl_launch();
}

extern "C" __global__ void glm_copy_bf16(bf16* __restrict__ dst, const bf16* __restrict__ src, int n) {
    pdl_wait();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
    pdl_launch();
}

// fp8 variant of the KV append: same positions-in-device-memory contract, e4m3 payload with
// one power-of-two e8m0 scale per 32 values along the head dimension.
extern "C" __global__ void glm_rope_kv_q8(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                            uint8_t* __restrict__ ks, uint8_t* __restrict__ vs,
                                            bf16* __restrict__ q, const bf16* __restrict__ k,
                                            const bf16* __restrict__ v, const int* __restrict__ pos,
                                            int rows, int nq, int nkv, int hd, int rot, float theta,
                                            int max_seq) {
    pdl_wait();
    const float lgt = log2f(theta);
    const int half = rot >> 1;
    // rotate Q in place; K is rotated on the fly below, so it is never written back
    const int qtotal = rows * nq * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < qtotal; idx += gridDim.x * blockDim.x) {
        const int e = idx % half, t = idx / half;
        const int qr = t / nq, qh = t % nq;
        bf16* qb = q + ((size_t)qr * nq + qh) * hd;
        const float ang = (float)pos[qr] * exp2f(-(float)(2 * e) / (float)rot * lgt);
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = b2f(qb[e]), bb = b2f(qb[e + half]);
        qb[e] = f2b(a * c - bb * s);
        qb[e + half] = f2b(bb * c + a * s);
    }
    const int r = blockIdx.x / nkv, h = blockIdx.x % nkv;
    if (r >= rows) return;
    const int p = pos[0] + r;
    const size_t src = ((size_t)r * nkv + h) * hd;
    const size_t dst = ((size_t)h * max_seq + p) * hd;
    const size_t sdst = ((size_t)h * max_seq + p) * (hd / 32);
    for (int kb = threadIdx.x; kb < hd / 32; kb += blockDim.x) {
        float ak[32], av[32];
        float mk = 0.f, mv = 0.f;
        for (int j = 0; j < 32; ++j) {
            const int col = kb * 32 + j;
            float kvv = b2f(k[src + col]);
            if (col < rot) {
                const int e = col < half ? col : col - half;
                const bf16* head = k + src;
                const float aa = b2f(head[e]), bb2 = b2f(head[e + half]);
                const float ang = (float)pos[r] * exp2f(-(float)(2 * e) / (float)rot * lgt);
                float c, s;
                __sincosf(ang, &s, &c);
                kvv = col < half ? aa * c - bb2 * s : bb2 * c + aa * s;
            }
            ak[j] = kvv;
            av[j] = b2f(v[src + kb * 32 + j]);
            mk = fmaxf(mk, fabsf(ak[j]));
            mv = fmaxf(mv, fabsf(av[j]));
        }
        const int ek = mk > 0.f ? (int)ceilf(log2f(mk / 448.f)) : -127;
        const int ev = mv > 0.f ? (int)ceilf(log2f(mv / 448.f)) : -127;
        const float ik = 1.f / exp2f((float)ek), iv = 1.f / exp2f((float)ev);
        for (int j = 0; j < 32; ++j) {
            kc[dst + kb * 32 + j] = (uint8_t)__nv_cvt_float_to_fp8(ak[j] * ik, __NV_SATFINITE, __NV_E4M3);
            vc[dst + kb * 32 + j] = (uint8_t)__nv_cvt_float_to_fp8(av[j] * iv, __NV_SATFINITE, __NV_E4M3);
        }
        ks[sdst + kb] = (uint8_t)(ek + 127);
        vs[sdst + kb] = (uint8_t)(ev + 127);
    }
    pdl_launch();
}

extern "C" __global__ void glm_kv_append_q8(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                            uint8_t* __restrict__ ks, uint8_t* __restrict__ vs,
                                            const bf16* __restrict__ k, const bf16* __restrict__ v,
                                            const int* __restrict__ pos, int rows, int nkv, int hd,
                                            int max_seq) {
    pdl_wait();
    const int r = blockIdx.x / nkv, h = blockIdx.x % nkv;
    if (r >= rows) return;
    const int p = pos[0] + r;
    const size_t src = ((size_t)r * nkv + h) * hd;
    const size_t dst = ((size_t)h * max_seq + p) * hd;
    const size_t sdst = ((size_t)h * max_seq + p) * (hd / 32);
    for (int kb = threadIdx.x; kb < hd / 32; kb += blockDim.x) {
        float ak[32], av[32];
        float mk = 0.f, mv = 0.f;
        for (int j = 0; j < 32; ++j) {
            ak[j] = b2f(k[src + kb * 32 + j]);
            av[j] = b2f(v[src + kb * 32 + j]);
            mk = fmaxf(mk, fabsf(ak[j]));
            mv = fmaxf(mv, fabsf(av[j]));
        }
        const int ek = mk > 0.f ? (int)ceilf(log2f(mk / 448.f)) : -127;
        const int ev = mv > 0.f ? (int)ceilf(log2f(mv / 448.f)) : -127;
        const float ik = 1.f / exp2f((float)ek), iv = 1.f / exp2f((float)ev);
        for (int j = 0; j < 32; ++j) {
            kc[dst + kb * 32 + j] = (uint8_t)__nv_cvt_float_to_fp8(ak[j] * ik, __NV_SATFINITE, __NV_E4M3);
            vc[dst + kb * 32 + j] = (uint8_t)__nv_cvt_float_to_fp8(av[j] * iv, __NV_SATFINITE, __NV_E4M3);
        }
        ks[sdst + kb] = (uint8_t)(ek + 127);
        vs[sdst + kb] = (uint8_t)(ev + 127);
    }
    pdl_launch();
}

static DEV uint8_t e2m1_code(float x) {
    static const float g[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float a = fabsf(x);
    uint8_t best = 0;
    float err = INFINITY;
    for (int i = 0; i < 8; ++i) {
        const float e = fabsf(g[i] - a);
        if (e < err) { err = e; best = (uint8_t)i; }
    }
    return (uint8_t)(best | (x < 0.f ? 8 : 0));
}

// fp4 KV append: e2m1 nibbles with one e8m0 scale per 32 values, so a head-position costs
// hd/2 payload bytes plus hd/32 exponent bytes instead of 2*hd.
extern "C" __global__ void glm_rope_kv_q4(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                            uint8_t* __restrict__ ks, uint8_t* __restrict__ vs,
                                            bf16* __restrict__ q, const bf16* __restrict__ k,
                                            const bf16* __restrict__ v, const int* __restrict__ pos,
                                            int rows, int nq, int nkv, int hd, int rot, float theta,
                                            int max_seq) {
    pdl_wait();
    const float lgt = log2f(theta);
    const int half = rot >> 1;
    // rotate Q in place; K is rotated on the fly below, so it is never written back
    const int qtotal = rows * nq * half;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < qtotal; idx += gridDim.x * blockDim.x) {
        const int e = idx % half, t = idx / half;
        const int qr = t / nq, qh = t % nq;
        bf16* qb = q + ((size_t)qr * nq + qh) * hd;
        const float ang = (float)pos[qr] * exp2f(-(float)(2 * e) / (float)rot * lgt);
        float c, s;
        __sincosf(ang, &s, &c);
        const float a = b2f(qb[e]), bb = b2f(qb[e + half]);
        qb[e] = f2b(a * c - bb * s);
        qb[e + half] = f2b(bb * c + a * s);
    }
    const int r = blockIdx.x / nkv, h = blockIdx.x % nkv;
    if (r >= rows) return;
    const int p = pos[0] + r;
    const size_t src = ((size_t)r * nkv + h) * hd;
    const size_t dst = ((size_t)h * max_seq + p) * (hd / 2);
    const size_t sdst = ((size_t)h * max_seq + p) * (hd / 32);
    for (int kb = threadIdx.x; kb < hd / 32; kb += blockDim.x) {
        float ak[32], av[32];
        float mk = 0.f, mv = 0.f;
        for (int j = 0; j < 32; ++j) {
            const int col = kb * 32 + j;
            float kvv = b2f(k[src + col]);
            if (col < rot) {
                const int e = col < half ? col : col - half;
                const bf16* head = k + src;
                const float aa = b2f(head[e]), bb2 = b2f(head[e + half]);
                const float ang = (float)pos[r] * exp2f(-(float)(2 * e) / (float)rot * lgt);
                float c, s;
                __sincosf(ang, &s, &c);
                kvv = col < half ? aa * c - bb2 * s : bb2 * c + aa * s;
            }
            ak[j] = kvv;
            av[j] = b2f(v[src + kb * 32 + j]);
            mk = fmaxf(mk, fabsf(ak[j]));
            mv = fmaxf(mv, fabsf(av[j]));
        }
        const int ek = mk > 0.f ? (int)ceilf(log2f(mk / 6.f)) : -127;
        const int ev = mv > 0.f ? (int)ceilf(log2f(mv / 6.f)) : -127;
        const float ik = 1.f / exp2f((float)ek), iv = 1.f / exp2f((float)ev);
        for (int j = 0; j < 16; ++j) {
            const uint8_t kk = e2m1_code(ak[2 * j] * ik) | (e2m1_code(ak[2 * j + 1] * ik) << 4);
            const uint8_t vv = e2m1_code(av[2 * j] * iv) | (e2m1_code(av[2 * j + 1] * iv) << 4);
            kc[dst + kb * 16 + j] = kk;
            vc[dst + kb * 16 + j] = vv;
        }
        ks[sdst + kb] = (uint8_t)(ek + 127);
        vs[sdst + kb] = (uint8_t)(ev + 127);
    }
    pdl_launch();
}

extern "C" __global__ void glm_kv_append_q4(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                            uint8_t* __restrict__ ks, uint8_t* __restrict__ vs,
                                            const bf16* __restrict__ k, const bf16* __restrict__ v,
                                            const int* __restrict__ pos, int rows, int nkv, int hd,
                                            int max_seq) {
    pdl_wait();
    const int r = blockIdx.x / nkv, h = blockIdx.x % nkv;
    if (r >= rows) return;
    const int p = pos[0] + r;
    const size_t src = ((size_t)r * nkv + h) * hd;
    const size_t dst = ((size_t)h * max_seq + p) * (hd / 2);
    const size_t sdst = ((size_t)h * max_seq + p) * (hd / 32);
    for (int kb = threadIdx.x; kb < hd / 32; kb += blockDim.x) {
        float ak[32], av[32];
        float mk = 0.f, mv = 0.f;
        for (int j = 0; j < 32; ++j) {
            ak[j] = b2f(k[src + kb * 32 + j]);
            av[j] = b2f(v[src + kb * 32 + j]);
            mk = fmaxf(mk, fabsf(ak[j]));
            mv = fmaxf(mv, fabsf(av[j]));
        }
        const int ek = mk > 0.f ? (int)ceilf(log2f(mk / 6.f)) : -127;
        const int ev = mv > 0.f ? (int)ceilf(log2f(mv / 6.f)) : -127;
        const float ik = 1.f / exp2f((float)ek), iv = 1.f / exp2f((float)ev);
        for (int j = 0; j < 16; ++j) {
            const uint8_t kk = e2m1_code(ak[2 * j] * ik) | (e2m1_code(ak[2 * j + 1] * ik) << 4);
            const uint8_t vv = e2m1_code(av[2 * j] * iv) | (e2m1_code(av[2 * j + 1] * iv) << 4);
            kc[dst + kb * 16 + j] = kk;
            vc[dst + kb * 16 + j] = vv;
        }
        ks[sdst + kb] = (uint8_t)(ek + 127);
        vs[sdst + kb] = (uint8_t)(ev + 127);
    }
    pdl_launch();
}
