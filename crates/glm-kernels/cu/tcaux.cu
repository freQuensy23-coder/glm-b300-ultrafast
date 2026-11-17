#include "common.cuh"

// Activation quantization for the tensor-core path.
//
// tcgen05 consumes e4m3 activations with a **power-of-two** e8m0 scale per 32 values
// along K, streamed through `tcgen05.cp` in the same 512-byte-per-128x128 tile order the
// weights use. A free-form fp32 scale cannot be expressed in that pipe, so the block scale
// is snapped to 2^e here; that is the mxfp8 format and costs at most one bit of accuracy.
struct QmxParams {
    const bf16* x;
    const bf16* pw; // optional pre-norm weight: when set the RMSNorm is fused here
    uint8_t* aq;
    uint8_t* sfa;
    int rows, real, k, lda, kt;
    float eps;
    int norm;
};

DEV float bfv(bf16 v) { return __bfloat162float(v); }

DEV uint8_t q_e4m3(float v) {
    // Same rounding the CPU reference uses: round-to-nearest, saturating at 448.
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

DEV int q_exp(float amax) {
    if (!(amax > 0.0f)) return 0;
    int e = (int)ceilf(log2f(amax / 448.0f));
    return e < -127 ? -127 : (e > 127 ? 127 : e);
}

extern "C" __global__ void glm_quant_mx(const __grid_constant__ QmxParams p) {
    pdl_wait();
    const int row = blockIdx.x;
    if (row >= p.rows) return;
    const int nkb = p.k / 32;
    // Fused RMSNorm: one block owns a row, so the sum of squares reduces in shared memory
    // and the scale is ready before any value is quantized.
    __shared__ float red[256];
    float rs = 1.0f;
    if (p.norm) {
        float ss = 0.0f;
        if (row < p.real) {
            const bf16* xr = p.x + (size_t)row * p.lda;
            for (int i = threadIdx.x; i < p.k; i += blockDim.x) ss += bfv(xr[i]) * bfv(xr[i]);
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        rs = rsqrtf(red[0] / (float)p.k + p.eps);
        __syncthreads();
    }
    if (row >= p.real) {
        // Padding rows keep the scale tile well defined: scale 1, value 0.
        for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
            uint8_t* dst = p.aq + (size_t)row * p.k + kb * 32;
            for (int j = 0; j < 32; ++j) dst[j] = 0;
            const int t = row / 128, rr = row % 128, kti = kb / 4, kc = kb % 4;
            p.sfa[(size_t)(t * p.kt + kti) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + kc] = 127;
        }
        return;
    }
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        const bf16* bsrc = p.x + (size_t)row * p.lda + kb * 32;
        const float f = p.pw ? rs : 1.0f;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(f * bfv(bsrc[j]) * (p.pw ? bfv(p.pw[kb * 32 + j]) : 1.0f)));
        const int e = q_exp(amax);
        const float inv = 1.0f / ldexpf(1.0f, e);
        uint8_t* dst = p.aq + (size_t)row * p.k + kb * 32;
        for (int j = 0; j < 32; ++j) {
            const float v = f * bfv(bsrc[j]) * (p.pw ? bfv(p.pw[kb * 32 + j]) : 1.0f);
            dst[j] = q_e4m3(v * inv);
        }
        const int t = row / 128, rr = row % 128, kti = kb / 4, kc = kb % 4;
        p.sfa[(size_t)(t * p.kt + kti) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + kc] = (uint8_t)(e + 127);
    }
    pdl_launch();
}

// Gather the routed rows into the MoE image and quantize them in one pass.
//
// The upstream MoE prologue expects a pre-computed norm partial-sum buffer whose layout is
// 26 floats per row per slice; recomputing the reduction here removes that dependency and
// keeps the normalized row out of memory entirely.
struct GqParams {
    const bf16* x;
    const bf16* pw;
    const int* row_pair; // slot -> source row
    uint8_t* xq;
    uint8_t* sf;
    int rows;   // slots
    int topk;
    int k;
    int kt;     // k / 128
    float eps;
};

extern "C" __global__ void glm_gather_quant(const __grid_constant__ GqParams p) {
    pdl_wait();
    const int r = blockIdx.x;
    if (r >= p.rows) return;
    // row_pair stores the (token, slot) index; the token row it gathers from is that / topk
    const int src = p.row_pair[r] / p.topk;
    const int nkb = p.k / 32;
    __shared__ float red[256];
    const bool do_norm = p.pw != nullptr;
    float rs = 1.0f;
    const bf16* xr = p.x + (size_t)src * p.k;
    if (do_norm) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < p.k; i += blockDim.x) ss += bfv(xr[i]) * bfv(xr[i]);
        red[threadIdx.x] = ss;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        rs = rsqrtf(red[0] / (float)p.k + p.eps);
    }
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        const bf16* src32 = xr + kb * 32;
        float v[32];
        float amax = 0.0f;
        const float f = do_norm ? rs : 1.0f;
        for (int j = 0; j < 32; ++j) {
            // normalize the gathered row and apply the norm weight when it is supplied
            v[j] = f * bfv(src32[j]) * (do_norm ? bfv(p.pw[kb * 32 + j]) : 1.0f);
            amax = fmaxf(amax, fabsf(v[j]));
        }
        const int e = q_exp(amax);
        const float inv = 1.0f / ldexpf(1.0f, e);
        uint8_t* dst = p.xq + (size_t)r * p.k + kb * 32;
        for (int j = 0; j < 32; ++j) dst[j] = q_e4m3(v[j] * inv);
        const int t = r / 128, rr = r % 128, kti = kb / 4, kc = kb % 4;
        p.sf[(size_t)(t * p.kt + kti) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + kc] = (uint8_t)(e + 127);
    }
    pdl_launch();
}
