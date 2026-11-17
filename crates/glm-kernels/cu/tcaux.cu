#include "common.cuh"

// Activation quantization for the tensor-core path.
//
// tcgen05 consumes e4m3 activations with a **power-of-two** e8m0 scale per 32 values
// along K, streamed through `tcgen05.cp` in the same 512-byte-per-128x128 tile order the
// weights use. A free-form fp32 scale cannot be expressed in that pipe, so the block scale
// is snapped to 2^e here; that is the mxfp8 format and costs at most one bit of accuracy.
struct QmxParams {
    const bf16* x;
    uint8_t* aq;
    uint8_t* sfa;
    int rows, real, k, lda, kt;
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
    const int row = blockIdx.x;
    if (row >= p.rows) return;
    const int nkb = p.k / 32;
    if (row >= p.real) {
        // Padding rows keep the scale tile well defined: scale 1, value 0.
        for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
            uint8_t* dst = p.aq + (size_t)row * p.lda + kb * 32;
            for (int j = 0; j < 32; ++j) dst[j] = 0;
            const int t = row / 128, rr = row % 128, kti = kb / 4, kc = kb % 4;
            p.sfa[(size_t)(t * p.kt + kti) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + kc] = 127;
        }
        return;
    }
    for (int kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        const bf16* bsrc = p.x + (size_t)row * p.lda + kb * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(bfv(bsrc[j])));
        const int e = q_exp(amax);
        const float inv = 1.0f / ldexpf(1.0f, e);
        uint8_t* dst = p.aq + (size_t)row * p.lda + kb * 32;
        for (int j = 0; j < 32; ++j) dst[j] = q_e4m3(bfv(bsrc[j]) * inv);
        const int t = row / 128, rr = row % 128, kti = kb / 4, kc = kb % 4;
        p.sfa[(size_t)(t * p.kt + kti) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + kc] = (uint8_t)(e + 127);
    }
}
