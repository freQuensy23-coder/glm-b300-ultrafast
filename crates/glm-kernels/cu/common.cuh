// Shared helpers for the single-GPU GLM kernels (sm_100a/sm_103a).
//
// Weight format assumed everywhere: row-major [n_out, k_in] in fp8 e4m3 with
// fp32-per-128x128-block scales stored as e8m0 bytes, i.e. the layout Hugging
// Face uses for `*-block-fp8` checkpoints. Activations stay bf16.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define DEV __device__ __forceinline__
#define GMAX 16 // widest decode/verify row block (draft block + 1)

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;

DEV float b2f(bf16 v) { return __bfloat162float(v); }
DEV bf16 f2b(float v) { return __float2bfloat16(v); }

// e8m0 scale byte -> float
DEV float e8m0(unsigned char b) { return __uint_as_float((uint32_t)b << 23); }

DEV float fp8f(uint8_t b) { return float(*(fp8*)&b); }

// Convert one fp8 byte to bf16 without the fp8 header's exception paths.
DEV bf16 cvt(uint8_t b) { return __float2bfloat16(float(*(fp8*)&b)); }

DEV float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

DEV float block_sum(float v, float* s) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) s[warp] = v;
    __syncthreads();
    int n = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        float t = (lane < n) ? s[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) s[0] = t;
    }
    __syncthreads();
    return s[0]; // broadcast to the whole CTA
}

DEV float block_max(float v, float* s) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) s[warp] = v;
    __syncthreads();
    int n = (blockDim.x + 31) >> 5;
    if (warp == 0) {
        float t = (lane < n) ? s[lane] : -INFINITY;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_xor_sync(0xffffffffu, t, o));
        if (lane == 0) s[0] = t;
    }
    __syncthreads();
    return s[0];
}

DEV float silu(float x) { return x / (1.f + __expf(-x)); }

// 32-bit Philox-free counter RNG: enough for sampling, portable, host-reproducible.
DEV uint32_t rng_next(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (uint32_t)(z >> 32);
}

DEV float rng_unit(uint64_t* st) { return (rng_next(st) >> 8) * (1.0f / 16777216.0f); }

// 16-byte global->shared copy through the async engine, so the weight math below overlaps the
// staging of the activation slice instead of waiting on it.
DEV void cp_async16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}

DEV void cp_async_wait() { asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory"); }
