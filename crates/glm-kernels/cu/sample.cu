// Token selection: greedy argmax, temperature sampling, and the speculative
// verification step (row-wise acceptance with a residual-correction draw).
//
// Both samplers partition the vocabulary into `blockDim.x` contiguous slices, so
// their CDF order equals index order and the host reference in `glm::sampling`
// reproduces them bit for bit.
#include "common.cuh"

// Block-wide argmax over `row`; the result is broadcast through shared memory.
DEV int argmax_row(const float* row, int n, float* sbuf, int* sibuf) {
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const int lo = (int)((int64_t)tid * n / B), hi = (int)((int64_t)(tid + 1) * n / B);
    float bv = -INFINITY;
    int bi = lo < hi ? lo : 0;
    for (int i = lo; i < hi; ++i) {
        float v = row[i];
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    sbuf[tid] = bv;
    sibuf[tid] = bi;
    __syncthreads();
    for (int stride = B >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sbuf[tid + stride] > sbuf[tid] || (sbuf[tid + stride] == sbuf[tid] && sibuf[tid + stride] < sibuf[tid])) {
                sbuf[tid] = sbuf[tid + stride];
                sibuf[tid] = sibuf[tid + stride];
            }
        }
        __syncthreads();
    }
    int r = sibuf[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void glm_argmax(const float* __restrict__ logits, int n, int* __restrict__ out) {
    extern __shared__ char smem[];
    float* sb = (float*)smem;
    int* si = (int*)(sb + blockDim.x);
    const int r = argmax_row(logits, n, sb, si);
    if (threadIdx.x == 0) out[0] = r;
}

extern "C" __global__ void glm_sample(const float* __restrict__ logits, int n, float temp, uint64_t seed,
                                      unsigned int* __restrict__ step, int* __restrict__ out) {
    extern __shared__ char smem[];
    float* sslice = (float*)smem;         // B
    float* sscan = sslice + blockDim.x;   // B
    float* smax = sscan + blockDim.x;     // 8
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const float inv = temp > 0.f ? 1.f / temp : 1.f;

    float mx = -INFINITY;
    for (int i = tid; i < n; i += B) mx = fmaxf(mx, logits[i]);
    mx = block_max(mx, smax);

    const int lo = (int)((int64_t)tid * n / B), hi = (int)((int64_t)(tid + 1) * n / B);
    float part = 0.f;
    for (int i = lo; i < hi; ++i) part += __expf((logits[i] - mx) * inv);
    sslice[tid] = part;
    __syncthreads();
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < B; ++i) {
            sscan[i] = acc;
            acc += sslice[i];
        }
        smax[0] = acc;
    }
    __syncthreads();
    const float total = smax[0];
    // one counter shared by every sampler launch, advanced on the device so a captured
    // graph keeps producing fresh draws without host involvement
    __shared__ uint32_t sstep;
    if (threadIdx.x == 0) sstep = atomicAdd(step, 1u);
    __syncthreads();
    uint64_t st = seed ^ ((uint64_t)sstep * 0x9E3779B97F4A7C15ull);
    const float u = rng_unit(&st) * (total > 0.f ? total : 1.f);
    int pick = B - 1;
    for (int i = 0; i < B; ++i)
        if (u >= sscan[i] && u < sscan[i] + sslice[i]) {
            pick = i;
            break;
        }
    if (tid == pick) {
        float c = 0.f;
        int chosen = hi > lo ? hi - 1 : 0;
        for (int i = lo; i < hi; ++i) {
            c += __expf((logits[i] - mx) * inv);
            if (u < sscan[pick] + c) {
                chosen = i;
                break;
            }
        }
        out[0] = chosen;
    }
}

// Speculative verification.
//
//   target: (blk+1) rows of target logits, draft: blk proposed token ids,
//   qlogits: blk rows of drafter logits (null in greedy mode).
//
// For i in 0..blk accept draft[i] with probability min(1, p_i(draft[i])/q_i(draft[i]));
// at the first rejection draw from normalised max(0, p - q) and stop. If all drafts
// are accepted, draw one extra token from the final target row.
// `committed` = number of target rows whose KV is authoritative.
extern "C" __global__ void glm_spec_accept(const float* __restrict__ target, const int* __restrict__ draft,
                                           const float* __restrict__ qlogits, int blk, int vocab, float temp,
                                           uint64_t seed, unsigned int* __restrict__ step, int greedy,
                                           int* __restrict__ out, int* __restrict__ count,
                                           int* __restrict__ committed) {
    extern __shared__ char smem[];
    __shared__ uint32_t sstep;
    if (threadIdx.x == 0) sstep = atomicAdd(step, 1u);
    __syncthreads();
    float* smax = (float*)smem;           // 8
    float* sbuf = smax + 8;               // B  (argmax values)
    int* sibuf = (int*)(sbuf + blockDim.x); // B
    float* sslice = (float*)(sibuf + blockDim.x); // B
    float* sscan = sslice + blockDim.x;   // B
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const float inv = temp > 0.f ? 1.f / temp : 1.f;

    auto row_max = [&](const float* row) {
        float mx = -INFINITY;
        for (int i = tid; i < vocab; i += B) mx = fmaxf(mx, row[i]);
        return block_max(mx, smax);
    };
    auto row_sum = [&](const float* row, float mx) {
        float s = 0.f;
        for (int i = tid; i < vocab; i += B) s += __expf((row[i] - mx) * inv);
        return block_sum(s, smax);
    };
    auto prob_at = [&](const float* row, float mx, float z, int tok) {
        return __expf((row[tok] - mx) * inv) / (z > 0.f ? z : 1.f);
    };
    auto draw_residual = [&](const float* p_row, const float* q_row, float pm, float qm, float pz, float qz) {
        const int lo = (int)((int64_t)tid * vocab / B), hi = (int)((int64_t)(tid + 1) * vocab / B);
        float part = 0.f;
        for (int i = lo; i < hi; ++i) {
            float p = __expf((p_row[i] - pm) * inv) / (pz > 0.f ? pz : 1.f);
            float q = q_row ? __expf((q_row[i] - qm) * inv) / (qz > 0.f ? qz : 1.f) : 0.f;
            part += fmaxf(0.f, p - q);
        }
        sslice[tid] = part;
        __syncthreads();
        if (tid == 0) {
            float acc = 0.f;
            for (int i = 0; i < B; ++i) {
                sscan[i] = acc;
                acc += sslice[i];
            }
            smax[0] = acc;
        }
        __syncthreads();
        uint64_t st = seed ^ ((uint64_t)sstep * 0xD1B54A32D192ED03ull);
        const float u = rng_unit(&st) * (smax[0] > 0.f ? smax[0] : 1.f);
        int pick = B - 1;
        for (int i = 0; i < B; ++i)
            if (u >= sscan[i] && u < sscan[i] + sslice[i]) {
                pick = i;
                break;
            }
        int chosen = hi > lo ? hi - 1 : 0;
        if (tid == pick) {
            float c = 0.f;
            for (int i = lo; i < hi; ++i) {
                float p = __expf((p_row[i] - pm) * inv) / (pz > 0.f ? pz : 1.f);
                float q = q_row ? __expf((q_row[i] - qm) * inv) / (qz > 0.f ? qz : 1.f) : 0.f;
                c += fmaxf(0.f, p - q);
                if (u < sscan[pick] + c) {
                    chosen = i;
                    break;
                }
            }
        }
        __syncthreads();
        return chosen;
    };

    int emitted = 0;
    for (int i = 0; i < blk; ++i) {
        const float* prow = target + (size_t)i * vocab;
        const int d = draft[i];
        if (d < 0 || d >= vocab) {
            if (tid == 0) {
                committed[0] = i;
                count[0] = emitted;
            }
            return;
        }
        if (greedy) {
            const int am = argmax_row(prow, vocab, sbuf, sibuf);
            if (am == d) {
                if (tid == 0) out[emitted] = d;
                ++emitted;
                if (tid == 0) committed[0] = i + 1;
                __syncthreads();
                continue;
            }
            if (tid == 0) {
                out[emitted] = am;
                committed[0] = i;
                count[0] = emitted + 1;
            }
            return;
        }
        const float pm = row_max(prow);
        const float pz = row_sum(prow, pm);
        const float* qrow = qlogits ? qlogits + (size_t)i * vocab : nullptr;
        float qm = 0.f, qz = 0.f;
        if (qrow) {
            qm = row_max(qrow);
            qz = row_sum(qrow, qm);
        }
        const float p = prob_at(prow, pm, pz, d);
        const float q = qrow ? prob_at(qrow, qm, qz, d) : p;
        uint64_t st = seed ^ ((uint64_t)(sstep + (uint32_t)i) * 0x2545F4914F6CDD1Dull);
        const float u = rng_unit(&st);
        const float acc = (q > 0.f) ? fminf(1.f, p / q) : 1.f;
        if (u < acc) {
            if (tid == 0) out[emitted] = d;
            ++emitted;
            if (tid == 0) committed[0] = i + 1;
            __syncthreads();
            continue;
        }
        const int r = draw_residual(prow, qrow, pm, qm, pz, qz);
        if (tid == 0) {
            out[emitted] = r;
            committed[0] = i;
            count[0] = emitted + 1;
        }
        return;
    }
    const float* prow = target + (size_t)blk * vocab;
    int extra;
    if (greedy) {
        extra = argmax_row(prow, vocab, sbuf, sibuf);
    } else {
        const float pm = row_max(prow);
        const float pz = row_sum(prow, pm);
        extra = draw_residual(prow, nullptr, pm, 0.f, pz, 0.f);
    }
    if (tid == 0) {
        out[emitted] = extra;
        committed[0] = blk + 1;
        count[0] = emitted + 1;
    }
}

// Choose how many drafts the next verification pass should cover and hand the answer to
// the graph's conditional node. Runs on the device inside a captured graph, so the branch
// is picked without a host round-trip.
//
// `spec` is the accept kernel's state: [0] emitted, [1] committed, [4] smoothed acceptance
// rate in 1/256 units, [5] the L chosen last round. committed counts the anchor row, so
// accepted drafts are committed - 1.
extern "C" __global__ void glm_spec_pick(int* __restrict__ spec, unsigned long long h, int blk) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int drafted = spec[5] > 0 ? spec[5] : blk;
    int accepted = spec[1] - 1;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    int inst = (int)((256 * (long long)accepted) / drafted);
    int ema = spec[4];
    ema = ema <= 0 ? inst : (ema * 3 + inst) / 4;
    spec[4] = ema;
    // full width at high acceptance, a single row when drafts rarely land
    int l = 1 + (int)(((long long)ema * (blk - 1) + 128) / 256);
    if (l < 1) l = 1;
    if (l > blk) l = blk;
    spec[5] = l;
    cudaGraphSetConditional((cudaGraphConditionalHandle)h, (unsigned)(l - 1));
}
