//! KV cache allocation and capacity planning for one GPU.
//!
//! The cache is contiguous per layer: `[n_kv_heads, max_seq, head_dim]` in bf16, which
//! is what `glm_kv_append` and the attention kernels read. Sliding-window layers still
//! allocate the full ring; the kernels mask by position instead of compacting.

use crate::config::GlmConfig;
use anyhow::{ensure, Result};
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_cuda::DevPtr;

/// Storage dtype of the KV cache. Both narrow modes keep one e8m0 scale per 32 values and
/// cut the bytes the attention pass has to pull from HBM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvMode {
    Bf16,
    Fp8,
    Fp4,
}

impl KvMode {
    pub fn from_env() -> Self {
        static M: std::sync::OnceLock<KvMode> = std::sync::OnceLock::new();
        *M.get_or_init(|| match std::env::var("GLM_KV").as_deref() {
            Ok("fp8") | Ok("q8") => KvMode::Fp8,
            Ok("fp4") | Ok("q4") => KvMode::Fp4,
            _ => KvMode::Bf16,
        })
    }

    /// Payload bytes per head-position for K **and** V together.
    pub fn payload(self, hd: usize) -> usize {
        match self {
            KvMode::Bf16 => 4 * hd,
            KvMode::Fp8 => 2 * hd,
            KvMode::Fp4 => hd,
        }
    }

    /// Exponent bytes per head-position, both stripes counted.
    pub fn scales(self, hd: usize) -> usize {
        match self {
            KvMode::Bf16 => 0,
            _ => 2 * (hd / 32),
        }
    }
}

pub struct KvCache {
    k: Vec<cu::Buffer>,
    v: Vec<cu::Buffer>,
    /// e8m0 scale stripes, present only in the fp8 cache mode.
    ks: Vec<cu::Buffer>,
    vs: Vec<cu::Buffer>,
    pub mode: KvMode,
    pub max_seq: usize,
    pub nkv: usize,
    pub hd: usize,
}

impl KvCache {
    /// Bytes for one layer of `max_seq` tokens (K plus V) in the requested dtype.
    ///
    /// fp8 halves the payload and adds one exponent byte per 32 values, so the cache costs
    /// `hd + hd/32` bytes per head-position instead of `2*hd`.
    pub fn layer_bytes(nkv: usize, hd: usize, max_seq: usize, mode: KvMode) -> usize {
        nkv * max_seq * (mode.payload(hd) + mode.scales(hd))
    }

    /// Largest context that fits `budget` bytes across `n_layers` (+1 MTP slot).
    pub fn max_seq_for(nkv: usize, hd: usize, n_layers: usize, budget: usize, cap: usize, mode: KvMode) -> usize {
        let per_token = Self::layer_bytes(nkv, hd, 1, mode) * (n_layers + 1).max(1);
        (budget / per_token.max(1)).clamp(0, cap)
    }

    fn stripe_bytes(nkv: usize, hd: usize, max_seq: usize, mode: KvMode) -> (usize, usize) {
        (nkv * max_seq * mode.payload(hd) / 2, nkv * max_seq * mode.scales(hd) / 2)
    }

    pub fn alloc(cfg: &GlmConfig, max_seq: usize, with_mtp: bool, mode: KvMode) -> Result<Self> {
        ensure!(max_seq > 0, "KV cache needs a positive context length");
        ensure!(mode == KvMode::Bf16 || cfg.head_dim % 32 == 0, "quantized KV needs a head_dim multiple of 32");
        ensure!(mode != KvMode::Fp4 || cfg.head_dim % 2 == 0, "fp4 KV needs an even head_dim");
        let slots = cfg.n_layers + usize::from(with_mtp);
        let (bytes, sbytes) = Self::stripe_bytes(cfg.n_kv, cfg.head_dim, max_seq, mode);
        let mut k = Vec::with_capacity(slots);
        let mut v = Vec::with_capacity(slots);
        let mut ks = Vec::with_capacity(slots);
        let mut vs = Vec::with_capacity(slots);
        for _ in 0..slots {
            k.push(cu::Buffer::zeroed(bytes)?);
            v.push(cu::Buffer::zeroed(bytes)?);
            if mode != KvMode::Bf16 {
                ks.push(cu::Buffer::zeroed(sbytes.max(1))?);
                vs.push(cu::Buffer::zeroed(sbytes.max(1))?);
            }
        }
        Ok(KvCache { k, v, ks, vs, mode, max_seq, nkv: cfg.n_kv, hd: cfg.head_dim })
    }

    /// e8m0 stripes for one layer (only meaningful in fp8 mode).
    pub fn scale_ptrs(&self, layer: usize) -> (DevPtr, DevPtr) {
        if self.mode == KvMode::Bf16 {
            (0, 0)
        } else {
            (self.ks[layer].ptr, self.vs[layer].ptr)
        }
    }

    pub fn ptrs(&self, layer: usize) -> (DevPtr, DevPtr) {
        (self.k[layer].ptr, self.v[layer].ptr)
    }

    pub fn bytes(&self) -> usize {
        self.k.iter().chain(self.v.iter()).chain(self.ks.iter()).chain(self.vs.iter()).map(|b| b.bytes).sum()
    }

    pub fn slots(&self) -> usize {
        self.k.len()
    }
}

/// Decode-time split count: enough CTAs to fill the device, without splitting a short
/// context into useless fragments.
pub fn decode_splits(tokens: usize, nq: usize, rows: usize, sms: usize) -> usize {
    let heads = nq.max(1) * rows.max(1);
    let want = (sms * 2).saturating_div(heads).max(1);
    let by_len = tokens.div_ceil(512).max(1);
    want.min(by_len).clamp(1, crate::engine::SPLITS_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_cache_has_no_scale_stripes() {
        let cache = KvCache {
            k: Vec::new(), v: Vec::new(), ks: Vec::new(), vs: Vec::new(),
            mode: KvMode::Bf16, max_seq: 1024, nkv: 8, hd: 128,
        };
        assert_eq!(cache.scale_ptrs(0), (0, 0));
    }

    #[test]
    fn layer_bytes_matches_the_kernel_layout() {
        // 8 kv heads x 128 lanes x 2 bytes, doubled for K and V
        assert_eq!(KvCache::layer_bytes(8, 128, 1024, KvMode::Bf16), 8 * 128 * 1024 * 2 * 2);
        // fp8 halves the payload and pays one exponent byte per 32 values
        assert_eq!(KvCache::layer_bytes(8, 128, 1024, KvMode::Fp8), 8 * 1024 * (256 + 8));
        // fp4 quarters it
        assert_eq!(KvCache::layer_bytes(8, 128, 1024, KvMode::Fp4), 8 * 1024 * (128 + 8));
    }

    #[test]
    fn allocated_stripes_match_the_capacity_budget() {
        for mode in [KvMode::Bf16, KvMode::Fp8, KvMode::Fp4] {
            let (payload, scales) = KvCache::stripe_bytes(8, 128, 1024, mode);
            assert_eq!(payload * 2 + scales * 2, KvCache::layer_bytes(8, 128, 1024, mode));
            assert_eq!(payload, 8 * 1024 * mode.payload(128) / 2);
            assert_eq!(scales, 8 * 1024 * mode.scales(128) / 2);
        }
    }

    #[test]
    fn context_length_is_capped_by_the_budget() {
        let budget = 100 * 1024 * 1024 * 1024usize;
        let s = KvCache::max_seq_for(8, 128, 92, budget, 1 << 20, KvMode::Bf16);
        assert!(s >= 100_000, "{s}");
        // a tiny budget still yields a usable (small) context
        assert!(KvCache::max_seq_for(8, 128, 92, 1 << 20, 1 << 20, KvMode::Bf16) < 1024);
        assert_eq!(KvCache::max_seq_for(8, 128, 92, 0, 1 << 20, KvMode::Bf16), 0);
    }

    #[test]
    fn split_count_follows_context_and_occupancy() {
        assert_eq!(decode_splits(64, 96, 1, 148), 1);
        assert!(decode_splits(100_000, 96, 1, 148) > 1);
        assert!(decode_splits(100_000, 96, 16, 148) <= decode_splits(100_000, 96, 1, 148));
        assert_eq!(decode_splits(1_000_000, 96, 1, 148), 64.min(148 * 2 / 96));
    }
}
