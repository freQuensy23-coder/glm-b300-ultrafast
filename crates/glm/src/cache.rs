//! KV cache allocation and capacity planning for one GPU.
//!
//! The cache is contiguous per layer: `[n_kv_heads, max_seq, head_dim]` in bf16, which
//! is what `glm_kv_append` and the attention kernels read. Sliding-window layers still
//! allocate the full ring; the kernels mask by position instead of compacting.

use crate::config::GlmConfig;
use anyhow::{ensure, Result};
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_cuda::DevPtr;

pub struct KvCache {
    k: Vec<cu::Buffer>,
    v: Vec<cu::Buffer>,
    pub max_seq: usize,
    pub nkv: usize,
    pub hd: usize,
}

impl KvCache {
    /// Bytes for one layer of `max_seq` tokens (K plus V).
    pub fn layer_bytes(nkv: usize, hd: usize, max_seq: usize) -> usize {
        2 * nkv * hd * max_seq * 2
    }

    /// Largest context that fits `budget` bytes across `n_layers` (+1 MTP slot).
    pub fn max_seq_for(nkv: usize, hd: usize, n_layers: usize, budget: usize, cap: usize) -> usize {
        let per_token = 2 * nkv * hd * 2 * (n_layers + 1).max(1);
        (budget / per_token.max(1)).clamp(0, cap)
    }

    pub fn alloc(cfg: &GlmConfig, max_seq: usize, with_mtp: bool) -> Result<Self> {
        ensure!(max_seq > 0, "KV cache needs a positive context length");
        let slots = cfg.n_layers + usize::from(with_mtp);
        let bytes = Self::layer_bytes(cfg.n_kv, cfg.head_dim, max_seq);
        let mut k = Vec::with_capacity(slots);
        let mut v = Vec::with_capacity(slots);
        for _ in 0..slots {
            k.push(cu::Buffer::zeroed(bytes)?);
            v.push(cu::Buffer::zeroed(bytes)?);
        }
        Ok(KvCache { k, v, max_seq, nkv: cfg.n_kv, hd: cfg.head_dim })
    }

    pub fn ptrs(&self, layer: usize) -> (DevPtr, DevPtr) {
        (self.k[layer].ptr, self.v[layer].ptr)
    }

    pub fn bytes(&self) -> usize {
        self.k.iter().chain(self.v.iter()).map(|b| b.bytes).sum()
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
    want.min(by_len).clamp(1, 64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_bytes_matches_the_kernel_layout() {
        // 8 kv heads x 128 lanes x 2 bytes, doubled for K and V
        assert_eq!(KvCache::layer_bytes(8, 128, 1024), 8 * 128 * 1024 * 2 * 2);
    }

    #[test]
    fn context_length_is_capped_by_the_budget() {
        let budget = 100 * 1024 * 1024 * 1024usize;
        let s = KvCache::max_seq_for(8, 128, 92, budget, 1 << 20);
        assert!(s > 0 && s >= 100_000, "{s}");
        // a tiny budget still yields a usable (small) context
        assert!(KvCache::max_seq_for(8, 128, 92, 1 << 20, 1 << 20) < 1024);
        assert_eq!(KvCache::max_seq_for(8, 128, 92, 0, 1 << 20), 0);
    }

    #[test]
    fn split_count_follows_context_and_occupancy() {
        assert_eq!(decode_splits(64, 96, 1, 148), 1);
        assert!(decode_splits(100_000, 96, 1, 148) > 1);
        assert!(decode_splits(100_000, 96, 16, 148) <= decode_splits(100_000, 96, 1, 148));
        assert_eq!(decode_splits(1_000_000, 96, 1, 148), 64.min(148 * 2 / 96));
    }
}
