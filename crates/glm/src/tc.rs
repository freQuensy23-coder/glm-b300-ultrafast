//! Tensor-core (tcgen05) path for the decode GEMV.
//!
//! The kernels themselves are compiled from the upstream corpus (`crates/kernels/cu/tcgemv.cu`)
//! rather than re-derived: they carry the TMEM allocation, the mbarrier choreography and the
//! cluster K-split. What has to be reproduced on this side is the ABI: the `TvParams` struct,
//! the TMA descriptor for the weights, and the e8m0 scale tiles.
//!
//! Layout the kernel expects:
//!   * weights row-major `[n, kp]`, e4m3, `kp = k` padded to 128, streamed by a 128-B swizzled
//!     TMA descriptor whose box is `nt` rows;
//!   * scales as e8m0, one per 32 values along K, folded into 512-byte tiles per 128x128 block;
//!   * activations e4m3 with the same 32-value scale granularity, `m <= 8`.

use anyhow::{ensure, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Set once from the CLI (`--tc`) or the environment; read on the hot path.
static TC: AtomicBool = AtomicBool::new(false);

pub fn set_on(v: bool) {
    TC.store(v, Ordering::Relaxed);
}

pub fn on() -> bool {
    TC.load(Ordering::Relaxed)
}
use deepseek_ultrafast_cuda as cu;

/// Byte-compatible placeholder for the kernel's `RopeParams`. The fused KV-rope branch is
/// disabled (`kvr_on = 0`), so the body of this blob is never read.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopeBlob {
    raw: [u8; 104],
}

impl Default for RopeBlob {
    fn default() -> Self {
        RopeBlob { raw: [0; 104] }
    }
}

/// Mirror of `TvParams` in `tcgemv.cu`. Offsets were taken from the compiled C++ struct,
/// so the size asserts below are the real contract, not a guess.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TvParams {
    pub tmap: u64,
    pub sfa: u64,
    pub x: u64,
    pub prenorm_w: u64,
    pub norm_part: u64,
    pub out: u64,
    pub out2: u64,
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub meta: u64,
    pub alpha: f32,
    pub eps: f32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub kpt: i32,
    pub ldx: i32,
    pub ldo: i32,
    pub ldo2: i32,
    pub n_split: i32,
    pub out_f32: i32,
    pub nslices: i32,
    pub rope_head: i32,
    pub rope_off: i32,
    pub partial: i32,
    pub group_rows: i32,
    pub kvr: RopeBlob,
    pub kvr_on: i32,
    pub pad2: i32,
}

const _: () = assert!(std::mem::size_of::<TvParams>() == 256);
const _: () = assert!(std::mem::size_of::<RopeBlob>() == 104);

/// Mirror of `QmxParams` in `cu/tcaux.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QmxParams {
    pub x: u64,
    pub aq: u64,
    pub sfa: u64,
    pub rows: i32,
    pub real: i32,
    pub k: i32,
    pub lda: i32,
    pub kt: i32,
}

/// Cluster split along K, ported from the upstream selector: the split has to keep
/// `k / cs <= 1024` (one K-tile pass in TMEM) while still filling the SMs.
pub fn cluster_for(n: usize, k: usize, sm_count: usize) -> Option<usize> {
    [8usize, 5, 4, 2, 1].into_iter().find(|&c| {
        k % (128 * c) == 0 && k / c <= 1024 && (n / 128) * c <= sm_count && (n / 128) * c >= 96
    })
}

/// One weight matrix in the tensor-core layout.
pub struct TcWeight {
    pub tmap: cu::Buffer,
    pub sf: cu::Buffer,
    pub n: usize,
    pub k: usize,
    pub cs: usize,
    pub smem: usize,
    /// `(rows, k)` sized activation buffers, kept across steps. Reallocating per decoded
    /// token would cost more than the kernel itself.
    scratch: std::sync::Mutex<Option<(usize, cu::Buffer, cu::Buffer)>>,
}

impl TcWeight {
    /// `base` is the already-padded row-major fp8 weight, `sf_tiled` its e8m0 scale tiles
    /// (`formats::tile_sf_blocks`), `nt` the TMA box height in rows (64, 128 or 256).
    pub fn new(base: cu::DevPtr, n: usize, k: usize, sf_tiled: &[u8], nt: u32) -> Result<Self> {
        ensure!(n % 128 == 0 && k % 128 == 0, "tc: n and k must be multiples of 128");
        ensure!(sf_tiled.len() == (n / 128) * (k / 128) * 512, "tc: scale tile has wrong size");
        let sm_count = crate::kernels::sm_count();
        let cs = cluster_for(n, k, sm_count)
            .ok_or_else(|| anyhow::anyhow!("tc: no cluster split fits n={n} k={k}"))?;
        let map = cu::tensor_map_u8_sw128(base, k as u64, n as u64, k as u64, nt)?;
        let ks = k / cs;
        let kt = ks / 128;
        let smem = kt * (16384 + 1024 + 512 + 512) + 8 * ks * 2 + ks * 2 + 8 * 128 * 4 + 1040 * 4 + 1024;
        Ok(TcWeight {
            tmap: cu::Buffer::from_host(&map)?,
            sf: cu::Buffer::from_host(sf_tiled)?,
            n,
            k,
            cs,
            smem,
            scratch: std::sync::Mutex::new(None),
        })
    }

    /// Decode GEMV: `out[m, n] = x[m, k] @ w[n, k]^T`. `m` is the verification width.
    pub fn gemv(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        x: cu::DevPtr,
        ldx: usize,
        m: usize,
        out: cu::DevPtr,
        ldo: usize,
    ) -> Result<()> {
        ensure!(m <= 8, "tc gemv: m <= 8 (kernel bound)");
        ensure!(ldx == self.k, "tc gemv: x must be contiguous over k");
        let f = kern.get("tc_gemv_fp8")?;
        f.set_max_dynamic_smem(self.smem as u32)?;
        let p = TvParams {
            tmap: self.tmap.ptr,
            sfa: self.sf.ptr,
            x,
            out,
            alpha: 1.0,
            m: m as i32,
            n: self.n as i32,
            k: self.k as i32,
            kpt: (self.k / 128) as i32,
            ldx: ldx as i32,
            ldo: ldo as i32,
            ..Default::default()
        };
        let gx = (self.n / 128) as u32;
        let mut cfg = cu::Launch::new(gx, 128).grid3((gx, self.cs as u32, 1)).smem(self.smem as u32);
        cfg.cluster = Some((1, self.cs as u32, 1));
        cu::launch(f, &cfg, s, cu::Args::new().push(p))
    }
}

impl TcWeight {
    /// Quantize bf16 activations and run the decode GEMV in one call.
    pub fn gemv_bf16(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        x: cu::DevPtr,
        m: usize,
        out: cu::DevPtr,
        ldo: usize,
    ) -> Result<()> {
        ensure!(m <= 8, "tc gemv: m <= 8 (kernel bound)");
        let rows = m.next_multiple_of(128);
        let mut slot = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let stale = slot.as_ref().map(|(r, _, _)| *r != rows).unwrap_or(true);
        if stale {
            *slot = Some((rows, cu::Buffer::alloc(rows * self.k)?, cu::Buffer::alloc((rows / 128) * (self.k / 128) * 512)?));
        }
        let (_, aq, sfa) = slot.as_ref().unwrap();
        quant_mx_into(kern, s, x, self.k, m, self.k, aq.ptr, sfa.ptr)?;
        self.gemv(kern, s, aq.ptr, self.k, m, out, ldo)
    }
}

/// Quantize `rows x k` bf16 activations into e4m3 plus e8m0 scale tiles.
pub fn quant_mx(
    kern: &glm_kernels::Kernels,
    s: &cu::Stream,
    x: cu::DevPtr,
    lda: usize,
    real: usize,
    k: usize,
) -> Result<(cu::Buffer, cu::Buffer)> {
    let rows = real.next_multiple_of(128);
    ensure!(rows % 128 == 0 && k % 128 == 0, "quant_mx: rows and k must be multiples of 128");
    let aq = cu::Buffer::alloc(rows * k)?;
    let sfa = cu::Buffer::alloc((rows / 128) * (k / 128) * 512)?;
    quant_mx_into(kern, s, x, lda, real, k, aq.ptr, sfa.ptr)?;
    Ok((aq, sfa))
}

/// Same as [`quant_mx`] but writes into caller-owned buffers.
pub fn quant_mx_into(
    kern: &glm_kernels::Kernels,
    s: &cu::Stream,
    x: cu::DevPtr,
    lda: usize,
    real: usize,
    k: usize,
    aq: cu::DevPtr,
    sfa: cu::DevPtr,
) -> Result<()> {
    let rows = real.next_multiple_of(128);
    ensure!(rows % 128 == 0 && k % 128 == 0, "quant_mx: rows and k must be multiples of 128");
    let p = QmxParams {
        x,
        aq,
        sfa,
        rows: rows as i32,
        real: real as i32,
        k: k as i32,
        lda: lda as i32,
        kt: (k / 128) as i32,
    };
    let cfg = cu::Launch::new(rows as u32, 256);
    cu::launch(kern.get("glm_quant_mx")?, &cfg, s, cu::Args::new().push(p))
}
