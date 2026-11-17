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

/// Gen-2 GEMV switch (`GLM_TC2=0` pins the first generation).
pub fn gemv2_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("GLM_TC2").as_deref() != Ok("0"))
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
    /// Pre-norm weight: when `norm` is set the RMSNorm is fused into the quantizer.
    pub pw: u64,
    pub aq: u64,
    pub sfa: u64,
    pub rows: i32,
    pub real: i32,
    pub k: i32,
    pub lda: i32,
    pub kt: i32,
    pub eps: f32,
    pub norm: i32,
}

/// Mirror of `GemvParams` in `gemv.cu` — the weight-tiled `mma.sync` GEMV family
/// (`gemv_fp8_n*_p*`, `gemv_fp4_n*_p*`). The struct is passed by value, so the field order and
/// every scalar width have to match the C definition exactly; the size assert below is the
/// contract.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GemvParams {
    pub w: u64,
    pub ws: u64,
    pub x: u64,
    pub out: u64,
    pub M: i32,
    pub N: i32,
    pub K: i32,
    pub ldx: i32,
    pub ldo: i32,
    pub out_f32: i32,
    pub x_group_rows: i32,
    pub alpha: f32,
    pub stage_steps: i32,
    pub plan: u64,
    pub w_stride: u64,
    pub mode: i32,
    pub swiglu_limit: f32,
    pub prenorm_w: u64,
    pub prenorm_eps: f32,
    pub norm_part: u64,
    pub norm_nslices: i32,
    pub ksplit_slots: i32,
    pub xq: u64,
    pub xsq: u64,
    pub oq: u64,
    pub osq: u64,
    pub sched: u64,
    pub out2: u64,
    pub ldo2: i32,
    pub n_split: i32,
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub rope_pos: u64,
    pub rope_head: i32,
    pub rope_off: i32,
    pub w_sh: u64,
    pub r_scores: u64,
    pub r_bias: u64,
    pub r_ids: u64,
    pub r_w: u64,
    pub r_nexp: i32,
    pub r_topk: i32,
    pub r_scale: f32,
    pub r_nsum: i32,
    pub kvr: RopeBlob,
    pub kvr_on: i32,
    pub out_pos: u64,
    pub iq_head: i32,
    pub iq_off: i32,
    pub sp: u64,
    pub spart: u64,
    pub sp_vloc: i32,
    pub sp_off: i32,
    pub xshift: i32,
    pub _padx: i32,
}

const _: () = assert!(std::mem::size_of::<GemvParams>() == 432);

/// Mirror of `Tg2Params` in tcgemv2.cu (352 bytes) — the second-generation tensor-core GEMV.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Tg2Params {
    pub tmap: u64,
    pub sfa: u64,
    pub x: u64,
    pub gamma: u64,
    pub bimg: u64,
    pub rs_part: u64,
    pub out: u64,
    pub out2: u64,
    pub rope_cos: u64,
    pub rope_sin: u64,
    pub meta: u64,
    pub qimg: u64,
    pub qgamma: u64,
    pub qssq: u64,
    pub kvr: RopeBlob,
    pub alpha: f32,
    pub eps: f32,
    pub M: i32,
    pub N: i32,
    pub K: i32,
    pub kpt: i32,
    pub ldx: i32,
    pub ldo: i32,
    pub ldo2: i32,
    pub n_split: i32,
    pub out_f32: i32,
    pub bsrc: i32,
    pub rs_n: i32,
    pub rs_stride: i32,
    pub rs_rowstride: i32,
    pub rs_off: i32,
    pub rope_head: i32,
    pub rope_off: i32,
    pub iq_head: i32,
    pub iq_off: i32,
    pub partial: i32,
    pub group_rows: i32,
    pub qrows: i32,
    pub kvr_on: i32,
    pub xq: u64,
    pub xsq: u64,
    pub ssp: i32,
    pub rs_pre: i32,
    pub nacc: i32,
    pub bload: i32,
    pub flags: i32,
}

const _: () = assert!(std::mem::size_of::<Tg2Params>() == 352);

impl TcWeight {
    /// Second-generation decode GEMV (`tc_gemv2_*`): same contract as [`TcWeight::gemv`], but the
    /// kernel pipelines the weight come in one persistent pass and takes pre-quantised rows.
    /// `aq`/`sfa` are the pre-quantised activation image and its e8m0 scales (`bsrc == 0`).
    pub fn gemv2(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        aq: cu::DevPtr,
        sfa: cu::DevPtr,
        m: usize,
        out: cu::DevPtr,
        ldo: usize,
    ) -> Result<()> {
        ensure!(m <= 8, "tc gemv2: m <= 8 (kernel bound)");
        let f = kern.get("tc_gemv2_fp8")?;
        f.set_max_dynamic_smem(self.smem as u32)?;
        let p = Tg2Params {
            tmap: self.tmap.ptr,
            sfa: self.sf.ptr,
            out,
            alpha: 1.0,
            M: m as i32,
            N: self.n as i32,
            K: self.k as i32,
            kpt: (self.k / 128) as i32,
            ldx: self.k as i32,
            ldo: ldo as i32,
            out_f32: 1,
            // bsrc 0 selects the pre-quantised activation image
            bsrc: 0,
            xq: aq,
            xsq: sfa,
            ..Default::default()
        };
        let gx = (self.n / 128) as u32;
        let mut cfg = cu::Launch::new(gx, 128).grid3((gx, self.cs as u32, 1)).smem(self.smem as u32);
        cfg.cluster = Some((1, self.cs as u32, 1));
        cu::launch(f, &cfg, s, cu::Args::new().push(p))
    }

    /// Quantise `x` in place into the activation image, then run the gen-2 GEMV.
    pub fn gemv2_bf16(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        x: cu::DevPtr,
        m: usize,
        out: cu::DevPtr,
        ldo: usize,
    ) -> Result<()> {
        ensure!(m <= 8, "tc gemv2: m <= 8 (kernel bound)");
        let rows = m.next_multiple_of(128);
        let mut slot = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let stale = slot.as_ref().map(|(r, _, _)| *r != rows).unwrap_or(true);
        if stale {
            *slot = Some((rows, cu::Buffer::alloc(rows * self.k)?, cu::Buffer::alloc((rows / 128) * (self.k / 128) * 512)?));
        }
        let (_, aq, sfa) = slot.as_ref().unwrap();
        quant_mx_into(kern, s, x, self.k, m, self.k, aq.ptr, sfa.ptr, None)?;
        self.gemv2(kern, s, aq.ptr, sfa.ptr, m, out, ldo)
    }
}

/// Mirror of `TgParams` in `tcgemm.cu` (prefill GEMM). Offsets taken from the compiled
/// C++ struct; `ta` is the activation map, `tb` the weight map.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TgParams {
    pub ta: u64,
    pub tb: u64,
    pub sfa: u64,
    pub sfb: u64,
    pub out: u64,
    pub ldo: i32,
    pub out_f32: i32,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub nt: i32,
    pub alpha: f32,
    pub out2: u64,
    pub ldo2: i32,
    pub n_split: i32,
    pub swiglu: i32,
    pub swiglu_limit: f32,
}

const _: () = assert!(std::mem::size_of::<TgParams>() == 96);

/// Mirror of `TqParams`: the norm+quant prologue that turns bf16 activations into the
/// e4m3 payload and e8m0 tiles the GEMM streams.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TqParams {
    pub x: u64,
    pub ldx: i32,
    pub m: i32,
    pub k: i32,
    pub pad: i32,
    pub pw: u64,
    pub norm_part: u64,
    pub nslices: i32,
    pub eps: f32,
    pub aq: u64,
    pub sfa: u64,
}

const _: () = assert!(std::mem::size_of::<TqParams>() == 64);

/// Shared-memory footprint of `tc_gemm_fp8` for a given box height.
pub fn tg_smem(nt: usize) -> u32 {
    let sb = (16384 + nt * 128 + 512 + (nt / 128).max(1) * 512 + 1023) & !1023;
    (4 * sb + 1024) as u32
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
    /// Prefill scratch: activation payload, its scale tiles, and the TMA map over them.
    gemm_scratch: std::sync::Mutex<Option<(usize, cu::Buffer, cu::Buffer, cu::Buffer)>>,
}

impl TcWeight {
    /// `base` is the already-padded row-major fp8 weight, `sf_tiled` its e8m0 scale tiles
    /// (`formats::tile_sf_blocks`), `nt` the TMA box height in rows (64, 128 or 256).
    pub fn new(base: cu::DevPtr, n: usize, k: usize, sf_tiled: &[u8], nt: u32) -> Result<Option<Self>> {
        ensure!(n % 128 == 0 && k % 128 == 0, "tc: n and k must be multiples of 128");
        ensure!(sf_tiled.len() == (n / 128) * (k / 128) * 512, "tc: scale tile has wrong size");
        let sm_count = crate::kernels::sm_count();
        // Narrow shapes (router, kv projections) have no cluster split that covers them; those
        // matrices stay on the FMA path instead of failing the whole load.
        let Some(cs) = cluster_for(n, k, sm_count) else {
            // shape too narrow for a cluster split: stays on the FMA path, but say so once
            static NOTE: std::sync::Once = std::sync::Once::new();
            NOTE.call_once(|| {
                eprintln!("note: some dense projections are too narrow for a tc cluster split and stay on the FMA kernels")
            });
            return Ok(None);
        };
        let map = cu::tensor_map_u8_sw128(base, k as u64, n as u64, k as u64, nt)?;
        let ks = k / cs;
        let kt = ks / 128;
        let smem = kt * (16384 + 1024 + 512 + 512) + 8 * ks * 2 + ks * 2 + 8 * 128 * 4 + 1040 * 4 + 1024;
        Ok(Some(TcWeight {
            tmap: cu::Buffer::from_host(&map)?,
            sf: cu::Buffer::from_host(sf_tiled)?,
            n,
            k,
            cs,
            smem,
            scratch: std::sync::Mutex::new(None),
            gemm_scratch: std::sync::Mutex::new(None),
        }))
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
            out_f32: 1, // every caller passes an f32 destination and an f32 pitch
            ..Default::default()
        };
        let gx = (self.n / 128) as u32;
        let mut cfg = cu::Launch::new(gx, 128).grid3((gx, self.cs as u32, 1)).smem(self.smem as u32);
        cfg.cluster = Some((1, self.cs as u32, 1));
        cu::launch(f, &cfg, s, cu::Args::new().push(p))
    }
}

/// Four-stage weight-box pipeline lives inside `tc_gemm_fp8`; this wrapper only has to
/// hand it the two tensor maps and the matching scale tiles.
impl TcWeight {
    /// Prefill GEMM: `out[m, n] = x[m, k] @ w[n, k]^T`, activations quantized on the fly.
    ///
    /// `m` is bounded by the caller's prefill chunk. The kernel reads full 128-row TMA
    /// boxes, so the activation image is padded up to that boundary; the epilogue is bounded
    /// by `m`, which is why the caller's output buffer needs no padding of its own.
    pub fn gemm(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        x: cu::DevPtr,
        m: usize,
        out: cu::DevPtr,
        ldo: usize,
    ) -> Result<()> {
        self.gemm_norm(kern, s, x, m, None, out, ldo)
    }

    /// Same GEMM with the RMSNorm fused into the activation quantizer: the row never has to
    /// be written back as bf16 and re-read, which is one launch and one full pass saved.
    pub fn gemm_norm(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        x: cu::DevPtr,
        m: usize,
        norm: Option<(cu::DevPtr, f32)>,
        out: cu::DevPtr,
        ldo: usize,
    ) -> Result<()> {
        ensure!(self.n % 128 == 0 && self.k % 128 == 0, "tc gemm: alignment");
        let mp = m.next_multiple_of(128);
        let ta = {
            let mut slot = self.gemm_scratch.lock().unwrap_or_else(|e| e.into_inner());
            let stale = slot.as_ref().map(|(r, _, _, _)| *r < mp).unwrap_or(true);
            if stale {
                let aq = cu::Buffer::alloc(mp * self.k)?;
                let sfa = cu::Buffer::alloc((mp / 128) * (self.k / 128) * 512)?;
                let map = cu::tensor_map_u8_sw128(aq.ptr, self.k as u64, mp as u64, self.k as u64, 128)?;
                // the map has to outlive every launch that reads it, so it is owned here
                let tmap = cu::Buffer::from_host(&map)?;
                *slot = Some((mp, aq, sfa, tmap));
            }
            let g = slot.as_ref().unwrap();
            quant_mx_into(kern, s, x, self.k, m, self.k, g.1.ptr, g.2.ptr, norm)?;
            (g.3.ptr, g.2.ptr)
        };
        let (ta, sfa) = ta;
        let gp = TgParams {
            ta,
            tb: self.tmap.ptr,
            sfa,
            sfb: self.sf.ptr,
            out,
            ldo: ldo as i32,
            m: m as i32,
            n: self.n as i32,
            k: self.k as i32,
            nt: 128,
            alpha: 1.0,
            ldo2: self.n as i32,
            n_split: self.n as i32,
            out_f32: 1, // f32 destination, f32 pitch
            ..Default::default()
        };
        let smem = tg_smem(128);
        let f = kern.get("tc_gemm_fp8")?;
        f.set_max_dynamic_smem(smem)?;
        let gx = (self.n / 128) as u32;
        let mt = (m as u32).div_ceil(128);
        cu::launch(f, &cu::Launch::new(gx, 128).grid3((gx, mt, 1)).smem(smem), s, cu::Args::new().push(gp))
    }

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
        quant_mx_into(kern, s, x, self.k, m, self.k, aq.ptr, sfa.ptr, None)?;
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
    quant_mx_into(kern, s, x, lda, real, k, aq.ptr, sfa.ptr, None)?;
    Ok((aq, sfa))
}

/// Same as [`quant_mx`] but writes into caller-owned buffers.
#[allow(clippy::too_many_arguments)]
pub fn quant_mx_into(
    kern: &glm_kernels::Kernels,
    s: &cu::Stream,
    x: cu::DevPtr,
    lda: usize,
    real: usize,
    k: usize,
    aq: cu::DevPtr,
    sfa: cu::DevPtr,
    norm: Option<(cu::DevPtr, f32)>,
) -> Result<()> {
    let rows = real.next_multiple_of(128);
    ensure!(rows % 128 == 0 && k % 128 == 0, "quant_mx: rows and k must be multiples of 128");
    let p = QmxParams {
        x,
        pw: norm.map_or(0, |n| n.0),
        aq,
        sfa,
        rows: rows as i32,
        real: real as i32,
        k: k as i32,
        lda: lda as i32,
        kt: (k / 128) as i32,
        eps: norm.map_or(0.0, |n| n.1),
        norm: i32::from(norm.is_some()),
    };
    let cfg = cu::Launch::new(rows as u32, 256);
    cu::launch(kern.get("glm_quant_mx")?, &cfg, s, cu::Args::new().push(p))
}
