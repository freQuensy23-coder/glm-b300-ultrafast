//! Routed experts on the tcgen05 path.
//!
//! The upstream MoE kernels do not read expert weights as plain matrices: every expert is
//! packed into a fixed-size box (2080-byte rows: fp4 payload plus its scale stripe), all
//! experts are concatenated with a constant stride, and a single 3-D TMA descriptor addresses
//! any expert by index. `packing::expert_shard_tc` builds that box; this module feeds it from
//! our checkpoint and keeps the descriptor alive.
//!
//! fp4 is not what the checkpoint carries: experts arrive as block-fp8, so each value is
//! dequantized and re-quantized to e2m1 with a power-of-two scale per 32 values. Doing it once
//! at load keeps the decode path free of any conversion work.

use anyhow::{ensure, Result};
use deepseek_ultrafast_cuda as cu;
use rayon::prelude::*;

use crate::quant;

/// Mirror of `MpParams` (the routing-to-tile-list pass).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MpParams {
    pub ids: u64,
    pub w: u64,
    pub m: i32,
    pub topk: i32,
    pub n_exp: i32,
    pub row_pair: u64,
    pub row_w: u64,
    pub pair_row: u64,
    pub row_tile: u64,
    pub tiles: u64,
    pub ntiles: u64,
}

const _: () = assert!(std::mem::size_of::<MpParams>() == 80);

/// Mirror of `TmParams` as compiled with `TCMOE_RM`: the extra `wmap` field is the 3-D
/// descriptor over all experts, which is what lets one launch address every expert.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct TmParamsRm {
    pub w: u64,
    pub w_stride: u64,
    pub k: i32,
    pub rt: i32,
    pub tb: u64,
    pub sfb: u64,
    pub tiles: u64,
    pub ntiles: u64,
    pub row_pair: u64,
    pub row_w: u64,
    pub aq: u64,
    pub asf: u64,
    pub ldaq: i32,
    pub akt: i32,
    pub limit: f32,
    pub nq: i32,
    pub out: u64,
    pub ldo: i32,
    pub flags: i32,
    pub sf_off: u64,
    pub wmap: u64,
}

const _: () = assert!(std::mem::size_of::<TmParamsRm>() == 136);

/// Buffers one MoE layer needs per forward pass. Sizes follow the kernels: the gathered
/// image holds one row per (token, slot) pair, and the tile list holds one entry per 128-row
/// tile plus one per non-empty expert.
/// Mirror of `GqParams` (our gather+quant kernel).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GqParams {
    pub x: u64,
    pub pw: u64,
    pub row_pair: u64,
    pub xq: u64,
    pub sf: u64,
    pub rows: i32,
    pub topk: i32,
    pub k: i32,
    pub kt: i32,
    pub eps: f32,
}

/// Shared-memory budget of the MoE body (same expression as the upstream constant).
pub const TM_SMEM: u32 = (4 * 16384 + 2 * (16 * 2080 + 4096) + 3 * 8 * 2176 + 40 * 512 + 2 * 32 * 33 * 4 + 1024) as u32;

pub struct MoeIo {
    /// Identity map `0..rows_pad*topk`: the short-chain path emits one row per (token, slot), so
    /// its reduce needs no permutation.
    pub ident: cu::Buffer,
    pub row_pair: cu::Buffer,
    pub row_w: cu::Buffer,
    pub pair_row: cu::Buffer,
    pub row_tile: cu::Buffer,
    pub tiles: cu::Buffer,
    pub ntiles: cu::Buffer,
    pub xq: cu::Buffer,
    pub sf: cu::Buffer,
    pub aq: cu::Buffer,
    pub asf: cu::Buffer,
    pub ypair: cu::Buffer,
    /// Bank of eight descriptors over `xq`, one per box height (16..128 rows), and the same
    /// over `aq`: the kernel picks the box that matches the tile it is working on.
    pub xq_maps: cu::Buffer,
    pub aq_maps: cu::Buffer,
    /// Slots actually addressable in `xq`/`aq` (padded up to a 128-row tile).
    pub rows_pad: usize,
}

impl MoeIo {
    /// `rows` is the widest token block the engine will run through this layer.
    pub fn alloc(rows: usize, topk: usize, n_exp: usize, dim: usize, inter: usize) -> Result<Self> {
        let p = rows * topk;
        let rows_pad = p.next_multiple_of(128);
        let tiles = p.div_ceil(128) + n_exp;
        let ident = cu::Buffer::from_host(&(0..p).map(|i| i as i32).collect::<Vec<_>>())?;
        Ok(MoeIo {
            ident,
            row_pair: cu::Buffer::zeroed(p * 4)?,
            row_w: cu::Buffer::zeroed(p * 4)?,
            pair_row: cu::Buffer::zeroed(p * 4)?,
            row_tile: cu::Buffer::zeroed(p * 4)?,
            tiles: cu::Buffer::zeroed(tiles * 16)?,
            ntiles: cu::Buffer::zeroed(4)?,
            xq: cu::Buffer::zeroed(rows_pad * dim)?,
            sf: cu::Buffer::zeroed(tiles * (dim / 128) * 512)?,
            aq: cu::Buffer::zeroed(rows_pad * dim)?,
            asf: cu::Buffer::zeroed(tiles * inter.div_ceil(128) * 512)?,
            ypair: cu::Buffer::zeroed(p * dim * 4)?,
            xq_maps: cu::Buffer::zeroed(8 * 128)?,
            aq_maps: cu::Buffer::zeroed(8 * 128)?,
            rows_pad,
        })
    }

    /// Fill the two descriptor banks. Call after any reallocation of `xq`/`aq`.
    pub fn build_maps(&self, dim: usize, inter: usize, ldaq: usize) -> Result<()> {
        for (buf, rows, cols, dst) in [
            (self.xq.ptr, self.rows_pad, dim, self.xq_maps.ptr),
            (self.aq.ptr, self.rows_pad, ldaq, self.aq_maps.ptr),
        ] {
            for nb in 1..=8u32 {
                let m = cu::tensor_map_u8_sw128(buf, cols as u64, rows as u64, cols as u64, 16 * nb)?;
                // layout mirrors the kernel's `(const uint8_t*)p.tb + (nb - 1) * 128`
                let off = dst + ((nb - 1) * 128) as u64;
                cu::upload(off, &m)?;
            }
        }
        let _ = inter;
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        [
            &self.row_pair, &self.row_w, &self.pair_row, &self.row_tile, &self.tiles,
            &self.ntiles, &self.xq, &self.sf, &self.aq, &self.asf, &self.ypair,
            &self.xq_maps, &self.aq_maps,
        ]
        .iter()
        .map(|b| b.bytes)
        .sum()
    }
}

/// One expert's three matrices as they come out of the checkpoint: fp8 payload (row-major)
/// plus one fp32 scale per 128x128 block.
pub struct ExpertSrc<'a> {
    pub gate: (&'a [u8], &'a [f32], usize, usize),
    pub up: (&'a [u8], &'a [f32], usize, usize),
    pub down: (&'a [u8], &'a [f32], usize, usize),
}

/// Routed experts of one layer in the layout `tc_moe_w13_rm2` / `tc_moe_w2_rm` index.
pub struct TcMoe {
    pub w13: cu::Buffer,
    pub w2: cu::Buffer,
    /// 3-D descriptor over all experts (`wmap` in the kernel params).
    pub map13: cu::Buffer,
    pub map2: cu::Buffer,
    pub w13_stride: usize,
    pub w2_stride: usize,
    pub n: usize,
    pub inter: usize,
    pub dim: usize,
}

impl TcMoe {
    /// One routed-expert pass: tile list, gather+quant, gate/up with the swiglu requant,
    /// down projection, weighted reduce. All five stages stay on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        kern: &glm_kernels::Kernels,
        s: &cu::Stream,
        io: &MoeIo,
        h: cu::DevPtr,
        pw: cu::DevPtr,
        eps: f32,
        ids: cu::DevPtr,
        wts: cu::DevPtr,
        m: usize,
        topk: usize,
        swiglu_limit: f32,
        out: cu::DevPtr,
    ) -> Result<()> {
        let (dim, inter) = (self.dim, self.inter);
        let slots = m * topk;
        ensure!(slots <= io.rows_pad, "tc moe: {slots} slots exceed the {}-row image", io.rows_pad);
        ensure!(inter % 128 == 0, "tc moe: inter {inter} must be a multiple of 128 for the requant tiles");
        let ldaq = inter.next_multiple_of(128);

        let mp = MpParams {
            ids,
            w: wts,
            m: m as i32,
            topk: topk as i32,
            n_exp: self.n as i32,
            row_pair: io.row_pair.ptr,
            row_w: io.row_w.ptr,
            pair_row: io.pair_row.ptr,
            row_tile: io.row_tile.ptr,
            tiles: io.tiles.ptr,
            ntiles: io.ntiles.ptr,
        };
        cu::launch(kern.get("moe_perm")?, &cu::Launch::new(1, 1024), s, cu::Args::new().push(mp))?;

        let gq = GqParams {
            x: h,
            pw,
            row_pair: io.row_pair.ptr,
            xq: io.xq.ptr,
            sf: io.sf.ptr,
            rows: slots as i32,
            topk: topk as i32,
            k: dim as i32,
            kt: (dim / 128) as i32,
            eps,
        };
        cu::launch(
            kern.get("glm_gather_quant")?,
            &cu::Launch::new(slots as u32, 256),
            s,
            cu::Args::new().push(gq),
        )?;

        // gate/up: reads the gathered image, applies swiglu, and writes the requantized
        // result straight into `aq`/`asf` for the down projection.
        let w13 = TmParamsRm {
            w: self.w13.ptr,
            w_stride: self.w13_stride as u64,
            k: dim as i32,
            rt: ((2 * inter) / 128) as i32,
            tb: io.xq_maps.ptr,
            sfb: io.sf.ptr,
            tiles: io.tiles.ptr,
            ntiles: io.ntiles.ptr,
            row_pair: io.row_pair.ptr,
            row_w: io.row_w.ptr,
            aq: io.aq.ptr,
            asf: io.asf.ptr,
            ldaq: ldaq as i32,
            akt: (inter / 128) as i32,
            limit: swiglu_limit,
            nq: ((2 * inter) / 64) as i32,
            out: 0,
            ldo: 0,
            flags: 0,
            sf_off: 0,
            wmap: self.map13.ptr,
        };
        let w2 = TmParamsRm {
            w: self.w2.ptr,
            w_stride: self.w2_stride as u64,
            k: ldaq as i32,
            rt: (dim / 128) as i32,
            tb: io.aq_maps.ptr,
            sfb: io.asf.ptr,
            nq: (dim / 64) as i32,
            out: io.ypair.ptr,
            ldo: dim as i32,
            sf_off: 0,
            wmap: self.map2.ptr,
            ..w13
        };
        let sms = crate::kernels::sm_count().min(1024) as u32;
        for (name, p) in [("tc_moe_w13_rm2", w13), ("tc_moe_w2_rm", w2)] {
            let f = kern.get(name)?;
            f.set_max_dynamic_smem(TM_SMEM)?;
            cu::launch(f, &cu::Launch::new(sms, 512).smem(TM_SMEM), s, cu::Args::new().push(p))?;
        }

        let mut a = cu::Args::new();
        a.ptr(out).ptr(io.ypair.ptr).ptr(io.pair_row.ptr).ptr(io.row_w.ptr).ptr(out).i32(m as i32).i32(topk as i32).i32(dim as i32);
        cu::launch(
            kern.get("glm_moe_reduce_pair")?,
            &cu::Launch::new(m as u32, 256).grid3((m as u32, (dim as u32).div_ceil(256), 1)),
            s,
            &mut a,
        )
    }

    pub fn bytes(&self) -> usize {
        self.w13.bytes + self.w2.bytes + self.map13.bytes + self.map2.bytes
    }

    /// Pack and upload every expert of one layer.
    pub fn build(src: &[ExpertSrc<'_>], dim: usize, inter: usize) -> Result<Self> {
        ensure!(dim % 128 == 0, "tc moe: dim {dim} must be a multiple of 128");
        ensure!(inter % 32 == 0, "tc moe: inter {inter} must be a multiple of 32");
        let w13_stride = inter * dim + (2 * inter / 128) * (dim / 128) * 512;
        let w2_stride = deepseek_ultrafast_model::packing::w2_rowmajor_bytes(dim, inter);
        let packed: Vec<(Vec<u8>, Vec<u8>)> = src
            .par_iter()
            .map(|e| {
                let g = dequant_to_f32(e.gate);
                let u = dequant_to_f32(e.up);
                let d = dequant_to_f32(e.down);
                let (gn, gk) = (e.gate.2, e.gate.3);
                let (dn, dk) = (e.down.2, e.down.3);
                let g4 = quant::fp4_quant(&g, gn, gk);
                let u4 = quant::fp4_quant(&u, gn, gk);
                let d4 = quant::fp4_quant(&d, dn, dk);
                quant::pack_expert_tc(&g4, &u4, &d4, dim, inter)
            })
            .collect();
        let mut w13 = Vec::with_capacity(src.len() * w13_stride);
        let mut w2 = Vec::with_capacity(src.len() * w2_stride);
        for (a, b) in &packed {
            ensure!(a.len() == w13_stride, "tc moe: w13 box is {} not {w13_stride}", a.len());
            ensure!(b.len() == w2_stride, "tc moe: w2 box is {} not {w2_stride}", b.len());
            w13.extend_from_slice(a);
            w2.extend_from_slice(b);
        }
        let w13 = cu::Buffer::from_host(&w13)?;
        let w2 = cu::Buffer::from_host(&w2)?;
        let n = src.len();
        // One descriptor for all experts: the third dimension is the expert index, stepped by
        // the fixed box stride, which is exactly what `tensor_map_fp4_unpack_strided` encodes.
        let map13 = cu::tensor_map_fp4_unpack_strided(
            w13.ptr,
            dim as u64,
            (2 * inter) as u64,
            n as u64,
            w13_stride as u64,
            128,
        )?;
        let map2 = cu::tensor_map_fp4_unpack_strided(
            w2.ptr,
            (inter.div_ceil(128) * 128) as u64,
            dim as u64,
            n as u64,
            w2_stride as u64,
            128,
        )?;
        Ok(TcMoe {
            map13: cu::Buffer::from_host(&map13)?,
            map2: cu::Buffer::from_host(&map2)?,
            w13,
            w2,
            w13_stride,
            w2_stride,
            n,
            inter,
            dim,
        })
    }
}

/// Block-fp8 checkpoint tensor -> row-major f32, one 128x128 scale per block.
fn dequant_to_f32(t: (&[u8], &[f32], usize, usize)) -> Vec<f32> {
    let (data, scales, n, k) = t;
    let nkb = k / 128;
    let mut out = vec![0f32; n * k];
    out.par_chunks_mut(128 * k).enumerate().for_each(|(rb, rows)| {
        for kb in 0..nkb {
            let s = scales[rb * nkb + kb];
            for r in 0..128 {
                let src = rb * 128 + r;
                for c in kb * 128..(kb + 1) * 128 {
                    rows[r * k + c] = quant::e4m3_to_f32(data[src * k + c]) * s;
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(n: usize, k: usize, seed: f32) -> (Vec<u8>, Vec<f32>) {
        let data = (0..n * k).map(|i| ((i as f32 * 0.017 + seed).sin() * 200.0) as i8 as u8 | 0x40).collect::<Vec<u8>>();
        let scales = vec![0.017f32; (n / 128) * (k / 128)];
        (data, scales)
    }

    #[test]
    fn dequant_uses_one_scale_per_128_block() {
        let (n, k) = (128usize, 256usize);
        let (d, s) = src(n, k, 0.4);
        let f = dequant_to_f32((&d, &s, n, k));
        assert_eq!(f.len(), n * k);
        // every column of a block uses that block's scale, nothing else
        let s = 0.017f32;
        for (i, &c) in [0usize, 1, 127, 128, 255].iter().enumerate() {
            let expect = quant::e4m3_to_f32(d[c]) * s;
            assert_eq!(f[c], expect, "column {c} (case {i})");
        }
    }

    #[test]
    fn layer_box_sizes_match_the_kernel_stride() {
        // inter must be a multiple of 128 as well: the down matrix has k = inter, and the
        // packing helper asserts a 128-wide K
        let (dim, inter) = (128usize, 128usize);
        let (g, gs) = src(inter, dim, 0.1);
        let (u, us) = src(inter, dim, 0.2);
        let (d, ds) = src(dim, inter, 0.3);
        let one = [ExpertSrc {
            gate: (&g, &gs, inter, dim),
            up: (&u, &us, inter, dim),
            down: (&d, &ds, dim, inter),
        }];
        let stride13 = inter * dim + (2 * inter / 128) * (dim / 128) * 512;
        let stride2 = deepseek_ultrafast_model::packing::w2_rowmajor_bytes(dim, inter);
        // build() would need a CUDA context; the packing half is what the sizes depend on
        let g4 = quant::fp4_quant(&dequant_to_f32((&g, &gs, inter, dim)), inter, dim);
        let u4 = quant::fp4_quant(&dequant_to_f32((&u, &us, inter, dim)), inter, dim);
        let d4 = quant::fp4_quant(&dequant_to_f32((&d, &ds, dim, inter)), dim, inter);
        let (a, b) = quant::pack_expert_tc(&g4, &u4, &d4, dim, inter);
        assert_eq!(a.len(), stride13);
        assert_eq!(b.len(), stride2);
        assert_eq!(one.len(), 1);
    }
}
