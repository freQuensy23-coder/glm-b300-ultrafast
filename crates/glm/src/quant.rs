//! Weight quantization to the layout the kernels consume: row-major fp8 e4m3 with
//! one fp32 scale per 128x128 block, plus zero padding to multiples of 128.
//!
//! This is the layout Hugging Face `*-block-fp8` checkpoints already use, so an fp8
//! checkpoint is uploaded without requantization and a bf16 checkpoint is quantized
//! on load with the same block grid.

use anyhow::{ensure, Result};
use deepseek_ultrafast_model::formats::{e4m3_to_f32, f32_to_e4m3_fast};
use half::bf16;
use rayon::prelude::*;

pub const BLK: usize = 128;

#[derive(Clone, Debug)]
pub struct Fp8Block {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    /// Padded output-row count and reduction width.
    pub n: usize,
    pub k: usize,
}

impl Fp8Block {
    pub fn rows_of_scales(&self) -> usize {
        self.n / BLK
    }

    pub fn cols_of_scales(&self) -> usize {
        self.k / BLK
    }

    pub fn scale_at(&self, row: usize, kb: usize) -> f32 {
        self.scales[(row / BLK) * self.cols_of_scales() + kb]
    }

    /// Dequantize one element; used by tests and by the reference checker.
    pub fn dequant(&self, r: usize, c: usize) -> f32 {
        e4m3_to_f32(self.data[r * self.k + c]) * self.scale_at(r, c / BLK)
    }
}

fn pad_to(v: usize, m: usize) -> usize {
    v.next_multiple_of(m)
}

/// e4m3 encode with round-to-nearest, saturating at the format maximum.
pub fn e4m3_from_f32(v: f32) -> u8 {
    f32_to_e4m3_fast(v)
}

/// Quantize bf16 weights `[n, k]` (row-major) into the block-fp8 layout.
pub fn quantize_bf16(w: &[u16], n: usize, k: usize) -> Result<Fp8Block> {
    ensure!(w.len() == n * k, "weight has {} values, expected {n}x{k}", w.len());
    let (np, kp) = (pad_to(n, BLK), pad_to(k, BLK));
    let mut data = vec![0u8; np * kp];
    let nkb = kp / BLK;
    let mut scales = vec![0f32; (np / BLK) * nkb];

    data.par_chunks_mut(BLK * kp)
        .zip(scales.par_chunks_mut(nkb))
        .enumerate()
        .for_each(|(rb, (dst, srow))| {
            for kb in 0..nkb {
                let mut amax = 0f32;
                for r in 0..BLK {
                    let row = rb * BLK + r;
                    if row >= n {
                        continue;
                    }
                    for c in kb * BLK..kb * BLK + BLK {
                        if c >= k {
                            break;
                        }
                        amax = amax.max(bf16::from_bits(w[row * k + c]).to_f32().abs());
                    }
                }
                let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
                srow[kb] = scale;
                let inv = 1.0 / scale;
                for r in 0..BLK {
                    let row = rb * BLK + r;
                    if row >= n {
                        break;
                    }
                    for c in kb * BLK..kb * BLK + BLK {
                        if c >= k {
                            break;
                        }
                        let v = bf16::from_bits(w[row * k + c]).to_f32() * inv;
                        dst[r * kp + c] = f32_to_e4m3_fast(v);
                    }
                }
            }
        });
    Ok(Fp8Block { data, scales, n: np, k: kp })
}

/// Adopt an fp8 checkpoint tensor: bytes stay as they are, only the scale tensor is
/// reinterpreted. `scale_inv[r, c]` is the multiplier that dequantizes block (r, c).
pub fn adopt_fp8(w: &[u8], scale_inv: &[f32], n: usize, k: usize) -> Result<Fp8Block> {
    ensure!(n % BLK == 0 && k % BLK == 0, "fp8 checkpoint tensor {n}x{k} is not 128-aligned; requantize from bf16");
    ensure!(w.len() == n * k, "fp8 tensor has {} bytes, expected {}", w.len(), n * k);
    let expect = (n / BLK) * (k / BLK);
    ensure!(scale_inv.len() == expect, "scale tensor has {} values, expected {expect}", scale_inv.len());
    ensure!(scale_inv.iter().all(|s| s.is_finite() && *s > 0.0), "scale tensor contains non-positive values");
    Ok(Fp8Block { data: w.to_vec(), scales: scale_inv.to_vec(), n, k })
}

/// Zero-pad a bf16 matrix `[n, k]` to `[np, kp]` (used for embeddings and lm_head).
pub fn pad_bf16(w: &[u16], n: usize, k: usize, np: usize, kp: usize) -> Vec<u16> {
    if n == np && k == kp {
        return w.to_vec();
    }
    let mut out = vec![0u16; np * kp];
    out.par_chunks_mut(kp).enumerate().for_each(|(r, row)| {
        if r < n {
            row[..k].copy_from_slice(&w[r * k..(r + 1) * k]);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize, k: usize) -> Vec<u16> {
        let mut s = 0x1234_5678_9abc_def0u64;
        (0..n * k)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let base = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4;
                // one large outlier per 128x128 block stresses the block scale
                let v = if i % (BLK * BLK) == 0 { base * 40.0 } else { base };
                bf16::from_f32(v).to_bits()
            })
            .collect()
    }

    #[test]
    fn quantize_roundtrip_stays_within_fp8_error() {
        let (n, k) = (256, 384);
        let w = sample(n, k);
        let q = quantize_bf16(&w, n, k).unwrap();
        assert_eq!(q.n, n);
        assert_eq!(q.k, k);
        assert_eq!(q.scales.len(), (n / BLK) * (k / BLK));
        let mut worst = 0f32;
        for r in 0..n {
            for c in 0..k {
                let x = bf16::from_bits(w[r * k + c]).to_f32();
                let y = q.dequant(r, c);
                let scale = q.scale_at(r, c / BLK);
                // e4m3 has 3 mantissa bits (relative half-ulp 1/16) plus one subnormal step
                assert!(
                    (x - y).abs() <= x.abs() / 16.0 + scale / 512.0 + 1e-6,
                    "{r},{c}: {x} vs {y} (scale {scale})"
                );
                if x.abs() > 1e-3 {
                    worst = worst.max((x - y).abs() / x.abs());
                }
            }
        }
        assert!(worst < 0.1, "worst relative error {worst}");
    }

    #[test]
    fn pads_ragged_shapes() {
        let w = sample(130, 200);
        let q = quantize_bf16(&w, 130, 200).unwrap();
        assert_eq!((q.n, q.k), (256, 256));
        assert_eq!(q.data.len(), 256 * 256);
        // padding is zero and every padded block still has a usable scale
        assert_eq!(q.dequant(200, 100), 0.0);
        assert!(q.scales.iter().all(|s| *s > 0.0));
        let p = pad_bf16(&w, 130, 200, 256, 256);
        assert_eq!(p.len(), 256 * 256);
        assert_eq!(p[129 * 256 + 199], w[129 * 200 + 199]);
        assert_eq!(p[130 * 256 + 5], 0);
    }

    #[test]
    fn rejects_unaligned_fp8_tensors() {
        let e = adopt_fp8(&[0u8; 100 * 100], &vec![1.0; 4], 100, 100).unwrap_err().to_string();
        assert!(e.contains("128-aligned"), "{e}");
        let e = adopt_fp8(&[0u8; 128 * 128], &vec![1.0; 3], 128, 128).unwrap_err().to_string();
        assert!(e.contains("scale tensor"), "{e}");
    }
}

/// MX-style quantization for the tensor-core path: e4m3 data with one **power-of-two**
/// e8m0 scale per 32-element block along K.
///
/// The tcgen05 kernels consume scales as `[n][k/32]` e8m0 bytes and stream them through a
/// `tcgen05.cp` tile, so the scale must be exactly `2^(e-127)`; a free-form fp32 scale
/// cannot be expressed there. Snapping to a power of two costs at most one extra bit of
/// error versus a per-block fp32 scale, which is what the mxfp8 format does anyway.
pub struct Mx {
    /// `[n][k]` e4m3 bytes, row-major.
    pub data: Vec<u8>,
    /// `[n][k/32]` e8m0 scale bytes.
    pub sf: Vec<u8>,
}

/// e4m3 max magnitude; blocks are scaled so the largest element lands just under it.
const E4M3_MAX: f32 = 448.0;

pub fn quantize_mx(src: &[f32], n: usize, k: usize) -> Mx {
    assert_eq!(src.len(), n * k, "mx quantize: shape mismatch");
    assert!(k % 32 == 0, "mx quantize: k must be a multiple of 32");
    let mut data = vec![0u8; n * k];
    let mut sf = vec![0u8; n * (k / 32)];
    for r in 0..n {
        for b in 0..k / 32 {
            let base = r * k + b * 32;
            let mut amax = 0f32;
            for j in 0..32 {
                amax = amax.max(src[base + j].abs());
            }
            // scale = 2^e with 2^e >= amax / E4M3_MAX
            let mut e = 0i32;
            if amax > 0.0 {
                e = (amax / E4M3_MAX).log2().ceil() as i32;
                e = e.clamp(-127, 127);
            }
            let scale = pow2(e);
            sf[r * (k / 32) + b] = (e + 127) as u8;
            let inv = 1.0 / scale;
            for j in 0..32 {
                data[base + j] = e4m3_from_f32(src[base + j] * inv);
            }
        }
    }
    Mx { data, sf }
}

/// Re-express a block-fp8 matrix in the mxfp8 form the tensor-core kernels consume:
/// every scale becomes an exact power of two, the fp8 payload is rescaled to match, and the
/// scales are folded into the 512-byte-per-128x128 tiles `tcgen05.cp` streams.
///
/// `scales` is rewritten in place (so the FMA fallback keeps agreeing with the tiles) and the
/// tiled e8m0 bytes are returned. Rescaling moves each value by a factor in (0.5, 1], i.e. at
/// most one e4m3 ulp.
pub fn mx_rescale(data: &mut [u8], scales: &mut [f32], n: usize, k: usize) -> Vec<u8> {
    assert_eq!(data.len(), n * k, "mx rescale: payload shape mismatch");
    let (nrb, nkb) = (n / 128, k / 128);
    assert_eq!(scales.len(), nrb * nkb, "mx rescale: scale count mismatch");
    let mut sf = vec![127u8; n * (k / 32)];
    for rb in 0..nrb {
        for kb in 0..nkb {
            let s = scales[rb * nkb + kb];
            let e = if s > 0.0 { (s.log2().ceil() as i32).clamp(-127, 127) } else { 0 };
            let scale = pow2(e);
            scales[rb * nkb + kb] = scale;
            let f = if s > 0.0 { s / scale } else { 1.0 };
            // one LUT per block: table lookup beats re-rounding every element through f32 math
            let mut lut = [0u8; 256];
            for (i, slot) in lut.iter_mut().enumerate() {
                *slot = e4m3_from_f32(e4m3_to_f32(i as u8) * f);
            }
            for r in rb * 128..(rb + 1) * 128 {
                let base = r * k + kb * 128;
                for c in 0..128 {
                    data[base + c] = lut[data[base + c] as usize];
                }
                for sub in 0..4 {
                    sf[r * (k / 32) + kb * 4 + sub] = (e + 127) as u8;
                }
            }
        }
    }
    deepseek_ultrafast_model::formats::tile_sf_blocks(&sf, n, k)
}

fn pow2(e: i32) -> f32 {
    f32::from_bits((((e + 127) as u32) & 0xff) << 23)
}

#[cfg(test)]
mod mx_tests {
    use super::*;

    #[test]
    fn scales_are_powers_of_two_and_lossless() {
        let (n, k) = (128usize, 128usize);
        let src: Vec<f32> = (0..n * k).map(|i| ((i % 97) as f32 - 48.0) * 0.037).collect();
        let q = quantize_mx(&src, n, k);
        assert_eq!(q.data.len(), n * k);
        assert_eq!(q.sf.len(), n * k / 32);
        for (i, &b) in q.sf.iter().enumerate() {
            let e = b as i32 - 127;
            assert!((-127..=127).contains(&e), "block {i} exponent {e}");
        }
    }

    #[test]
    fn round_trip_stays_inside_the_e4m3_bound() {
        let (n, k) = (64usize, 64usize);
        let src: Vec<f32> = (0..n * k).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 6.0).collect();
        let q = quantize_mx(&src, n, k);
        for r in 0..n {
            for b in 0..k / 32 {
                let e = q.sf[r * (k / 32) + b] as i32 - 127;
                let scale = pow2(e);
                for j in 0..32 {
                    let x = src[r * k + b * 32 + j];
                    let y = e4m3_to_f32(q.data[r * k + b * 32 + j]) * scale;
                    assert!(
                        (x - y).abs() <= x.abs() / 16.0 + scale / 512.0 + 1e-6,
                        "{x} vs {y} (scale {scale})"
                    );
                }
            }
        }
    }

    #[test]
    fn mx_rescale_makes_scales_exact_powers_of_two() {
        let (n, k) = (128usize, 256usize);
        let src: Vec<f32> = (0..n * k).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 4.0).collect();
        let mut blk = quantize_bf16(&src.iter().map(|v| bf16::from_f32(*v).to_bits()).collect::<Vec<_>>(), n, k).unwrap();
        let before: Vec<f32> = (0..n * k).map(|i| blk.dequant(i / k, i % k)).collect();
        let tiled = mx_rescale(&mut blk.data, &mut blk.scales, n, k);
        for &s in &blk.scales {
            assert_eq!(s, pow2((s.log2().round()) as i32), "scale {s} is not 2^e");
        }
        assert_eq!(tiled.len(), (n / 128) * (k / 128) * 512);
        let mut worst = 0f32;
        for i in 0..n * k {
            let after = blk.dequant(i / k, i % k);
            worst = worst.max((after - before[i]).abs() / (before[i].abs() + 1e-9));
        }
        assert!(worst < 0.13, "rescaling moved a value by {worst}");
    }

    #[test]
    fn sf_tiles_have_the_layout_the_kernel_streams() {
        use deepseek_ultrafast_model::formats::tile_sf_blocks;
        let (n, k) = (256usize, 256usize);
        let src = vec![0.25f32; n * k];
        let q = quantize_mx(&src, n, k);
        let tiled = tile_sf_blocks(&q.sf, n, k);
        assert_eq!(tiled.len(), (n / 128) * (k / 128) * 512);
    }
}
