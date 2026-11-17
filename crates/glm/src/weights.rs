//! Weight loading for the single-GPU GLM engine.
//!
//! `HostLinear` is the pure-CPU half (read + quantize), so it can be tested against a
//! synthetic checkpoint without a GPU; `Linear::upload` moves it to the device.
//!
//! A `Linear` may borrow another buffer (the tied-embedding head reuses the embedding
//! table), so ownership lives in an `owned` vector rather than in the `DevPtr`.

use crate::config::GlmConfig;
use crate::kernels::Kernels;
use crate::naming::{self, Resolution};
use crate::quant::{adopt_fp8, pad_bf16, quantize_bf16, Fp8Block, BLK};
use anyhow::{bail, ensure, Context, Result};
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_cuda::{Args, DevPtr, Launch, Stream};
use deepseek_ultrafast_model::st::Checkpoint;
use half::bf16;
use std::collections::{BTreeSet, HashMap};

const KDIM: usize = 16; // widest decode row block

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    Fp8,
    Bf16,
}

#[derive(Debug)]
pub enum HostLinear {
    Fp8(Fp8Block),
    Bf16 { data: Vec<u16>, n: usize, k: usize },
}

impl HostLinear {
    pub fn device_bytes(&self) -> usize {
        match self {
            HostLinear::Fp8(b) => b.data.len() + b.scales.len() * 4,
            HostLinear::Bf16 { data, .. } => data.len() * 2,
        }
    }
}

/// A linear layer resident on the device.
pub struct Linear {
    pub w: DevPtr,
    pub sf: Option<DevPtr>,
    owned: Vec<cu::Buffer>,
    pub n: usize,
    pub k: usize,
    pub fmt: Fmt,
    /// Present when the tensor-core path is enabled: same matrix, seen through the TMA
    /// descriptor and mxfp8 scale tiles.
    pub tc: Option<crate::tc::TcWeight>,
}

impl Linear {
    pub fn upload(h: &HostLinear) -> Result<Self> {
        match h {
            HostLinear::Fp8(b) => {
                let (mut data, mut scales) = (b.data.clone(), b.scales.clone());
                let tiled = crate::tc::on().then(|| crate::quant::mx_rescale(&mut data, &mut scales, b.n, b.k));
                let w = cu::Buffer::from_host(&data)?;
                let sf = cu::Buffer::from_host(&scales)?;
                let tc = match tiled {
                    Some(t) if b.n % 128 == 0 && b.k % 128 == 0 => {
                        Some(crate::tc::TcWeight::new(w.ptr, b.n, b.k, &t, 128)?)
                    }
                    _ => None,
                };
                Ok(Linear { w: w.ptr, sf: Some(sf.ptr), owned: vec![w, sf], n: b.n, k: b.k, fmt: Fmt::Fp8, tc })
            }
            HostLinear::Bf16 { data, n, k } => {
                let w = cu::Buffer::from_host(data)?;
                Ok(Linear { w: w.ptr, sf: None, owned: vec![w], n: *n, k: *k, fmt: Fmt::Bf16, tc: None })
            }
        }
    }

    /// Borrow an existing bf16 matrix (used for a tied embedding / lm_head).
    pub fn borrow(w: DevPtr, n: usize, k: usize) -> Self {
        Linear { w, sf: None, owned: Vec::new(), n, k, fmt: Fmt::Bf16, tc: None }
    }

    pub fn bytes(&self) -> usize {
        self.owned.iter().map(|b| b.bytes).sum()
    }

    /// y[t, :] = x[t, :] @ W^T for at most 16 rows.
    pub fn gemv(&self, k: &Kernels, s: &Stream, x: DevPtr, m: usize, y: DevPtr, ld_y: usize) -> Result<()> {
        ensure!(m <= KDIM, "decode GEMV supports {KDIM} rows, got {m}");
        // Tensor-core path: same contract, but the kernel caps the row count at 8 and wants
        // contiguous activations, so anything wider stays on the FMA kernels.
        if let Some(tc) = &self.tc {
            if m <= 8 {
                tc.gemv_bf16(k, s, x, m, y, ld_y)?;
                return Ok(());
            }
        }
        ensure!(self.k % 128 == 0 && self.n % 64 == 0, "{}x{} is not kernel aligned", self.n, self.k);
        let grid = (self.n / 64) as u32;
        let mut a = Args::new();
        match self.fmt {
            Fmt::Fp8 => {
                a.ptr(self.w).ptr(self.sf.unwrap()).ptr(x).ptr(y);
            }
            Fmt::Bf16 => {
                a.ptr(self.w).ptr(x).ptr(y);
            }
        }
        a.i32(self.n as i32).i32(self.k as i32).i32(m as i32).i32(self.k as i32).i32(ld_y as i32);
        let name = if self.fmt == Fmt::Fp8 { "glm_gemv_fp8" } else { "glm_gemv_bf16" };
        cu::launch(k.get(name)?, &Launch::new(grid, 256), s, &mut a)
    }

    /// Prefill path; bf16 layers fall back to the decode GEMV.
    pub fn gemm(&self, k: &Kernels, s: &Stream, x: DevPtr, m: usize, y: DevPtr, ld_y: usize) -> Result<()> {
        if self.fmt == Fmt::Bf16 {
            return self.gemv(k, s, x, m.min(KDIM), y, ld_y);
        }
        if m <= KDIM {
            return self.gemv(k, s, x, m, y, ld_y);
        }
        let mut a = Args::new();
        a.ptr(self.w).ptr(self.sf.unwrap()).ptr(x).ptr(y);
        a.i32(self.n as i32).i32(self.k as i32).i32(m as i32).i32(self.k as i32).i32(ld_y as i32);
        let grid = ((self.n / 64) as u32, m.div_ceil(32) as u32, 1);
        cu::launch(k.get("glm_gemm_fp8")?, &Launch::new(1, 256).grid3(grid), s, &mut a)
    }
}

/// All experts of one layer packed back to back: `[n][2*inter, dim]` (gate rows then
/// up rows per expert) and `[n][dim, inter]`, each with its own 128-block scales.
pub struct Experts {
    pub w13: DevPtr,
    pub sf13: DevPtr,
    pub w2: DevPtr,
    pub sf2: DevPtr,
    owned: Vec<cu::Buffer>,
    pub n: usize,
    pub inter: usize,
}

impl Experts {
    pub fn bytes(&self) -> usize {
        self.owned.iter().map(|b| b.bytes).sum()
    }
}

pub enum Ffn {
    Dense { gate: Linear, up: Linear, down: Linear },
    Moe { router: Linear, router_bias: Option<cu::Buffer>, shared: Option<(Linear, Linear, Linear)>, experts: Experts },
}

pub struct Layer {
    /// Index in the decoder stack (the MTP layer sits at `n_layers`).
    pub id: usize,
    pub input_norm: cu::Buffer,
    pub post_norm: cu::Buffer,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub q_norm: cu::Buffer,
    pub k_norm: cu::Buffer,
    pub qkv_bias: Option<cu::Buffer>,
    pub ffn: Ffn,
}

pub struct Mtp {
    pub eh_proj: Linear,
    pub enorm: cu::Buffer,
    pub hnorm: cu::Buffer,
    pub norm: cu::Buffer,
    pub layer: Layer,
}

pub struct Weights {
    pub embed: cu::Buffer,
    pub final_norm: cu::Buffer,
    pub head: Linear,
    pub layers: Vec<Layer>,
    pub mtp: Option<Mtp>,
    pub bytes: usize,
}

/// Read one linear layer, quantizing bf16 to 128-block fp8.
pub fn load_host_linear(ck: &Checkpoint, name: &str, n: usize, k: usize) -> Result<HostLinear> {
    let info = ck.info(name)?;
    ensure!(info.shape == [n, k], "{name}: shape {:?} does not match the config's [{n}, {k}]", info.shape);
    match info.dtype.as_str() {
        "F8_E4M3" | "F8_E4M3FN" | "FLOAT8_E4M3FN" => {
            let bytes = ck.bytes(name)?;
            let scale_name = scale_tensor_name(ck, name)?;
            let scales = ck.f32s(&scale_name).with_context(|| scale_name.clone())?;
            ensure!(bytes.len() == n * k, "{name}: expected {} fp8 bytes, found {}", n * k, bytes.len());
            if n % BLK == 0 && k % BLK == 0 {
                Ok(HostLinear::Fp8(adopt_fp8(bytes, &scales, n, k)?))
            } else {
                let nkb = k / BLK;
                ensure!(scales.len() == (n / BLK) * nkb, "{name}: unexpected scale count {}", scales.len());
                let mut w = vec![0u16; n * k];
                for r in 0..n {
                    for c in 0..k {
                        let s = scales[(r / BLK) * nkb + c / BLK];
                        let v = deepseek_ultrafast_model::formats::e4m3_to_f32(bytes[r * k + c]) * s;
                        w[r * k + c] = bf16::from_f32(v).to_bits();
                    }
                }
                Ok(HostLinear::Fp8(quantize_bf16(&w, n, k)?))
            }
        }
        "BF16" => {
            let bytes = ck.bytes(name)?;
            ensure!(bytes.len() == n * k * 2, "{name}: expected {} bytes, found {}", n * k * 2, bytes.len());
            let w: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Ok(HostLinear::Fp8(quantize_bf16(&w, n, k)?))
        }
        d => bail!("{name}: unsupported dtype {d} (expected BF16 or F8_E4M3)"),
    }
}

fn scale_tensor_name(ck: &Checkpoint, name: &str) -> Result<String> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    for cand in [format!("{stem}.weight_scale_inv"), format!("{name}_scale_inv"), format!("{stem}.scale_inv")] {
        if ck.has(&cand) {
            return Ok(cand);
        }
    }
    bail!("{name}: fp8 tensor without a *_scale_inv tensor (128x128 block scales required)")
}

fn read_bf16(ck: &Checkpoint, name: &str, len: usize) -> Result<Vec<u16>> {
    let bytes = ck.bytes(name)?;
    let info = ck.info(name)?;
    match info.dtype.as_str() {
        "BF16" => {
            ensure!(bytes.len() == len * 2, "{name}: expected {} bytes, found {}", len * 2, bytes.len());
            Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
        }
        "F32" => {
            ensure!(bytes.len() == len * 4, "{name}: expected {} bytes, found {}", len * 4, bytes.len());
            Ok(bytes.chunks_exact(4).map(|c| bf16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_bits()).collect())
        }
        d => bail!("{name}: unsupported dtype {d}"),
    }
}

pub struct Loader<'a> {
    pub ck: &'a Checkpoint,
    pub cfg: &'a GlmConfig,
    pub res: Resolution,
    map: HashMap<String, String>,
}

impl<'a> Loader<'a> {
    pub fn new(ck: &'a Checkpoint, cfg: &'a GlmConfig) -> Result<Self> {
        let present: BTreeSet<String> = ck.tensors.keys().cloned().collect();
        let res = naming::resolve(&naming::slots(cfg), &present);
        if !res.missing.is_empty() {
            bail!("checkpoint does not match the GLM layout:\n  {}", res.missing.join("\n  "));
        }
        let map = res.chosen.iter().cloned().collect();
        Ok(Loader { ck, cfg, res, map })
    }

    pub fn name(&self, key: &str) -> Result<&str> {
        self.map.get(key).map(|s| s.as_str()).with_context(|| format!("no tensor resolved for {key}"))
    }

    fn linear(&self, key: &str, n: usize, k: usize) -> Result<Linear> {
        let name = self.name(key)?.to_string();
        Linear::upload(&load_host_linear(self.ck, &name, n, k).with_context(|| name.clone())?)
    }

    fn vec(&self, key: &str, len: usize, padded: usize) -> Result<cu::Buffer> {
        let name = self.name(key)?.to_string();
        let v = read_bf16(self.ck, &name, len)?;
        let mut out = vec![0u16; padded];
        out[..len].copy_from_slice(&v);
        cu::Buffer::from_host(&out)
    }

    /// First candidate that exists in the checkpoint; otherwise a grouped error that
    /// names every alternative that was tried.
    fn pick(&self, candidates: &[String]) -> Result<String> {
        for c in candidates {
            if self.ck.has(c) {
                return Ok(c.clone());
            }
        }
        bail!(
            "none of these tensors exist: {} (layer checkpoint truncated?)",
            candidates.join(", ")
        )
    }

    fn experts(&self, layer: usize) -> Result<Experts> {
        let c = self.cfg;
        let (np, inter, dim) = (c.n_experts, c.moe_inter, c.hidden);
        let (nkb, ekb) = (dim / 128, inter / 128);
        let mut w13 = Vec::with_capacity(np * 2 * inter * dim);
        let mut sf13 = Vec::with_capacity(np * 2 * ekb * nkb);
        let mut w2 = Vec::with_capacity(np * dim * inter);
        let mut sf2 = Vec::with_capacity(np * (dim / 128) * ekb);
        for e in 0..np {
            let names = naming::expert(layer, e);
            let g = load_host_linear(self.ck, &self.pick(&names[0])?, inter, dim)?;
            let u = load_host_linear(self.ck, &self.pick(&names[1])?, inter, dim)?;
            let d = load_host_linear(self.ck, &self.pick(&names[2])?, dim, inter)?;
            for h in [g, u, d] {
                match h {
                    HostLinear::Fp8(b) => {
                        if b.n == inter {
                            w13.extend_from_slice(&b.data);
                            sf13.extend_from_slice(&b.scales);
                        } else {
                            w2.extend_from_slice(&b.data);
                            sf2.extend_from_slice(&b.scales);
                        }
                    }
                    HostLinear::Bf16 { .. } => unreachable!("load_host_linear returns fp8"),
                }
            }
        }
        let (a, b, d, s) = (
            cu::Buffer::from_host(&w13)?,
            cu::Buffer::from_host(&sf13)?,
            cu::Buffer::from_host(&w2)?,
            cu::Buffer::from_host(&sf2)?,
        );
        Ok(Experts { w13: a.ptr, sf13: b.ptr, w2: d.ptr, sf2: s.ptr, owned: vec![a, b, d, s], n: np, inter })
    }

    fn layer(&self, i: usize) -> Result<Layer> {
        let c = self.cfg;
        let (dim, qd, kd) = (c.hidden, c.q_dim(), c.kv_dim());
        let (dp, qp, kp) = (GlmConfig::npad(dim), GlmConfig::npad(qd), GlmConfig::npad(kd));
        let qkv_bias = if c.qkv_bias {
            let mut v = Vec::with_capacity(dp * 3);
            for (key, len, padded) in [
                (format!("layers.{i}.self_attn.q_proj.bias"), qd, qp),
                (format!("layers.{i}.self_attn.k_proj.bias"), kd, kp),
                (format!("layers.{i}.self_attn.v_proj.bias"), kd, kp),
            ] {
                if let Some(name) = self.map.get(&key) {
                    let b = read_bf16(self.ck, name, len)?;
                    let mut padded_v = vec![0u16; padded];
                    padded_v[..len].copy_from_slice(&b);
                    v.extend_from_slice(&padded_v);
                } else {
                    v.extend(std::iter::repeat_n(0u16, padded));
                }
            }
            Some(cu::Buffer::from_host(&v)?)
        } else {
            None
        };
        let ffn = if c.is_moe(i) {
            let router_bias = {
                let name = format!("model.layers.{i}.mlp.gate.e_score_correction_bias");
                if self.ck.has(&name) {
                    let v = self.ck.f32s(&name)?;
                    ensure!(v.len() == c.n_experts, "{name}: expected {} values", c.n_experts);
                    Some(cu::Buffer::from_host(&v)?)
                } else {
                    None
                }
            };
            let shared = if c.n_shared > 0 {
                let si = c.n_shared * c.moe_inter;
                let sg = naming::layer(c, i).shared_gate[0].clone();
                let su = naming::layer(c, i).shared_up[0].clone();
                let sd = naming::layer(c, i).shared_down[0].clone();
                Some((
                    Linear::upload(&load_host_linear(self.ck, &sg, si, dim)?)?,
                    Linear::upload(&load_host_linear(self.ck, &su, si, dim)?)?,
                    Linear::upload(&load_host_linear(self.ck, &sd, dim, si)?)?,
                ))
            } else {
                None
            };
            Ffn::Moe {
                router: self.linear(&format!("layers.{i}.mlp.gate"), c.n_experts, dim)?,
                router_bias,
                shared,
                experts: self.experts(i)?,
            }
        } else {
            Ffn::Dense {
                gate: self.linear(&format!("layers.{i}.mlp.gate_proj"), c.inter, dim)?,
                up: self.linear(&format!("layers.{i}.mlp.up_proj"), c.inter, dim)?,
                down: self.linear(&format!("layers.{i}.mlp.down_proj"), dim, c.inter)?,
            }
        };
        Ok(Layer {
            id: i,
            input_norm: self.vec(&format!("layers.{i}.input_layernorm"), dim, dp)?,
            post_norm: self.vec(&format!("layers.{i}.post_attention_layernorm"), dim, dp)?,
            q: self.linear(&format!("layers.{i}.self_attn.q_proj"), qd, dim)?,
            k: self.linear(&format!("layers.{i}.self_attn.k_proj"), kd, dim)?,
            v: self.linear(&format!("layers.{i}.self_attn.v_proj"), kd, dim)?,
            o: self.linear(&format!("layers.{i}.self_attn.o_proj"), dim, qd)?,
            q_norm: self.vec(&format!("layers.{i}.self_attn.q_norm"), c.head_dim, GlmConfig::npad(c.head_dim))?,
            k_norm: self.vec(&format!("layers.{i}.self_attn.k_norm"), c.head_dim, GlmConfig::npad(c.head_dim))?,
            qkv_bias,
            ffn,
        })
    }
}

fn layer_bytes(l: &Layer) -> usize {
    let attn = l.q.bytes() + l.k.bytes() + l.v.bytes() + l.o.bytes();
    let ffn = match &l.ffn {
        Ffn::Dense { gate, up, down } => gate.bytes() + up.bytes() + down.bytes(),
        Ffn::Moe { router, shared, experts, .. } => {
            router.bytes() + experts.bytes() + shared.as_ref().map_or(0, |(g, u, d)| g.bytes() + u.bytes() + d.bytes())
        }
    };
    attn + ffn + 8 * 4096
}

/// Load every tensor the config requires. Norms and the LM head stay bf16, all
/// projections become 128-block fp8 unless the checkpoint already ships fp8.
pub fn load(ck: &Checkpoint, cfg: &GlmConfig, with_mtp: bool) -> Result<Weights> {
    let l = Loader::new(ck, cfg)?;
    let (dim, dp, vp) = (cfg.hidden, GlmConfig::npad(cfg.hidden), GlmConfig::npad(cfg.vocab_size));
    let embed_name = l.name("embed_tokens")?.to_string();
    let embed_host = read_bf16(ck, &embed_name, cfg.vocab_size * dim)?;
    let embed = cu::Buffer::from_host(&pad_bf16(&embed_host, cfg.vocab_size, dim, vp, dp))?;
    let final_norm = l.vec("final_norm", dim, dp)?;
    let head = if cfg.tie_embed {
        Linear::borrow(embed.ptr, vp, dp)
    } else {
        let name = l.name("lm_head")?.to_string();
        let w = read_bf16(ck, &name, cfg.vocab_size * dim).with_context(|| name.clone())?;
        Linear::upload(&HostLinear::Bf16 { data: pad_bf16(&w, cfg.vocab_size, dim, vp, dp), n: vp, k: dp })?
    };
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        layers.push(l.layer(i)?);
    }
    let mtp = if with_mtp && cfg.n_mtp > 0 {
        let i = cfg.n_layers;
        let eh = l.linear("mtp.eh_proj", dim, 2 * dim)?;
        let enorm = l.vec("mtp.enorm", dim, dp)?;
        let hnorm = l.vec("mtp.hnorm", dim, dp)?;
        let norm_key = "mtp.norm";
        let norm = if l.map.contains_key(norm_key) {
            l.vec(norm_key, dim, dp)?
        } else {
            l.vec("final_norm", dim, dp)?
        };
        Some(Mtp { eh_proj: eh, enorm, hnorm, norm, layer: l.layer(i)? })
    } else {
        None
    };
    let mut bytes = embed.bytes + final_norm.bytes + head.bytes() + layers.iter().map(layer_bytes).sum::<usize>();
    if let Some(m) = &mtp {
        bytes += layer_bytes(&m.layer) + m.eh_proj.bytes() + m.enorm.bytes + m.hnorm.bytes + m.norm.bytes;
    }
    Ok(Weights { embed, final_norm, head, layers, mtp, bytes })
}

/// Copy of `npad` that also guards against a zero vocabulary.
pub fn padded(n: usize, minimum: usize) -> usize {
    n.max(minimum).next_multiple_of(128)
}
