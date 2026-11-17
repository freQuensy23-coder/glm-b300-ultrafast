//! Single-GPU GLM engine: prefill, block verification and speculative decode.
//!
//! One `Engine` owns the whole device. There is no tensor parallelism, no NVLS and no
//! RDMA: every collective in the original engine is replaced by the full matrix, and
//! the vocabulary is not sharded, so sampling needs no cross-rank exchange.
//!
//! Row semantics follow the original engine: a decode step processes at most 16 rows,
//! which for speculative decoding are `[anchor, draft_0 .. draft_{blk-1}]`.

use crate::config::GlmConfig;
use crate::kernels::Kernels;
use crate::weights::{Ffn, Layer, Weights};
use anyhow::{ensure, Context, Result};
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_cuda::{Args, Buffer, DevPtr, Launch, Stream};
use glm_kernels as gk;

pub const ROWS: usize = 16; // widest verify block
pub const SPLITS_MAX: usize = 32;
pub const OUT_CAP: usize = 1 << 16;

struct Bufs {
    dp: usize,
    qp: usize,
    kp: usize,
    vp: usize,
    ip: usize,
    mip: usize,
    rows: usize,
    tok: Buffer,
    tob: Buffer,
    h: Buffer,
    hn: Buffer,
    q: Buffer,
    kn: Buffer,
    vn: Buffer,
    ao: Buffer,
    pf: Buffer,
    g1: Buffer,
    g2: Buffer,
    act: Buffer,
    logits: Buffer,
    ids: Buffer,
    wts: Buffer,
    gscore: Buffer,
    offs: Buffer,
    idx: Buffer,
    cursor: Buffer,
    mh: Buffer,
    mpart: Buffer,
    apart: Buffer,
    out: Buffer,
    spec: Buffer,
    draft: Buffer,
    dlogits: Buffer,
    vr: Buffer,
    rng: Buffer,
    cat: Buffer,
    hold: Buffer,
    hprev: Buffer,
}

fn zeroed_f32(n: usize) -> Result<Buffer> {
    Buffer::zeroed(n * 4)
}

fn zeroed_bf16(n: usize) -> Result<Buffer> {
    Buffer::zeroed(n * 2)
}

pub struct Engine {
    pub cfg: GlmConfig,
    k: Kernels,
    s: Stream,
    pub w: Weights,
    pub kv: crate::cache::KvCache,
    b: Bufs,
    pub max_seq: usize,
    pub chunk: usize,
    /// Number of committed positions in both KV caches.
    pub pos: usize,
    /// How far the drafter's cache has been advanced (<= pos, settles on demand).
    pub mtp_pos: usize,
    /// Captured round, when the graph path is enabled.
    graph: Option<SpecGraph>,
    graph_committed: usize,
    pub seed: u64,
    pub step: u32,
    pub mtp_concat_embed_first: bool,
    pub window: usize,
}

impl Engine {
    pub fn new(cfg: GlmConfig, w: Weights, max_seq: usize, chunk: usize) -> Result<Self> {
        let chunk = chunk.clamp(ROWS, 8192);
        let rows = chunk.max(ROWS);
        let (dp, qp, kp, vp) = (
            GlmConfig::npad(cfg.hidden),
            GlmConfig::npad(cfg.q_dim()),
            GlmConfig::npad(cfg.kv_dim()),
            GlmConfig::npad(cfg.vocab_size),
        );
        let (ip, mip) = (GlmConfig::npad(cfg.inter), GlmConfig::npad(cfg.moe_inter));
        let topk = cfg.top_k.max(1);
        let ne = cfg.n_experts.max(1);
        let b = Bufs {
            dp,
            qp,
            kp,
            vp,
            ip,
            mip,
            rows,
            tok: Buffer::zeroed(rows * 4)?,
            tob: Buffer::zeroed(rows * 4)?,
            h: zeroed_bf16(rows * dp)?,
            hn: zeroed_bf16(rows * dp)?,
            q: zeroed_bf16(rows * qp)?,
            kn: zeroed_bf16(rows * kp)?,
            vn: zeroed_bf16(rows * kp)?,
            ao: zeroed_bf16(rows * qp)?,
            pf: zeroed_f32(rows * qp.max(dp))?,
            g1: zeroed_f32(rows * ip)?,
            g2: zeroed_f32(rows * ip)?,
            act: zeroed_bf16(rows * ip)?,
            logits: zeroed_f32(ROWS * vp)?,
            ids: Buffer::zeroed(rows * topk * 4)?,
            wts: zeroed_f32(rows * topk)?,
            gscore: zeroed_f32(ne.max(64))?,
            offs: Buffer::zeroed((ne + 2) * 4)?,
            idx: Buffer::zeroed(rows * topk * 4)?,
            cursor: Buffer::zeroed((ne + 2) * 4)?,
            mh: zeroed_bf16(rows * topk * mip)?,
            mpart: zeroed_f32(rows * topk * dp)?,
            apart: zeroed_f32(ROWS * cfg.n_heads * SPLITS_MAX * (cfg.head_dim + 2))?,
            out: Buffer::zeroed(OUT_CAP * 4)?,
            spec: Buffer::zeroed(8 * 4)?,
            draft: Buffer::zeroed(ROWS * 4)?,
            dlogits: zeroed_f32(ROWS * vp)?,
            vr: Buffer::zeroed(ROWS * 4)?,
            rng: Buffer::zeroed(4 * 4)?,
            cat: zeroed_bf16(2 * dp)?,
            hold: zeroed_bf16(ROWS * dp)?,
            hprev: zeroed_bf16(ROWS * dp)?,
        };
        let kv = crate::cache::KvCache::alloc(&cfg, max_seq, w.mtp.is_some())?;
        let k = Kernels::load()?;
        // Dynamic shared memory above 48 KB needs an explicit opt-in per function, and
        // `cu::launch` only forwards the byte count. Set it once for the two attention
        // kernels, whose scratch is sized from the head dimension.
        for (name, bytes) in [
            ("glm_attn_decode_split", gk::attn_decode_smem(cfg.head_dim)),
            ("glm_attn_prefill", gk::attn_prefill_smem(cfg.head_dim)),
        ] {
            let f = k.get(name)?;
            if bytes > 48 * 1024 {
                f.set_max_dynamic_smem(bytes)?;
            }
        }
        let window = cfg.sliding_window.unwrap_or(0).min(max_seq);
        Ok(Engine {
            cfg,
            k,
            s: Stream::new()?,
            w,
            kv,
            b,
            max_seq,
            chunk,
            pos: 0,
            mtp_pos: 0,
            graph: None,
            graph_committed: 0,
            seed: 0,
            step: 0,
            mtp_concat_embed_first: true,
            window,
        })
    }

    pub fn scratch_bytes(&self) -> usize {
        let b = &self.b;
        [
            b.tok.bytes,
            b.h.bytes,
            b.hn.bytes,
            b.q.bytes,
            b.kn.bytes,
            b.vn.bytes,
            b.ao.bytes,
            b.pf.bytes,
            b.g1.bytes,
            b.g2.bytes,
            b.act.bytes,
            b.logits.bytes,
            b.mh.bytes,
            b.mpart.bytes,
            b.apart.bytes,
            b.dlogits.bytes,
        ]
        .iter()
        .sum()
    }

    fn launch(&self, name: &str, cfg: Launch, a: &mut Args) -> Result<()> {
        cu::launch(self.k.get(name)?, &cfg, &self.s, a)
    }

    /// Upload token ids and their absolute positions for the next `m` rows.
    pub fn set_rows(&self, start: usize, tokens: &[i32]) -> Result<()> {
        ensure!(tokens.len() <= self.b.rows, "{} rows exceed the {}-row buffer", tokens.len(), self.b.rows);
        let pos: Vec<i32> = (0..tokens.len()).map(|i| (start + i) as i32).collect();
        cu::upload(self.b.tok.ptr, tokens)?;
        cu::upload(self.b.tob.ptr, &pos)?;
        Ok(())
    }

    fn embed(&self, m: usize) -> Result<()> {
        self.embed_ids(self.b.tok.ptr, m)
    }

    fn embed_ids(&self, ids: DevPtr, m: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(self.b.h.ptr).ptr(self.w.embed.ptr).ptr(ids).i32(m as i32).i32(self.b.dp as i32);
        self.launch("glm_embed", Launch::new(m as u32, 256), &mut a)
    }

    fn rmsnorm(&self, x: DevPtr, w: DevPtr, out: DevPtr, m: usize, ldx: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(x).ptr(w).ptr(out).i32(m as i32).i32(self.b.dp as i32).f32(self.cfg.rms_eps).i32(ldx as i32).i32(self.b.dp as i32);
        self.launch("glm_rmsnorm", Launch::new(m as u32, 256), &mut a)
    }

    fn cast(&self, dst: DevPtr, src: DevPtr, n: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(dst).ptr(src).i32(n as i32);
        self.launch("glm_cast_f32_bf16", Launch::new((n as u32).div_ceil(256).min(65535), 256), &mut a)
    }

    fn add_residual(&self, h: DevPtr, y: DevPtr, n: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(h).ptr(y).i32(n as i32);
        self.launch("glm_add_residual", Launch::new((n as u32).div_ceil(256).min(65535), 256), &mut a)
    }

    fn norm_heads(&self, x: DevPtr, w: DevPtr, heads: usize, m: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(x).ptr(w).ptr(x).i32(heads as i32).i32(self.cfg.head_dim as i32).f32(self.cfg.rms_eps);
        self.launch("glm_rmsnorm_heads", Launch::new(heads as u32, 128).grid3((heads as u32, m as u32, 1)), &mut a)
    }

    fn rope(&self, q: DevPtr, k: DevPtr, m: usize) -> Result<()> {
        let half = self.cfg.rope_dim / 2;
        let total = m * (self.cfg.n_heads + self.cfg.n_kv) * half;
        let mut a = Args::new();
        a.ptr(q).ptr(k).ptr(self.b.tob.ptr).i32(m as i32).i32(self.cfg.n_heads as i32).i32(self.cfg.n_kv as i32).i32(self.cfg.head_dim as i32).i32(self.cfg.rope_dim as i32).f32(self.cfg.rope_theta);
        self.launch("glm_rope_qk", Launch::new((total as u32).div_ceil(256).min(4096).max(1), 256), &mut a)
    }

    fn kv_append(&self, slot: usize, _start: usize, m: usize) -> Result<()> {
        let (kc, vc) = self.kv.ptrs(slot);
        let total = (m * self.cfg.n_kv * self.cfg.head_dim / 8) as u32;
        let mut a = Args::new();
        a.ptr(kc).ptr(vc).ptr(self.b.kn.ptr).ptr(self.b.vn.ptr).ptr(self.b.tob.ptr).i32(m as i32).i32(self.cfg.n_kv as i32).i32(self.cfg.head_dim as i32).i32(self.kv.max_seq as i32);
        self.launch("glm_kv_append", Launch::new(total.div_ceil(128).max(1), 128), &mut a)
    }

    /// Decode attention for `m <= ROWS` rows starting at `start`.
    fn attn_decode(&self, slot: usize, start: usize, m: usize) -> Result<()> {
        let c = &self.cfg;
        let sms = crate::kernels::sm_count();
        let (kc, vc) = self.kv.ptrs(slot);
        let splits = crate::cache::decode_splits(start + m, c.n_heads, m, sms);
        let smem = gk::attn_decode_smem(c.head_dim);
        let mut a = Args::new();
        a.ptr(self.b.q.ptr).ptr(kc).ptr(vc).ptr(self.b.apart.ptr).ptr(self.b.tob.ptr).i32(m as i32).i32(self.kv.max_seq as i32).i32(c.n_heads as i32).i32(c.n_kv as i32).i32(c.head_dim as i32).f32(c.attn_scale()).i32(self.window as i32).i32(0).i32(splits as i32).i32(c.gqa() as i32);
        self.launch("glm_attn_decode_split", Launch::new(c.n_heads as u32, 256).grid3((c.n_heads as u32, m as u32, splits as u32)).smem(smem), &mut a)?;
        let mut b = Args::new();
        b.ptr(self.b.apart.ptr).ptr(self.b.ao.ptr).i32(m as i32).i32(c.n_heads as i32).i32(c.head_dim as i32).i32(splits as i32);
        self.launch("glm_attn_combine", Launch::new(c.n_heads as u32, 128).grid3((c.n_heads as u32, m as u32, 1)), &mut b)
    }

    fn attn_prefill(&self, slot: usize, _start: usize, m: usize) -> Result<()> {
        let c = &self.cfg;
        let (kc, vc) = self.kv.ptrs(slot);
        let smem = gk::attn_prefill_smem(c.head_dim);
        let mut a = Args::new();
        a.ptr(self.b.q.ptr).ptr(kc).ptr(vc).ptr(self.b.ao.ptr).ptr(self.b.tob.ptr).i32(m as i32).i32(self.kv.max_seq as i32).i32(c.n_heads as i32).i32(c.n_kv as i32).i32(c.head_dim as i32).f32(c.attn_scale()).i32(self.window as i32).i32(c.gqa() as i32);
        self.launch("glm_attn_prefill", Launch::new(c.n_heads as u32, 256).grid3((c.n_heads as u32, m.div_ceil(16) as u32, 1)).smem(smem), &mut a)
    }

    fn swiglu(&self, out: DevPtr, gate: DevPtr, up: DevPtr, n: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(out).ptr(gate).ptr(up).i32(n as i32);
        self.launch("glm_swiglu_mul", Launch::new((n as u32).div_ceil(256).min(65535), 256), &mut a)
    }

    fn moe(&self, layer: &Layer, hn: DevPtr, h: DevPtr, m: usize) -> Result<()> {
        let c = &self.cfg;
        let Ffn::Moe { router, router_bias, shared, experts } = &layer.ffn else {
            anyhow::bail!("moe() called on a dense layer");
        };
        let ne = c.n_experts;
        let topk = c.top_k;
        let smem = ((ne + 8 + topk) * 4) as u32;
        let mut a = Args::new();
        a.ptr(hn).ptr(router.w).ptr(router.sf.unwrap()).ptr(router_bias.as_ref().map_or(0, |b| b.ptr)).i32(m as i32).i32(self.b.dp as i32).i32(ne as i32).i32(topk as i32).i32(c.n_group as i32).i32(c.topk_group as i32).f32(c.route_scale).ptr(self.b.ids.ptr).ptr(self.b.wts.ptr).ptr(self.b.gscore.ptr).i32(i32::from(c.scoring == crate::config::Scoring::Sigmoid));
        self.launch("glm_router", Launch::new(m as u32, 256).smem(smem), &mut a)?;

        let mut a = Args::new();
        a.ptr(self.b.ids.ptr).i32(m as i32).i32(topk as i32).i32(ne as i32).ptr(self.b.offs.ptr).ptr(self.b.idx.ptr).ptr(self.b.cursor.ptr);
        self.launch("glm_route_lists", Launch::new(1, 32), &mut a)?;

        let mut a = Args::new();
        a.ptr(self.b.mh.ptr).ptr(hn).ptr(experts.w13).ptr(experts.sf13).ptr(self.b.idx.ptr).ptr(self.b.offs.ptr).i32(topk as i32).i32(self.b.dp as i32).i32(self.b.mip as i32);
        self.launch("glm_moe_gate_up", Launch::new(ne as u32, 128).grid3((ne as u32, (self.b.mip / 128) as u32, 1)).smem((self.b.dp * 4) as u32), &mut a)?;

        let mut a = Args::new();
        a.ptr(self.b.mpart.ptr).ptr(self.b.mh.ptr).ptr(experts.w2).ptr(experts.sf2).ptr(self.b.idx.ptr).ptr(self.b.offs.ptr).i32(self.b.mip as i32).i32(self.b.dp as i32);
        self.launch("glm_moe_down", Launch::new(ne as u32, 128).grid3((ne as u32, (self.b.dp / 128) as u32, 1)), &mut a)?;

        let mut a = Args::new();
        a.ptr(h).ptr(self.b.mpart.ptr).ptr(self.b.wts.ptr).ptr(h).i32(m as i32).i32(topk as i32).i32(self.b.dp as i32);
        self.launch("glm_moe_reduce", Launch::new((m * self.b.dp).div_ceil(256) as u32, 256), &mut a)?;

        if let Some((sg, su, sd)) = shared {
            sg.gemm(&self.k, &self.s, hn, m, self.b.g1.ptr, self.b.ip)?;
            su.gemm(&self.k, &self.s, hn, m, self.b.g2.ptr, self.b.ip)?;
            self.swiglu(self.b.act.ptr, self.b.g1.ptr, self.b.g2.ptr, m * self.b.ip)?;
            sd.gemm(&self.k, &self.s, self.b.act.ptr, m, self.b.pf.ptr, self.b.dp)?;
            self.add_residual(h, self.b.pf.ptr, m * self.b.dp)?;
        }
        Ok(())
    }

    fn dense_ffn(&self, layer: &Layer, hn: DevPtr, h: DevPtr, m: usize) -> Result<()> {
        let Ffn::Dense { gate, up, down } = &layer.ffn else {
            anyhow::bail!("dense_ffn() called on a MoE layer");
        };
        gate.gemm(&self.k, &self.s, hn, m, self.b.g1.ptr, self.b.ip)?;
        up.gemm(&self.k, &self.s, hn, m, self.b.g2.ptr, self.b.ip)?;
        self.swiglu(self.b.act.ptr, self.b.g1.ptr, self.b.g2.ptr, m * self.b.ip)?;
        down.gemm(&self.k, &self.s, self.b.act.ptr, m, self.b.pf.ptr, self.b.dp)?;
        self.add_residual(h, self.b.pf.ptr, m * self.b.dp)
    }

    /// One decoder layer over `m` rows at positions `start..start+m`, writing back to `h`.
    fn layer(&self, layer: &Layer, slot: usize, start: usize, m: usize, prefill: bool) -> Result<()> {
        let c = &self.cfg;
        let h = self.b.h.ptr;
        self.rmsnorm(h, layer.input_norm.ptr, self.b.hn.ptr, m, self.b.dp)?;
        let x = self.b.hn.ptr;
        layer.q.gemm(&self.k, &self.s, x, m, self.b.pf.ptr, self.b.qp)?;
        self.cast(self.b.q.ptr, self.b.pf.ptr, m * self.b.qp)?;
        layer.k.gemm(&self.k, &self.s, x, m, self.b.pf.ptr, self.b.kp)?;
        self.cast(self.b.kn.ptr, self.b.pf.ptr, m * self.b.kp)?;
        layer.v.gemm(&self.k, &self.s, x, m, self.b.pf.ptr, self.b.kp)?;
        self.cast(self.b.vn.ptr, self.b.pf.ptr, m * self.b.kp)?;
        if let Some(bias) = &layer.qkv_bias {
            self.add_qkv_bias(bias.ptr, m)?;
        }
        self.norm_heads(self.b.q.ptr, layer.q_norm.ptr, c.n_heads, m)?;
        self.norm_heads(self.b.kn.ptr, layer.k_norm.ptr, c.n_kv, m)?;
        self.rope(self.b.q.ptr, self.b.kn.ptr, m)?;
        self.kv_append(slot, start, m)?;
        if prefill {
            self.attn_prefill(slot, start, m)?;
        } else {
            self.attn_decode(slot, start, m)?;
        }
        layer.o.gemm(&self.k, &self.s, self.b.ao.ptr, m, self.b.pf.ptr, self.b.dp)?;
        self.add_residual(h, self.b.pf.ptr, m * self.b.dp)?;
        self.rmsnorm(h, layer.post_norm.ptr, self.b.hn.ptr, m, self.b.dp)?;
        if c.is_moe(slot_layer_index(layer)) {
            self.moe(layer, self.b.hn.ptr, h, m)
        } else {
            self.dense_ffn(layer, self.b.hn.ptr, h, m)
        }
    }

    /// Add the fused q/k/v bias to the q, k and v rows (already cast to bf16).
    ///
    /// GLM packs the three biases into one `[q_dim + 2*kv_dim]` tensor, so each buffer
    /// gets its own slice with its own row stride (`qp` for q, `kp` for k and v).
    fn add_qkv_bias(&self, bias: DevPtr, m: usize) -> Result<()> {
        let c = &self.cfg;
        let q_len = c.n_heads * c.head_dim;
        let kv_len = c.n_kv * c.head_dim;
        for (dst, stride, len, off) in [
            (self.b.q.ptr, self.b.qp, q_len, 0usize),
            (self.b.kn.ptr, self.b.kp, kv_len, q_len),
            (self.b.vn.ptr, self.b.kp, kv_len, q_len + kv_len),
        ] {
            let mut a = Args::new();
            a.ptr(dst)
                .ptr(bias + (off * 2) as u64)
                .i32(m as i32)
                .i32(stride as i32)
                .i32(len as i32);
            self.launch("glm_add_bias", Launch::new(m as u32, 256), &mut a)?;
        }
        Ok(())
    }

    /// Final norm + lm_head for the first `m` rows of `h`.
    pub fn head_logits(&self, m: usize, first_row: usize) -> Result<()> {
        self.rmsnorm(
            self.b.h.ptr + (first_row * self.b.dp * 2) as u64,
            self.w.final_norm.ptr,
            self.b.hn.ptr,
            m,
            self.b.dp,
        )?;
        self.w.head.gemv(&self.k, &self.s, self.b.hn.ptr, m, self.b.logits.ptr, self.b.vp)
    }

    /// Full forward over `m` rows starting at `start`, writing logits for every row.
    pub fn forward(&mut self, start: usize, tokens: &[i32]) -> Result<()> {
        let m = tokens.len();
        ensure!(m > 0 && m <= self.b.rows, "forward needs 1..={} rows, got {m}", self.b.rows);
        self.set_rows(start, tokens)?;
        self.embed(m)?;
        let prefill = m > ROWS;
        for i in 0..self.cfg.n_layers {
            let layer = &self.w.layers[i];
            self.layer(layer, i, start, m, prefill)?;
        }
        if prefill {
            self.head_logits(1, m - 1)
        } else {
            self.head_logits(m, 0)
        }
    }

    pub fn read_logits(&self, rows: usize) -> Result<Vec<f32>> {
        self.s.synchronize()?;
        cu::download(self.b.logits.ptr, rows * self.b.vp)
    }

    pub fn read_out(&self, n: usize) -> Result<Vec<i32>> {
        self.s.synchronize()?;
        cu::download(self.b.out.ptr, n)
    }

    pub fn argmax_row(&self, row: usize) -> Result<i32> {
        let mut a = Args::new();
        a.ptr(self.b.logits.ptr + (row * self.b.vp * 4) as u64).i32(self.cfg.vocab_size as i32).ptr(self.b.draft.ptr);
        let smem = ((1024 + 1024) * 4) as u32;
        self.launch("glm_argmax", Launch::new(1, 1024).smem(smem), &mut a)?;
        self.s.synchronize()?;
        Ok(cu::download::<i32>(self.b.draft.ptr, 1)?[0])
    }

    pub fn sample_row(&self, row: usize, temp: f32) -> Result<i32> {
        let mut a = Args::new();
        a.ptr(self.b.logits.ptr + (row * self.b.vp * 4) as u64).i32(self.cfg.vocab_size as i32).f32(temp).push(self.seed).ptr(self.b.rng.ptr).ptr(self.b.draft.ptr);
        let smem = ((1024 * 3 + 8) * 4) as u32;
        self.launch("glm_sample", Launch::new(1, 1024).smem(smem), &mut a)?;
        self.s.synchronize()?;
        Ok(cu::download::<i32>(self.b.draft.ptr, 1)?[0])
    }
}


impl Engine {
    fn copy_row(&self, dst: DevPtr, src: DevPtr, n: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(dst).ptr(src).i32(n as i32);
        self.launch("glm_copy_bf16", Launch::new((n as u32).div_ceil(256).max(1), 256), &mut a)
    }

    /// Copy element `row` of a `[rows][stride]` bf16 buffer into `dst`.
    fn copy_row_of(&self, dst: DevPtr, src: DevPtr, row: usize, stride: usize) -> Result<()> {
        let off = (row * stride * 2) as u64;
        self.copy_row(dst, src + off, stride)
    }

    fn head_gemv_into(&self, dst: DevPtr) -> Result<()> {
        self.w.head.gemv(&self.k, &self.s, self.b.hn.ptr, 1, dst, self.b.vp)
    }

    /// The MTP body for one row: `eh_proj(cat(norm(embed(tok)), norm(h_prev)))` through
    /// the extra decoder layer at `pos`, then the shared head into `logits`.
    ///
    /// Token ids come from `ids` (one entry) so the same code serves the host-driven path
    /// and a captured graph, where the previous draft's sampled id is only ever in device
    /// memory.
    fn mtp_body(&self, pos: usize, ids: DevPtr, h_prev: DevPtr) -> Result<()> {
        let mtp = self.w.mtp.as_ref().context("this checkpoint has no MTP layer")?;
        let slot = self.cfg.n_layers;
        self.embed_ids(ids, 1)?;
        // embed_norm -> cat[0:dp], hidden_norm -> cat[dp:2dp]
        self.rmsnorm(self.b.h.ptr, mtp.enorm.ptr, self.b.cat.ptr, 1, self.b.dp)?;
        self.rmsnorm(h_prev, mtp.hnorm.ptr, self.b.cat.ptr + (self.b.dp * 2) as u64, 1, self.b.dp)?;
        if !self.mtp_concat_embed_first {
            // [hidden ; embed] order: swap the two halves
            self.copy_row(self.b.hold.ptr, self.b.cat.ptr, self.b.dp)?;
            self.copy_row(self.b.cat.ptr, self.b.cat.ptr + (self.b.dp * 2) as u64, self.b.dp)?;
            self.copy_row(self.b.cat.ptr + (self.b.dp * 2) as u64, self.b.hold.ptr, self.b.dp)?;
        }
        mtp.eh_proj.gemv(&self.k, &self.s, self.b.cat.ptr, 1, self.b.pf.ptr, self.b.dp)?;
        self.cast(self.b.h.ptr, self.b.pf.ptr, self.b.dp)?;
        self.layer(&mtp.layer, slot, pos, 1, false)?;
        self.rmsnorm(self.b.h.ptr, mtp.norm.ptr, self.b.hn.ptr, 1, self.b.dp)?;
        self.head_gemv_into(self.b.logits.ptr)
    }

    /// Run the MTP layer for one row and sample a token into `out`.
    fn mtp_row(&self, pos: usize, tok: i32, h_prev: DevPtr) -> Result<()> {
        self.set_rows(pos, &[tok])?;
        self.mtp_body(pos, self.b.tok.ptr, h_prev)
    }

    /// Sample the first logits row into `out` without a host round-trip.
    fn sample_into(&self, out: DevPtr, temp: f32) -> Result<()> {
        let mut a = Args::new();
        a.ptr(self.b.logits.ptr)
            .i32(self.cfg.vocab_size as i32)
            .f32(temp)
            .push(self.seed)
            .ptr(self.b.rng.ptr)
            .ptr(out);
        let smem = ((1024 * 3 + 8) * 4) as u32;
        self.launch("glm_sample", Launch::new(1, 1024).smem(smem), &mut a)
    }

    /// Advance the drafter's cache to the target's committed position.
    ///
    /// Drafting appends `blk` rows but only `committed` are authoritative, so the
    /// drafter is short by one row exactly when every draft was accepted (the extra
    /// token sampled from the last verify row). That row is replayed here with the
    /// emitted token and the hidden state of the row before it.
    fn mtp_settle(&mut self, last_emitted: i32) -> Result<()> {
        let hprev = self.b.hprev.ptr;
        while self.mtp_pos < self.pos {
            let p = self.mtp_pos;
            self.mtp_row(p, last_emitted, hprev)?;
            self.copy_row(hprev, self.b.h.ptr, self.b.dp)?;
            self.mtp_pos += 1;
        }
        Ok(())
    }

    /// Draft `blk` tokens from the MTP layer starting after the committed prefix.
    /// `h_prev` is the target hidden state of the last committed token.
    pub fn draft(&mut self, blk: usize, anchor: i32, h_prev: DevPtr, temp: f32) -> Result<Vec<i32>> {
        ensure!(blk >= 1 && blk < ROWS, "draft block must be 1..{}, got {blk}", ROWS);
        let mut out = Vec::with_capacity(blk);
        let mut tok = anchor;
        let mut src = h_prev;
        // the drafter scores its own rows in `dlogits`
        for i in 0..blk {
            self.mtp_row(self.mtp_pos + i, tok, src)?;
            let d = self.sample_row(0, temp)?;
            cu::memcpy_dtod_async(self.b.dlogits.ptr + (i * self.b.vp * 4) as u64, self.b.logits.ptr, self.b.vp * 4, &self.s)?;
            out.push(d);
            tok = d;
            src = self.b.h.ptr; // the MTP layer's own hidden row
        }
        self.mtp_pos += blk;
        cu::upload(self.b.draft.ptr, &out)?;
        Ok(out)
    }

    /// Verify a draft block and return (emitted tokens, committed positions).
    pub fn verify(&mut self, anchor: i32, drafts: &[i32], temp: f32, greedy: bool) -> Result<(Vec<i32>, usize)> {
        let blk = drafts.len();
        ensure!(blk + 1 <= ROWS, "verify block {} exceeds {ROWS} rows", blk + 1);
        let mut rows = Vec::with_capacity(blk + 1);
        rows.push(anchor);
        rows.extend_from_slice(drafts);
        let start = self.pos;
        self.forward(start, &rows)?;
        let (count, committed) = self.accept(blk, temp, greedy)?;
        let emitted = self.read_out(count)?;
        self.graph_committed = committed;
        self.pos += committed;
        Ok((emitted, committed))
    }

    fn accept(&mut self, blk: usize, temp: f32, greedy: bool) -> Result<(usize, usize)> {
        let smem = ((8 + 4 * 1024) * 4) as u32;
        let ql = if greedy { 0 } else { self.b.dlogits.ptr };
        let mut a = Args::new();
        a.ptr(self.b.logits.ptr)
            .ptr(self.b.draft.ptr)
            .ptr(ql)
            .i32(blk as i32)
            .i32(self.cfg.vocab_size as i32)
            .f32(temp)
            .push(self.seed)
            .ptr(self.b.rng.ptr)
            .i32(i32::from(greedy))
            .ptr(self.b.out.ptr)
            .ptr(self.b.spec.ptr)
            .ptr(self.b.spec.ptr + 4);
        self.launch("glm_spec_accept", Launch::new(1, 1024).smem(smem), &mut a)?;
        self.s.synchronize()?;
        let v = cu::download::<i32>(self.b.spec.ptr, 2)?;
        Ok((v[0].max(0) as usize, v[1].max(0) as usize))
    }

    /// Prompt processing in `chunk`-sized passes; returns the argmax of the last row.
    pub fn prefill(&mut self, tokens: &[i32]) -> Result<i32> {
        ensure!(!tokens.is_empty(), "empty prompt");
        ensure!(self.pos == 0, "prefill expects an empty cache, pos = {}", self.pos);
        let n = tokens.len();
        ensure!(n < self.max_seq, "prompt {} does not fit the {}-token cache", n, self.max_seq);
        let mut start = 0usize;
        let mut last_len = 0usize;
        while start < n {
            let len = (n - start).min(self.chunk);
            self.forward(start, &tokens[start..start + len])?;
            start += len;
            last_len = len;
        }
        self.pos = n;
        // the hidden state of the last prompt token feeds the first MTP draft
        self.copy_row_of(self.b.hprev.ptr, self.b.h.ptr, last_len - 1, self.b.dp)?;
        self.mtp_pos = n;
        self.argmax_row(0)
    }

    /// One speculative round. Returns the emitted tokens.
    pub fn step(&mut self, anchor: i32, policy: &mut crate::spec::SpecPolicy, temp: f32, greedy: bool) -> Result<Vec<i32>> {
        if let Some(g) = self.graph.take() {
            let r = self.graph_round(&g, anchor);
            let blk = g.blk;
            self.graph = Some(g);
            let emitted = r?;
            let committed = self.graph_committed;
            policy.observe(committed.saturating_sub(1), blk);
            return Ok(emitted);
        }
        let blk = policy.block();
        let hprev = self.b.hprev.ptr;
        let drafts = self.draft(blk, anchor, hprev, temp)?;
        let (emitted, committed) = self.verify(anchor, &drafts, temp, greedy)?;
        let accepted = committed.saturating_sub(1);
        policy.observe(accepted, blk);
        self.step += 1;
        // the hidden state of the last committed row becomes the next draft's input
        if committed > 0 {
            self.copy_row_of(self.b.hprev.ptr, self.b.h.ptr, committed - 1, self.b.dp)?;
        }
        // the drafter advanced `blk` rows; replay anything the target committed past it
        if let Some(last) = emitted.last() {
            self.mtp_settle(*last)?;
        }
        Ok(emitted)
    }

    /// Greedy temperature-0 generation used by tests and by `--check`.
    pub fn generate(
        &mut self,
        prompt: &[i32],
        max_new: usize,
        temp: f32,
        greedy: bool,
        mut on_token: impl FnMut(i32) -> bool,
    ) -> Result<Vec<i32>> {
        let first = self.prefill(prompt)?;
        let mut out = Vec::with_capacity(max_new);
        let mut policy = crate::spec::SpecPolicy::new(1, 7.min(ROWS - 1));
        let mut anchor = first;
        while out.len() < max_new {
            if !on_token(anchor) {
                break;
            }
            out.push(anchor);
            if out.len() >= max_new {
                break;
            }
            let emitted = self.step(anchor, &mut policy, temp, greedy)?;
            let Some((last, rest)) = emitted.split_last() else {
                anyhow::bail!("speculative step emitted no token");
            };
            for t in rest {
                if !on_token(*t) {
                    return Ok(out);
                }
                out.push(*t);
                if out.len() >= max_new {
                    break;
                }
            }
            anchor = *last;
        }
        Ok(out)
    }
}

/// Layer index within the stack, used to decide dense vs MoE. The MTP layer is at
/// `n_layers`, which is always a MoE layer in GLM checkpoints with MTP.
fn slot_layer_index(layer: &Layer) -> usize {
    layer.id
}

impl Engine {
    /// Positions for `n` consecutive rows starting at `start`, staged for the graph.
    pub fn stage_positions(&self, start: usize, n: usize) -> Result<()> {
        let pos: Vec<i32> = (0..n).map(|i| (start + i) as i32).collect();
        cu::upload(self.b.tob.ptr, &pos)
    }

    /// Device-only draft: no host read between rows, so it can be captured.
    pub fn draft_dev(&self, blk: usize, temp: f32) -> Result<()> {
        ensure!(self.w.mtp.is_some(), "this checkpoint has no MTP layer");
        let mut src = self.b.hprev.ptr;
        for i in 0..blk {
            let ids = if i == 0 { self.b.tok.ptr } else { self.b.draft.ptr + ((i - 1) * 4) as u64 };
            let out = self.b.draft.ptr + (i * 4) as u64;
            self.mtp_body(self.pos + i, ids, src)?;
            cu::memcpy_dtod_async(
                self.b.dlogits.ptr + (i * self.b.vp * 4) as u64,
                self.b.logits.ptr,
                self.b.vp * 4,
                &self.s,
            )?;
            self.sample_into(out, temp)?;
            src = self.b.h.ptr; // the MTP layer's own hidden row feeds the next draft
        }
        Ok(())
    }

    /// `rows[0] = tok[0]`, `rows[1..=n] = draft[0..n]`: the verification token block.
    pub fn build_rows(&self, n: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(self.b.vr.ptr).ptr(self.b.tok.ptr).ptr(self.b.draft.ptr).i32(n as i32);
        self.launch("glm_build_rows", Launch::new(1, 32), &mut a)
    }

    /// Forward over `m` rows whose tokens are already in the verification buffer.
    pub fn forward_ids(&self, m: usize) -> Result<()> {
        ensure!(m > 0 && m <= self.b.rows, "forward needs 1..={} rows, got {m}", self.b.rows);
        self.embed_ids(self.b.vr.ptr, m)?;
        let prefill = m > ROWS;
        for i in 0..self.cfg.n_layers {
            self.layer(&self.w.layers[i], i, self.pos, m, prefill)?;
        }
        if prefill {
            self.head_logits(1, m - 1)
        } else {
            self.head_logits(m, 0)
        }
    }

    /// Launch acceptance without reading the result back.
    pub fn accept_launch(&self, blk: usize, temp: f32, greedy: bool) -> Result<()> {
        let smem = ((8 + 4 * 1024) * 4) as u32;
        let ql = if greedy { 0 } else { self.b.dlogits.ptr };
        let mut a = Args::new();
        a.ptr(self.b.logits.ptr)
            .ptr(self.b.draft.ptr)
            .ptr(ql)
            .i32(blk as i32)
            .i32(self.cfg.vocab_size as i32)
            .f32(temp)
            .push(self.seed)
            .ptr(self.b.rng.ptr)
            .i32(i32::from(greedy))
            .ptr(self.b.out.ptr)
            .ptr(self.b.spec.ptr)
            .ptr(self.b.spec.ptr + 4);
        self.launch("glm_spec_accept", Launch::new(1, 1024).smem(smem), &mut a)
    }

    /// Ask the device to pick the next verification width.
    pub fn pick_launch(&self, handle: u64, blk: usize) -> Result<()> {
        let mut a = Args::new();
        a.ptr(self.b.spec.ptr).push(handle).i32(blk as i32);
        self.launch("glm_spec_pick", Launch::new(1, 32), &mut a)
    }
}

/// A captured speculative round: drafting, the device-side width choice and every
/// verification branch live in one `CUgraph`, so a step is one launch plus one small copy.
pub struct SpecGraph {
    exec: cu::GraphExec,
    pub blk: usize,
}

impl Engine {
    /// Capture the whole round once per draft width.
    ///
    /// Shape: `draft(blk rows)` → `spec_pick` → switch on the handle with `blk` bodies,
    /// where body `i` verifies `i + 1` drafts and accepts them. Positions and token ids
    /// reach the graph through the staging buffers, never through kernel arguments, so the
    /// same `GraphExec` replays at any offset.
    pub fn capture_spec_graph(&mut self, blk: usize, temp: f32, greedy: bool) -> Result<SpecGraph> {
        ensure!(self.w.mtp.is_some(), "the graph path needs an MTP drafter");
        ensure!(blk >= 1 && blk + 1 <= ROWS, "graph block must be 1..{}", ROWS - 1);
        let g = cu::Graph::empty()?;
        let handle = g.cond_handle_flags((blk - 1) as u32, true)?;
        self.s.synchronize()?;

        self.s.begin_capture_into(g.raw)?;
        let drafted = self.draft_dev(blk, temp).and_then(|()| self.pick_launch(handle, blk));
        let cap = self.s.end_capture_into();
        drafted?;
        cap?;

        let sinks = g.sinks()?;
        let bodies = g.add_switch(&sinks, handle, blk as u32)?;
        for (i, body) in bodies.iter().enumerate() {
            let m = i + 2; // value i selects L = i + 1 drafts, i.e. m rows including the anchor
            self.s.begin_capture_into(*body)?;
            let r = self
                .build_rows(m - 1)
                .and_then(|()| self.forward_ids(m))
                .and_then(|()| self.accept_launch(m - 1, temp, greedy));
            let cap = self.s.end_capture_into();
            r?;
            cap?;
        }
        let exec = g.instantiate()?;
        exec.upload(&self.s)?;
        Ok(SpecGraph { exec, blk })
    }

    /// Replay one captured round starting at `self.pos`, returning the emitted tokens.
    ///
    /// The drafter and the target end the round with the same token prefix in their
    /// caches, so `mtp_pos` simply follows `pos`: no settle step is needed in this path.
    pub fn graph_round(&mut self, g: &SpecGraph, anchor: i32) -> Result<Vec<i32>> {
        ensure!(self.pos + g.blk + 2 <= self.kv.max_seq, "round would pass the KV capacity");
        self.set_rows(self.pos, &[anchor])?;
        self.stage_positions(self.pos, g.blk + 1)?;
        g.exec.launch(&self.s)?;
        self.s.synchronize()?;
        let v = cu::download::<i32>(self.b.spec.ptr, 2)?;
        let (count, committed) = (v[0].max(0) as usize, v[1].max(0) as usize);
        let emitted = self.read_out(count)?;
        self.graph_committed = committed;
        self.pos += committed;
        self.mtp_pos = self.pos;
        Ok(emitted)
    }

    /// Capture the round once and route every subsequent step through it.
    pub fn enable_graph(&mut self, blk: usize, temp: f32, greedy: bool) -> Result<()> {
        let g = self.capture_spec_graph(blk, temp, greedy)?;
        self.graph = Some(g);
        Ok(())
    }

    /// Width the device chose last round (0 before the first round).
    pub fn last_pick(&self) -> Result<usize> {
        Ok(cu::download::<i32>(self.b.spec.ptr + 20, 1)?[0].max(0) as usize)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn split_plan_matches_the_cache_helper() {
        assert!(crate::cache::decode_splits(1, 96, 1, 148) >= 1);
    }
}
