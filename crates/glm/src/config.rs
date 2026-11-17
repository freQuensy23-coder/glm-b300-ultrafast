//! GLM (GLM-4.5/5-family MoE, e.g. GLM-5.3-Flash) configuration.
//!
//! The engine is driven entirely by the checkpoint's `config.json`: no architecture
//! value is hardcoded. Missing optional keys fall back to the GLM defaults below,
//! anything the kernels cannot serve is rejected at load time with a message that
//! names the offending key.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scoring {
    Sigmoid,
    Softmax,
}

#[derive(Clone, Debug)]
pub struct GlmConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    /// Dense-FFN intermediate width (layers `0..first_dense`).
    pub inter: usize,
    pub moe_inter: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Rotary lanes per head (`head_dim * partial_rotary_factor`, even).
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub max_seq: usize,
    // MoE
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub first_dense: usize,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub scoring: Scoring,
    pub qkv_bias: bool,
    pub tie_embed: bool,
    /// Number of extra MTP layers used as the speculative drafter.
    pub n_mtp: usize,
    /// >0 turns on sliding-window attention for every layer (GLM-5 style hybrid
    /// attention uses per-layer values; this port serves uniform windows).
    pub sliding_window: Option<usize>,
}

fn get<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| v.get(*k)).filter(|x| !x.is_null())
}

fn num(v: &Value, keys: &[&str], default: Option<f64>) -> Result<f64> {
    match get(v, keys) {
        Some(x) => x
            .as_f64()
            .or_else(|| x.as_i64().map(|i| i as f64))
            .with_context(|| format!("{} must be a number", keys[0])),
        None => default.with_context(|| format!("missing required key {}", keys[0])),
    }
}

fn boolean(v: &Value, keys: &[&str], default: bool) -> bool {
    get(v, keys).and_then(|x| x.as_bool()).unwrap_or(default)
}

impl GlmConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("config.json")).with_context(|| "config.json")?;
        let v: Value = serde_json::from_slice(&raw)?;
        Self::from_json(&v)
    }

    pub fn from_json(v: &Value) -> Result<Self> {
        let hidden = num(v, &["hidden_size"], None)? as usize;
        let n_heads = num(v, &["num_attention_heads"], None)? as usize;
        let n_kv = num(v, &["num_key_value_heads", "multi_query_group_num"], Some(n_heads as f64))? as usize;
        let head_dim = num(v, &["head_dim", "kv_channels"], Some((hidden / n_heads) as f64))? as usize;
        let partial = num(v, &["partial_rotary_factor"], Some(1.0))? as f32;
        let rope_dim = ((head_dim as f32 * partial) as usize).next_multiple_of(2).min(head_dim);
        let moe_inter = num(v, &["moe_intermediate_size"], Some(num(v, &["intermediate_size"], None)?))? as usize;
        let n_experts = num(v, &["n_routed_experts", "num_experts", "num_local_experts"], Some(0.0))? as usize;
        let n_group_default = if n_experts > 0 { n_experts } else { 1 };
        let scoring = match get(v, &["scoring_func", "score_function"]).and_then(|x| x.as_str()) {
            Some("softmax") => Scoring::Softmax,
            _ => Scoring::Sigmoid,
        };
        let cfg = GlmConfig {
            vocab_size: num(v, &["vocab_size"], None)? as usize,
            hidden,
            inter: num(v, &["intermediate_size", "ffn_hidden_size"], None)? as usize,
            moe_inter,
            n_layers: num(v, &["num_hidden_layers"], None)? as usize,
            n_heads,
            n_kv,
            head_dim,
            rope_dim,
            rope_theta: num(v, &["rope_theta"], Some(10_000.0))? as f32,
            rms_eps: num(v, &["rms_norm_eps", "layernorm_epsilon"], Some(1e-5))? as f32,
            max_seq: num(v, &["max_position_embeddings"], Some(131_072.0))? as usize,
            n_experts,
            top_k: num(v, &["num_experts_per_tok", "moe_topk"], Some(1.0))? as usize,
            n_shared: num(v, &["n_shared_experts", "shared_expert_num"], Some(0.0))? as usize,
            n_group: num(v, &["n_group"], Some(n_group_default as f64))? as usize,
            topk_group: num(v, &["topk_group"], Some(n_group_default as f64))? as usize,
            first_dense: num(v, &["first_k_dense_replace"], Some(0.0))? as usize,
            norm_topk_prob: boolean(v, &["norm_topk_prob"], false),
            route_scale: num(v, &["routed_scaling_factor"], Some(1.0))? as f32,
            scoring,
            qkv_bias: boolean(v, &["attention_bias"], false),
            tie_embed: boolean(v, &["tie_word_embeddings"], false),
            n_mtp: num(v, &["num_nextn_predict_layers"], Some(0.0))? as usize,
            sliding_window: get(v, &["sliding_window", "window_size"]).and_then(|x| x.as_u64()).filter(|&w| w > 0).map(|w| w as usize),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.hidden % 128 == 0, "hidden_size {} must be a multiple of 128 (fp8 block scales)", self.hidden);
        // Dense and MoE intermediate widths are free-form (GLM uses 10944 and 1408);
        // the loader pads them to 128 so the block kernels can stream them.
        ensure!(self.inter > 0, "intermediate_size must be positive");
        if self.n_experts > 0 {
            ensure!(self.moe_inter > 0, "moe_intermediate_size must be positive");
        }
        ensure!(self.head_dim % 2 == 0 && self.rope_dim % 2 == 0, "head_dim/rope_dim must be even");
        ensure!(self.rope_dim <= self.head_dim, "rope_dim {} exceeds head_dim {}", self.rope_dim, self.head_dim);
        ensure!(self.n_kv > 0 && self.n_heads % self.n_kv == 0, "num_attention_heads must be a multiple of num_key_value_heads");
        ensure!(self.first_dense <= self.n_layers, "first_k_dense_replace {} exceeds the layer count", self.first_dense);
        if self.n_experts > 0 {
            ensure!(self.top_k > 0 && self.top_k <= self.n_experts, "num_experts_per_tok {} is out of range", self.top_k);
            ensure!(self.n_experts % self.n_group == 0, "n_routed_experts must be divisible by n_group");
            ensure!(self.topk_group >= 1 && self.topk_group <= self.n_group, "topk_group out of range");
            ensure!(self.top_k >= self.topk_group, "top-k must cover at least one expert per routed group");
        }
        if self.n_mtp > 1 {
            bail!("num_nextn_predict_layers {} > 1: this port drafts with a single MTP layer", self.n_mtp);
        }
        Ok(())
    }

    pub fn is_moe(&self, layer: usize) -> bool {
        self.n_experts > 0 && layer >= self.first_dense
    }

    pub fn gqa(&self) -> usize {
        self.n_heads / self.n_kv
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }

    pub fn attn_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }

    /// k padded to the 128-wide fp8 block the kernels expect.
    pub fn kpad(k: usize) -> usize {
        k.next_multiple_of(128)
    }

    pub fn npad(n: usize) -> usize {
        n.next_multiple_of(128)
    }

    /// KV cache bytes per token per layer (bf16 K and V).
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.kv_dim() * 2
    }

    /// Upper bound on cached tokens for a given device budget.
    pub fn max_seq_for_budget(&self, bytes: usize) -> usize {
        let per_layer = self.kv_bytes_per_token();
        (bytes / (per_layer * self.n_layers.max(1) + 16 * per_layer)).min(self.max_seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn parses_glm_moe_config() {
        let c = GlmConfig::from_json(&json(
            r#"{"vocab_size":151552,"hidden_size":4096,"intermediate_size":10944,"moe_intermediate_size":1408,
                "num_hidden_layers":92,"num_attention_heads":96,"num_key_value_heads":8,"head_dim":128,
                "partial_rotary_factor":0.5,"rope_theta":1000000,"rms_norm_eps":1e-5,
                "max_position_embeddings":131072,"n_routed_experts":128,"num_experts_per_tok":8,
                "n_shared_experts":1,"n_group":8,"topk_group":4,"first_k_dense_replace":1,
                "norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sigmoid",
                "num_nextn_predict_layers":1,"tie_word_embeddings":false}"#,
        ))
        .unwrap();
        assert_eq!(c.rope_dim, 64);
        assert_eq!(c.gqa(), 12);
        assert_eq!(c.q_dim(), 96 * 128);
        assert!(c.is_moe(1) && !c.is_moe(0));
        assert_eq!(c.scoring, Scoring::Sigmoid);
        assert_eq!(c.n_mtp, 1);
        assert!((c.route_scale - 2.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_shapes_the_kernels_cannot_serve() {
        let base = json(r#"{"vocab_size":1000,"hidden_size":4090,"intermediate_size":1000,"num_hidden_layers":2,
                           "num_attention_heads":8,"num_key_value_heads":8,"head_dim":128}"#);
        let err = GlmConfig::from_json(&base).unwrap_err().to_string();
        assert!(err.contains("multiple of 128"), "{err}");
        // a ragged intermediate width is accepted: the loader pads it
        let ragged = json(r#"{"vocab_size":1000,"hidden_size":4096,"intermediate_size":8576,"num_hidden_layers":2,
                             "num_attention_heads":8,"num_key_value_heads":8,"head_dim":128}"#);
        assert!(GlmConfig::from_json(&ragged).is_ok());

        let mut ok = json(r#"{"vocab_size":1000,"hidden_size":4096,"intermediate_size":1024,"num_hidden_layers":2,
                             "num_attention_heads":8,"num_key_value_heads":8,"head_dim":128}"#);
        assert!(GlmConfig::from_json(&ok).is_ok());
        // 8 heads / 3 kv heads is not a whole GQA group
        ok["num_key_value_heads"] = json("3");
        assert!(GlmConfig::from_json(&ok).unwrap_err().to_string().contains("multiple of num_key_value_heads"));
    }

    #[test]
    fn kv_budget_is_conservative() {
        let c = GlmConfig::from_json(&json(
            r#"{"vocab_size":151552,"hidden_size":4096,"intermediate_size":10944,"moe_intermediate_size":1408,
                "num_hidden_layers":4,"num_attention_heads":96,"num_key_value_heads":8,"head_dim":128,
                "max_position_embeddings":131072}"#,
        ))
        .unwrap();
        assert_eq!(c.kv_bytes_per_token(), 2 * 8 * 128 * 2);
        let fits = c.max_seq_for_budget(c.kv_bytes_per_token() * 4 * 1000);
        assert!(fits > 0 && fits <= 1000, "{fits}");
    }
}
