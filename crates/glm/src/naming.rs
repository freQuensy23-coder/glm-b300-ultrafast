//! Checkpoint tensor naming for the GLM MoE family, with alternative spellings.
//!
//! A port cannot rely on one exact spelling for every release (GLM-4.5, GLM-4.6 and
//! GLM-5 checkpoints differ in the MTP block and in router bias naming), so every
//! slot carries the alternatives the engine can serve and the loader takes the first
//! one present. When nothing matches, `missing_report` answers with the closest
//! tensor names actually in the checkpoint.

use crate::config::GlmConfig;
use std::collections::BTreeSet;

pub struct LayerNames {
    pub p: String,
    pub input_norm: Vec<String>,
    pub post_norm: Vec<String>,
    pub q: Vec<String>,
    pub k: Vec<String>,
    pub v: Vec<String>,
    pub o: Vec<String>,
    pub q_bias: Vec<String>,
    pub k_bias: Vec<String>,
    pub v_bias: Vec<String>,
    pub q_norm: Vec<String>,
    pub k_norm: Vec<String>,
    pub gate: Vec<String>,
    pub up: Vec<String>,
    pub down: Vec<String>,
    pub router: Vec<String>,
    pub router_bias: Vec<String>,
    pub shared_gate: Vec<String>,
    pub shared_up: Vec<String>,
    pub shared_down: Vec<String>,
}

fn alts(p: &str, suffixes: &[&str]) -> Vec<String> {
    suffixes.iter().map(|s| format!("{p}{s}")).collect()
}

pub fn layer(cfg: &GlmConfig, i: usize) -> LayerNames {
    let p = format!("model.layers.{i}.");
    let moe = cfg.is_moe(i);
    let mut n = LayerNames {
        input_norm: alts(&p, &["input_layernorm.weight"]),
        post_norm: alts(&p, &["post_attention_layernorm.weight"]),
        q: alts(&p, &["self_attn.q_proj.weight"]),
        k: alts(&p, &["self_attn.k_proj.weight"]),
        v: alts(&p, &["self_attn.v_proj.weight"]),
        o: alts(&p, &["self_attn.o_proj.weight", "self_attn.dense.weight"]),
        q_bias: alts(&p, &["self_attn.q_proj.bias"]),
        k_bias: alts(&p, &["self_attn.k_proj.bias"]),
        v_bias: alts(&p, &["self_attn.v_proj.bias"]),
        q_norm: alts(&p, &["self_attn.q_norm.weight"]),
        k_norm: alts(&p, &["self_attn.k_norm.weight"]),
        gate: vec![],
        up: vec![],
        down: vec![],
        router: vec![],
        router_bias: vec![],
        shared_gate: vec![],
        shared_up: vec![],
        shared_down: vec![],
        p: p.clone(),
    };
    if moe {
        n.router = alts(&p, &["mlp.gate.weight", "mlp.router.weight"]);
        n.router_bias = alts(&p, &["mlp.gate.e_score_correction_bias", "mlp.router.e_score_correction_bias"]);
        n.shared_gate = alts(&p, &["mlp.shared_experts.gate_proj.weight", "mlp.shared_expert.gate_proj.weight"]);
        n.shared_up = alts(&p, &["mlp.shared_experts.up_proj.weight", "mlp.shared_expert.up_proj.weight"]);
        n.shared_down = alts(&p, &["mlp.shared_experts.down_proj.weight", "mlp.shared_expert.down_proj.weight"]);
    } else {
        n.gate = alts(&p, &["mlp.gate_proj.weight"]);
        n.up = alts(&p, &["mlp.up_proj.weight"]);
        n.down = alts(&p, &["mlp.down_proj.weight"]);
    }
    n
}

/// Candidate names for one expert's gate, up and down matrices, in that order.
/// `w1`/`w3`/`w2` covers checkpoints that keep the original Mixtral-style names.
pub fn expert(i: usize, j: usize) -> [Vec<String>; 3] {
    let p = format!("model.layers.{i}.mlp.experts.{j}.");
    [
        alts(&p, &["gate_proj.weight", "w1.weight"]),
        alts(&p, &["up_proj.weight", "w3.weight"]),
        alts(&p, &["down_proj.weight", "w2.weight"]),
    ]
}

pub fn embed() -> Vec<String> {
    alts("model.", &["embed_tokens.weight"])
}

pub fn final_norm() -> Vec<String> {
    alts("model.", &["norm.weight"])
}

pub fn head(cfg: &GlmConfig) -> Vec<String> {
    if cfg.tie_embed {
        embed()
    } else {
        alts("", &["lm_head.weight"])
    }
}

/// The MTP layer is the first block after the decoder stack, with `eh_proj`/`enorm`/
/// `hnorm` fused into it.
pub fn mtp(i: usize) -> Vec<String> {
    let p = format!("model.layers.{i}.");
    alts(
        &p,
        &[
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "shared_head.norm.weight",
            "shared_head.head.weight",
        ],
    )
}

/// A required slot: `key` for diagnostics, `alternatives` in preference order.
pub struct Slot {
    pub key: String,
    pub alternatives: Vec<String>,
}

pub fn slots(cfg: &GlmConfig) -> Vec<Slot> {
    let mut out = Vec::new();
    let mut add = |key: String, alternatives: Vec<String>| out.push(Slot { key, alternatives });
    add("embed_tokens".into(), embed());
    add("final_norm".into(), final_norm());
    add("lm_head".into(), head(cfg));
    // The MTP block lives at `n_layers` and carries the same per-layer tensors as a
    // decoder layer, so it needs the same slots resolved.
    let stack = cfg.n_layers + usize::from(cfg.n_mtp > 0);
    for i in 0..stack {
        let n = layer(cfg, i);
        for (key, alts) in [
            ("input_layernorm", n.input_norm),
            ("post_attention_layernorm", n.post_norm),
            ("self_attn.q_proj", n.q),
            ("self_attn.k_proj", n.k),
            ("self_attn.v_proj", n.v),
            ("self_attn.o_proj", n.o),
            ("self_attn.q_norm", n.q_norm),
            ("self_attn.k_norm", n.k_norm),
        ] {
            add(format!("layers.{i}.{key}"), alts);
        }
        if cfg.is_moe(i) {
            add(format!("layers.{i}.mlp.gate"), n.router);
            if cfg.n_shared > 0 {
                add(format!("layers.{i}.mlp.shared_experts"), n.shared_gate);
            }
            // Expert tensors are uniform across the layer, so probing the first and the
            // last catches truncated downloads without enumerating hundreds of slots.
            let mut probes = vec![0usize];
            if cfg.n_experts > 1 {
                probes.push(cfg.n_experts - 1);
            }
            for e in probes {
                let names = expert(i, e);
                for (k, kind) in ["gate", "up", "down"].iter().enumerate() {
                    add(format!("layers.{i}.mlp.experts.{e}.{kind}"), names[k].clone());
                }
            }
        } else {
            for (key, alts) in [("gate_proj", n.gate), ("up_proj", n.up), ("down_proj", n.down)] {
                add(format!("layers.{i}.mlp.{key}"), alts);
            }
        }
    }
    if cfg.n_mtp > 0 {
        let m = mtp(cfg.n_layers);
        for (key, alts) in [
            ("eh_proj", vec![m[0].clone()]),
            ("enorm", vec![m[1].clone()]),
            ("hnorm", vec![m[2].clone()]),
        ] {
            add(format!("mtp.{key}"), alts);
        }
    }
    out
}

#[derive(Debug, Default)]
pub struct Resolution {
    pub chosen: Vec<(String, String)>,
    pub missing: Vec<String>,
}

/// Pick the first existing alternative per slot; report the rest with suggestions.
pub fn resolve(slots: &[Slot], present: &BTreeSet<String>) -> Resolution {
    let mut r = Resolution::default();
    for s in slots {
        match s.alternatives.iter().find(|a| present.contains(*a)) {
            Some(name) => r.chosen.push((s.key.clone(), name.clone())),
            None => {
                let hint = suggest(&s.alternatives[0], present);
                r.missing.push(format!("{}: none of {:?} present (did you mean {}?)", s.key, s.alternatives, hint));
            }
        }
    }
    r
}

/// Closest tensor by longest shared dotted-prefix, then by shared suffix.
pub fn suggest(want: &str, present: &BTreeSet<String>) -> String {
    let mut best: Option<(&String, usize)> = None;
    for p in present {
        let mut score = 0usize;
        for (a, b) in want.split('.').zip(p.split('.')) {
            if a == b {
                score += 1;
            } else {
                break;
            }
        }
        if want.rsplit('.').next() == p.rsplit('.').next() {
            score += 1;
        }
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((p, score));
        }
    }
    best.map(|(p, _)| p.clone()).unwrap_or_else(|| "<no tensors>".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GlmConfig {
        GlmConfig::from_json(&serde_json::json!({
            "vocab_size": 1024, "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 256,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
            "partial_rotary_factor": 0.5, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "n_shared_experts": 1, "n_group": 2, "topk_group": 1, "first_k_dense_replace": 1,
            "num_nextn_predict_layers": 1
        }))
        .unwrap()
    }

    #[test]
    fn dense_and_moe_layers_get_different_slots() {
        let c = cfg();
        let l0 = layer(&c, 0);
        assert!(l0.gate[0].ends_with("mlp.gate_proj.weight"));
        assert!(l0.router.is_empty());
        let l1 = layer(&c, 1);
        assert_eq!(l1.router[0], "model.layers.1.mlp.gate.weight");
        assert!(l1.gate.is_empty());
        assert_eq!(expert(1, 7)[2][0], "model.layers.1.mlp.experts.7.down_proj.weight");
        assert_eq!(mtp(2)[0], "model.layers.2.eh_proj.weight");
    }

    #[test]
    fn resolves_alternatives_and_keeps_the_first_present() {
        let c = cfg();
        let slots = slots(&c);
        let mut present = BTreeSet::new();
        for s in &slots {
            // every slot exists under its preferred spelling ...
            present.insert(s.alternatives[0].clone());
        }
        let r = resolve(&slots, &present);
        assert!(r.missing.is_empty(), "{:?}", r.missing);
        assert!(r.chosen.iter().any(|(k, v)| k == "lm_head" && v == "lm_head.weight"));
        // ... and a checkpoint spelling the output projection `dense` still resolves
        let mut alt = present.clone();
        alt.remove("model.layers.0.self_attn.o_proj.weight");
        alt.insert("model.layers.0.self_attn.dense.weight".into());
        let r = resolve(&slots, &alt);
        assert!(r.missing.is_empty(), "{:?}", r.missing);
        assert!(r
            .chosen
            .iter()
            .any(|(_, v)| v == "model.layers.0.self_attn.dense.weight"));
    }

    #[test]
    fn reports_missing_slots_with_a_suggestion() {
        let c = cfg();
        let slots = slots(&c);
        let mut present = BTreeSet::new();
        for s in &slots {
            present.insert(s.alternatives[0].clone());
        }
        present.remove("model.layers.1.mlp.gate.weight");
        let r = resolve(&slots, &present);
        assert_eq!(r.missing.len(), 1);
        assert!(r.missing[0].contains("layers.1.mlp.gate"), "{}", r.missing[0]);
        assert!(r.missing[0].contains("layers.1.mlp."), "{}", r.missing[0]);
    }

    #[test]
    fn tied_embeddings_reuse_the_embedding_tensor() {
        let mut c = cfg();
        c.tie_embed = true;
        assert_eq!(head(&c), embed());
    }
}
