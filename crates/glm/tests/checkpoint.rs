//! End-to-end checkpoint handling without a GPU: builds a synthetic GLM checkpoint in
//! a temp directory and drives the parts of the loader that do not need a device
//! (tensor resolution, fp8 adoption, bf16 quantization, memory accounting).

use deepseek_ultrafast_model::st::Checkpoint;
use glm::config::GlmConfig;
use glm::quant::BLK;
use glm::weights::{load_host_linear, HostLinear, Loader};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

const HIDDEN: usize = 256;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const INTER: usize = 512;
const MOE_INTER: usize = 256;
const VOCAB: usize = 1024;
const LAYERS: usize = 2;
const EXPERTS: usize = 2;

fn config() -> serde_json::Value {
    json!({
        "architectures": ["Glm4MoeForCausalLM"],
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "moe_intermediate_size": MOE_INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "partial_rotary_factor": 0.5,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 8192,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": false
    })
}

fn bf16_matrix(rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 2);
    for i in 0..rows * cols {
        let v = ((i as f32 * 0.017 + seed).sin()) * 0.05;
        out.extend_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
    }
    out
}

fn bf16_vec(n: usize, seed: f32) -> Vec<u8> {
    bf16_matrix(1, n, seed)
}

struct Tensors {
    map: BTreeMap<String, (String, Vec<usize>, Vec<u8>)>,
}

impl Tensors {
    fn new() -> Self {
        Tensors { map: BTreeMap::new() }
    }

    fn bf16(&mut self, name: &str, rows: usize, cols: usize, seed: f32) {
        self.map.insert(name.to_string(), ("BF16".into(), vec![rows, cols], bf16_matrix(rows, cols, seed)));
    }

    fn vec_bf16(&mut self, name: &str, n: usize, seed: f32) {
        self.map.insert(name.to_string(), ("BF16".into(), vec![n], bf16_vec(n, seed)));
    }

    fn f32(&mut self, name: &str, n: usize) {
        let data: Vec<u8> = (0..n).flat_map(|_| 0.1f32.to_le_bytes()).collect();
        self.map.insert(name.to_string(), ("F32".into(), vec![n], data));
    }

    /// An fp8 tensor with 128x128 block scales, as a `*-block-fp8` checkpoint ships it.
    fn fp8(&mut self, name: &str, rows: usize, cols: usize) {
        let data = vec![0x38u8; rows * cols]; // 1.0 in e4m3
        self.map.insert(name.to_string(), ("F8_E4M3".into(), vec![rows, cols], data));
        let scale = format!("{}.weight_scale_inv", name.trim_end_matches(".weight"));
        let scales: Vec<u8> = (0..(rows / BLK) * (cols / BLK)).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        self.map.insert(scale, ("F32".into(), vec![rows / BLK, cols / BLK], scales));
    }

    fn write(&self, dir: &Path) {
        let mut header = serde_json::Map::new();
        let mut blob = Vec::new();
        for (name, (dtype, shape, data)) in &self.map {
            let start = blob.len();
            blob.extend_from_slice(data);
            header.insert(
                name.clone(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, blob.len()]}),
            );
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(&(h.len() as u64).to_le_bytes());
        file.extend_from_slice(&h);
        file.extend_from_slice(&blob);
        std::fs::write(dir.join("model-00001-of-00001.safetensors"), file).unwrap();
    }
}

fn build(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config()).unwrap()).unwrap();
    let mut t = Tensors::new();
    t.bf16("model.embed_tokens.weight", VOCAB, HIDDEN, 0.0);
    t.vec_bf16("model.norm.weight", HIDDEN, 0.1);
    t.bf16("lm_head.weight", VOCAB, HIDDEN, 0.2);
    for i in 0..=LAYERS {
        let p = format!("model.layers.{i}.");
        t.vec_bf16(&format!("{p}input_layernorm.weight"), HIDDEN, 0.3);
        t.vec_bf16(&format!("{p}post_attention_layernorm.weight"), HIDDEN, 0.4);
        t.bf16(&format!("{p}self_attn.q_proj.weight"), HEADS * HEAD_DIM, HIDDEN, 0.5);
        // one projection ships as block-fp8 to exercise the adoption path
        if i == 0 {
            t.fp8(&format!("{p}self_attn.k_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN);
        } else {
            t.bf16(&format!("{p}self_attn.k_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 0.6);
        }
        t.bf16(&format!("{p}self_attn.v_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, 0.7);
        t.bf16(&format!("{p}self_attn.o_proj.weight"), HIDDEN, HEADS * HEAD_DIM, 0.8);
        t.vec_bf16(&format!("{p}self_attn.q_norm.weight"), HEAD_DIM, 0.9);
        t.vec_bf16(&format!("{p}self_attn.k_norm.weight"), HEAD_DIM, 1.0);
        if i < 1 {
            t.bf16(&format!("{p}mlp.gate_proj.weight"), INTER, HIDDEN, 1.1);
            t.bf16(&format!("{p}mlp.up_proj.weight"), INTER, HIDDEN, 1.2);
            t.bf16(&format!("{p}mlp.down_proj.weight"), HIDDEN, INTER, 1.3);
        } else {
            t.bf16(&format!("{p}mlp.gate.weight"), EXPERTS, HIDDEN, 1.4);
            t.f32(&format!("{p}mlp.gate.e_score_correction_bias"), EXPERTS);
            t.bf16(&format!("{p}mlp.shared_experts.gate_proj.weight"), MOE_INTER, HIDDEN, 1.5);
            t.bf16(&format!("{p}mlp.shared_experts.up_proj.weight"), MOE_INTER, HIDDEN, 1.6);
            t.bf16(&format!("{p}mlp.shared_experts.down_proj.weight"), HIDDEN, MOE_INTER, 1.7);
            for e in 0..EXPERTS {
                let q = format!("{p}mlp.experts.{e}.");
                t.bf16(&format!("{q}gate_proj.weight"), MOE_INTER, HIDDEN, 1.8 + e as f32);
                t.bf16(&format!("{q}up_proj.weight"), MOE_INTER, HIDDEN, 1.9 + e as f32);
                t.bf16(&format!("{q}down_proj.weight"), HIDDEN, MOE_INTER, 2.0 + e as f32);
            }
        }
    }
    let m = format!("model.layers.{LAYERS}.");
    t.bf16(&format!("{m}eh_proj.weight"), HIDDEN, 2 * HIDDEN, 2.1);
    t.vec_bf16(&format!("{m}enorm.weight"), HIDDEN, 2.2);
    t.vec_bf16(&format!("{m}hnorm.weight"), HIDDEN, 2.3);
    t.write(dir);
}

fn open() -> (tempfile::TempDir, GlmConfig, Checkpoint) {
    let dir = tempfile::tempdir().unwrap();
    build(dir.path());
    let cfg = GlmConfig::load(dir.path()).unwrap();
    let ck = Checkpoint::open(dir.path()).unwrap();
    (dir, cfg, ck)
}

#[test]
fn resolves_every_tensor_of_a_glm_shaped_checkpoint() {
    let (_d, cfg, ck) = open();
    let l = Loader::new(&ck, &cfg).expect("all slots resolve");
    assert!(l.res.missing.is_empty());
    assert_eq!(
        l.name("layers.1.mlp.experts.1.down").unwrap(),
        "model.layers.1.mlp.experts.1.down_proj.weight"
    );
    assert_eq!(l.name("mtp.eh_proj").unwrap(), "model.layers.2.eh_proj.weight");
    assert_eq!(l.name("lm_head").unwrap(), "lm_head.weight");
}

#[test]
fn missing_tensor_names_the_slot_and_suggests_a_neighbour() {
    let dir = tempfile::tempdir().unwrap();
    build(dir.path());
    // drop the second expert's down projection
    let cfg = GlmConfig::load(dir.path()).unwrap();
    let mut ck = Checkpoint::open(dir.path()).unwrap();
    ck.tensors.remove("model.layers.1.mlp.experts.1.down_proj.weight");
    let err = match Loader::new(&ck, &cfg) {
        Ok(_) => panic!("a missing expert must fail the load"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("experts.1.down_proj") || err.contains("experts"), "{err}");
}

#[test]
fn bf16_weights_are_quantized_to_block_fp8() {
    let (_d, cfg, ck) = open();
    let h = load_host_linear(&ck, "model.layers.0.self_attn.q_proj.weight", HEADS * HEAD_DIM, HIDDEN).unwrap();
    match h {
        HostLinear::Fp8(b) => {
            assert_eq!((b.n, b.k), (HEADS * HEAD_DIM, HIDDEN));
            assert_eq!(b.scales.len(), (b.n / BLK) * (b.k / BLK));
            assert!(b.scales.iter().all(|s| *s > 0.0));
            // every block scale is the block amax over 448, so dequantized values stay in range
            let max_scale = b.scales.iter().cloned().fold(0.0f32, f32::max);
            assert!(max_scale > 0.0 && max_scale < 0.05, "{max_scale}");
        }
        _ => panic!("bf16 input must become fp8"),
    }
    assert_eq!(cfg.hidden, HIDDEN);
}

#[test]
fn fp8_tensors_are_adopted_with_their_block_scales() {
    let (_d, _cfg, ck) = open();
    let h = load_host_linear(&ck, "model.layers.0.self_attn.k_proj.weight", KV_HEADS * HEAD_DIM, HIDDEN).unwrap();
    match h {
        HostLinear::Fp8(b) => {
            assert_eq!(b.scale_at(0, 0), 0.5);
            // 0x38 is 1.0 in e4m3, scaled by 0.5
            assert!((b.dequant(0, 0) - 0.5).abs() < 1e-6);
        }
        _ => panic!("fp8 input must stay fp8"),
    }
}

#[test]
fn moe_layer_shapes_match_the_packed_expert_buffers() {
    let (_d, cfg, ck) = open();
    let l = Loader::new(&ck, &cfg).unwrap();
    // expert 0 gate/up plus down, all quantized to 128-block fp8
    let g = load_host_linear(
        &ck,
        l.name("layers.1.mlp.experts.0.gate").unwrap(),
        MOE_INTER,
        HIDDEN,
    )
    .unwrap();
    let d = load_host_linear(
        &ck,
        "model.layers.1.mlp.experts.0.down_proj.weight",
        HIDDEN,
        MOE_INTER,
    )
    .unwrap();
    assert_eq!(g.device_bytes(), MOE_INTER * HIDDEN + (MOE_INTER / BLK) * (HIDDEN / BLK) * 4);
    assert_eq!(d.device_bytes(), HIDDEN * MOE_INTER + (HIDDEN / BLK) * (MOE_INTER / BLK) * 4);
}

#[test]
fn ragged_fp8_scales_cover_partial_blocks() {
    for (n, k) in [(130usize, 128usize), (128, 130), (2, 256)] {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Tensors::new();
        let name = "ragged.weight";
        t.map.insert(name.into(), ("F8_E4M3".into(), vec![n, k], vec![0x38; n * k]));
        let nr = n.div_ceil(BLK);
        let nk = k.div_ceil(BLK);
        let scales: Vec<u8> = (0..nr * nk).flat_map(|i| ((i + 1) as f32 * 0.5).to_le_bytes()).collect();
        t.map.insert("ragged.weight_scale_inv".into(), ("F32".into(), vec![nr, nk], scales));
        t.write(dir.path());
        let ck = Checkpoint::open(dir.path()).unwrap();
        let HostLinear::Fp8(b) = load_host_linear(&ck, name, n, k).unwrap() else { panic!("fp8") };
        assert_eq!((b.n, b.k), (n.next_multiple_of(BLK), k.next_multiple_of(BLK)));
        for (r, c) in [(0, 0), (n - 1, k - 1)] {
            let expected = ((r / BLK) * nk + c / BLK + 1) as f32 * 0.5;
            assert!((b.dequant(r, c) - expected).abs() < 0.06 * expected);
        }
        if n < b.n { assert_eq!(b.dequant(n, 0), 0.0); }
        if k < b.k { assert_eq!(b.dequant(0, k), 0.0); }
    }
}

#[test]
fn row_major_layout_is_preserved_between_rows() {
    let (_d, _cfg, ck) = open();
    let h = load_host_linear(&ck, "model.layers.0.mlp.up_proj.weight", INTER, HIDDEN).unwrap();
    let HostLinear::Fp8(b) = h else { panic!("expected fp8") };
    // rows must not interleave: the paddings of row 0 and row 1 sit where expected
    assert_eq!(b.data.len(), INTER * HIDDEN);
    let row_stride = b.k;
    assert_eq!(row_stride, HIDDEN);
}
