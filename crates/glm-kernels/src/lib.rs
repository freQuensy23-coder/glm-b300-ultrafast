//! Kernel registry for the single-GPU GLM engine.
//!
//! `cu/*.cu` is compiled to cubins by `build.rs` for `GLM_SM` (default `sm_103a`,
//! i.e. B300) and embedded in the binary. This registry keeps the exported symbol
//! list next to the launch code that uses it so a signature change fails loudly at
//! load time instead of silently at the first token.
use anyhow::{Context, Result};
use deepseek_ultrafast_cuda::{Function, Module};
use std::collections::HashMap;

include!(concat!(env!("OUT_DIR"), "/cubins.rs"));

/// Module -> exported kernel symbols. Kept in sync with the `.cu` files by tests.
const MODULES: &[(&str, &[&str])] = &[
    ("gemv", &["glm_gemv_fp8", "glm_gemv_bf16", "glm_gemv_fp4", "glm_gemm_fp8"]),
    (
        "misc",
        &[
            "glm_rmsnorm",
            "glm_rmsnorm_heads",
            "glm_rope_qk",
            "glm_swiglu_mul",
            "glm_add_residual",
            "glm_cast_f32_bf16",
            "glm_cat2",
            "glm_embed",
            "glm_kv_append", "glm_kv_append_q8", "glm_kv_append_q4",
            "glm_zero_f32",
            "glm_add_bias",
            "glm_copy_bf16",
        ],
    ),
    ("attn", &["glm_attn_decode_split", "glm_attn_decode_split_q8", "glm_attn_decode_split_q4", "glm_attn_combine", "glm_attn_prefill", "glm_attn_prefill_q8", "glm_attn_prefill_q4"]),
    (
        "moe",
        &["glm_router", "glm_route_lists", "glm_moe_gate_up", "glm_moe_down", "glm_moe_reduce", "glm_moe_reduce_pair", "glm_moe_sgl"],
    ),
    ("sample", &["glm_argmax", "glm_sample", "glm_spec_accept", "glm_spec_pick", "glm_build_rows", "glm_stage_pos", "glm_prev_pos", "glm_stage_pos_rel"]),
    // Reused upstream tensor-core kernels (compiled from ../kernels/cu).
    ("u_tcgemv", &["tc_gemv_fp8", "tc_gemv_bf16"]),
    (
        "u_gemv",
        &[
            "gemv_fp8_n1_p1",
            "gemv_fp8_n1_p2",
            "gemv_fp8_n1_p4",
            "gemv_fp4_n1_p1",
            "gemv_fp4_n1_p2",
            "gemv_fp4_n1_p4",
            "gemv_bf16_n1_p1",
        ],
    ),
    ("u_tcgemm", &["tc_gemm_fp8", "tc_quant_rows", "tc_row_sumsq", "tc_norm_rows", "tc_gemm_bf16"]),
    ("tcaux", &["glm_quant_mx", "glm_gather_quant"]),
    (
        "u_tcmoe",
        &["moe_perm", "moe_gather_quant", "tc_moe_w13", "tc_moe_w2"],
    ),
    // The persistent, per-expert-map variant of the same body: this is the path the
    // upstream engine actually runs, the plain `tcmoe.cu` build is its fallback.
    ("u_variant_zzzzzzzzzzmoe", &["tc_moe_w13_rm2", "tc_moe_w2_rm", "tmoe_fused"]),
];

pub struct Kernels {
    _modules: Vec<Module>,
    funcs: HashMap<&'static str, Function>,
}

impl Kernels {
    pub fn load() -> Result<Self> {
        let mut k = Kernels { _modules: Vec::new(), funcs: HashMap::new() };
        for (name, fns) in MODULES {
            let image = CUBINS
                .iter()
                .find(|(m, _)| m == name)
                .map(|(_, b)| *b)
                .with_context(|| format!("cubin {name} was not built (set GLM_SM, e.g. sm_103a)"))?;
            let m = Module::load(image).with_context(|| format!("loading module {name}"))?;
            for f in fns.iter() {
                let func = m.function(f).with_context(|| format!("{name}/{f}"))?;
                if k.funcs.insert(f, func).is_some() {
                    anyhow::bail!("duplicate kernel symbol {f}");
                }
            }
            k._modules.push(m);
        }
        Ok(k)
    }

    pub fn get(&self, name: &str) -> Result<&Function> {
        self.funcs.get(name).with_context(|| format!("no kernel named {name}"))
    }

    pub fn names(&self) -> impl Iterator<Item = &&'static str> {
        self.funcs.keys()
    }
}

/// Dynamic shared memory for `glm_attn_decode_split`: q[hd], K/V tiles, scores,
/// output row and the reduction scratch.
pub fn attn_decode_smem(hd: usize) -> u32 {
    let tile = 64usize;
    ((hd + 2 * tile * hd + tile + hd + 8) * 4) as u32
}

/// Dynamic shared memory for `glm_attn_prefill` (TQ=16 query rows, TILE=64 KV).
pub fn attn_prefill_smem(hd: usize) -> u32 {
    let (tq, tile) = (16usize, 64usize);
    ((tq * hd + 2 * tile * hd + tq * tile + tq * hd + 2 * tq) * 4) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_table_covers_every_built_cubin() {
        for (m, _) in CUBINS {
            assert!(MODULES.iter().any(|(n, _)| n == m), "cubin {m} missing from MODULES");
        }
        for (m, _) in MODULES {
            assert!(CUBINS.iter().any(|(n, _)| n == m), "MODULES lists {m} but it was not built");
        }
    }

    /// A kernel name that never made it into the cubin (`#ifdef`-gated variants are the easy
    /// way to lose one) must fail here rather than at the first launch on the GPU.
    #[test]
    fn every_listed_kernel_is_exported_by_its_cubin() {
        for (module, names) in MODULES {
            let image = CUBINS.iter().find(|(n, _)| n == module).map(|(_, b)| *b).expect("cubin");
            for name in *names {
                let needle = name.as_bytes();
                assert!(
                    image.windows(needle.len()).any(|w| w == needle),
                    "cubin {module} does not export {name}"
                );
            }
        }
    }

    #[test]
    fn shared_memory_budgets_fit_blackwell() {
        // B300 allows 227 KB of dynamic shared memory per SM.
        assert!(attn_decode_smem(128) < 227 << 10);
        assert!(attn_prefill_smem(128) < 227 << 10);
    }
}
