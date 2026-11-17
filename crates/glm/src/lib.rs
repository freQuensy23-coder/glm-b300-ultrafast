//! Single-GPU inference for the GLM MoE family (GLM-5.3-Flash class models) on
//! sm_103a (B300), derived from `deepseek-ultrafast`.
//!
//! What was kept from the original engine: the block-verification speculative loop
//! with GPU-side acceptance, the weight-stationary fp8 decode GEMV, block-fp8/tiled
//! quantization, the OpenAI-compatible serving surface and the "draft block is rows,
//! not a batch" execution model.
//!
//! What was dropped because a single GPU does not need it: tensor parallelism, NVLS
//! multicast, all-reduce/all-gather, RDMA KV transfer, the Redis ownership directory
//! and the per-release-version kernel variants.
//!
//! What was replaced because GLM is not DeepSeek-V4.1: MLA + DSA indexer -> GQA with
//! partial RoPE and per-head qk-norm, hyper-connections -> plain residual streams,
//! engram -> none, DSpark Markov drafter -> a standard MTP layer.

pub mod cache;
pub mod config;
pub mod engine;
pub mod kernels;
pub mod naming;
pub mod quant;
pub mod sampling;
pub mod server;
pub mod spec;
pub mod tc;
pub mod weights;

pub use config::GlmConfig;
pub use engine::Engine;
pub use weights::Weights;
