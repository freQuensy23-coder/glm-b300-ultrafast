//! `glm-serve`: single-B300 serving for GLM-class MoE checkpoints.
//!
//! Examples:
//!   glm-serve --ckpt /models/GLM-5.3-Flash --plan          # verify the checkpoint layout
//!   glm-serve --ckpt /models/GLM-5.3-Flash --prompt "hi"   # one-shot generation
//!   glm-serve --ckpt /models/GLM-5.3-Flash --port 8000     # OpenAI-compatible server

use anyhow::{bail, Context, Result};
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_model::st::Checkpoint;
use glm::config::GlmConfig;
use glm::engine::Engine;
use glm::weights;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const USAGE: &str = "glm-serve --ckpt DIR [--gpus 0] [--port 8000] [--max-seq N] [--chunk N] \
[--model-name NAME] [--temp F] [--seed N] [--prompt TEXT] [--max-new N] [--mtp-concat eh|he] [--graph BLK] [--tc] [--plan] [--no-mtp]";

struct Args {
    ckpt: PathBuf,
    gpus: Vec<i32>,
    port: u16,
    max_seq: Option<usize>,
    chunk: usize,
    model_name: String,
    temp: f32,
    seed: u64,
    prompt: Option<String>,
    max_new: usize,
    mtp_concat_embed_first: bool,
    plan: bool,
    no_mtp: bool,
    graph: Option<usize>,
    tc: bool,
}

fn parse() -> Result<Args> {
    let v: Vec<String> = std::env::args().collect();
    let mut a = Args {
        ckpt: PathBuf::from("/models"),
        gpus: vec![0],
        port: 8000,
        max_seq: None,
        chunk: 2048,
        model_name: "glm-5.3-flash".into(),
        temp: 1.0,
        seed: 0,
        prompt: None,
        max_new: 256,
        mtp_concat_embed_first: true,
        plan: false,
        no_mtp: false,
        graph: None,
        tc: false,
    };
    let mut i = 1;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().with_context(|| format!("{} needs a value\n{USAGE}", v[i]));
        match v[i].as_str() {
            "--ckpt" => a.ckpt = val()?.into(),
            "--gpus" => {
                let g: Vec<i32> = val()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()?;
                if g.len() != 1 {
                    bail!("this build serves one GPU; pass a single device such as --gpus 0\n{USAGE}");
                }
                a.gpus = g;
            }
            "--port" => a.port = val()?.parse()?,
            "--max-seq" => a.max_seq = Some(val()?.parse()?),
            "--chunk" => a.chunk = val()?.parse()?,
            "--model-name" => a.model_name = val()?,
            "--temp" | "--temperature" => a.temp = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--prompt" => a.prompt = Some(val()?),
            "--max-new" | "--max-tokens" => a.max_new = val()?.parse()?,
            "--mtp-concat" => {
                let s = val()?;
                a.mtp_concat_embed_first = match s.as_str() {
                    "eh" | "embed-first" => true,
                    "he" | "hidden-first" => false,
                    _ => bail!("--mtp-concat must be eh or he"),
                };
            }
            "--plan" => a.plan = true,
            "--tc" => a.tc = true,
            "--no-mtp" => a.no_mtp = true,
            "--graph" => {
                a.graph = Some(val()?.parse()?);
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            x => bail!("unknown argument {x}\n{USAGE}"),
        }
        i += 2;
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse()?;
    let cfg = GlmConfig::load(&args.ckpt).with_context(|| args.ckpt.display().to_string())?;
    let ck = Checkpoint::open(&args.ckpt)?;

    if args.plan {
        return plan(&cfg, &ck);
    }

    cu::init()?;
    let dev = cu::Device::new(args.gpus[0])?;
    let (free, total) = dev.mem_info()?;
    let (cc_major, cc_minor) = dev.compute_capability()?;
    println!(
        "device {}: compute {cc_major}.{cc_minor} sm={} {:.1}/{:.1} GB free",
        args.gpus[0],
        dev.sm_count()?,
        free as f64 / 1e9,
        total as f64 / 1e9
    );

    let t0 = Instant::now();
    let w = weights::load(&ck, &cfg, !args.no_mtp)?;
    println!("weights {:.1} GB loaded in {:.1}s", w.bytes as f64 / 1e9, t0.elapsed().as_secs_f32());
    let has_mtp = w.mtp.is_some();

    let (free, _) = dev.mem_info()?;
    let reserve = (free / 10).max(2 << 30);
    let budget = free.saturating_sub(reserve);
    let max_seq = match args.max_seq {
        Some(n) => n,
        None => glm::cache::KvCache::max_seq_for(cfg.n_kv, cfg.head_dim, cfg.n_layers, budget / 2, cfg.max_seq).max(4096),
    };
    println!(
        "kv cache: {max_seq} tokens ({:.1} GB)",
        (glm::cache::KvCache::layer_bytes(cfg.n_kv, cfg.head_dim, max_seq) * (cfg.n_layers + usize::from(has_mtp))) as f64
            / 1e9
    );

    let mut engine = Engine::new(cfg.clone(), w, max_seq, args.chunk)?;
    engine.seed = args.seed;
    engine.mtp_concat_embed_first = args.mtp_concat_embed_first;
    // Tensor-core path: mxfp8 weights + tcgen05 GEMV for the decode steps.
    crate::tc::set_on(args.tc || std::env::var("GLM_TC").is_ok_and(|v| v != "0"));
    if let Some(blk) = args.graph {
        // One captured round: drafting, the device-side width choice and every verify
        // branch in a single graph launch.
        engine.enable_graph(blk, args.temp, args.temp <= 0.0)?;
        println!("speculative round captured as a conditional graph, block {blk}");
    }
    println!(
        "ready in {:.1}s ({} layers, {} experts, top-{} active, {}-token context)",
        t0.elapsed().as_secs_f32(),
        cfg.n_layers,
        cfg.n_experts,
        cfg.top_k,
        max_seq
    );

    let tok = Tokenizer::from_file(args.ckpt.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;

    match args.prompt {
        Some(p) => {
            let enc = tok.encode(p, false).map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            let ids: Vec<i32> = enc.get_ids().iter().map(|&i| i as i32).collect();
            let st = Instant::now();
            let mut out = Vec::new();
            engine.generate(&ids, args.max_new, args.temp, args.temp <= 0.0, |t| {
                out.push(t);
                true
            })?;
            let dt = st.elapsed().as_secs_f64();
            let text = tok.decode(&out.iter().map(|&i| i as u32).collect::<Vec<_>>(), true).unwrap_or_default();
            println!("{text}\n");
            println!("{} tokens in {:.2}s ({:.1} tok/s)", out.len(), dt, out.len() as f64 / dt);
            Ok(())
        }
        None => {
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            rt.block_on(glm::server::run(engine, tok, args.model_name, args.port))
        }
    }
}

/// `--plan`: resolve every tensor the config needs, report what is missing, and print a
/// memory estimate. Touches no GPU, so it can run on a CPU-only box before renting one.
fn plan(cfg: &GlmConfig, ck: &Checkpoint) -> Result<()> {
    let loader = weights::Loader::new(ck, cfg)?;
    let mut total = 0usize;
    for name in loader.res.chosen.iter().map(|(_, n)| n) {
        let info = ck.info(name)?;
        let elems: usize = info.shape.iter().product();
        total += match info.dtype.as_str() {
            "F8_E4M3" | "F8_E4M3FN" | "FLOAT8_E4M3FN" => elems,
            "BF16" => elems * 2,
            "F32" => elems * 4,
            _ => 0,
        };
    }
    println!("resolved {} tensor slots", loader.res.chosen.len());
    println!("checkpoint payload {:.1} GB", total as f64 / 1e9);
    println!(
        "kv per token {:.1} KB | layers {} (+{} mtp) | experts {} top-{} | gqa {} | rope dim {}",
        cfg.kv_bytes_per_token() as f64 / 1024.0,
        cfg.n_layers,
        cfg.n_mtp,
        cfg.n_experts,
        cfg.top_k,
        cfg.gqa(),
        cfg.rope_dim
    );
    let per_token = cfg.kv_bytes_per_token() * (cfg.n_layers + cfg.n_mtp.max(0));
    for ctx in [32_768usize, 131_072, 524_288, 1_048_576] {
        println!("  ctx {ctx:>8}: kv {:.1} GB", (per_token * ctx) as f64 / 1e9);
    }
    Ok(())
}
