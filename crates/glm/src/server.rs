//! OpenAI-compatible HTTP surface for the single-GPU engine.
//!
//! The engine object lives on its own OS thread and is driven by requests over a
//! channel: `Engine` holds CUDA handles that are meant to be used from the thread that
//! created them, and serialising the work here also mirrors the original server, which
//! admits one job at a time. Concurrency comes from the number of server processes,
//! not from batching unrelated requests into one step.

use crate::config::GlmConfig;
use crate::engine::Engine;
use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;

pub struct GenRequest {
    pub prompt: Vec<i32>,
    pub max_new: usize,
    pub temp: f32,
    pub greedy: bool,
    pub tokens: tokio::sync::mpsc::UnboundedSender<i32>,
}

pub struct Handle {
    pub cfg: GlmConfig,
    pub tok: Arc<Tokenizer>,
    pub model_name: String,
    tx: std::sync::mpsc::Sender<GenRequest>,
    pub ready: Arc<AtomicBool>,
    pub sessions: Mutex<Option<String>>,
}

impl Handle {
    /// Spawn the engine on a dedicated thread and return a handle for the HTTP layer.
    pub fn spawn(mut engine: Engine, tok: Tokenizer, model_name: String) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<GenRequest>();
        let cfg = engine.cfg.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let r = ready.clone();
        std::thread::Builder::new()
            .name("glm-engine".into())
            .stack_size(64 << 20)
            .spawn(move || {
                r.store(true, Ordering::SeqCst);
                for req in rx {
                    let GenRequest { prompt, max_new, temp, greedy, tokens } = req;
                    let res = engine.generate(&prompt, max_new, temp, greedy, |t| {
                        tokens.send(t).is_ok()
                    });
                    if let Err(e) = res {
                        eprintln!("generation failed: {e:#}");
                    }
                }
            })
            .context("spawning the engine thread")?;
        Ok(Handle {
            cfg,
            tok: Arc::new(tok),
            model_name,
            tx,
            ready,
            sessions: Mutex::new(None),
        })
    }

    fn submit(&self, req: GenRequest) -> Result<()> {
        self.tx.send(req).map_err(|_| anyhow::anyhow!("the engine thread has stopped"))
    }
}

#[derive(Deserialize, Default)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    prompt: Value,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

fn template(messages: &[ChatMessage]) -> String {
    // GLM chat format: [gMASK]<sop> then per-turn role markers.
    let mut s = String::from("[gMASK]<sop>");
    for m in messages {
        s.push_str(&format!("<|{}|>{}", m.role, m.content));
    }
    s.push_str("<|assistant|>");
    s
}

fn decode_prompt(tok: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let enc = tok.encode(text, false).map_err(|e| anyhow::anyhow!("tokenizing the prompt: {e}"))?;
    Ok(enc.get_ids().iter().map(|&i| i as i32).collect())
}

fn stream_tokens(
    handle: &Handle,
    prompt: Vec<i32>,
    max_new: usize,
    temp: f32,
    chat: bool,
) -> Result<tokio::sync::mpsc::UnboundedReceiver<i32>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    handle.submit(GenRequest { prompt, max_new, temp, greedy: temp <= 0.0, tokens: tx })?;
    let _ = chat;
    Ok(rx)
}

/// Incremental detokenizer: re-decodes the emitted prefix and forwards only the delta,
/// so multi-byte tokens are never split across chunks.
struct Detok {
    tok: Arc<Tokenizer>,
    all: Vec<u32>,
    emitted: String,
}

impl Detok {
    fn new(tok: Arc<Tokenizer>) -> Self {
        Detok { tok, all: Vec::new(), emitted: String::new() }
    }

    /// Decode the accumulated ids and return only the new suffix. A decoded string that
    /// ends in U+FFFD means the last token ended mid-character, so the tail is held back
    /// until the remaining bytes arrive — otherwise a client would see mojibake.
    fn push(&mut self, id: i32) -> Option<String> {
        self.all.push(id as u32);
        let mut text = self.tok.decode(&self.all, true).ok()?;
        while text.ends_with('\u{FFFD}') {
            text.pop();
        }
        if text.len() <= self.emitted.len() {
            return None;
        }
        if !text.starts_with(&self.emitted) {
            self.emitted = text.clone();
            return Some(text);
        }
        let delta = text[self.emitted.len()..].to_string();
        self.emitted = text;
        Some(delta)
    }
}

fn now_id() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

async fn health(State(h): State<Arc<Handle>>) -> Response {
    if h.ready.load(Ordering::SeqCst) {
        (StatusCode::OK, Json(json!({"status": "ok"}))).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "loading"}))).into_response()
    }
}

async fn readyz(State(h): State<Arc<Handle>>) -> Response {
    health(State(h)).await
}

async fn models(State(h): State<Arc<Handle>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": h.model_name, "object": "model", "owned_by": "local"}]}))
}

async fn chat(State(h): State<Arc<Handle>>, Json(req): Json<ChatRequest>) -> Response {
    let max_new = req.max_completion_tokens.or(req.max_tokens).unwrap_or(512).min(32_768);
    let temp = req.temperature.unwrap_or(1.0);
    let prompt = match decode_prompt(&h.tok, &template(&req.messages)) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
    };
    if prompt.len() + max_new + 16 >= h.cfg.max_seq {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "prompt plus output exceeds the KV capacity"}))).into_response();
    }
    let model = req.model.clone().unwrap_or_else(|| h.model_name.to_string());
    if req.stream.unwrap_or(false) {
        let rx = match stream_tokens(&h, prompt, max_new, temp, true) {
            Ok(r) => r,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
        };
        let handle = h.clone();
        let detok = Detok::new(h.tok.clone());
        let stream = futures::stream::unfold((rx, detok), move |(mut rx, mut detok)| {
            let handle = handle.clone();
            let model = model.clone();
            async move {
                match rx.recv().await {
                    Some(id) => {
                        let text = detok.push(id).unwrap_or_default();
                        let chunk = json!({
                            "id": format!("chatcmpl-{}", now_id()),
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
                        });
                        let _ = &handle;
                        Some((Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string())), (rx, detok)))
                    }
                    None => {
                        let done = json!({
                            "id": format!("chatcmpl-{}", now_id()),
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        });
                        Some((Ok(Event::default().data(done.to_string())), (rx, detok)))
                    }
                }
            }
        });
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }
    let mut rx = match stream_tokens(&h, prompt, max_new, temp, true) {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    };
    let mut detok = Detok::new(h.tok.clone());
    let mut text = String::new();
    let mut n = 0usize;
    while let Some(id) = rx.recv().await {
        n += 1;
        if let Some(d) = detok.push(id) {
            text.push_str(&d);
        }
    }
    Json(json!({
        "id": format!("chatcmpl-{}", now_id()),
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": n},
    }))
    .into_response()
}

async fn completions(State(h): State<Arc<Handle>>, Json(req): Json<CompletionRequest>) -> Response {
    let max_new = req.max_tokens.unwrap_or(16).min(32_768);
    let temp = req.temperature.unwrap_or(1.0);
    let prompt = match &req.prompt {
        Value::String(s) => match decode_prompt(&h.tok, s) {
            Ok(p) => p,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))).into_response(),
        },
        Value::Array(a) => a.iter().filter_map(|v| v.as_i64().map(|i| i as i32)).collect(),
        _ => return (StatusCode::BAD_REQUEST, Json(json!({"error": "prompt must be a string or token array"}))).into_response(),
    };
    if prompt.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "empty prompt"}))).into_response();
    }
    let model = req.model.clone().unwrap_or_else(|| h.model_name.to_string());
    let rx = match stream_tokens(&h, prompt, max_new, temp, false) {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response(),
    };
    if req.stream.unwrap_or(false) {
        let detok = Detok::new(h.tok.clone());
        let stream = futures::stream::unfold((rx, detok), move |(mut rx, mut detok)| {
            let model = model.clone();
            async move {
                match rx.recv().await {
                    Some(id) => {
                        let text = detok.push(id).unwrap_or_default();
                        let chunk = json!({
                            "id": format!("cmpl-{}", now_id()),
                            "object": "text_completion",
                            "model": model,
                            "choices": [{"index": 0, "text": text, "finish_reason": null}],
                        });
                        Some((Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string())), (rx, detok)))
                    }
                    None => {
                        let done = json!({
                            "id": format!("cmpl-{}", now_id()),
                            "object": "text_completion",
                            "model": model,
                            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                        });
                        Some((Ok(Event::default().data(done.to_string())), (rx, detok)))
                    }
                }
            }
        });
        return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
    }
    let mut rx = rx;
    let mut detok = Detok::new(h.tok.clone());
    let mut text = String::new();
    let mut n = 0usize;
    while let Some(id) = rx.recv().await {
        n += 1;
        if let Some(d) = detok.push(id) {
            text.push_str(&d);
        }
    }
    Json(json!({
        "id": format!("cmpl-{}", now_id()),
        "object": "text_completion",
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"completion_tokens": n},
    }))
    .into_response()
}

pub async fn run(engine: Engine, tok: Tokenizer, model_name: String, port: u16) -> Result<()> {
    let handle = Arc::new(Handle::spawn(engine, tok, model_name)?);
    let app = Router::new()
        .route("/health", get(health))
        .route("/readyz", get(readyz))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .with_state(handle);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    println!("glm-serve listening on 0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GPT-2 byte-level alphabet: byte -> printable char, so a BPE vocabulary can
    /// carry arbitrary bytes. Same table `tokenizers` uses in `ByteLevel`.
    fn byte_chars() -> Vec<char> {
        let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        let mut cs: Vec<u32> = bs.clone();
        let mut n = 0;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        let mut out = vec![' '; 256];
        for (b, c) in bs.iter().zip(cs) {
            out[*b as usize] = char::from_u32(c).unwrap();
        }
        out
    }

    fn tok() -> Arc<Tokenizer> {
        let chars = byte_chars();
        let vocab: std::collections::HashMap<String, u32> =
            chars.iter().enumerate().map(|(i, c)| (c.to_string(), i as u32)).collect();
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": true,
                "vocab": vocab,
                "merges": [],
            },
        });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        Arc::new(Tokenizer::from_file(&path).expect("tokenizer"))
    }

    #[test]
    fn chat_template_wraps_every_turn() {
        let msgs = vec![
            ChatMessage { role: "system".into(), content: "be brief".into() },
            ChatMessage { role: "user".into(), content: "hi".into() },
        ];
        let p = template(&msgs);
        assert!(p.starts_with("[gMASK]<sop>"), "{p}");
        assert!(p.contains("<|system|>") && p.contains("be brief"), "{p}");
        assert!(p.contains("<|user|>") && p.contains("hi"), "{p}");
        assert!(p.ends_with("<|assistant|>"), "{p}");
    }

    #[test]
    fn detokenizer_emits_only_the_delta() {
        let t = tok();
        let mut d = Detok::new(t.clone());
        let ids: Vec<u32> = t.encode("hello", false).unwrap().get_ids().to_vec();
        assert_eq!(ids.len(), 5);
        let mut out = String::new();
        for id in ids {
            if let Some(delta) = d.push(id as i32) {
                out.push_str(&delta);
            }
        }
        assert_eq!(out, "hello");
    }

    #[test]
    fn detokenizer_holds_back_partial_characters() {
        let t = tok();
        let mut d = Detok::new(t.clone());
        let ids: Vec<u32> = t.encode("€", false).unwrap().get_ids().to_vec();
        assert_eq!(ids.len(), 3, "a three-byte character");
        assert_eq!(d.push(ids[0] as i32), None);
        assert_eq!(d.push(ids[1] as i32), None);
        assert_eq!(d.push(ids[2] as i32).as_deref(), Some("€"));
    }
}
