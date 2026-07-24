//! Local LLM inference adapter (2026 Tauri pattern).
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LocalInferenceError {
    #[error("backend {backend} unreachable at {url}: {detail}")] BackendUnreachable { backend: String, url: String, detail: String },
    #[error("backend error: {0}")] BackendError(String),
    #[error("no model loaded")] NoModelLoaded,
    #[error("timeout after {0:?}")] Timeout(Duration),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalBackend { LlamaCpp, Ollama, LmStudio, Vllm, Custom }
impl LocalBackend { #[must_use] pub fn default_url(&self) -> &str { match self { LocalBackend::LlamaCpp => "http://localhost:8080", LocalBackend::Ollama => "http://localhost:11434", LocalBackend::LmStudio => "http://localhost:1234", LocalBackend::Vllm => "http://localhost:8000", LocalBackend::Custom => "http://localhost:8080" } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalBackendConfig { pub backend: LocalBackend, pub url: String, pub model: String, pub temperature: f32, pub max_tokens: usize, pub timeout_secs: u64, pub stream: bool }
impl Default for LocalBackendConfig { fn default() -> Self { Self { backend: LocalBackend::Ollama, url: LocalBackend::Ollama.default_url().into(), model: "llama3.2:3b".into(), temperature: 0.7, max_tokens: 2048, timeout_secs: 60, stream: false } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest { pub prompt: String, pub system: Option<String>, pub model: Option<String>, pub temperature: Option<f32>, pub max_tokens: Option<usize> }
impl Default for GenerateRequest { fn default() -> Self { Self { prompt: String::new(), system: None, model: None, temperature: None, max_tokens: None } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse { pub text: String, pub model: String, pub backend: LocalBackend, pub latency_ms: u64, pub tokens_generated: u64, pub tokens_per_second: f64 }

pub trait HttpClient: Send + Sync { fn post_json(&self, url: &str, body: &str) -> Result<String, String>; }

pub struct StubHttpClient { pub response: String, pub should_fail: bool }
impl HttpClient for StubHttpClient { fn post_json(&self, _url: &str, _body: &str) -> Result<String, String> { if self.should_fail { Err("fail".into()) } else { Ok(self.response.clone()) } } }

pub struct LocalInferenceAdapter { config: RwLock<LocalBackendConfig>, client: Arc<dyn HttpClient> }

impl LocalInferenceAdapter {
    #[must_use] pub fn new(config: LocalBackendConfig, client: Arc<dyn HttpClient>) -> Self { Self { config: RwLock::new(config), client } }
    pub fn set_config(&self, config: LocalBackendConfig) { *self.config.write() = config; }
    #[must_use] pub fn config(&self) -> LocalBackendConfig { self.config.read().clone() }
    pub fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse, LocalInferenceError> {
        let cfg = self.config.read().clone();
        let model = req.model.clone().unwrap_or_else(|| cfg.model.clone());
        let temp = req.temperature.unwrap_or(cfg.temperature);
        let mt = req.max_tokens.unwrap_or(cfg.max_tokens);
        let body = self.build_body(&cfg, &model, req, temp, mt);
        let url = self.build_url(&cfg);
        let start = Instant::now();
        let resp = self.client.post_json(&url, &body).map_err(|e| LocalInferenceError::BackendUnreachable { backend: format!("{:?}", cfg.backend), url: url.clone(), detail: e })?;
        let lat = start.elapsed(); if lat > Duration::from_secs(cfg.timeout_secs) { return Err(LocalInferenceError::Timeout(lat)); }
        let parsed = self.parse(&resp, &cfg.backend).map_err(LocalInferenceError::BackendError)?;
        let lat_ms = lat.as_millis() as u64; let tps = if lat_ms > 0 { parsed.tokens_generated as f64 / (lat_ms as f64 / 1000.0) } else { 0.0 };
        Ok(GenerateResponse { text: parsed.text, model, backend: cfg.backend, latency_ms: lat_ms, tokens_generated: parsed.tokens_generated, tokens_per_second: tps })
    }
    pub fn health_check(&self) -> Result<(), LocalInferenceError> { let cfg = self.config.read().clone(); let url = format!("{}/health", cfg.url); match self.client.post_json(&url, "{}") { Ok(_) => Ok(()), Err(e) => Err(LocalInferenceError::BackendUnreachable { backend: format!("{:?}", cfg.backend), url: cfg.url, detail: e }) } }
    fn build_body(&self, cfg: &LocalBackendConfig, model: &str, req: &GenerateRequest, temp: f32, mt: usize) -> String {
        match cfg.backend {
            LocalBackend::Ollama => { let p = match &req.system { Some(s) => format!("{s}\n\n{}", req.prompt), None => req.prompt.clone() }; serde_json::json!({"model": model, "prompt": p, "temperature": temp, "num_predict": mt, "stream": cfg.stream}).to_string() }
            _ => { let msgs = match &req.system { Some(s) => serde_json::json!([{"role":"system","content":s},{"role":"user","content":req.prompt}]), None => serde_json::json!([{"role":"user","content":req.prompt}]) }; serde_json::json!({"model": model, "messages": msgs, "temperature": temp, "max_tokens": mt, "stream": cfg.stream}).to_string() }
        }
    }
    fn build_url(&self, cfg: &LocalBackendConfig) -> String { match cfg.backend { LocalBackend::Ollama => format!("{}/api/generate", cfg.url), _ => format!("{}/v1/chat/completions", cfg.url) } }
    fn parse(&self, body: &str, backend: &LocalBackend) -> Result<ParsedResponse, String> {
        let j: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("invalid JSON: {e}"))?;
        match backend {
            LocalBackend::Ollama => { let t = j.get("response").and_then(|v| v.as_str()).ok_or("missing response")?.to_string(); let tok = j.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0); Ok(ParsedResponse { text: t, tokens_generated: tok }) }
            _ => { let t = j.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("message")).and_then(|m| m.get("content")).and_then(|v| v.as_str()).ok_or("missing content")?.to_string(); let tok = j.get("usage").and_then(|u| u.get("completion_tokens")).and_then(|v| v.as_u64()).unwrap_or(0); Ok(ParsedResponse { text: t, tokens_generated: tok }) }
        }
    }
}
struct ParsedResponse { text: String, tokens_generated: u64 }
pub type SharedLocalInferenceAdapter = Arc<LocalInferenceAdapter>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ollama() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: serde_json::json!({"response":"Hello!","eval_count":10}).to_string(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); let r = a.generate(&GenerateRequest { prompt: "hi".into(), ..Default::default() }).unwrap(); assert_eq!(r.text, "Hello!"); assert_eq!(r.tokens_generated, 10); }
    #[test] fn openai_compatible() { let cfg = LocalBackendConfig { backend: LocalBackend::LmStudio, ..Default::default() }; let stub = StubHttpClient { response: serde_json::json!({"choices":[{"message":{"content":"Hi!"}}],"usage":{"completion_tokens":15}}).to_string(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); let r = a.generate(&GenerateRequest { prompt: "hi".into(), ..Default::default() }).unwrap(); assert_eq!(r.text, "Hi!"); assert_eq!(r.tokens_generated, 15); }
    #[test] fn system_prompt() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: serde_json::json!({"response":"ok","eval_count":1}).to_string(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); let _ = a.generate(&GenerateRequest { prompt: "hi".into(), system: Some("be helpful".into()), ..Default::default() }).unwrap(); }
    #[test] fn backend_fail() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: String::new(), should_fail: true }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); assert!(matches!(a.generate(&GenerateRequest::default()), Err(LocalInferenceError::BackendUnreachable { .. }))); }
    #[test] fn health_ok() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: "ok".into(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); assert!(a.health_check().is_ok()); }
    #[test] fn health_fail() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: String::new(), should_fail: true }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); assert!(a.health_check().is_err()); }
    #[test] fn config_update() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: serde_json::json!({"response":"ok","eval_count":1}).to_string(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); assert_eq!(a.config().backend, LocalBackend::Ollama); a.set_config(LocalBackendConfig { backend: LocalBackend::LlamaCpp, ..Default::default() }); assert_eq!(a.config().backend, LocalBackend::LlamaCpp); }
    #[test] fn tps() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: serde_json::json!({"choices":[{"message":{"content":"hello"}}],"usage":{"completion_tokens":100}}).to_string(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); let r = a.generate(&GenerateRequest { prompt: "hi".into(), ..Default::default() }).unwrap(); assert_eq!(r.tokens_generated, 100); assert!(r.tokens_per_second >= 0.0); }
    #[test] fn overrides() { let cfg = LocalBackendConfig { model: "default".into(), ..Default::default() }; let stub = StubHttpClient { response: serde_json::json!({"choices":[{"message":{"content":"ok"}}],"usage":{"completion_tokens":1}}).to_string(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); let r = a.generate(&GenerateRequest { prompt: "hi".into(), model: Some("custom".into()), temperature: Some(0.9), max_tokens: Some(500), ..Default::default() }).unwrap(); assert_eq!(r.model, "custom"); }
    #[test] fn default_urls() { assert_eq!(LocalBackend::Ollama.default_url(), "http://localhost:11434"); assert_eq!(LocalBackend::LlamaCpp.default_url(), "http://localhost:8080"); assert_eq!(LocalBackend::LmStudio.default_url(), "http://localhost:1234"); assert_eq!(LocalBackend::Vllm.default_url(), "http://localhost:8000"); }
    #[test] fn invalid_json() { let cfg = LocalBackendConfig::default(); let stub = StubHttpClient { response: "not json".into(), should_fail: false }; let a = LocalInferenceAdapter::new(cfg, Arc::new(stub)); assert!(matches!(a.generate(&GenerateRequest::default()), Err(LocalInferenceError::BackendError(_)))); }
}
