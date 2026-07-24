//! LLM Gateway Router (2026 TrueFoundry/Braintrust guides).
#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GatewayError { #[error("no providers")] NoProviders, #[error("all failed: {0}")] AllProvidersFailed(String), #[error("cost limit for {tenant}: {spent}/{limit}")] CostLimitExceeded { tenant: String, spent: f64, limit: f64 }, }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Provider { Openai, Anthropic, Google, Local, Cohere, Mistral }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig { pub provider: Provider, pub priority: u32, pub enabled: bool, pub cost_per_million_input_tokens: f64, pub cost_per_million_output_tokens: f64, pub avg_latency_ms: u64, pub max_tokens: usize }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest { pub prompt: String, pub model: String, pub max_tokens: usize, pub temperature: f32, pub tenant_id: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse { pub text: String, pub provider: Provider, pub input_tokens: u64, pub output_tokens: u64, pub cost_cents: f64, pub latency_ms: u64, pub cached: bool }

#[async_trait::async_trait]
pub trait LlmBackend: Send + Sync { async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, String>; fn name(&self) -> &str; }

pub struct StubLlmBackend { pub provider: Provider, pub should_fail: bool }
#[async_trait::async_trait]
impl LlmBackend for StubLlmBackend { async fn generate(&self, r: &LlmRequest) -> Result<LlmResponse, String> { if self.should_fail { return Err("fail".into()); } Ok(LlmResponse { text: format!("Response: {}", &r.prompt[..r.prompt.len().min(50)]), provider: self.provider, input_tokens: r.prompt.len() as u64 / 4, output_tokens: r.max_tokens as u64 / 4, cost_cents: 0.0, latency_ms: 100, cached: false }) } fn name(&self) -> &str { "stub" } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig { pub cache_enabled: bool, pub max_cache_entries: usize, pub failover_enabled: bool, pub max_failover_attempts: usize, pub tenant_cost_limit: f64, pub observability_enabled: bool }
impl Default for GatewayConfig { fn default() -> Self { Self { cache_enabled: true, max_cache_entries: 10_000, failover_enabled: true, max_failover_attempts: 3, tenant_cost_limit: 10_000.0, observability_enabled: true } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayLogEntry { pub timestamp_ms: u64, pub tenant_id: String, pub provider: Provider, pub model: String, pub input_tokens: u64, pub output_tokens: u64, pub cost_cents: f64, pub latency_ms: u64, pub cached: bool, pub success: bool }

pub struct LlmGateway { cfg: GatewayConfig, providers: RwLock<Vec<ProviderConfig>>, backends: RwLock<AHashMap<Provider, Arc<dyn LlmBackend>>>, cache: RwLock<AHashMap<u64, LlmResponse>>, spend: RwLock<AHashMap<String, f64>>, log: RwLock<Vec<GatewayLogEntry>> }

impl LlmGateway {
    #[must_use] pub fn new(cfg: GatewayConfig) -> Self { Self { cfg, providers: RwLock::new(Vec::new()), backends: RwLock::new(AHashMap::new()), cache: RwLock::new(AHashMap::new()), spend: RwLock::new(AHashMap::new()), log: RwLock::new(Vec::new()) } }
    pub fn register_provider(&self, cfg: ProviderConfig, backend: Arc<dyn LlmBackend>) { self.backends.write().insert(cfg.provider, backend); self.providers.write().push(cfg); self.providers.write().sort_by_key(|p| p.priority); }
    pub async fn generate(&self, req: &LlmRequest) -> Result<LlmResponse, GatewayError> {
        if self.cfg.tenant_cost_limit > 0.0 { let sp = *self.spend.read().get(&req.tenant_id).unwrap_or(&0.0); if sp >= self.cfg.tenant_cost_limit { return Err(GatewayError::CostLimitExceeded { tenant: req.tenant_id.clone(), spent: sp, limit: self.cfg.tenant_cost_limit }); } }
        let key = hash_req(req);
        if self.cfg.cache_enabled { if let Some(c) = self.cache.read().get(&key).cloned() { self.log_entry(&req.tenant_id, c.provider, &req.model, c.input_tokens, c.output_tokens, 0.0, 0, true, true); let mut c = c; c.cached = true; return Ok(c); } }
        let providers = self.providers.read().clone();
        if providers.is_empty() { return Err(GatewayError::NoProviders); }
        let mut errors = Vec::new();
        let max = if self.cfg.failover_enabled { self.cfg.max_failover_attempts.min(providers.len()) } else { 1 };
        for cfg in providers.iter().take(max) {
            if !cfg.enabled { continue; }
            let backend = self.backends.read().get(&cfg.provider).cloned();
            let Some(backend) = backend else { errors.push(format!("no backend {:?}", cfg.provider)); continue; };
            let start = Instant::now();
            match backend.generate(req).await {
                Ok(mut r) => { r.latency_ms = start.elapsed().as_millis() as u64; let cost = (r.input_tokens as f64 / 1e6) * cfg.cost_per_million_input_tokens + (r.output_tokens as f64 / 1e6) * cfg.cost_per_million_output_tokens; r.cost_cents = cost; *self.spend.write().entry(req.tenant_id.clone()).or_insert(0.0) += cost; if self.cfg.cache_enabled { let mut c = self.cache.write(); if c.len() >= self.cfg.max_cache_entries { let evict = (self.cfg.max_cache_entries / 10).max(1); let keys: Vec<u64> = c.keys().copied().take(evict).collect(); for k in keys { c.remove(&k); } } c.insert(key, r.clone()); } self.log_entry(&req.tenant_id, r.provider, &req.model, r.input_tokens, r.output_tokens, cost, r.latency_ms, false, true); return Ok(r); }
                Err(e) => { errors.push(format!("{:?}: {e}", cfg.provider)); self.log_entry(&req.tenant_id, cfg.provider, &req.model, 0, 0, 0.0, 0, false, false); }
            }
        }
        Err(GatewayError::AllProvidersFailed(errors.join("; ")))
    }
    fn log_entry(&self, tid: &str, p: Provider, m: &str, it: u64, ot: u64, cost: f64, lat: u64, cached: bool, success: bool) { if self.cfg.observability_enabled { self.log.write().push(GatewayLogEntry { timestamp_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0), tenant_id: tid.into(), provider: p, model: m.into(), input_tokens: it, output_tokens: ot, cost_cents: cost, latency_ms: lat, cached, success }); } }
    #[must_use] pub fn tenant_spend(&self, tid: &str) -> f64 { *self.spend.read().get(tid).unwrap_or(&0.0) }
    #[must_use] pub fn log(&self) -> Vec<GatewayLogEntry> { self.log.read().clone() }
    pub fn clear_cache(&self) { self.cache.write().clear(); }
    #[must_use] pub fn cache_size(&self) -> usize { self.cache.read().len() }
    #[must_use] pub fn provider_count(&self) -> usize { self.providers.read().len() }
}
pub type SharedLlmGateway = Arc<LlmGateway>;
fn hash_req(r: &LlmRequest) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for b in r.prompt.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } for b in r.model.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } h }

#[cfg(test)]
mod tests {
    use crate::ServiceConfig;
    use super::*;
    fn req(t:&str,p:&str)->LlmRequest{LlmRequest{prompt:p.into(),model:"m".into(),max_tokens:100,temperature:0.7,tenant_id:t.into()}}
    #[tokio::test] async fn routes() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); let r = g.generate(&req("t1","hello")).await.unwrap(); assert_eq!(r.provider, Provider::Openai); }
    #[tokio::test] async fn failover() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:true})); g.register_provider(ProviderConfig{provider:Provider::Anthropic,priority:1,enabled:true,cost_per_million_input_tokens:300.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:600,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Anthropic,should_fail:false})); let r = g.generate(&req("t1","hello")).await.unwrap(); assert_eq!(r.provider, Provider::Anthropic); }
    #[tokio::test] async fn cache() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); let r = req("t1","hello"); g.generate(&r).await.unwrap(); let r2 = g.generate(&r).await.unwrap(); assert!(r2.cached); }
    #[tokio::test] async fn spend() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); g.generate(&req("t1","hello world longer prompt")).await.unwrap(); assert!(g.tenant_spend("t1") > 0.0); }
    #[tokio::test] async fn cost_limit() { let g = LlmGateway::new(GatewayConfig{tenant_cost_limit:0.01,..Default::default()}); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); g.generate(&req("t1","hello world longer prompt")).await; assert!(matches!(g.generate(&req("t1","another")).await, Err(GatewayError::CostLimitExceeded { .. }))); }
    #[tokio::test] async fn log() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); g.generate(&req("t1","hello")).await.unwrap(); g.generate(&req("t1","hello")).await.unwrap(); assert_eq!(g.log().len(), 2); }
    #[tokio::test] async fn no_providers() { let g = LlmGateway::new(GatewayConfig::default()); assert!(matches!(g.generate(&req("t1","hi")).await, Err(GatewayError::NoProviders))); }
    #[tokio::test] async fn all_failed() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:true})); assert!(matches!(g.generate(&req("t1","hi")).await, Err(GatewayError::AllProvidersFailed(_)))); }
    #[tokio::test] async fn cache_clear() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:true,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); let r = req("t1","hi"); g.generate(&r).await.unwrap(); assert_eq!(g.cache_size(), 1); g.clear_cache(); assert_eq!(g.cache_size(), 0); }
    #[tokio::test] async fn skip_disabled() { let g = LlmGateway::new(GatewayConfig::default()); g.register_provider(ProviderConfig{provider:Provider::Openai,priority:0,enabled:false,cost_per_million_input_tokens:500.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:500,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Openai,should_fail:false})); g.register_provider(ProviderConfig{provider:Provider::Anthropic,priority:1,enabled:true,cost_per_million_input_tokens:300.0,cost_per_million_output_tokens:1500.0,avg_latency_ms:600,max_tokens:4096}, Arc::new(StubLlmBackend{provider:Provider::Anthropic,should_fail:false})); let r = g.generate(&req("t1","hi")).await.unwrap(); assert_eq!(r.provider, Provider::Anthropic); }
}
