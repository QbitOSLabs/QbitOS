//! Agent observability tracing (OpenTelemetry GenAI 2026).
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObservabilityError { #[error("span `{0}` not found")] SpanNotFound(String), #[error("span `{0}` already ended")] SpanAlreadyEnded(String) }

pub type SpanId = String; pub type TraceId = String;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus { InProgress, Ok, Error }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanKind { Internal, Client, Server, Producer, Consumer }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent { pub name: String, pub timestamp_ns: u64, pub attributes: AHashMap<String, serde_json::Value> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span { pub id: SpanId, pub trace_id: TraceId, pub parent_id: Option<SpanId>, pub name: String, pub kind: SpanKind, pub start_time_ns: u64, pub end_time_ns: Option<u64>, pub status: SpanStatus, pub attributes: AHashMap<String, serde_json::Value>, pub events: Vec<SpanEvent> }
impl Span { #[must_use] pub fn is_in_progress(&self) -> bool { self.end_time_ns.is_none() } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace { pub id: TraceId, pub spans: Vec<Span>, pub attributes: AHashMap<String, serde_json::Value>, pub start_time_ns: u64 }
impl Trace { #[must_use] pub fn llm_calls(&self) -> Vec<&Span> { self.spans.iter().filter(|s| s.name == "llm_call").collect() } #[must_use] pub fn total_tokens(&self) -> u64 { self.llm_calls().iter().filter_map(|s| s.attributes.get("gen_ai.usage.total_tokens")).filter_map(|v| v.as_u64()).sum() } }

pub struct Tracer { traces: RwLock<AHashMap<TraceId, Trace>>, enabled: RwLock<bool>, max_traces: usize }
impl Tracer {
    #[must_use] pub fn new(max: usize) -> Self { Self { traces: RwLock::new(AHashMap::new()), enabled: RwLock::new(true), max_traces: max } }
    pub fn start_trace(&self, attrs: AHashMap<String, serde_json::Value>) -> TraceId { if !*self.enabled.read() { return String::new(); } let id = format!("trace_{}", now_ns()); let t = Trace { id: id.clone(), spans: Vec::new(), attributes: attrs, start_time_ns: now_ns() }; let mut tr = self.traces.write(); if tr.len() >= self.max_traces { if let Some(oldest) = tr.iter().min_by_key(|(_, t)| t.start_time_ns).map(|(k, _)| k.clone()) { tr.remove(&oldest); } } tr.insert(id.clone(), t); id }
    pub fn start_span(&self, tid: &str, pid: Option<&str>, name: &str, kind: SpanKind, attrs: AHashMap<String, serde_json::Value>) -> Result<SpanId, ObservabilityError> { if !*self.enabled.read() { return Ok(String::new()); } let mut tr = self.traces.write(); let t = tr.get_mut(tid).ok_or(ObservabilityError::SpanNotFound(tid.into()))?; let sid = format!("span_{}", now_ns()); t.spans.push(Span { id: sid.clone(), trace_id: tid.into(), parent_id: pid.map(String::from), name: name.into(), kind, start_time_ns: now_ns(), end_time_ns: None, status: SpanStatus::InProgress, attributes: attrs, events: Vec::new() }); Ok(sid) }
    pub fn end_span(&self, tid: &str, sid: &str, status: SpanStatus) -> Result<(), ObservabilityError> { if !*self.enabled.read() { return Ok(()); } let mut tr = self.traces.write(); let t = tr.get_mut(tid).ok_or(ObservabilityError::SpanNotFound(tid.into()))?; let s = t.spans.iter_mut().find(|s| s.id == sid).ok_or(ObservabilityError::SpanNotFound(sid.into()))?; if s.end_time_ns.is_some() { return Err(ObservabilityError::SpanAlreadyEnded(sid.into())); } s.end_time_ns = Some(now_ns()); s.status = status; Ok(()) }
    pub fn set_attribute(&self, tid: &str, sid: &str, k: &str, v: serde_json::Value) -> Result<(), ObservabilityError> { if !*self.enabled.read() { return Ok(()); } let mut tr = self.traces.write(); let t = tr.get_mut(tid).ok_or(ObservabilityError::SpanNotFound(tid.into()))?; let s = t.spans.iter_mut().find(|s| s.id == sid).ok_or(ObservabilityError::SpanNotFound(sid.into()))?; s.attributes.insert(k.into(), v); Ok(()) }
    pub fn add_event(&self, tid: &str, sid: &str, name: &str, attrs: AHashMap<String, serde_json::Value>) -> Result<(), ObservabilityError> { if !*self.enabled.read() { return Ok(()); } let mut tr = self.traces.write(); let t = tr.get_mut(tid).ok_or(ObservabilityError::SpanNotFound(tid.into()))?; let s = t.spans.iter_mut().find(|s| s.id == sid).ok_or(ObservabilityError::SpanNotFound(sid.into()))?; s.events.push(SpanEvent { name: name.into(), timestamp_ns: now_ns(), attributes: attrs }); Ok(()) }
    #[must_use] pub fn get_trace(&self, tid: &str) -> Option<Trace> { self.traces.read().get(tid).cloned() }
    #[must_use] pub fn trace_count(&self) -> usize { self.traces.read().len() }
    pub fn set_enabled(&self, e: bool) { *self.enabled.write() = e; }
    pub fn clear(&self) { self.traces.write().clear(); }
    #[must_use] pub fn export_json(&self, tid: &str) -> Option<String> { self.get_trace(tid).and_then(|t| serde_json::to_string_pretty(&t).ok()) }
}

pub struct SpanGuard { tracer: Arc<Tracer>, trace_id: TraceId, span_id: SpanId, status: SpanStatus }
impl SpanGuard {
    pub fn start(tracer: Arc<Tracer>, tid: &str, pid: Option<&str>, name: &str, kind: SpanKind) -> Self { let sid = tracer.start_span(tid, pid, name, kind, AHashMap::new()).unwrap_or_default(); Self { tracer, trace_id: tid.into(), span_id: sid, status: SpanStatus::Ok } }
    pub fn set_status(&mut self, s: SpanStatus) { self.status = s; }
    pub fn set_attribute(&self, k: &str, v: serde_json::Value) { let _ = self.tracer.set_attribute(&self.trace_id, &self.span_id, k, v); }
}
impl Drop for SpanGuard { fn drop(&mut self) { let _ = self.tracer.end_span(&self.trace_id, &self.span_id, self.status); } }

fn now_ns() -> u64 { SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0) }
pub type SharedTracer = Arc<Tracer>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn start_trace() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); assert!(!tid.is_empty()); assert_eq!(t.trace_count(), 1); }
    #[test] fn start_end_span() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); let sid = t.start_span(&tid, None, "test", SpanKind::Internal, AHashMap::new()).unwrap(); assert!(t.get_trace(&tid).unwrap().spans[0].is_in_progress()); t.end_span(&tid, &sid, SpanStatus::Ok).unwrap(); assert!(!t.get_trace(&tid).unwrap().spans[0].is_in_progress()); }
    #[test] fn attributes() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); let sid = t.start_span(&tid, None, "llm_call", SpanKind::Client, AHashMap::new()).unwrap(); t.set_attribute(&tid, &sid, "gen_ai.system", "openai".into()).unwrap(); t.set_attribute(&tid, &sid, "gen_ai.usage.total_tokens", 1500u64.into()).unwrap(); assert_eq!(t.get_trace(&tid).unwrap().spans[0].attributes.get("gen_ai.system").unwrap(), "openai"); }
    #[test] fn llm_calls() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); for tok in [1000u64, 2000] { let sid = t.start_span(&tid, None, "llm_call", SpanKind::Client, AHashMap::new()).unwrap(); t.set_attribute(&tid, &sid, "gen_ai.usage.total_tokens", tok.into()).unwrap(); t.end_span(&tid, &sid, SpanStatus::Ok).unwrap(); } let tr = t.get_trace(&tid).unwrap(); assert_eq!(tr.llm_calls().len(), 2); assert_eq!(tr.total_tokens(), 3000); }
    #[test] fn span_guard() { let t = Arc::new(Tracer::new(100)); let tid = t.start_trace(AHashMap::new()); { let _g = SpanGuard::start(Arc::clone(&t), &tid, None, "test", SpanKind::Internal); } let tr = t.get_trace(&tid).unwrap(); assert!(!tr.spans[0].is_in_progress()); assert_eq!(tr.spans[0].status, SpanStatus::Ok); }
    #[test] fn guard_error() { let t = Arc::new(Tracer::new(100)); let tid = t.start_trace(AHashMap::new()); { let mut g = SpanGuard::start(Arc::clone(&t), &tid, None, "test", SpanKind::Internal); g.set_status(SpanStatus::Error); } assert_eq!(t.get_trace(&tid).unwrap().spans[0].status, SpanStatus::Error); }
    #[test] fn lru_evict() { let t = Tracer::new(2); t.start_trace(AHashMap::new()); std::thread::sleep(std::time::Duration::from_millis(1)); t.start_trace(AHashMap::new()); std::thread::sleep(std::time::Duration::from_millis(1)); t.start_trace(AHashMap::new()); assert_eq!(t.trace_count(), 2); }
    #[test] fn disable() { let t = Tracer::new(100); t.set_enabled(false); let tid = t.start_trace(AHashMap::new()); assert!(tid.is_empty()); }
    #[test] fn export_json() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); t.start_span(&tid, None, "test", SpanKind::Internal, AHashMap::new()).unwrap(); let j = t.export_json(&tid).unwrap(); assert!(j.contains("test")); }
    #[test] fn double_end() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); let sid = t.start_span(&tid, None, "test", SpanKind::Internal, AHashMap::new()).unwrap(); t.end_span(&tid, &sid, SpanStatus::Ok).unwrap(); assert!(t.end_span(&tid, &sid, SpanStatus::Ok).is_err()); }
    #[test] fn events() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); let sid = t.start_span(&tid, None, "test", SpanKind::Internal, AHashMap::new()).unwrap(); t.add_event(&tid, &sid, "tool.start", AHashMap::new()).unwrap(); t.add_event(&tid, &sid, "tool.end", AHashMap::new()).unwrap(); assert_eq!(t.get_trace(&tid).unwrap().spans[0].events.len(), 2); }
    #[test] fn clear() { let t = Tracer::new(100); t.start_trace(AHashMap::new()); assert_eq!(t.trace_count(), 1); t.clear(); assert_eq!(t.trace_count(), 0); }
    #[test] fn count_by_status() { let t = Tracer::new(100); let tid = t.start_trace(AHashMap::new()); for s in [SpanStatus::Ok, SpanStatus::Error, SpanStatus::Ok] { let sid = t.start_span(&tid, None, "test", SpanKind::Internal, AHashMap::new()).unwrap(); t.end_span(&tid, &sid, s).unwrap(); } let tr = t.get_trace(&tid).unwrap(); assert_eq!(tr.spans.iter().filter(|s| s.status == SpanStatus::Ok).count(), 2); assert_eq!(tr.spans.iter().filter(|s| s.status == SpanStatus::Error).count(), 1); }
}
