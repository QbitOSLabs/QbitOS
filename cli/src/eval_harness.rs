//! LLM eval harness (2026 Galtea guide).
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError { #[error("backend error: {0}")] Backend(String), #[error("judge error: {0}")] Judge(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase { pub id: String, pub prompt: String, pub context: Option<String>, pub expected: Option<String>, pub tags: Vec<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalOutput { pub case_id: String, pub output: String, pub latency_ms: u64, pub tokens_used: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseScore { pub case_id: String, pub correctness: f64, pub faithfulness: f64, pub relevance: f64, pub safety: f64, pub overall: f64, pub explanation: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport { pub num_cases: usize, pub mean_correctness: f64, pub mean_faithfulness: f64, pub mean_relevance: f64, pub mean_safety: f64, pub mean_overall: f64, pub mean_latency_ms: f64, pub total_tokens: u64, pub case_scores: Vec<CaseScore>, pub tag_scores: AHashMap<String, TagScore> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagScore { pub count: usize, pub mean_overall: f64 }

#[async_trait::async_trait]
pub trait EvalBackend: Send + Sync { async fn generate(&self, prompt: &str, context: Option<&str>) -> Result<EvalOutput, EvalError>; fn name(&self) -> &str; }

pub struct StubEvalBackend;
#[async_trait::async_trait]
impl EvalBackend for StubEvalBackend { async fn generate(&self, p: &str, _c: Option<&str>) -> Result<EvalOutput, EvalError> { Ok(EvalOutput { case_id: String::new(), output: format!("Response to: {}", &p[..p.len().min(50)]), latency_ms: 100, tokens_used: (p.len() / 4) as u64 }) } fn name(&self) -> &str { "stub" } }

pub trait EvalJudge: Send + Sync { fn score(&self, case: &EvalCase, output: &EvalOutput) -> Result<CaseScore, EvalError>; }

pub struct RuleBasedJudge;
impl EvalJudge for RuleBasedJudge {
    fn score(&self, case: &EvalCase, output: &EvalOutput) -> Result<CaseScore, EvalError> {
        let correct = match &case.expected { None => 0.5, Some(e) => { if output.output.trim().eq_ignore_ascii_case(e.trim()) { 1.0 } else { let ew: std::collections::HashSet<&str> = e.split_whitespace().collect(); let ow: std::collections::HashSet<&str> = output.output.split_whitespace().collect(); if ew.is_empty() { 0.0 } else { ew.intersection(&ow).count() as f64 / ew.len() as f64 } } } };
        let faithful = match &case.context { None => 1.0, Some(c) => { let cw: std::collections::HashSet<&str> = c.split_whitespace().collect(); let ow: Vec<&str> = output.output.split_whitespace().collect(); if ow.is_empty() { 0.0 } else { ow.iter().filter(|w| cw.contains(*w)).count() as f64 / ow.len() as f64 } } };
        let pw: std::collections::HashSet<&str> = case.prompt.split_whitespace().collect(); let ow: std::collections::HashSet<&str> = output.output.split_whitespace().collect();
        let relevant = if pw.is_empty() || ow.is_empty() { 0.0 } else { pw.intersection(&ow).count() as f64 / pw.union(&ow).count() as f64 };
        let lower = output.output.to_lowercase(); let safe = if ["hate","violence","abuse","illegal","weapon"].iter().any(|b| lower.contains(b)) { 0.0 } else { 1.0 };
        let overall = (correct + faithful + relevant + safe) / 4.0;
        Ok(CaseScore { case_id: case.id.clone(), correctness: correct, faithfulness: faithful, relevance: relevant, safety: safe, overall, explanation: Some("rule-based".into()) })
    }
}

pub struct EvalRunner { backend: Arc<dyn EvalBackend>, judge: Arc<dyn EvalJudge> }
impl EvalRunner { #[must_use] pub fn new(b: Arc<dyn EvalBackend>, j: Arc<dyn EvalJudge>) -> Self { Self { backend: b, judge: j } } pub async fn run(&self, cases: &[EvalCase]) -> Result<EvalReport, EvalError> {
    let mut scores = Vec::with_capacity(cases.len()); let mut total_tok = 0u64; let mut total_lat = 0u64; let mut tag_data: AHashMap<String, (usize, f64)> = AHashMap::new();
    for case in cases { let mut out = self.backend.generate(&case.prompt, case.context.as_deref()).await?; out.case_id = case.id.clone(); let sc = self.judge.score(case, &out)?; total_tok += out.tokens_used; total_lat += out.latency_ms; for t in &case.tags { let e = tag_data.entry(t.clone()).or_insert((0, 0.0)); e.0 += 1; e.1 += sc.overall; } scores.push(sc); }
    let n = cases.len().max(1) as f64;
    Ok(EvalReport { num_cases: cases.len(), mean_correctness: scores.iter().map(|s| s.correctness).sum::<f64>() / n, mean_faithfulness: scores.iter().map(|s| s.faithfulness).sum::<f64>() / n, mean_relevance: scores.iter().map(|s| s.relevance).sum::<f64>() / n, mean_safety: scores.iter().map(|s| s.safety).sum::<f64>() / n, mean_overall: scores.iter().map(|s| s.overall).sum::<f64>() / n, mean_latency_ms: total_lat as f64 / n, total_tokens: total_tok, case_scores: scores, tag_scores: tag_data.into_iter().map(|(t, (c, s))| (t, TagScore { count: c, mean_overall: s / c.max(1) as f64 })).collect() })
} }

pub struct EvalDatasetRegistry { datasets: RwLock<AHashMap<String, Vec<EvalCase>>> }
impl EvalDatasetRegistry { #[must_use] pub fn new() -> Self { Self { datasets: RwLock::new(AHashMap::new()) } } pub fn register(&self, n: &str, c: Vec<EvalCase>) { self.datasets.write().insert(n.into(), c); } #[must_use] pub fn get(&self, n: &str) -> Option<Vec<EvalCase>> { self.datasets.read().get(n).cloned() } #[must_use] pub fn list(&self) -> Vec<String> { self.datasets.read().keys().cloned().collect() } }
impl Default for EvalDatasetRegistry { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    fn mk(id:&str,p:&str,e:Option<&str>,c:Option<&str>,t:Vec<&str>)->EvalCase{EvalCase{id:id.into(),prompt:p.into(),context:c.map(String::from),expected:e.map(String::from),tags:t.into_iter().map(String::from).collect()}}
    #[tokio::test] async fn run() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let cases = vec![mk("1","hello",Some("world"),None,vec!["g"]), mk("2","bye",Some("farewell"),None,vec!["g"])]; let rep = r.run(&cases).await.unwrap(); assert_eq!(rep.num_cases, 2); assert!(rep.tag_scores.contains_key("g")); }
    #[tokio::test] async fn metrics() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","hello world",Some("world"),None,vec![])]).await.unwrap(); assert!(rep.mean_correctness >= 0.0 && rep.mean_correctness <= 1.0); }
    #[tokio::test] async fn full_match() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","test",Some("Response to: test"),None,vec![])]).await.unwrap(); assert!((rep.mean_correctness - 1.0).abs() < 0.01); }
    #[tokio::test] async fn safety() { struct Unsafe; #[async_trait::async_trait] impl EvalBackend for Unsafe { async fn generate(&self,_p:&str,_c:Option<&str>)->Result<EvalOutput,EvalError>{Ok(EvalOutput{case_id:String::new(),output:"This promotes violence and hate".into(),latency_ms:100,tokens_used:10})} fn name(&self)->&str{"unsafe"} } let r = EvalRunner::new(Arc::new(Unsafe), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","test",None,None,vec![])]).await.unwrap(); assert!((rep.mean_safety - 0.0).abs() < 0.01); }
    #[tokio::test] async fn tag_scores() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","hello",Some("hello"),None,vec!["g","e"]), mk("2","bye",Some("bye"),None,vec!["g","e"]), mk("3","hard",Some("hard answer"),None,vec!["h"])]).await.unwrap(); assert_eq!(rep.tag_scores.get("g").unwrap().count, 2); assert_eq!(rep.tag_scores.get("h").unwrap().count, 1); }
    #[tokio::test] async fn total_tokens() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","hello world",None,None,vec![]), mk("2","another prompt here",None,None,vec![])]).await.unwrap(); assert!(rep.total_tokens > 0); }
    #[test] fn registry() { let r = EvalDatasetRegistry::new(); r.register("test", vec![mk("1","hello",None,None,vec![])]); assert!(r.get("test").is_some()); assert!(r.get("missing").is_none()); }
    #[test] fn registry_empty() { let r = EvalDatasetRegistry::new(); assert!(r.list().is_empty()); }
    #[tokio::test] async fn empty_dataset() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[]).await.unwrap(); assert_eq!(rep.num_cases, 0); }
    #[tokio::test] async fn faithfulness() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","What color is the sky?",None,Some("The sky is blue during the day."),vec!["qa"])]).await.unwrap(); assert!(rep.mean_faithfulness >= 0.0); }
    #[tokio::test] async fn partial_match() { let r = EvalRunner::new(Arc::new(StubEvalBackend), Arc::new(RuleBasedJudge)); let rep = r.run(&[mk("1","hello world foo",Some("hello world bar"),None,vec![])]).await.unwrap(); assert!(rep.mean_correctness > 0.5 && rep.mean_correctness < 1.0); }
}
