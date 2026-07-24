//! Low-resource translation adapter (WMT 2026 Indic + LoResMT EACL 2026).
#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TranslationError { #[error("unsupported source: {0}")] UnsupportedSource(String), #[error("unsupported target: {0}")] UnsupportedTarget(String), #[error("backend error: {0}")] Backend(String), #[error("no examples for {src}->{tgt}")] NoExamples { src: String, tgt: String } }

pub type LangCode = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPair { pub source: String, pub target: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest { pub text: String, pub source_lang: LangCode, pub target_lang: LangCode, pub num_few_shot: usize, pub use_transliteration: bool, pub verify_back_translation: bool }
impl Default for TranslationRequest { fn default() -> Self { Self { text: String::new(), source_lang: "en".into(), target_lang: "es".into(), num_few_shot: 3, use_transliteration: true, verify_back_translation: false } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse { pub translation: String, pub source_lang: LangCode, pub target_lang: LangCode, pub num_examples_used: usize, pub transliterated: bool, pub back_translation_score: Option<f64>, pub backend: String }

#[async_trait::async_trait]
pub trait TranslationBackend: Send + Sync { async fn translate(&self, prompt: &str) -> Result<String, TranslationError>; fn name(&self) -> &str; }

pub struct StubTranslationBackend;
#[async_trait::async_trait]
impl TranslationBackend for StubTranslationBackend { async fn translate(&self, p: &str) -> Result<String, TranslationError> { Ok(format!("[tr: {}]", &p[..p.len().min(50)])) } fn name(&self) -> &str { "stub" } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageFamily { pub family: String, pub subfamily: String, pub script: String, pub related_high_resource: Option<LangCode> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig { pub max_few_shot: usize, pub use_family_routing: bool, pub default_back_translation_verify: bool, pub min_back_translation_score: f64 }
impl Default for TranslationConfig { fn default() -> Self { Self { max_few_shot: 5, use_family_routing: true, default_back_translation_verify: false, min_back_translation_score: 0.5 } } }

pub struct LowResourceTranslator { cfg: TranslationConfig, backend: Arc<dyn TranslationBackend>, examples: RwLock<AHashMap<(LangCode, LangCode), Vec<ParallelPair>>>, families: RwLock<AHashMap<LangCode, LanguageFamily>> }

impl LowResourceTranslator {
    #[must_use] pub fn new(cfg: TranslationConfig, backend: Arc<dyn TranslationBackend>) -> Self { Self { cfg, backend, examples: RwLock::new(AHashMap::new()), families: RwLock::new(AHashMap::new()) } }
    pub fn add_examples(&self, sl: &str, tl: &str, pairs: Vec<ParallelPair>) { self.examples.write().entry((sl.into(), tl.into())).or_default().extend(pairs); }
    pub fn register_language_family(&self, l: &str, f: LanguageFamily) { self.families.write().insert(l.into(), f); }
    pub async fn translate(&self, req: &TranslationRequest) -> Result<TranslationResponse, TranslationError> {
        let examples = self.get_examples(&req.source_lang, &req.target_lang, req.num_few_shot); let num = examples.len();
        let prompt = self.build_prompt(req, &examples); let translation = self.backend.translate(&prompt).await?;
        let bt_score = if req.verify_back_translation { Some(self.verify_bt(req, &translation).await) } else { None };
        Ok(TranslationResponse { translation, source_lang: req.source_lang.clone(), target_lang: req.target_lang.clone(), num_examples_used: num, transliterated: req.use_transliteration, back_translation_score: bt_score, backend: self.backend.name().to_string() })
    }
    fn get_examples(&self, sl: &str, tl: &str, n: usize) -> Vec<ParallelPair> { if n == 0 { return Vec::new(); } let ex = self.examples.read(); let k = (sl.to_string(), tl.to_string()); if let Some(p) = ex.get(&k) { p.iter().take(n).cloned().collect() } else { let rk = (tl.to_string(), sl.to_string()); if let Some(p) = ex.get(&rk) { p.iter().take(n).map(|x| ParallelPair { source: x.target.clone(), target: x.source.clone() }).collect() } else { Vec::new() } } }
    fn build_prompt(&self, req: &TranslationRequest, ex: &[ParallelPair]) -> String { let mut p = format!("Translate from {} to {}.\n", req.source_lang, req.target_lang); if !ex.is_empty() { p.push_str("Examples:\n"); for e in ex { p.push_str(&format!("{}: {}\n{}: {}\n\n", req.source_lang, e.source, req.target_lang, e.target)); } } p.push_str(&format!("Now translate:\n{}: {}\n{}: ", req.source_lang, req.text, req.target_lang)); p }
    async fn verify_bt(&self, req: &TranslationRequest, tr: &str) -> f64 { let br = TranslationRequest { text: tr.into(), source_lang: req.target_lang.clone(), target_lang: req.source_lang.clone(), num_few_shot: 0, use_transliteration: false, verify_back_translation: false }; let bp = self.build_prompt(&br, &[]); let bt = self.backend.translate(&bp).await.unwrap_or_default(); similarity(&req.text, &bt) }
    #[must_use] pub fn example_count(&self, sl: &str, tl: &str) -> usize { self.examples.read().get(&(sl.into(), tl.into())).map(Vec::len).unwrap_or(0) }
    #[must_use] pub fn language_count(&self) -> usize { self.families.read().len() }
    /// Returns the translator's configuration.
    #[must_use] pub fn config(&self) -> &TranslationConfig { &self.cfg }
}
pub type SharedTranslator = Arc<LowResourceTranslator>;
fn similarity(a: &str, b: &str) -> f64 { let aw: std::collections::HashSet<&str> = a.split_whitespace().collect(); let bw: std::collections::HashSet<&str> = b.split_whitespace().collect(); if aw.is_empty() && bw.is_empty() { return 1.0; } let i = aw.intersection(&bw).count(); let u = aw.union(&bw).count(); if u == 0 { 0.0 } else { i as f64 / u as f64 } }

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test] async fn zero_shot() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); let r = t.translate(&TranslationRequest { text: "hello".into(), source_lang: "en".into(), target_lang: "es".into(), num_few_shot: 0, ..Default::default() }).await.unwrap(); assert!(!r.translation.is_empty()); assert_eq!(r.num_examples_used, 0); }
    #[tokio::test] async fn few_shot() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); t.add_examples("en", "es", vec![ParallelPair{source:"hello".into(),target:"hola".into()}, ParallelPair{source:"bye".into(),target:"adios".into()}]); let r = t.translate(&TranslationRequest { text: "hello world".into(), source_lang: "en".into(), target_lang: "es".into(), num_few_shot: 2, ..Default::default() }).await.unwrap(); assert_eq!(r.num_examples_used, 2); }
    #[tokio::test] async fn reverse_examples() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); t.add_examples("es", "en", vec![ParallelPair{source:"hola".into(),target:"hello".into()}]); let r = t.translate(&TranslationRequest { text: "hello".into(), source_lang: "en".into(), target_lang: "es".into(), num_few_shot: 1, ..Default::default() }).await.unwrap(); assert_eq!(r.num_examples_used, 1); }
    #[test] fn example_count() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); t.add_examples("en", "fr", vec![ParallelPair{source:"a".into(),target:"b".into()}, ParallelPair{source:"c".into(),target:"d".into()}]); assert_eq!(t.example_count("en", "fr"), 2); assert_eq!(t.example_count("fr", "en"), 0); }
    #[test] fn family_count() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); t.register_language_family("hi", LanguageFamily{family:"IE".into(),subfamily:"IA".into(),script:"Dev".into(),related_high_resource:Some("en".into())}); assert_eq!(t.language_count(), 1); }
    #[tokio::test] async fn bt_verify() { let t = LowResourceTranslator::new(TranslationConfig { min_back_translation_score: 0.0, ..Default::default() }, Arc::new(StubTranslationBackend)); let r = t.translate(&TranslationRequest { text: "hello world".into(), source_lang: "en".into(), target_lang: "es".into(), num_few_shot: 0, verify_back_translation: true, ..Default::default() }).await.unwrap(); assert!(r.back_translation_score.is_some()); }
    #[test] fn sim_identical() { assert!((similarity("hello world", "hello world") - 1.0).abs() < 1e-5); }
    #[test] fn sim_disjoint() { assert!(similarity("alpha", "beta") < 0.01); }
    #[test] fn sim_partial() { let s = similarity("hello world foo", "hello world bar"); assert!(s > 0.4 && s < 0.8); }
    #[tokio::test] async fn empty_text() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); let r = t.translate(&TranslationRequest { text: String::new(), ..Default::default() }).await.unwrap(); assert!(!r.translation.is_empty()); }
    #[tokio::test] async fn no_examples() { let t = LowResourceTranslator::new(TranslationConfig::default(), Arc::new(StubTranslationBackend)); let r = t.translate(&TranslationRequest { text: "test".into(), source_lang: "en".into(), target_lang: "ja".into(), num_few_shot: 3, ..Default::default() }).await.unwrap(); assert_eq!(r.num_examples_used, 0); }
}
