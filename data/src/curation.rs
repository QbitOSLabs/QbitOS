//! Online reweighting data curation (ICLR 2026).
#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CurationError { #[error("empty pipeline")] EmptyPipeline, #[error("stage `{stage}` failed: {error}")] StageFailed { stage: String, error: String } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample { pub id: u64, pub content: String, pub quality: f64, pub importance: f64, pub last_loss: f64, pub presentation_count: u64, pub metadata: AHashMap<String, String> }
impl DataSample { #[must_use] pub fn new(id: u64, content: String) -> Self { Self { id, content, quality: 0.5, importance: 1.0, last_loss: 0.0, presentation_count: 0, metadata: AHashMap::new() } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationConfig { pub dedup_exact: bool, pub dedup_minhash: bool, pub minhash_threshold: f64, pub min_quality: f64, pub max_content_length: usize, pub importance_ema_decay: f64, pub filter_toxic: bool, pub toxic_keywords: Vec<String>, pub min_content_length: usize }
impl Default for CurationConfig { fn default() -> Self { Self { dedup_exact: true, dedup_minhash: true, minhash_threshold: 0.9, min_quality: 0.3, max_content_length: 100_000, importance_ema_decay: 0.95, filter_toxic: true, toxic_keywords: vec!["hate".into(), "violence".into(), "abuse".into()], min_content_length: 10 } } }

pub struct CurationPipeline { cfg: CurationConfig, seen: RwLock<AHashMap<u64, ()>>, seen_mh: RwLock<Vec<(u64, u64)>>, processed: RwLock<u64>, dedup_rej: RwLock<u64>, quality_rej: RwLock<u64>, toxic_rej: RwLock<u64>, length_rej: RwLock<u64> }

impl CurationPipeline {
    #[must_use] pub fn new(cfg: CurationConfig) -> Self { Self { cfg, seen: RwLock::new(AHashMap::new()), seen_mh: RwLock::new(Vec::new()), processed: RwLock::new(0), dedup_rej: RwLock::new(0), quality_rej: RwLock::new(0), toxic_rej: RwLock::new(0), length_rej: RwLock::new(0) } }
    pub fn process(&self, mut s: DataSample) -> Result<Option<DataSample>, CurationError> { *self.processed.write() += 1;
        if s.content.len() < self.cfg.min_content_length || s.content.len() > self.cfg.max_content_length { *self.length_rej.write() += 1; return Ok(None); }
        let h = hash_content(&s.content);
        if self.cfg.dedup_exact { let mut sn = self.seen.write(); if sn.contains_key(&h) { *self.dedup_rej.write() += 1; return Ok(None); } sn.insert(h, ()); }
        if self.cfg.dedup_minhash { let mh = minhash(&s.content); let mut sn = self.seen_mh.write(); for &(eh, _) in sn.iter() { if mh == eh { *self.dedup_rej.write() += 1; return Ok(None); } } sn.push((mh, s.id)); }
        s.quality = self.score_quality(&s.content);
        if s.quality < self.cfg.min_quality { *self.quality_rej.write() += 1; return Ok(None); }
        if self.cfg.filter_toxic && self.is_toxic(&s.content) { *self.toxic_rej.write() += 1; return Ok(None); }
        s.importance = s.quality; Ok(Some(s))
    }
    pub fn update_importance(&self, s: &mut DataSample, loss: f64) { let a = 1.0 - self.cfg.importance_ema_decay; s.last_loss = self.cfg.importance_ema_decay * s.last_loss + a * loss; s.importance = s.quality * (1.0 + s.last_loss); }
    fn score_quality(&self, c: &str) -> f64 { let mut s = 0.0; let l = c.len(); if (100..=10_000).contains(&l) { s += 0.3; } else if l >= 50 { s += 0.1; } let w: Vec<&str> = c.split_whitespace().collect(); if !w.is_empty() { let u: std::collections::HashSet<&str> = w.iter().copied().collect(); s += 0.4 * (u.len() as f64 / w.len() as f64); } if c.matches('.').count() >= 1 { s += 0.2; } let dr = c.chars().filter(|c| c.is_ascii_digit()).count() as f64 / c.len().max(1) as f64; if dr < 0.3 { s += 0.1; } s.min(1.0) }
    fn is_toxic(&self, c: &str) -> bool { let l = c.to_lowercase(); self.cfg.toxic_keywords.iter().any(|k| l.contains(k)) }
    #[must_use] pub fn processed_count(&self) -> u64 { *self.processed.read() }
    #[must_use] pub fn dedup_rejections(&self) -> u64 { *self.dedup_rej.read() }
    #[must_use] pub fn quality_rejections(&self) -> u64 { *self.quality_rej.read() }
    #[must_use] pub fn toxic_rejections(&self) -> u64 { *self.toxic_rej.read() }
    #[must_use] pub fn length_rejections(&self) -> u64 { *self.length_rej.read() }
    pub fn reset(&self) { self.seen.write().clear(); self.seen_mh.write().clear(); *self.processed.write() = 0; *self.dedup_rej.write() = 0; *self.quality_rej.write() = 0; *self.toxic_rej.write() = 0; *self.length_rej.write() = 0; }
}

pub struct ImportanceSampler { samples: RwLock<Vec<DataSample>>, total_weight: RwLock<f64> }
impl ImportanceSampler { #[must_use] pub fn new() -> Self { Self { samples: RwLock::new(Vec::new()), total_weight: RwLock::new(0.0) } } pub fn add(&self, s: DataSample) { *self.total_weight.write() += s.importance; self.samples.write().push(s); } #[must_use] pub fn draw(&self) -> Option<DataSample> { let mut sm = self.samples.write(); if sm.is_empty() { return None; } let mut bi = 0; let mut bw = sm[0].importance; for (i, s) in sm.iter().enumerate().skip(1) { if s.importance > bw { bw = s.importance; bi = i; } } let s = sm.remove(bi); *self.total_weight.write() -= s.importance; Some(s) } #[must_use] pub fn len(&self) -> usize { self.samples.read().len() } #[must_use] pub fn is_empty(&self) -> bool { self.samples.read().is_empty() } }
impl Default for ImportanceSampler { fn default() -> Self { Self::new() } }

fn hash_content(c: &str) -> u64 { let mut h: u64 = 0xcbf29ce484222325; for b in c.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } h }
fn minhash(c: &str) -> u64 { let mut mh: u64 = u64::MAX; for t in c.split_whitespace() { let h = hash_content(t); if h < mh { mh = h; } } if mh == u64::MAX { 0 } else { mh } }
pub type SharedCurationPipeline = Arc<CurationPipeline>;

#[cfg(test)]
mod tests {
    use super::*;
    fn mk(id: u64, c: &str) -> DataSample { DataSample::new(id, c.into()) }
    #[test] fn accepts() { let p = CurationPipeline::new(CurationConfig::default()); let r = p.process(mk(1, "The quick brown fox jumps over the lazy dog. Well-formed text.")).unwrap(); assert!(r.is_some()); }
    #[test] fn rejects_short() { let p = CurationPipeline::new(CurationConfig::default()); assert!(p.process(mk(1, "short")).unwrap().is_none()); }
    #[test] fn rejects_exact_dup() { let p = CurationPipeline::new(CurationConfig::default()); let c = "A unique sample with enough words to pass length checks and decent quality."; p.process(mk(1, c)).unwrap(); assert!(p.process(mk(2, c)).unwrap().is_none()); }
    #[test] fn rejects_toxic() { let p = CurationPipeline::new(CurationConfig::default()); assert!(p.process(mk(1, "This content contains hate speech and should be filtered")).unwrap().is_none()); }
    #[test] fn importance() { let p = CurationPipeline::new(CurationConfig::default()); let mut s = mk(1, "Good sample with diverse vocabulary."); s.quality = 0.8; s.importance = 0.8; p.update_importance(&mut s, 2.0); assert!(s.importance > 0.8); }
    #[test] fn sampler() { let s = ImportanceSampler::new(); let mut s1 = mk(1, "a"); s1.importance = 0.1; let mut s2 = mk(2, "b"); s2.importance = 0.9; s.add(s1); s.add(s2); let d = s.draw().unwrap(); assert_eq!(d.id, 2); }
    #[test] fn reset() { let p = CurationPipeline::new(CurationConfig::default()); p.process(mk(1, "short")).unwrap(); assert_eq!(p.processed_count(), 1); p.reset(); assert_eq!(p.processed_count(), 0); }
    #[test] fn quality_diverse() { let p = CurationPipeline::new(CurationConfig { min_quality: 0.0, ..Default::default() }); let d = p.process(mk(1, "The quick brown fox jumps over the lazy dog. Many unique words here.")).unwrap().unwrap(); let r = p.process(mk(2, "the the the the the the the the the the the the the the the.")).unwrap().unwrap(); assert!(d.quality > r.quality); }
    #[test] fn processed_count() { let p = CurationPipeline::new(CurationConfig::default()); for i in 0..10 { p.process(mk(i, "A unique sample with diverse vocabulary and structure.")).unwrap(); } assert_eq!(p.processed_count(), 10); }
    #[test] fn empty_sampler() { let s = ImportanceSampler::new(); assert!(s.draw().is_none()); }
    #[test] fn empty_batch() { let p = CurationPipeline::new(CurationConfig::default()); assert_eq!(p.processed_count(), 0); }
}
