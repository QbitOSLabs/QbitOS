//! Graph of Thoughts (Besta AAAI 2024, refined 2026).
#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GotError {
    #[error("thought `{0}` not found")] NotFound(String),
    #[error("invalid score: {0}")] InvalidScore(f64),
    #[error("generator error: {0}")] Generator(String),
    #[error("maximum graph size {0} exceeded")] MaxSizeExceeded(usize),
}

pub type ThoughtId = String;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransformKind { Initial, Decompose, Aggregate, Refine, Backtrack }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    pub id: ThoughtId, pub content: String, pub score: f64,
    pub transform: TransformKind, pub parents: Vec<ThoughtId>, pub children: Vec<ThoughtId>, pub depth: usize,
}

pub trait ThoughtGenerator: Send + Sync {
    fn generate(&self, prompt: &str) -> Result<String, GotError>;
    fn decompose(&self, thought: &str, n: usize) -> Result<Vec<String>, GotError> { (0..n).map(|i| self.generate(&format!("Part {}: {}", i+1, thought))).collect() }
    fn aggregate(&self, thoughts: &[String]) -> Result<String, GotError> { self.generate(&format!("Synthesize: {}", thoughts.join("; "))) }
    fn refine(&self, thought: &str) -> Result<String, GotError> { self.generate(&format!("Refine: {}", thought)) }
}

pub trait ThoughtScorer: Send + Sync { fn score(&self, thought: &str) -> Result<f64, GotError>; }

#[derive(Debug, Clone, Default)]
pub struct HeuristicScorer { pub keywords: Vec<String> }
impl ThoughtScorer for HeuristicScorer {
    fn score(&self, thought: &str) -> Result<f64, GotError> {
        if thought.is_empty() { return Ok(0.0); }
        let mut s = 0.3_f64;
        if thought.len() >= 100 && thought.len() <= 10000 { s += 0.3; } else if thought.len() >= 50 { s += 0.1; }
        let words: Vec<&str> = thought.split_whitespace().collect();
        if !words.is_empty() { let u: std::collections::HashSet<&str> = words.iter().copied().collect(); s += 0.4 * (u.len() as f64 / words.len() as f64); }
        for kw in &self.keywords { if thought.to_lowercase().contains(kw) { s += 0.1; } }
        Ok(s.min(1.0))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GotConfig { pub max_graph_size: usize, pub max_depth: usize, pub beam_width: usize, pub decompose_count: usize, pub final_aggregation: bool }
impl Default for GotConfig { fn default() -> Self { Self { max_graph_size: 64, max_depth: 4, beam_width: 3, decompose_count: 3, final_aggregation: true } } }

pub struct GraphOfThoughts { cfg: GotConfig, thoughts: RwLock<AHashMap<ThoughtId, Thought>>, next_id: RwLock<usize> }

impl GraphOfThoughts {
    #[must_use] pub fn new(cfg: GotConfig) -> Self { Self { cfg, thoughts: RwLock::new(AHashMap::new()), next_id: RwLock::new(0) } }

    pub fn solve(&self, prompt: &str, generator: &dyn ThoughtGenerator, scorer: &dyn ThoughtScorer) -> Result<GotSolution, GotError> {
        let init = generator.generate(prompt)?;
        let init_id = self.add(init, TransformKind::Initial, vec![], 0, scorer)?;
        let decomp = generator.decompose(&self.get(&init_id).unwrap().content, self.cfg.decompose_count)?;
        let mut current = Vec::new();
        for c in decomp {
            match self.add(c, TransformKind::Decompose, vec![init_id.clone()], 1, scorer) {
                Ok(id) => current.push(id),
                Err(GotError::MaxSizeExceeded(_)) => break,
                Err(e) => return Err(e),
            }
        }
        for depth in 2..=self.cfg.max_depth {
            if self.thoughts.read().len() >= self.cfg.max_graph_size { break; }
            let top = self.top_k(self.cfg.beam_width, depth - 1);
            if top.is_empty() { break; }
            let mut next = Vec::new();
            for pid in top {
                let p = self.thoughts.read().get(&pid).cloned();
                let Some(p) = p else { continue };
                let r = generator.refine(&p.content)?;
                match self.add(r, TransformKind::Refine, vec![pid], depth, scorer) {
                    Ok(id) => next.push(id),
                    Err(GotError::MaxSizeExceeded(_)) => break,
                    Err(e) => return Err(e),
                }
                if self.thoughts.read().len() >= self.cfg.max_graph_size { break; }
            }
            if next.is_empty() { break; }
            current = next;
        }
        // `current` holds the deepest layer of thoughts — currently unused after
        // the loop but retained for diagnostics and future aggregation heuristics.
        drop(current);
        let final_id = if self.cfg.final_aggregation {
            let top = self.top_k(self.cfg.beam_width, self.cfg.max_depth);
            let thoughts = self.thoughts.read();
            let contents: Vec<String> = top.iter().filter_map(|id| thoughts.get(id).map(|t| t.content.clone())).collect();
            drop(thoughts);
            if contents.is_empty() {
                init_id
            } else {
                let agg = generator.aggregate(&contents)?;
                match self.add(agg, TransformKind::Aggregate, top, self.cfg.max_depth + 1, scorer) {
                    Ok(id) => id,
                    Err(GotError::MaxSizeExceeded(_)) => init_id,
                    Err(e) => return Err(e),
                }
            }
        } else { init_id };
        let graph: Vec<Thought> = self.thoughts.read().values().cloned().collect();
        let best = self.thoughts.read().get(&final_id).cloned().unwrap_or(Thought { id: "empty".into(), content: String::new(), score: 0.0, transform: TransformKind::Initial, parents: vec![], children: vec![], depth: 0 });
        Ok(GotSolution { final_thought: best, graph })
    }

    fn add(&self, content: String, transform: TransformKind, parents: Vec<ThoughtId>, depth: usize, scorer: &dyn ThoughtScorer) -> Result<ThoughtId, GotError> {
        let mut nid = self.next_id.write(); *nid += 1; let id = format!("t{}", *nid); drop(nid);
        let score = scorer.score(&content)?; if !(0.0..=1.0).contains(&score) { return Err(GotError::InvalidScore(score)); }
        let mut thoughts = self.thoughts.write();
        for p in &parents { if let Some(n) = thoughts.get_mut(p) { n.children.push(id.clone()); } }
        if thoughts.len() >= self.cfg.max_graph_size { return Err(GotError::MaxSizeExceeded(self.cfg.max_graph_size)); }
        thoughts.insert(id.clone(), Thought { id: id.clone(), content, score, transform, parents, children: vec![], depth });
        Ok(id)
    }

    fn top_k(&self, k: usize, depth: usize) -> Vec<ThoughtId> {
        let thoughts = self.thoughts.read();
        let mut f: Vec<&Thought> = thoughts.values().filter(|t| t.depth == depth).collect();
        f.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        f.into_iter().take(k).map(|t| t.id.clone()).collect()
    }

    pub fn thoughts(&self) -> Vec<Thought> { self.thoughts.read().values().cloned().collect() }
    pub fn get(&self, id: &str) -> Option<Thought> { self.thoughts.read().get(id).cloned() }
    pub fn reset(&self) { self.thoughts.write().clear(); *self.next_id.write() = 0; }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GotSolution { pub final_thought: Thought, pub graph: Vec<Thought> }

#[cfg(test)]
mod tests {
    use super::*;
    struct Stub;
    impl ThoughtGenerator for Stub {
        fn generate(&self, p: &str) -> Result<String, GotError> { Ok(format!("Thought: {}", &p[..p.len().min(40)])) }
    }
    #[test] fn solve() { let g = GraphOfThoughts::new(GotConfig { max_depth: 2, beam_width: 2, decompose_count: 2, final_aggregation: true, max_graph_size: 32 }); let s = g.solve("test", &Stub, &HeuristicScorer::default()).unwrap(); assert!(!s.graph.is_empty()); }
    #[test] fn graph_has_nodes() { let g = GraphOfThoughts::new(GotConfig { max_depth: 1, beam_width: 1, decompose_count: 3, final_aggregation: false, max_graph_size: 32 }); let s = g.solve("test", &Stub, &HeuristicScorer::default()).unwrap(); assert!(s.graph.len() >= 4); }
    #[test] fn max_size() { let g = GraphOfThoughts::new(GotConfig { max_depth: 10, beam_width: 5, decompose_count: 5, final_aggregation: false, max_graph_size: 5 }); let s = g.solve("test", &Stub, &HeuristicScorer::default()).unwrap(); assert!(s.graph.len() <= 6); }
    #[test] fn aggregation() { let g = GraphOfThoughts::new(GotConfig { max_depth: 2, beam_width: 2, decompose_count: 2, final_aggregation: true, max_graph_size: 32 }); let s = g.solve("test", &Stub, &HeuristicScorer::default()).unwrap(); assert!(s.graph.iter().any(|t| t.transform == TransformKind::Aggregate)); }
    #[test] fn scorer() { let s = HeuristicScorer { keywords: vec!["answer".into()] }; assert!(s.score("the answer is 42").unwrap() > 0.3); }
    #[test] fn invalid_score() { struct Bad; impl ThoughtScorer for Bad { fn score(&self, _: &str) -> Result<f64, GotError> { Ok(2.0) } } let g = GraphOfThoughts::new(GotConfig::default()); assert!(matches!(g.solve("t", &Stub, &Bad), Err(GotError::InvalidScore(2.0)))); }
    #[test] fn reset() { let g = GraphOfThoughts::new(GotConfig::default()); let _ = g.solve("test", &Stub, &HeuristicScorer::default()); assert!(!g.thoughts().is_empty()); g.reset(); assert!(g.thoughts().is_empty()); }
}
