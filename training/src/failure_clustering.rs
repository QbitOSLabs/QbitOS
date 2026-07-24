#![allow(missing_docs)]
//! Failure clustering: groups failures by root cause for class-level fixes.
//! Failures are tokenised (alphanumeric words, length > 2) and grouped using
//! union-find with a Jaccard similarity threshold. Each cluster's root cause
//! is the most frequent significant token across its members; the cluster also
//! stores the union of all member tokens for retrieval matching.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FailureClusteringError { #[error("no failures recorded")] NoFailures }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    pub id: String, pub error_message: String, pub context: serde_json::Value,
    pub tool: String, pub timestamp: chrono::DateTime<chrono::Utc>, pub root_cause: Option<String>,
}

impl FailureRecord {
    #[must_use]
    pub fn new(error: impl Into<String>, context: serde_json::Value, tool: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(), error_message: error.into(), context, tool: tool.into(),
            timestamp: chrono::Utc::now(), root_cause: None,
        }
    }
    fn tokens(&self) -> HashSet<String> {
        self.error_message.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| s.len() > 2).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureCluster {
    pub cluster_id: String, pub root_cause: String, pub failures: Vec<String>,
    pub frequency: u32, pub suggested_fix: String,
    #[serde(default)] pub keywords: Vec<String>,
}

pub struct FailureClusterer {
    failures: Arc<RwLock<Vec<FailureRecord>>>,
    clusters: Arc<RwLock<Vec<FailureCluster>>>,
    similarity_threshold: f64,
}

impl Default for FailureClusterer { fn default() -> Self { Self::new() } }

impl FailureClusterer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            failures: Arc::new(RwLock::new(Vec::new())),
            clusters: Arc::new(RwLock::new(Vec::new())),
            similarity_threshold: 0.3,
        }
    }
    #[must_use] pub fn with_threshold(mut self, t: f64) -> Self { self.similarity_threshold = t.clamp(0.0, 1.0); self }

    pub fn record_failure(&self, error: String, context: serde_json::Value, tool: String) -> String {
        let r = FailureRecord::new(error, context, tool);
        let id = r.id.clone();
        self.failures.write().push(r);
        id
    }

    pub fn cluster(&self) {
        let failures = self.failures.read();
        let n = failures.len();
        if n == 0 { return; }
        let token_sets: Vec<HashSet<String>> = failures.iter().map(|f| f.tokens()).collect();
        let mut parent: Vec<usize> = (0..n).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                let inter = token_sets[i].intersection(&token_sets[j]).count() as f64;
                let union = token_sets[i].union(&token_sets[j]).count() as f64;
                let sim = if union > 0.0 { inter / union } else { 0.0 };
                if sim >= self.similarity_threshold { union_merge(&mut parent, i, j); }
            }
        }
        let mut groups: AHashMap<usize, Vec<usize>> = AHashMap::new();
        for i in 0..n { groups.entry(union_find(&mut parent, i)).or_default().push(i); }
        let mut clusters = Vec::with_capacity(groups.len());
        for (_, indices) in groups {
            let mut freq: AHashMap<String, u32> = AHashMap::new();
            let mut kw: HashSet<String> = HashSet::new();
            for &i in &indices {
                for t in &token_sets[i] {
                    *freq.entry(t.clone()).or_default() += 1;
                    kw.insert(t.clone());
                }
            }
            let root_cause = freq.iter().max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
                .map(|(t, _)| t.clone()).unwrap_or_else(|| "unknown".to_string());
            let tool = failures[indices[0]].tool.clone();
            clusters.push(FailureCluster {
                cluster_id: uuid::Uuid::new_v4().to_string(), root_cause: root_cause.clone(),
                failures: indices.iter().map(|&i| failures[i].id.clone()).collect(),
                frequency: indices.len() as u32,
                suggested_fix: suggest_fix(&root_cause, &tool),
                keywords: kw.into_iter().collect(),
            });
        }
        drop(failures);
        let mut failures = self.failures.write();
        for c in &clusters {
            for fid in &c.failures {
                if let Some(f) = failures.iter_mut().find(|fr| &fr.id == fid) {
                    f.root_cause = Some(c.root_cause.clone());
                }
            }
        }
        *self.clusters.write() = clusters;
    }

    #[must_use] pub fn get_clusters(&self) -> Vec<FailureCluster> { self.clusters.read().clone() }

    #[must_use]
    pub fn get_cluster_for_error(&self, error: &str) -> Option<FailureCluster> {
        let tokens: HashSet<String> = error.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| s.len() > 2).collect();
        self.clusters.read().iter()
            .filter(|c| !tokens.is_disjoint(&c.keywords.iter().cloned().collect::<HashSet<_>>()))
            .max_by_key(|c| c.frequency).cloned()
    }
    #[must_use] pub fn len(&self) -> usize { self.failures.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.failures.read().is_empty() }
}

fn union_find(p: &mut Vec<usize>, i: usize) -> usize {
    let mut c = i;
    while p[c] != c { p[c] = p[p[c]]; c = p[c]; }
    c
}
fn union_merge(p: &mut Vec<usize>, a: usize, b: usize) {
    let (ra, rb) = (union_find(p, a), union_find(p, b));
    if ra != rb { p[ra] = rb; }
}
fn suggest_fix(rc: &str, tool: &str) -> String {
    let r = rc.to_lowercase();
    if r.contains("timeout") { format!("Increase timeout or retry with backoff in {tool}") }
    else if r.contains("null") || r.contains("none") { format!("Add null-check validation before invoking {tool}") }
    else if r.contains("permission") || r.contains("denied") { format!("Verify permissions and credentials for {tool}") }
    else if r.contains("network") || r.contains("connection") { format!("Add retry/circuit-breaker for network calls in {tool}") }
    else if r.contains("parse") || r.contains("format") { format!("Improve input validation and schema for {tool}") }
    else { format!("Investigate root cause '{rc}' in {tool}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn record_failure_returns_id_and_increments_len() {
        let c = FailureClusterer::new();
        let id = c.record_failure("connection timeout".into(), json!({}), "http".into());
        assert!(!id.is_empty());
        assert_eq!(c.len(), 1);
    }
    #[test]
    fn cluster_groups_similar_errors() {
        let c = FailureClusterer::new();
        c.record_failure("network connection timeout error".into(), json!({}), "http".into());
        c.record_failure("network connection refused".into(), json!({}), "http".into());
        c.record_failure("cooking pasta recipe delicious".into(), json!({}), "chef".into());
        c.cluster();
        let clusters = c.get_clusters();
        assert!(clusters.len() >= 2);
        assert_eq!(clusters.iter().map(|c| c.frequency).sum::<u32>(), 3);
    }
    #[test]
    fn get_cluster_for_error_finds_match() {
        let c = FailureClusterer::new();
        c.record_failure("timeout error occurred while waiting".into(), json!({}), "tool".into());
        c.cluster();
        let found = c.get_cluster_for_error("timeout error again").unwrap();
        assert!(found.keywords.iter().any(|k| k == "timeout"));
    }
    #[test]
    fn suggest_fix_returns_meaningful_text() {
        let fix = suggest_fix("timeout", "http");
        assert!(fix.contains("timeout") && fix.contains("http"));
    }
    #[test]
    fn cluster_assigns_root_cause_to_records() {
        let c = FailureClusterer::new();
        let id = c.record_failure("connection timeout error".into(), json!({}), "http".into());
        c.cluster();
        let failures = c.failures.read();
        assert!(failures.iter().find(|f| f.id == id).unwrap().root_cause.is_some());
    }
}
