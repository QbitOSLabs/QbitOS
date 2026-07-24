#![allow(missing_docs)]
//! Reflection memory: every failure creates a reflection stored for later
//! retrieval. Reflections can be retrieved by keyword overlap (Jaccard
//! similarity) with a new question, or by tag. Retrieval counts are tracked so
//! frequently-used lessons can be promoted to procedural knowledge.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReflectionError { #[error("reflection not found: {0}")] NotFound(String) }

/// A reflection on a failed attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reflection {
    pub id: String,
    pub question: String,
    pub failed_answer: String,
    pub failure_reason: String,
    pub correct_approach: String,
    pub lesson: String,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub retrieval_count: u32,
}

impl Reflection {
    #[must_use]
    pub fn new(
        question: impl Into<String>, failed_answer: impl Into<String>,
        failure_reason: impl Into<String>, correct_approach: impl Into<String>,
        lesson: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(), question: question.into(),
            failed_answer: failed_answer.into(), failure_reason: failure_reason.into(),
            correct_approach: correct_approach.into(), lesson: lesson.into(),
            tags: Vec::new(), created_at: chrono::Utc::now(), retrieval_count: 0,
        }
    }
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self { self.tags = tags; self }
    fn question_tokens(&self) -> HashSet<String> {
        self.question.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty()).collect()
    }
}

/// A store of reflections with keyword and tag-based retrieval.
pub struct ReflectionStore {
    reflections: Arc<RwLock<Vec<Reflection>>>,
    by_tag: Arc<RwLock<AHashMap<String, Vec<String>>>>,
}

impl Default for ReflectionStore { fn default() -> Self { Self::new() } }

impl ReflectionStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            reflections: Arc::new(RwLock::new(Vec::new())),
            by_tag: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    /// Add a reflection. Returns the assigned id.
    pub fn add(&self, reflection: Reflection) -> String {
        let id = reflection.id.clone();
        for tag in &reflection.tags {
            self.by_tag.write().entry(tag.clone()).or_default().push(id.clone());
        }
        self.reflections.write().push(reflection);
        id
    }

    /// Retrieve the top-k reflections whose question overlaps with `question`
    /// (Jaccard similarity on whitespace tokens).
    #[must_use]
    pub fn retrieve_similar(&self, question: &str, k: usize) -> Vec<Reflection> {
        let query: HashSet<String> = question.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty()).collect();
        let reflections = self.reflections.read();
        let mut scored: Vec<(f64, &Reflection)> = reflections.iter().map(|r| {
            let rt = r.question_tokens();
            if rt.is_empty() || query.is_empty() { return (0.0, r); }
            let inter = rt.intersection(&query).count() as f64;
            let union = rt.union(&query).count() as f64;
            (inter / union, r)
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().filter(|(s, _)| *s > 0.0).take(k).map(|(_, r)| r.clone()).collect()
    }

    /// Retrieve all reflections tagged with `tag`.
    #[must_use]
    pub fn retrieve_by_tag(&self, tag: &str) -> Vec<Reflection> {
        let ids = self.by_tag.read().get(tag).cloned().unwrap_or_default();
        if ids.is_empty() { return Vec::new(); }
        self.reflections.read().iter().filter(|r| ids.contains(&r.id)).cloned().collect()
    }

    /// Increment the retrieval counter for a reflection by id.
    pub fn record_retrieval(&self, id: &str) {
        let mut reflections = self.reflections.write();
        for r in reflections.iter_mut() {
            if r.id == id { r.retrieval_count += 1; break; }
        }
    }
    #[must_use] pub fn len(&self) -> usize { self.reflections.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.reflections.read().is_empty() }
    #[must_use] pub fn tag_count(&self) -> usize { self.by_tag.read().len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn add_returns_id_and_increments_len() {
        let s = ReflectionStore::new();
        let id = s.add(Reflection::new("What is 2+2?", "5", "Arithmetic error", "Compute 2+2=4", "Always verify arithmetic"));
        assert!(!id.is_empty());
        assert_eq!(s.len(), 1);
    }
    #[test]
    fn retrieve_similar_finds_overlap() {
        let s = ReflectionStore::new();
        s.add(Reflection::new("How to handle null pointer in rust safely", "panic", "No safety check", "Use Option", "Check before unwrap"));
        let results = s.retrieve_similar("null pointer rust safety", 5);
        assert!(!results.is_empty());
        assert!(results[0].question.contains("null"));
    }
    #[test]
    fn retrieve_by_tag_filters_correctly() {
        let s = ReflectionStore::new();
        s.add(Reflection::new("q1", "a", "r", "c", "l").with_tags(vec!["math".into()]));
        s.add(Reflection::new("q2", "a", "r", "c", "l").with_tags(vec!["code".into()]));
        let math = s.retrieve_by_tag("math");
        assert_eq!(math.len(), 1);
        assert_eq!(math[0].question, "q1");
        assert!(s.retrieve_by_tag("missing").is_empty());
    }
    #[test]
    fn record_retrieval_increments_count() {
        let s = ReflectionStore::new();
        let id = s.add(Reflection::new("q", "a", "r", "c", "l"));
        s.record_retrieval(&id);
        s.record_retrieval(&id);
        assert_eq!(s.reflections.read()[0].retrieval_count, 2);
    }
    #[test]
    fn retrieve_similar_returns_empty_with_no_overlap() {
        let s = ReflectionStore::new();
        s.add(Reflection::new("debugging rust code", "a", "r", "c", "l"));
        assert!(s.retrieve_similar("cooking pasta recipe", 5).is_empty());
    }
}
