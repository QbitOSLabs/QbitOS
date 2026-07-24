#![allow(missing_docs)]

//! Web search tool: returns simulated search results for offline operation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::tool_vm::{Tool, ToolKind};

/// Errors emitted by the web search tool.
#[derive(Debug, Error)]
pub enum WebSearchError {
    /// Missing `query` parameter.
    #[error("missing 'query' parameter")]
    MissingQuery,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title.
    pub title: String,
    /// Result URL.
    pub url: String,
    /// Snippet.
    pub snippet: String,
    /// Relevance score in [0, 1].
    pub score: f32,
}

/// Web search tool that operates against an in-memory mock corpus.
pub struct WebSearchTool { corpus: Vec<SearchResult> }

impl Default for WebSearchTool { fn default() -> Self { Self::new() } }

impl WebSearchTool {
    /// Create a new tool with the built-in mock corpus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            corpus: vec![
                sr("Rust Programming Language", "https://www.rust-lang.org/",
                   "A language empowering everyone to build reliable and efficient software. Rust offers memory safety without a garbage collector.", 1.0),
                sr("The Rust Book", "https://doc.rust-lang.org/book/",
                   "The official guide to Rust, covering ownership, borrowing, lifetimes, traits, async, and the standard library.", 0.95),
                sr("Cargo — The Rust Package Manager", "https://doc.rust-lang.org/cargo/",
                   "Cargo is the Rust package manager. It downloads dependencies, builds your project, and runs tests.", 0.9),
                sr("Wikipedia: Artificial Intelligence", "https://en.wikipedia.org/wiki/Artificial_intelligence",
                   "Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence of humans.", 0.92),
                sr("Wikipedia: Large Language Model", "https://en.wikipedia.org/wiki/Large_language_model",
                   "A large language model is a computational model notable for achieving general-purpose language generation.", 0.9),
                sr("QbitOS Project", "https://github.com/qbitos/qbitos",
                   "QbitOS is a Rust-native multimodal cognitive AI operating system featuring a cognitive kernel, skill genome, and tool VM.", 0.97),
                sr("GPT-4 Technical Report", "https://arxiv.org/abs/2303.08774",
                   "GPT-4 is a large multimodal model that accepts image and text inputs and emits text outputs.", 0.85),
                sr("Vector Database Comparison", "https://benchmark.vectorview.ai/",
                   "A comparison of pgvector, Qdrant, Weaviate, Milvus, and Pinecone for similarity search at scale.", 0.78),
                sr("Transformers Library — Hugging Face", "https://huggingface.co/docs/transformers",
                   "State-of-the-art machine learning for PyTorch, TensorFlow, and JAX. Provides thousands of pretrained models.", 0.83),
                sr("Embedded Rust", "https://www.rust-embedded.org/",
                   "Resources for embedded development in Rust, including no_std crates, hardware abstraction layers, and tooling.", 0.8),
            ],
        }
    }

    /// Build a tool with a custom corpus.
    #[must_use]
    pub fn with_corpus(corpus: Vec<SearchResult>) -> Self { Self { corpus } }

    /// Search the corpus for `query`, returning at most `max_results` hits.
    #[must_use]
    pub fn search(&self, query: &str, max_results: usize) -> Vec<SearchResult> {
        let q_toks = tokenize(query);
        if q_toks.is_empty() {
            let mut out = self.corpus.clone();
            out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(max_results);
            return out;
        }
        let mut scored: Vec<(f32, SearchResult)> = self.corpus.iter().map(|r| {
            let toks = tokenize(&format!("{} {}", r.title, r.snippet));
            let sim = jaccard(&q_toks, &toks);
            (0.6 * sim + 0.4 * r.score, r.clone())
        }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(max_results).map(|(s, mut r)| { r.score = s; r }).collect()
    }
}

fn sr(title: &str, url: &str, snippet: &str, score: f32) -> SearchResult {
    SearchResult { title: title.into(), url: url.into(), snippet: snippet.into(), score }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str { "web_search" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "search" }
    fn scope(&self, _: &Value) -> String { "internet".into() }
    fn schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object", "required": ["query"],
            "properties": {
                "query": { "type": "string" },
                "max_results": { "type": "integer", "minimum": 1, "maximum": 50, "default": 5 }
            }
        }))
    }
    async fn execute(&self, params: &Value) -> Result<Value, String> {
        let query = params.get("query").and_then(|v| v.as_str())
            .ok_or(WebSearchError::MissingQuery.to_string())?;
        let max = params.get("max_results").and_then(|v| v.as_u64())
            .map(|n| n.min(50) as usize).unwrap_or(5);
        let results = self.search(query, max);
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        Ok(json!({ "query": query, "results": results, "total": results.len(), "timestamp": ts }))
    }
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1).map(String::from).collect()
}

fn jaccard(query: &[String], doc: &[String]) -> f32 {
    if query.is_empty() || doc.is_empty() { return 0.0; }
    let q: std::collections::HashSet<&String> = query.iter().collect();
    let d: std::collections::HashSet<&String> = doc.iter().collect();
    let inter = q.intersection(&d).count() as f32;
    let union = q.union(&d).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn searches_rust_corpus() {
        let tool = WebSearchTool::new();
        let out = tool.execute(&json!({"query": "rust programming language", "max_results": 3})).await.unwrap();
        let results = out.get("results").unwrap().as_array().unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        assert!(results[0].get("title").is_some());
    }

    #[test]
    fn ranks_relevant_results_first() {
        let tool = WebSearchTool::new();
        let r = tool.search("rust cargo package manager", 3);
        assert!(!r.is_empty());
        assert!(r[0].title.to_lowercase().contains("cargo") || r[0].title.to_lowercase().contains("rust"));
    }

    #[test]
    fn empty_query_returns_top_by_base_score() {
        let r = WebSearchTool::new().search("", 2);
        assert_eq!(r.len(), 2);
        assert!(r[0].score >= r[1].score);
    }

    #[tokio::test]
    async fn missing_query_errors() {
        let tool = WebSearchTool::new();
        assert!(tool.execute(&json!({"max_results": 3})).await.is_err());
    }
}
