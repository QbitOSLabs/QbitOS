#![allow(missing_docs)]

//! Hallucination guard: fact-checks generated text against a knowledge base
//! and scores the probability of hallucination in [0, 1].

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// Errors emitted by the hallucination guard.
#[derive(Debug, Error)]
pub enum HallucinationError {
    /// Empty input.
    #[error("empty input")]
    Empty,
}

/// Configuration for the guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardConfig {
    /// Jaccard threshold above which a claim is considered supported.
    pub support_threshold: f32,
    /// Minimum source relevance to consider.
    pub min_relevance: f32,
    /// Confidence floor returned for supported claims.
    pub min_confidence: f32,
}

impl Default for GuardConfig {
    fn default() -> Self { Self { support_threshold: 0.18, min_relevance: 0.05, min_confidence: 0.5 } }
}

/// A source used to verify claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Source content.
    pub content: String,
    /// Pre-computed relevance of this source to the query (0..1).
    pub relevance_score: f32,
    /// Optional URL.
    #[serde(default)]
    pub url: Option<String>,
}

impl Source {
    /// Create a new source.
    #[must_use]
    pub fn new(content: impl Into<String>, relevance_score: f32) -> Self {
        Self { content: content.into(), relevance_score, url: None }
    }
}

/// Result of checking a single claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimCheck {
    /// Whether the claim is supported by at least one source.
    pub supported: bool,
    /// Confidence in the verdict in [0, 1].
    pub confidence: f32,
    /// Index of the best-matching source, if any.
    pub matching_source: Option<usize>,
    /// Best similarity score observed.
    pub best_similarity: f32,
}

/// Result of checking an entire response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCheck {
    /// Overall hallucination probability in [0, 1] (lower is better).
    pub overall_score: f32,
    /// Number of claims checked.
    pub claims_checked: usize,
    /// Unsupported claim texts.
    pub unsupported_claims: Vec<String>,
}

/// The hallucination guard.
#[derive(Debug, Clone)]
pub struct HallucinationGuard { config: GuardConfig }

impl Default for HallucinationGuard { fn default() -> Self { Self::new() } }

impl HallucinationGuard {
    /// Create a new guard with default config.
    #[must_use]
    pub fn new() -> Self { Self { config: GuardConfig::default() } }

    /// Create a new guard with explicit config.
    #[must_use]
    pub fn with_config(config: GuardConfig) -> Self { Self { config } }

    /// Check a single claim against a set of sources.
    pub fn check_claim(&self, claim: &str, sources: &[Source]) -> ClaimCheck {
        let toks = tokenize(claim);
        if toks.is_empty() {
            return ClaimCheck { supported: false, confidence: 0.0, matching_source: None, best_similarity: 0.0 };
        }
        let mut best_sim = 0.0_f32;
        let mut best_idx: Option<usize> = None;
        for (i, src) in sources.iter().enumerate() {
            if src.relevance_score < self.config.min_relevance { continue; }
            let src_toks = tokenize(&src.content);
            let sim = weighted_sim(&toks, &src_toks, src.relevance_score);
            if sim > best_sim { best_sim = sim; best_idx = Some(i); }
        }
        let supported = best_sim >= self.config.support_threshold;
        let confidence = if supported {
            let rel = best_idx.and_then(|i| sources.get(i)).map(|s| s.relevance_score).unwrap_or(0.0);
            (self.config.min_confidence + 0.4 * best_sim + 0.1 * rel).min(1.0)
        } else {
            (1.0 - best_sim).clamp(self.config.min_confidence, 1.0)
        };
        ClaimCheck { supported, confidence, matching_source: best_idx, best_similarity: best_sim }
    }

    /// Check an entire response, splitting it into claims (sentences).
    pub fn check_response(&self, response: &str, sources: &[Source]) -> ResponseCheck {
        let claims = split_claims(response);
        if claims.is_empty() { return ResponseCheck { overall_score: 0.0, claims_checked: 0, unsupported_claims: Vec::new() }; }
        let mut unsupported = Vec::new();
        let mut supported = 0usize;
        let mut total_conf = 0.0_f32;
        for claim in &claims {
            let c = self.check_claim(claim, sources);
            if c.supported { supported += 1; } else { unsupported.push(claim.clone()); }
            total_conf += c.confidence;
        }
        let n = claims.len() as f32;
        let overall = (1.0 - supported as f32 / n) * (1.0 - total_conf / n).clamp(0.0, 1.0);
        ResponseCheck { overall_score: overall.clamp(0.0, 1.0), claims_checked: claims.len(), unsupported_claims: unsupported }
    }
}

fn tokenize(s: &str) -> HashSet<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2).filter(|t| !is_stopword(t)).map(String::from).collect()
}

fn is_stopword(t: &str) -> bool {
    const SW: &[&str] = &[
        "the","and","but","for","are","was","were","have","has","had","this","that",
        "with","from","they","their","them","there","then","than","these","those",
        "will","would","could","should","into","not","you","your","she","his","her",
        "its","our","any","all"];
    SW.contains(&t)
}

fn weighted_sim(claim: &HashSet<String>, src: &HashSet<String>, relevance: f32) -> f32 {
    let j = jaccard(claim, src);
    (j * (0.3 + 0.7 * relevance)).sqrt()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() { return 0.0; }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn split_claims(s: &str) -> Vec<String> {
    s.split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(str::trim).filter(|c| c.len() > 8).map(String::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_claim_matches_source() {
        let g = HallucinationGuard::new();
        let sources = vec![
            Source::new("paris is the capital of france", 1.0),
            Source::new("the eiffel tower is located in paris", 0.9),
        ];
        let c = g.check_claim("paris is the capital of france", &sources);
        assert!(c.supported && c.confidence > 0.5 && c.matching_source.is_some());
    }

    #[test]
    fn unsupported_claim_detected() {
        let g = HallucinationGuard::new();
        let sources = vec![Source::new("the sky is blue", 1.0)];
        assert!(!g.check_claim("mars has two moons named phobos and deimos", &sources).supported);
    }

    #[test]
    fn response_and_empty_claims() {
        let g = HallucinationGuard::new();
        let sources = vec![
            Source::new("rust is a systems programming language", 1.0),
            Source::new("rust emphasizes memory safety without garbage collection", 0.9),
        ];
        let resp = "rust is a systems programming language. rust emphasizes memory safety without garbage collection. the moon is made of cheese.";
        let r = g.check_response(resp, &sources);
        assert!(r.claims_checked == 3 && r.unsupported_claims.len() == 1 && r.overall_score < 0.5);
        assert!(!g.check_claim("", &[]).supported);
    }
}
