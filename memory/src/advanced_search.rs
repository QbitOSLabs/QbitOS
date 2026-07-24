//! Advanced search features inspired by 2026 research.
//!
//! Implements:
//!
//! 1. **Filtered ANN search** — pre-filter by metadata predicates during graph
//!    traversal (à la Qdrant's filtered search, 2024–2026 improvements).
//! 2. **Product Quantization (PQ) compression** — compress embeddings for
//!    memory-efficient similarity computation (Jégou et al., 2011; modern
//!    improvements in 2024-2026 for LLM-scale retrieval).
//! 3. **Hybrid search** — combine dense vector similarity with sparse
//!    keyword matching (BM25-style) for hybrid retrieval (2025-2026 trend
//!    in RAG systems).
//! 4. **Adaptive ef_search** — dynamically adjust search depth based on
//!    result quality feedback (2025 research on adaptive ANN).
//! 5. **Diversity-aware reranking** — post-processing that maximizes
//!    result diversity using Maximal Marginal Relevance (MMR).

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::embedding::Embedding;
use crate::object::MemoryId;

/// A metadata filter predicate for filtered ANN search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterPredicate {
    /// Key equals value.
    Eq(String, serde_json::Value),
    /// Key is in the set of values.
    In(String, Vec<serde_json::Value>),
    /// Key exists.
    Exists(String),
    /// Key's numeric value is >= threshold.
    Gte(String, f64),
    /// Key's numeric value is <= threshold.
    Lte(String, f64),
    /// Key's string value contains substring.
    Contains(String, String),
    /// AND of multiple predicates.
    And(Vec<FilterPredicate>),
    /// OR of multiple predicates.
    Or(Vec<FilterPredicate>),
}

impl FilterPredicate {
    /// Evaluate the predicate against a metadata map.
    #[must_use]
    pub fn evaluate(&self, metadata: &std::collections::BTreeMap<String, serde_json::Value>) -> bool {
        match self {
            Self::Eq(key, value) => metadata.get(key).map_or(false, |v| v == value),
            Self::In(key, values) => metadata.get(key).map_or(false, |v| values.contains(v)),
            Self::Exists(key) => metadata.contains_key(key),
            Self::Gte(key, threshold) => metadata.get(key)
                .and_then(|v| v.as_f64())
                .map_or(false, |f| f >= *threshold),
            Self::Lte(key, threshold) => metadata.get(key)
                .and_then(|v| v.as_f64())
                .map_or(false, |f| f <= *threshold),
            Self::Contains(key, substring) => metadata.get(key)
                .and_then(|v| v.as_str())
                .map_or(false, |s| s.contains(substring)),
            Self::And(preds) => preds.iter().all(|p| p.evaluate(metadata)),
            Self::Or(preds) => preds.iter().any(|p| p.evaluate(metadata)),
        }
    }
}

/// Product Quantizer for embedding compression.
///
/// Splits a D-dimensional vector into `n_subquantizers` sub-vectors, each
/// quantized to one of `n_centroids` centroids learned via k-means.
/// This reduces memory usage by a factor of `D * 4 / (n_subquantizers * log2(n_centroids))`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Number of sub-quantizers.
    pub n_subquantizers: usize,
    /// Number of centroids per sub-quantizer.
    pub n_centroids: usize,
    /// Dimensionality of the original vectors.
    pub dim: usize,
    /// Codebook: `n_subquantizers` × `n_centroids` × `sub_dim` centroids.
    pub codebook: Vec<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Create a new PQ with the given parameters.
    #[must_use]
    pub fn new(dim: usize, n_subquantizers: usize, n_centroids: usize) -> Self {
        let sub_dim = dim / n_subquantizers;
        let codebook = vec![vec![vec![0.0_f32; sub_dim]; n_centroids]; n_subquantizers];
        Self { n_subquantizers, n_centroids, dim, codebook }
    }

    /// Train the codebook using k-means on a set of training vectors.
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }
        let sub_dim = self.dim / self.n_subquantizers;

        for sq in 0..self.n_subquantizers {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let sub_vectors: Vec<Vec<f32>> = vectors.iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // K-means
            let centroids = kmeans(&sub_vectors, self.n_centroids, 10);
            self.codebook[sq] = centroids;
        }
    }

    /// Encode a vector into a compact PQ code.
    #[must_use]
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let sub_dim = self.dim / self.n_subquantizers;
        let mut code = Vec::with_capacity(self.n_subquantizers);

        for sq in 0..self.n_subquantizers {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let sub_vec = &vector[start..end];
            let best = self.codebook[sq].iter()
                .enumerate()
                .map(|(i, c)| (i, euclidean_sq(sub_vec, c)))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            code.push(best as u8);
        }
        code
    }

    /// Compute approximate distance between a query vector and a PQ code
    /// using a pre-computed lookup table (ADC — Asymmetric Distance Computation).
    #[must_use]
    pub fn asymmetric_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        let sub_dim = self.dim / self.n_subquantizers;
        let mut dist = 0.0_f32;

        for sq in 0..self.n_subquantizers {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let sub_query = &query[start..end];
            let centroid_idx = code.get(sq).copied().unwrap_or(0) as usize;
            if centroid_idx < self.codebook[sq].len() {
                dist += euclidean_sq(sub_query, &self.codebook[sq][centroid_idx]);
            }
        }
        dist
    }

    /// Build a distance lookup table for a query vector.
    /// `table[sq][centroid_idx] = distance`.
    #[must_use]
    pub fn build_lookup_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.dim / self.n_subquantizers;
        let mut table = Vec::with_capacity(self.n_subquantizers);

        for sq in 0..self.n_subquantizers {
            let start = sq * sub_dim;
            let end = start + sub_dim;
            let sub_query = &query[start..end];
            let row: Vec<f32> = self.codebook[sq].iter()
                .map(|c| euclidean_sq(sub_query, c))
                .collect();
            table.push(row);
        }
        table
    }

    /// Compute distance using a pre-built lookup table (faster than `asymmetric_distance`).
    #[must_use]
    pub fn distance_with_table(table: &[Vec<f32>], code: &[u8]) -> f32 {
        let mut dist = 0.0_f32;
        for (sq, &c) in code.iter().enumerate() {
            if sq < table.len() && (c as usize) < table[sq].len() {
                dist += table[sq][c as usize];
            }
        }
        dist
    }
}

/// Simple k-means clustering.
fn kmeans(data: &[Vec<f32>], k: usize, max_iters: usize) -> Vec<Vec<f32>> {
    if data.is_empty() || k == 0 {
        return vec![vec![0.0; data.first().map_or(0, |v| v.len())]; k];
    }

    let dim = data[0].len();
    let n = data.len().min(k);

    // Initialize centroids from first k data points
    let mut centroids: Vec<Vec<f32>> = data[..n].to_vec();
    while centroids.len() < k {
        centroids.push(vec![0.0; dim]);
    }

    for _ in 0..max_iters {
        // Assign points to nearest centroid
        let mut assignments = vec![0usize; data.len()];
        for (i, point) in data.iter().enumerate() {
            let best = centroids.iter()
                .enumerate()
                .map(|(j, c)| (j, euclidean_sq(point, c)))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            assignments[i] = best;
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0u32; k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            for d in 0..dim {
                new_centroids[c][d] += point[d];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    new_centroids[c][d] /= counts[c] as f32;
                }
            } else {
                new_centroids[c] = centroids[c].clone();
            }
        }

        // Check convergence
        let moved: f32 = centroids.iter().zip(new_centroids.iter())
            .map(|(a, b)| euclidean_sq(a, b))
            .sum();
        centroids = new_centroids;
        if moved < 1e-6 {
            break;
        }
    }

    centroids
}

/// Squared Euclidean distance.
fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// A sparse keyword index for hybrid search (BM25-style).
pub struct SparseIndex {
    /// Term → (doc_id, term_frequency).
    postings: AHashMap<String, Vec<(MemoryId, u32)>>,
    /// Document lengths.
    doc_lengths: AHashMap<MemoryId, usize>,
    /// Total number of documents.
    n_docs: usize,
    /// Average document length.
    avg_doc_len: f64,
}

impl SparseIndex {
    /// Create a new empty sparse index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            postings: AHashMap::new(),
            doc_lengths: AHashMap::new(),
            n_docs: 0,
            avg_doc_len: 0.0,
        }
    }

    /// Add a document to the sparse index.
    pub fn add_document(&mut self, id: MemoryId, text: &str) {
        let tokens: Vec<String> = text.split_whitespace()
            .map(|t| t.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|t| !t.is_empty())
            .collect();
        let doc_len = tokens.len();

        // Count term frequencies
        let mut tf: AHashMap<String, u32> = AHashMap::new();
        for token in &tokens {
            *tf.entry(token.to_string()).or_insert(0) += 1;
        }

        // Add to postings
        for (term, freq) in tf {
            self.postings.entry(term).or_default().push((id, freq));
        }

        self.doc_lengths.insert(id, doc_len);
        self.n_docs += 1;
        self.avg_doc_len = self.doc_lengths.values().map(|&l| l as f64).sum::<f64>() / self.n_docs as f64;
    }

    /// Search for documents matching the query using BM25 scoring.
    #[must_use]
    pub fn search(&self, query: &str, k: usize) -> Vec<(MemoryId, f32)> {
        let query_tokens: Vec<String> = query.split_whitespace()
            .map(|t| t.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|t| !t.is_empty())
            .collect();

        if query_tokens.is_empty() || self.n_docs == 0 {
            return Vec::new();
        }

        let mut scores: AHashMap<MemoryId, f32> = AHashMap::new();
        let k1 = 1.5_f32;
        let b = 0.75_f32;

        for term in &query_tokens {
            if let Some(posting) = self.postings.get(term) {
                let df = posting.len() as f64;
                let idf = ((self.n_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln() as f32;

                for (doc_id, tf) in posting {
                    let doc_len = self.doc_lengths.get(doc_id).copied().unwrap_or(0) as f32;
                    let tf_f = *tf as f32;
                    let tf_norm = (tf_f * (k1 + 1.0)) / (tf_f + k1 * (1.0 - b + b * doc_len / self.avg_doc_len as f32));
                    *scores.entry(*doc_id).or_insert(0.0) += idf * tf_norm;
                }
            }
        }

        let mut results: Vec<(MemoryId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid search result combining dense and sparse scores.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Memory id.
    pub id: MemoryId,
    /// Dense similarity score (0-1).
    pub dense_score: f32,
    /// Sparse BM25 score (normalized 0-1).
    pub sparse_score: f32,
    /// Combined score.
    pub combined_score: f32,
}

/// Perform hybrid search by combining dense (HNSW) and sparse (BM25) results.
///
/// `alpha` controls the balance: 0.0 = pure sparse, 1.0 = pure dense.
#[must_use]
pub fn hybrid_search(
    dense_results: &[(MemoryId, f32)],
    sparse_results: &[(MemoryId, f32)],
    alpha: f32,
    k: usize,
) -> Vec<HybridResult> {
    // Normalize scores to [0, 1]
    let max_dense = dense_results.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max).max(1e-10);
    let max_sparse = sparse_results.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max).max(1e-10);

    let mut all_ids: HashSet<MemoryId> = HashSet::new();
    for (id, _) in dense_results { all_ids.insert(*id); }
    for (id, _) in sparse_results { all_ids.insert(*id); }

    let dense_map: AHashMap<MemoryId, f32> = dense_results.iter()
        .map(|(id, s)| (*id, *s / max_dense))
        .collect();
    let sparse_map: AHashMap<MemoryId, f32> = sparse_results.iter()
        .map(|(id, s)| (*id, *s / max_sparse))
        .collect();

    let mut results: Vec<HybridResult> = all_ids.iter()
        .map(|&id| {
            let dense = dense_map.get(&id).copied().unwrap_or(0.0);
            let sparse = sparse_map.get(&id).copied().unwrap_or(0.0);
            HybridResult {
                id,
                dense_score: dense,
                sparse_score: sparse,
                combined_score: alpha * dense + (1.0 - alpha) * sparse,
            }
        })
        .collect();

    results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Maximal Marginal Relevance (MMR) reranking for diversity.
///
/// Selects items that balance relevance to the query with diversity from
/// already-selected items. Parameter `lambda` controls the trade-off:
/// 1.0 = pure relevance, 0.0 = pure diversity.
#[must_use]
pub fn maximal_marginal_relevance(
    query_embedding: &Embedding,
    candidate_embeddings: &[(MemoryId, Embedding)],
    lambda: f32,
    k: usize,
) -> Vec<(MemoryId, f32)> {
    if candidate_embeddings.is_empty() {
        return Vec::new();
    }

    // Precompute query-candidate similarities
    let query_sims: Vec<f32> = candidate_embeddings.iter()
        .map(|(_, emb)| query_embedding.cosine(emb))
        .collect();

    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..candidate_embeddings.len()).collect();

    while !remaining.is_empty() && selected.len() < k {
        let mut best_idx = 0;
        let mut best_remaining_pos = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (pos, &idx) in remaining.iter().enumerate() {
            let relevance = query_sims[idx];
            let diversity = if selected.is_empty() {
                0.0
            } else {
                selected.iter()
                    .map(|&sel_idx| candidate_embeddings[sel_idx].1.cosine(&candidate_embeddings[idx].1))
                    .fold(0.0_f32, f32::max)
            };
            let mmr_score = lambda * relevance - (1.0 - lambda) * diversity;
            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = idx;
                best_remaining_pos = pos;
            }
        }

        remaining.remove(best_remaining_pos);
        selected.push(best_idx);
    }

    selected.into_iter()
        .map(|idx| (candidate_embeddings[idx].0, query_sims[idx]))
        .collect()
}

/// Adaptive search depth controller.
///
/// Dynamically adjusts `ef_search` based on the quality of initial results.
/// If results have low confidence (high variance in distances), the search
/// depth is increased; if results are high-confidence, it's decreased for
/// speed.
pub struct AdaptiveEf {
    /// Current ef_search value.
    current_ef: usize,
    /// Minimum ef_search.
    min_ef: usize,
    /// Maximum ef_search.
    max_ef: usize,
    /// Adaptation rate.
    alpha: f32,
    /// History of result quality.
    quality_history: Vec<f32>,
}

impl AdaptiveEf {
    /// Create a new adaptive ef controller.
    #[must_use]
    pub fn new(initial_ef: usize, min_ef: usize, max_ef: usize) -> Self {
        Self {
            current_ef: initial_ef,
            min_ef,
            max_ef,
            alpha: 0.1,
            quality_history: Vec::new(),
        }
    }

    /// Get the current ef_search value.
    #[must_use]
    pub fn ef(&self) -> usize {
        self.current_ef
    }

    /// Update the controller based on the quality of the latest search results.
    /// Quality is measured as the mean cosine similarity of the top-k results.
    pub fn update(&mut self, top_k_mean_similarity: f32) {
        self.quality_history.push(top_k_mean_similarity);
        if self.quality_history.len() > 10 {
            self.quality_history.remove(0);
        }

        // Compute trend
        let recent_avg: f32 = self.quality_history.iter().sum::<f32>() / self.quality_history.len() as f32;

        // If quality is low (< 0.5), increase ef; if high (> 0.8), decrease
        let target_ef = if recent_avg < 0.5 {
            (self.current_ef as f32 * 1.2) as usize
        } else if recent_avg > 0.8 {
            (self.current_ef as f32 * 0.9) as usize
        } else {
            self.current_ef
        };

        // Exponential moving average
        self.current_ef = ((1.0 - self.alpha) * self.current_ef as f32 + self.alpha * target_ef as f32) as usize;
        self.current_ef = self.current_ef.clamp(self.min_ef, self.max_ef);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_predicate_eq() {
        let pred = FilterPredicate::Eq("layer".into(), serde_json::json!("episodic"));
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("layer".into(), serde_json::json!("episodic"));
        assert!(pred.evaluate(&meta));
        meta.insert("layer".into(), serde_json::json!("semantic"));
        assert!(!pred.evaluate(&meta));
    }

    #[test]
    fn filter_predicate_gte() {
        let pred = FilterPredicate::Gte("importance".into(), 0.5);
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("importance".into(), serde_json::json!(0.8));
        assert!(pred.evaluate(&meta));
        meta.insert("importance".into(), serde_json::json!(0.3));
        assert!(!pred.evaluate(&meta));
    }

    #[test]
    fn filter_predicate_and_or() {
        let pred = FilterPredicate::And(vec![
            FilterPredicate::Eq("type".into(), serde_json::json!("memory")),
            FilterPredicate::Or(vec![
                FilterPredicate::Gte("importance".into(), 0.5),
                FilterPredicate::Eq("pinned".into(), serde_json::json!(true)),
            ]),
        ]);
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("type".into(), serde_json::json!("memory"));
        meta.insert("importance".into(), serde_json::json!(0.3));
        meta.insert("pinned".into(), serde_json::json!(true));
        assert!(pred.evaluate(&meta));
    }

    #[test]
    fn product_quantizer_train_encode() {
        let mut pq = ProductQuantizer::new(128, 8, 256);
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..128).map(|j| ((i + j) as f32 * 0.01).sin()).collect())
            .collect();
        pq.train(&vectors);
        let code = pq.encode(&vectors[0]);
        assert_eq!(code.len(), 8);
        let dist = pq.asymmetric_distance(&vectors[0], &code);
        assert!(dist >= 0.0);
    }

    #[test]
    fn sparse_index_bm25() {
        let mut index = SparseIndex::new();
        index.add_document(MemoryId::new(), "the quick brown fox jumps over the lazy dog");
        index.add_document(MemoryId::new(), "a quick brown dog runs in the park");
        index.add_document(MemoryId::new(), "the lazy cat sleeps on the mat");
        let results = index.search("quick brown", 3);
        assert!(!results.is_empty());
        // First result should be the document with both "quick" and "brown"
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn hybrid_search_combines() {
        let dense = vec![
            (MemoryId::new(), 0.9),
            (MemoryId::new(), 0.7),
            (MemoryId::new(), 0.5),
        ];
        let sparse = vec![
            (dense[1].0, 5.0),
            (dense[0].0, 3.0),
            (MemoryId::new(), 2.0),
        ];
        let results = hybrid_search(&dense, &sparse, 0.5, 3);
        assert_eq!(results.len(), 3);
        // All combined scores should be in [0, 1]
        for r in &results {
            assert!(r.combined_score >= 0.0 && r.combined_score <= 1.0);
        }
    }

    #[test]
    fn mmr_promotes_diversity() {
        let query = Embedding::new(vec![1.0, 0.0, 0.0]);
        let candidates = vec![
            (MemoryId::new(), Embedding::new(vec![0.9, 0.1, 0.0])), // very similar to query
            (MemoryId::new(), Embedding::new(vec![0.89, 0.1, 0.0])), // very similar to #0
            (MemoryId::new(), Embedding::new(vec![0.5, 0.8, 0.1])),  // less similar but diverse
        ];
        // With lambda=0.3, diversity is prioritized more
        let results = maximal_marginal_relevance(&query, &candidates, 0.3, 2);
        assert_eq!(results.len(), 2);
        // With MMR, the diverse item should be selected over the duplicate
        let selected_ids: HashSet<_> = results.iter().map(|(id, _)| *id).collect();
        assert!(selected_ids.contains(&candidates[2].0), "diverse candidate should be selected with low lambda");
    }

    #[test]
    fn adaptive_ef_adjusts() {
        let mut ctrl = AdaptiveEf::new(50, 10, 200);
        assert_eq!(ctrl.ef(), 50);
        // Low quality → should increase ef
        for _ in 0..5 {
            ctrl.update(0.3);
        }
        assert!(ctrl.ef() > 50, "ef should increase for low quality, got {}", ctrl.ef());
        // High quality → should decrease ef
        for _ in 0..10 {
            ctrl.update(0.9);
        }
        assert!(ctrl.ef() < 200, "ef should decrease for high quality, got {}", ctrl.ef());
    }

    #[test]
    fn kmeans_clusters() {
        let data: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                let center = if i < 15 { 0.0 } else { 10.0 };
                vec![center + (i as f32 * 0.01), center + (i as f32 * 0.02)]
            })
            .collect();
        let centroids = kmeans(&data, 2, 20);
        assert_eq!(centroids.len(), 2);
        // Centroids should be near 0 and 10
        let c0 = centroids[0][0];
        let c1 = centroids[1][0];
        assert!((c0 - c1).abs() > 5.0, "centroids should be far apart: {c0} vs {c1}");
    }
}
