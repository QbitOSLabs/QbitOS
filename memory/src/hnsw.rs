//! Hierarchical Navigable Small World (HNSW) vector index.
//!
//! A from-scratch implementation of the HNSW algorithm (Malkov & Yashunin,
//! 2016) for approximate nearest-neighbor search. This is the primary
//! in-process vector index used by the memory fabric when an external Qdrant
//! instance is not configured.

use ahash::AHashMap;
use parking_lot::RwLock;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use thiserror::Error;

use crate::embedding::Embedding;
use crate::object::MemoryId;

/// Errors emitted by the HNSW index.
#[derive(Debug, Error)]
pub enum HnswError {
    /// The queried vector dimensionality does not match the index's.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Provided dimensionality.
        got: usize,
    },
    /// The id is already in the index.
    #[error("id {0:?} already exists")]
    Duplicate(MemoryId),
    /// The id is not in the index.
    #[error("id {0:?} not found")]
    NotFound(MemoryId),
}

#[derive(Clone)]
struct Node {
    embedding: Embedding,
    /// Neighbors per layer. `connections[layer]` is the list of neighbors at
    /// that layer.
    connections: Vec<Vec<MemoryId>>,
    /// Highest layer this node appears in.
    max_layer: usize,
}

impl Node {
    fn new(embedding: Embedding, max_layer: usize) -> Self {
        Self {
            embedding,
            connections: vec![Vec::new(); max_layer + 1],
            max_layer,
        }
    }
}

/// A candidate in the search frontier.
#[derive(Clone, Copy)]
struct Candidate {
    id: MemoryId,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want a min-heap by distance, so reverse the order.
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

/// Configuration for the HNSW index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of neighbors per node at layers > 0.
    pub m: usize,
    /// Maximum number of neighbors per node at layer 0 (typically `2 * m`).
    pub m0: usize,
    /// Construction search depth (ef_construction).
    pub ef_construction: usize,
    /// Search depth (ef_search).
    pub ef_search: usize,
    /// Level multiplier (1 / ln(M)).
    pub level_multiplier: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            level_multiplier: 1.0 / (m as f64).ln(),
            seed: 0x71b1_7055_1142_u64,
        }
    }
}

/// The HNSW index.
pub struct HnswIndex {
    cfg: HnswConfig,
    nodes: RwLock<AHashMap<MemoryId, Node>>,
    entry_point: RwLock<Option<MemoryId>>,
    max_layer: RwLock<usize>,
    dim: RwLock<Option<usize>>,
    rng: parking_lot::Mutex<SmallRng>,
}

impl HnswIndex {
    /// Create a new empty index with the given configuration.
    #[must_use]
    pub fn new(cfg: HnswConfig) -> Self {
        let rng = SmallRng::seed_from_u64(cfg.seed);
        Self {
            cfg,
            nodes: RwLock::new(AHashMap::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
            dim: RwLock::new(None),
            rng: parking_lot::Mutex::new(rng),
        }
    }

    /// Create a new index with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(HnswConfig::default())
    }

    /// Returns the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    fn select_level(&self) -> usize {
        let mut rng = self.rng.lock();
        let r: f64 = rng.r#gen();
        let level = -(r.ln() * self.cfg.level_multiplier);
        level.floor() as usize
    }

    /// Insert a vector into the index.
    ///
    /// # Errors
    /// Returns [`HnswError::DimMismatch`] if the dimensionality is inconsistent
    /// with the rest of the index, or [`HnswError::Duplicate`] if the id is
    /// already present.
    pub fn insert(&self, id: MemoryId, embedding: Embedding) -> Result<(), HnswError> {
        // Validate dimension
        {
            let mut dim = self.dim.write();
            match *dim {
                None => *dim = Some(embedding.dim()),
                Some(d) if d != embedding.dim() => {
                    return Err(HnswError::DimMismatch { expected: d, got: embedding.dim() });
                }
                Some(_) => {}
            }
        }

        if self.nodes.read().contains_key(&id) {
            return Err(HnswError::Duplicate(id));
        }

        let level = self.select_level();
        let node = Node::new(embedding.clone(), level);

        // If this is the first node, make it the entry point.
        {
            let mut entry = self.entry_point.write();
            if entry.is_none() {
                *entry = Some(id);
                let mut max_layer = self.max_layer.write();
                *max_layer = level;
                self.nodes.write().insert(id, node);
                return Ok(());
            }
        }

        // Find the entry point and search down from the top.
        let entry_id = self.entry_point.read().unwrap();
        let mut current_entry = entry_id;
        let top_layer = *self.max_layer.read();

        // Greedy search from top down to `level + 1`
        for layer in (level + 1..=top_layer).rev() {
            let best = self.greedy_search_layer(&embedding, current_entry, layer, 1);
            if let Some(c) = best.first() {
                current_entry = c.id;
            }
        }

        // For each layer from `min(level, top_layer)` down to 0, find neighbors
        // and connect bidirectionally.
        let mut nodes = self.nodes.write();
        nodes.insert(id, node);

        for layer in (0..=level.min(top_layer)).rev() {
            let candidates = self.search_layer_with_lock(&embedding, current_entry, layer, self.cfg.ef_construction, &nodes);
            let neighbors = self.select_neighbors(&candidates, if layer == 0 { self.cfg.m0 } else { self.cfg.m });

            // Add forward edges
            if let Some(n) = nodes.get_mut(&id) {
                if n.connections.len() <= layer {
                    n.connections.resize(layer + 1, Vec::new());
                }
                n.connections[layer] = neighbors.clone();
            }

            // Add backward edges
            for n_id in &neighbors {
                // Gather all the info we need under an immutable borrow first
                let (_embedding_clone, already_present, needs_prune, cap, prune_candidates) = {
                    let Some(n) = nodes.get(n_id) else { continue };
                    let embedding_clone = n.embedding.clone();
                    let present = if n.connections.len() > layer {
                        n.connections[layer].contains(&id)
                    } else {
                        false
                    };
                    let cap = if layer == 0 { self.cfg.m0 } else { self.cfg.m };
                    let needs_prune = n.connections.len() > layer && n.connections[layer].len() >= cap;
                    let prune_candidates: Vec<(MemoryId, f32)> = if needs_prune {
                        n.connections[layer].iter()
                            .filter_map(|c| {
                                if c == n_id { return None; }
                                nodes.get(c).map(|cn| (*c, 1.0 - embedding_clone.cosine(&cn.embedding)))
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    (embedding_clone, present, needs_prune, cap, prune_candidates)
                };

                // Now mutate using the precomputed data (no immutable borrow outstanding)
                if let Some(n) = nodes.get_mut(n_id) {
                    if n.connections.len() <= layer {
                        n.connections.resize(layer + 1, Vec::new());
                    }
                    let conn = &mut n.connections[layer];
                    if !already_present {
                        conn.push(id);
                    }
                    if needs_prune && conn.len() > cap {
                        let mut cand = prune_candidates.clone();
                        cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        conn.clear();
                        conn.extend(cand.into_iter().take(cap).map(|(c, _)| c));
                    }
                }
            }

            if let Some(c) = candidates.first() {
                current_entry = c.id;
            }
        }

        // Update entry point if this node has a higher level
        if level > top_layer {
            *self.max_layer.write() = level;
            *self.entry_point.write() = Some(id);
        }

        Ok(())
    }

    fn greedy_search_layer(&self, query: &Embedding, entry: MemoryId, layer: usize, ef: usize) -> Vec<Candidate> {
        self.search_layer_with_lock(query, entry, layer, ef, &self.nodes.read())
    }

    fn search_layer_with_lock(
        &self,
        query: &Embedding,
        entry: MemoryId,
        layer: usize,
        ef: usize,
        nodes: &AHashMap<MemoryId, Node>,
    ) -> Vec<Candidate> {
        let mut visited: AHashMap<MemoryId, bool> = AHashMap::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();

        let entry_node = match nodes.get(&entry) {
            Some(n) => n,
            None => return Vec::new(),
        };
        let dist = 1.0 - query.cosine(&entry_node.embedding);
        candidates.push(Candidate { id: entry, dist });
        results.push(std::cmp::Reverse(Candidate { id: entry, dist }));
        visited.insert(entry, true);

        while let Some(c) = candidates.pop() {
            let worst = results.peek().map(|r| r.0.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            let node = match nodes.get(&c.id) {
                Some(n) => n,
                None => continue,
            };
            if layer >= node.connections.len() {
                continue;
            }
            for neighbor_id in &node.connections[layer] {
                if visited.contains_key(neighbor_id) {
                    continue;
                }
                visited.insert(*neighbor_id, true);
                let neighbor = match nodes.get(neighbor_id) {
                    Some(n) => n,
                    None => continue,
                };
                let d = 1.0 - query.cosine(&neighbor.embedding);
                candidates.push(Candidate { id: *neighbor_id, dist: d });
                let worst = results.peek().map(|r| r.0.dist).unwrap_or(f32::INFINITY);
                if d < worst || results.len() < ef {
                    results.push(std::cmp::Reverse(Candidate { id: *neighbor_id, dist: d }));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out: Vec<Candidate> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out
    }

    fn select_neighbors(&self, candidates: &[Candidate], k: usize) -> Vec<MemoryId> {
        candidates.iter().take(k).map(|c| c.id).collect()
    }

    /// Search the index for the `k` nearest neighbors of `query`.
    ///
    /// # Errors
    /// Returns [`HnswError::DimMismatch`] if the query dimensionality is wrong.
    pub fn search(&self, query: &Embedding, k: usize) -> Result<Vec<(MemoryId, f32)>, HnswError> {
        let dim = self.dim.read();
        match *dim {
            None => return Ok(Vec::new()),
            Some(d) if d != query.dim() => {
                return Err(HnswError::DimMismatch { expected: d, got: query.dim() });
            }
            Some(_) => {}
        }

        let entry_id = match *self.entry_point.read() {
            None => return Ok(Vec::new()),
            Some(id) => id,
        };
        let top_layer = *self.max_layer.read();

        let mut current_entry = entry_id;
        for layer in (1..=top_layer).rev() {
            let best = self.greedy_search_layer(query, current_entry, layer, 1);
            if let Some(c) = best.first() {
                current_entry = c.id;
            }
        }

        let nodes = self.nodes.read();
        let mut results = self.search_layer_with_lock(query, current_entry, 0, self.cfg.ef_search.max(k), &nodes);
        results.truncate(k);
        Ok(results.into_iter().map(|c| (c.id, c.dist)).collect())
    }

    /// Remove a vector from the index. Note: this is a lazy removal — the
    /// node is removed but its neighbors are not rewired. For production use
    /// prefer rebuilding the index periodically.
    ///
    /// # Errors
    /// Returns [`HnswError::NotFound`] if the id is not in the index.
    pub fn remove(&self, id: MemoryId) -> Result<(), HnswError> {
        let mut nodes = self.nodes.write();
        let node = nodes.remove(&id).ok_or(HnswError::NotFound(id))?;
        // Remove backward edges
        for layer in 0..=node.max_layer {
            for neighbor_id in &node.connections[layer] {
                if let Some(n) = nodes.get_mut(neighbor_id) {
                    if layer < n.connections.len() {
                        n.connections[layer].retain(|x| *x != id);
                    }
                }
            }
        }
        // If this was the entry point, pick a new one
        if self.entry_point.read().map_or(false, |e| e == id) {
            let new_entry = nodes.keys().next().copied();
            *self.entry_point.write() = new_entry;
            if let Some(ne) = new_entry {
                if let Some(n) = nodes.get(&ne) {
                    *self.max_layer.write() = n.max_layer;
                }
            } else {
                *self.max_layer.write() = 0;
                *self.dim.write() = None;
            }
        }
        Ok(())
    }
}

/// Make `Arc<HnswIndex>` shareable across threads.
pub type SharedHnswIndex = Arc<HnswIndex>;

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_embedding(seed: f32, dim: usize) -> Embedding {
        let data: Vec<f32> = (0..dim).map(|i| (seed + i as f32).sin()).collect();
        let mut e = Embedding::new(data);
        e.normalize();
        e
    }

    #[test]
    fn insert_and_search() {
        let index = HnswIndex::new(HnswConfig::default());
        let dim = 64;
        for i in 0..100 {
            let id = MemoryId(Uuid::from_u128(i as u128));
            let emb = make_embedding(i as f32, dim);
            index.insert(id, emb).unwrap();
        }
        let query = make_embedding(5.0, dim);
        let results = index.search(&query, 5).unwrap();
        assert_eq!(results.len(), 5);
        // The nearest neighbor should be id 5 (the seed we used).
        let nearest = results[0].0;
        assert_eq!(nearest, MemoryId(Uuid::from_u128(5)));
    }

    #[test]
    fn remove_id() {
        let index = HnswIndex::default_config();
        let id = MemoryId::new();
        let emb = make_embedding(1.0, 32);
        index.insert(id, emb).unwrap();
        assert_eq!(index.len(), 1);
        index.remove(id).unwrap();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn dim_mismatch() {
        let index = HnswIndex::default_config();
        index.insert(MemoryId::new(), make_embedding(1.0, 32)).unwrap();
        let r = index.insert(MemoryId::new(), make_embedding(1.0, 64));
        assert!(matches!(r, Err(HnswError::DimMismatch { .. })));
    }
}
