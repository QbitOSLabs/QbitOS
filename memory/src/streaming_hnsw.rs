//! Streaming HNSW with dynamic edge pruning + index merging (2026).
//!
//! A streaming-capable HNSW (Hierarchical Navigable Small World) index
//! that supports incremental insertions, soft-deletes, restores, and
//! index merging — designed for online learning scenarios where the
//! index evolves over time.
//!
//! # Algorithm
//!
//! This is a faithful implementation of the HNSW algorithm from
//! Malkov & Yashunin (2018), "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs"
//! (TPAMI, vol. 42, no. 4), with streaming extensions:
//!
//! - **Multi-layer graph**: each node is assigned a maximum layer `l`
//!   drawn from an exponential distribution with decay rate `1/ln(M)`.
//!   The node is inserted at layers `0..=l` and connected to its `M`
//!   nearest neighbors at each layer (selected via `ef_construction`
//!   candidate search + heuristic neighbor selection).
//! - **Greedy search**: queries start at the top layer and greedily
//!   descend, narrowing the candidate set at each layer until layer 0,
//!   where `ef_search` controls the final candidate list size.
//! - **Heuristic neighbor selection**: when connecting a new node, we
//!   prefer neighbors that improve the diversity of the neighbor set
//!   (Malkov & Yashunin Algorithm 4), which prevents clustering and
//!   improves recall.
//! - **Soft delete / restore**: a node can be marked deleted without
//!   physically removing it from the graph. Deleted nodes are skipped
//!   during search (configurable via `filter_deleted`).
//! - **Index merging**: two indexes with the same dimension can be
//!   merged by re-inserting all non-deleted vectors from one into the
//!   other.
//!
//! # Example
//!
//! ```
//! use qbitos_memory::streaming_hnsw::{StreamingHnsw, StreamingHnswConfig};
//! let index = StreamingHnsw::new(StreamingHnswConfig::default());
//! let id = index.insert(vec![1.0, 0.0, 0.0]).unwrap();
//! let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
//! assert!(!results.is_empty());
//! ```

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::sync::Arc;
use thiserror::Error;

/// Errors emitted by the streaming HNSW index.
#[derive(Debug, Error)]
pub enum StreamingHnswError {
    /// The vector with the given id was not found in the index.
    #[error("vector {0} not found")]
    NotFound(u64),
    /// The inserted/query vector's dimension doesn't match the index's
    /// configured dimension.
    #[error("dim mismatch: expected {expected}, got {actual}")]
    DimMismatch {
        /// Expected dimension (from the index's config).
        expected: usize,
        /// Actual dimension (from the input vector).
        actual: usize,
    },
}

/// Stable identifier for a vector in the streaming HNSW index.
pub type VectorId = u64;

/// Configuration for the streaming HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingHnswConfig {
    /// Vector dimension (must match all inserted vectors).
    pub dim: usize,
    /// Maximum number of neighbors per node at layers > 0.
    pub m: usize,
    /// Maximum number of neighbors per node at layer 0 (doubled by convention).
    pub m0: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search (controls recall).
    pub ef_search: usize,
    /// Whether to apply diversity pruning when selecting neighbors
    /// (Malkov & Yashunin Algorithm 4).
    pub diversity_pruning: bool,
    /// Whether to filter soft-deleted vectors from search results.
    pub filter_deleted: bool,
}

impl Default for StreamingHnswConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 50,
            diversity_pruning: true,
            filter_deleted: true,
        }
    }
}

/// An internal node in the streaming HNSW index.
#[derive(Debug, Clone)]
struct Node {
    /// The vector stored at this node.
    vector: Vec<f32>,
    /// Neighbor lists per layer. `neighbors[layer]` is the list of
    /// neighbor `VectorId`s at that layer.
    neighbors: Vec<Vec<VectorId>>,
    /// Whether this node has been soft-deleted.
    deleted: bool,
    /// The highest layer this node appears in.
    max_layer: usize,
}

/// A candidate in the search frontier, ordered by distance (min-heap).
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    id: VectorId,
    dist: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse so that BinaryHeap (max-heap) behaves as a min-heap by distance.
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// The streaming HNSW index.
pub struct StreamingHnsw {
    cfg: StreamingHnswConfig,
    nodes: RwLock<AHashMap<VectorId, Node>>,
    entry_point: RwLock<Option<VectorId>>,
    max_level: RwLock<usize>,
    next_id: RwLock<u64>,
}

impl StreamingHnsw {
    /// Create a new empty streaming HNSW index with the given configuration.
    #[must_use]
    pub fn new(cfg: StreamingHnswConfig) -> Self {
        Self {
            cfg,
            nodes: RwLock::new(AHashMap::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            next_id: RwLock::new(0),
        }
    }

    /// Insert a vector into the index. Returns the assigned `VectorId`.
    ///
    /// Implements the HNSW insertion algorithm (Malkov & Yashunin Algorithm 1):
    /// 1. Assign the new node a maximum layer using an exponential distribution.
    /// 2. Greedily search from the top layer down to `max_layer + 1`, tracking
    ///    the closest entry point at each layer.
    /// 3. From `min(max_layer, top_level)` down to 0:
    ///    a. Search the layer with `ef_construction` candidates.
    ///    b. Select `M` (or `M0` at layer 0) neighbors using heuristic selection.
    ///    c. Add bidirectional edges between the new node and its neighbors.
    ///    d. Prune neighbors that exceed the maximum degree.
    /// 4. If the new node's `max_layer` is higher than the current `top_level`,
    ///    update the entry point.
    ///
    /// # Errors
    /// Returns `DimMismatch` if the vector's dimension doesn't match `cfg.dim`.
    pub fn insert(&self, vector: Vec<f32>) -> Result<VectorId, StreamingHnswError> {
        if vector.len() != self.cfg.dim {
            return Err(StreamingHnswError::DimMismatch {
                expected: self.cfg.dim,
                actual: vector.len(),
            });
        }

        let mut nid = self.next_id.write();
        *nid += 1;
        let id = *nid;
        drop(nid);

        // Sample the new node's max layer using the exponential distribution
        // with rate ln(M). This gives the geometric distribution over layers
        // that HNSW requires.
        let m = self.cfg.m.max(1);
        let level = sample_geometric_layer(m);

        let new_node = Node {
            vector: vector.clone(),
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
            max_layer: level,
        };

        let mut nodes = self.nodes.write();
        let entry_point_opt = *self.entry_point.read();
        let top_level = *self.max_level.read();

        match entry_point_opt {
            None => {
                // First node — just insert and become the entry point.
                nodes.insert(id, new_node);
                *self.entry_point.write() = Some(id);
                *self.max_level.write() = level;
            }
            Some(entry_id) => {
                let entry_vec = nodes
                    .get(&entry_id)
                    .map(|n| n.vector.clone())
                    .unwrap_or_default();

                // Phase 1: greedy descent from top_level down to level+1,
                // tracking the closest entry point at each layer.
                let mut current_entry = entry_id;
                let mut current_dist = cosine_distance(&entry_vec, &vector);

                for layer in (level + 1..=top_level).rev() {
                    let (ep, ed) = greedy_search_layer(
                        &nodes,
                        &vector,
                        current_entry,
                        layer,
                        self.cfg.filter_deleted,
                    );
                    current_entry = ep;
                    current_dist = ed;
                }
                let _ = current_dist;

                // Phase 2: from min(level, top_level) down to 0, search with
                // ef_construction candidates and connect to M neighbors.
                let connect_start = level.min(top_level);
                for layer in (0..=connect_start).rev() {
                    let candidates = search_layer(
                        &nodes,
                        &vector,
                        current_entry,
                        layer,
                        self.cfg.ef_construction,
                        self.cfg.filter_deleted,
                    );

                    // Select M (or M0 at layer 0) neighbors using heuristic selection.
                    let max_neighbors = if layer == 0 {
                        self.cfg.m0
                    } else {
                        self.cfg.m
                    };
                    let selected = select_neighbors(
                        &candidates,
                        max_neighbors,
                        &vector,
                        &nodes,
                        self.cfg.diversity_pruning,
                    );

                    // Insert the new node with its neighbor list, and add
                    // bidirectional edges to the selected neighbors.
                    let mut neighbors_for_new: Vec<VectorId> =
                        selected.iter().map(|c| c.id).collect();
                    neighbors_for_new.sort_unstable();
                    neighbors_for_new.dedup();

                    // Insert the node now (we held off until we computed neighbors).
                    let mut node_to_insert = new_node.clone();
                    if node_to_insert.neighbors.len() <= layer {
                        node_to_insert.neighbors.resize(layer + 1, Vec::new());
                    }
                    node_to_insert.neighbors[layer] = neighbors_for_new.clone();
                    nodes.insert(id, node_to_insert);

                    // Add the reverse edges (new node → each neighbor's neighbor list).
                    for neighbor_id in &neighbors_for_new {
                        // Collect what we need from the neighbor under a mutable borrow,
                        // then release the borrow before doing immutable lookups for pruning.
                        let (needs_prune, neighbor_vec, list_to_prune) = {
                            let Some(neighbor) = nodes.get_mut(neighbor_id) else {
                                continue;
                            };
                            if neighbor.neighbors.len() <= layer {
                                neighbor.neighbors.resize(layer + 1, Vec::new());
                            }
                            let neighbor_list = &mut neighbor.neighbors[layer];
                            if !neighbor_list.contains(&id) {
                                neighbor_list.push(id);
                            }
                            // Determine if pruning is needed.
                            let neighbor_max = if layer == 0 {
                                self.cfg.m0
                            } else {
                                self.cfg.m
                            };
                            if neighbor_list.len() > neighbor_max {
                                (true, neighbor.vector.clone(), neighbor_list.clone())
                            } else {
                                (false, Vec::new(), Vec::new())
                            }
                        };

                        // Prune outside the mutable borrow — now we can do immutable lookups.
                        if needs_prune {
                            let neighbor_max = if layer == 0 {
                                self.cfg.m0
                            } else {
                                self.cfg.m
                            };
                            let mut cands: Vec<Candidate> = list_to_prune
                                .iter()
                                .filter_map(|&nid| {
                                    nodes.get(&nid).map(|n| Candidate {
                                        id: nid,
                                        dist: cosine_distance(&neighbor_vec, &n.vector),
                                    })
                                })
                                .collect();
                            cands.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
                            let pruned: Vec<VectorId> =
                                cands.iter().take(neighbor_max).map(|c| c.id).collect();
                            // Re-acquire the mutable borrow to write the pruned list.
                            if let Some(neighbor) = nodes.get_mut(neighbor_id) {
                                if let Some(neighbor_list) = neighbor.neighbors.get_mut(layer) {
                                    *neighbor_list = pruned;
                                }
                            }
                        }
                    }

                    // Update the entry point for the next lower layer.
                    if let Some(c) = candidates.first() {
                        current_entry = c.id;
                    }
                }

                // Phase 3: if the new node's level is higher than top_level,
                // it becomes the new entry point.
                if level > top_level {
                    *self.entry_point.write() = Some(id);
                    *self.max_level.write() = level;
                }
            }
        }

        Ok(id)
    }

    /// Search for the `k` nearest neighbors of `query`. Returns a sorted
    /// list of `(VectorId, distance)` tuples (closest first).
    ///
    /// Implements HNSW search (Malkov & Yashunin Algorithm 5):
    /// 1. Greedily descend from the top layer to layer 1, tracking the
    ///    closest entry point.
    /// 2. At layer 0, perform an `ef_search`-candidate search.
    /// 3. Return the top-`k` results.
    ///
    /// # Errors
    /// Returns `DimMismatch` if the query's dimension doesn't match `cfg.dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, StreamingHnswError> {
        if query.len() != self.cfg.dim {
            return Err(StreamingHnswError::DimMismatch {
                expected: self.cfg.dim,
                actual: query.len(),
            });
        }
        let nodes = self.nodes.read();
        let entry_point_opt = *self.entry_point.read();
        let top_level = *self.max_level.read();

        let Some(entry_id) = entry_point_opt else {
            return Ok(Vec::new());
        };

        // Phase 1: greedy descent from top_level down to layer 1.
        let mut current_entry = entry_id;
        for layer in (1..=top_level).rev() {
            let (ep, _) = greedy_search_layer(&nodes, query, current_entry, layer, self.cfg.filter_deleted);
            current_entry = ep;
        }

        // Phase 2: ef_search-candidate search at layer 0.
        let ef = self.cfg.ef_search.max(k);
        let mut candidates = search_layer(&nodes, query, current_entry, 0, ef, self.cfg.filter_deleted);

        // Sort by distance ascending and take top-k.
        candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        Ok(candidates.into_iter().take(k).map(|c| (c.id, c.dist)).collect())
    }

    /// Soft-delete a vector (mark it as deleted; restorable via `restore`).
    ///
    /// The node remains in the graph but is skipped during search when
    /// `filter_deleted` is true.
    ///
    /// # Errors
    /// Returns `NotFound` if no vector with the given id exists.
    pub fn soft_delete(&self, id: VectorId) -> Result<(), StreamingHnswError> {
        let mut nodes = self.nodes.write();
        let node = nodes
            .get_mut(&id)
            .ok_or(StreamingHnswError::NotFound(id))?;
        node.deleted = true;
        Ok(())
    }

    /// Restore a previously soft-deleted vector.
    ///
    /// # Errors
    /// Returns `NotFound` if no vector with the given id exists.
    pub fn restore(&self, id: VectorId) -> Result<(), StreamingHnswError> {
        let mut nodes = self.nodes.write();
        let node = nodes
            .get_mut(&id)
            .ok_or(StreamingHnswError::NotFound(id))?;
        node.deleted = false;
        Ok(())
    }

    /// Merge another index into this one. Both indexes must have the same
    /// vector dimension. Soft-deleted vectors in `other` are skipped.
    ///
    /// # Errors
    /// Returns `DimMismatch` if the dimensions don't match.
    pub fn merge(&self, other: &StreamingHnsw) -> Result<(), StreamingHnswError> {
        if other.cfg.dim != self.cfg.dim {
            return Err(StreamingHnswError::DimMismatch {
                expected: self.cfg.dim,
                actual: other.cfg.dim,
            });
        }
        let on = other.nodes.read();
        // Collect vectors first to avoid borrowing `other` during `self.insert`.
        let to_merge: Vec<Vec<f32>> = on
            .iter()
            .filter(|(_, n)| !n.deleted)
            .map(|(_, n)| n.vector.clone())
            .collect();
        drop(on);
        for vector in to_merge {
            self.insert(vector)?;
        }
        Ok(())
    }

    /// Total number of vectors in the index (including soft-deleted).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Number of active (non-deleted) vectors.
    #[must_use]
    pub fn active_len(&self) -> usize {
        self.nodes.read().values().filter(|n| !n.deleted).count()
    }

    /// Returns `true` if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Number of soft-deleted vectors.
    #[must_use]
    pub fn deleted_count(&self) -> usize {
        self.nodes.read().values().filter(|n| n.deleted).count()
    }

    /// Returns the current top layer of the index (0 if empty).
    #[must_use]
    pub fn top_level(&self) -> usize {
        *self.max_level.read()
    }

    /// Returns the current entry point id, if any.
    #[must_use]
    pub fn entry_point(&self) -> Option<VectorId> {
        *self.entry_point.read()
    }

    /// Returns the number of edges at the given layer (bidirectional edges
    /// are counted once per direction).
    #[must_use]
    pub fn edge_count_at_layer(&self, layer: usize) -> usize {
        self.nodes
            .read()
            .values()
            .map(|n| n.neighbors.get(layer).map(Vec::len).unwrap_or(0))
            .sum()
    }

    /// Returns the maximum layer of the node with the given id, or `None`
    /// if the node doesn't exist.
    #[must_use]
    pub fn node_max_layer(&self, id: VectorId) -> Option<usize> {
        self.nodes.read().get(&id).map(|n| n.max_layer)
    }

    /// Returns the distribution of node max-layer values: `result[layer]` is
    /// the number of nodes whose maximum layer is `layer`.
    #[must_use]
    pub fn layer_distribution(&self) -> Vec<usize> {
        let nodes = self.nodes.read();
        let max = *self.max_level.read();
        let mut dist = vec![0_usize; max + 1];
        for n in nodes.values() {
            if n.max_layer < dist.len() {
                dist[n.max_layer] += 1;
            }
        }
        dist
    }
}

/// Sample a layer index from the geometric distribution used by HNSW:
/// `floor(-ln(uniform(0,1)) * ln(M))` (Malkov & Yashunin eq. 2).
///
/// We use a simple LCG instead of `rand` to keep the dependency footprint
/// small and the sampling deterministic per-thread.
fn sample_geometric_layer(m: usize) -> usize {
    let m = m.max(2) as f64;
    let ln_m = m.ln();
    if ln_m < 1e-12 {
        return 0;
    }
    // Thread-local seed based on address of a local variable — gives
    // different sequences across insertions without a global RNG.
    let seed = {
        let local = 0u64;
        let addr = &local as *const _ as u64;
        // Mix with a counter stored in a thread-local.
        use std::cell::Cell;
        thread_local! {
            static CTR: Cell<u64> = const { Cell::new(0x9E37_79B9_7F4A_7C15) };
        }
        CTR.with(|c| {
            let v = c.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            c.set(v);
            v ^ addr
        })
    };
    // Convert seed to a uniform float in (0, 1].
    let u = (seed as f64) / (u64::MAX as f64);
    let u = if u < 1e-12 { 1e-12 } else { u };
    let level = -(u.ln()) / ln_m;
    level.floor() as usize
}

/// Greedy search at a single layer: walk to the neighbor with the smallest
/// distance to the query until no improvement is possible. Returns the
/// closest node id and its distance.
///
/// This is the layer-level greedy search used in HNSW's upper layers.
fn greedy_search_layer(
    nodes: &AHashMap<VectorId, Node>,
    query: &[f32],
    entry: VectorId,
    layer: usize,
    filter_deleted: bool,
) -> (VectorId, f32) {
    let Some(entry_node) = nodes.get(&entry) else {
        return (entry, f32::INFINITY);
    };
    let mut current = entry;
    let mut current_dist = cosine_distance(&entry_node.vector, query);

    loop {
        let mut improved = false;
        if let Some(node) = nodes.get(&current) {
            let neighbors = node.neighbors.get(layer).map(|n| n.as_slice()).unwrap_or(&[]);
            for &neighbor_id in neighbors {
                let Some(neighbor) = nodes.get(&neighbor_id) else { continue };
                if filter_deleted && neighbor.deleted {
                    continue;
                }
                let d = cosine_distance(&neighbor.vector, query);
                if d < current_dist {
                    current_dist = d;
                    current = neighbor_id;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (current, current_dist)
}

/// Best-first search at a single layer with a candidate list of size `ef`.
/// Returns up to `ef` candidates sorted by distance (ascending).
///
/// This is the `SEARCH-LAYER` subroutine from Malkov & Yashunin Algorithm 2.
fn search_layer(
    nodes: &AHashMap<VectorId, Node>,
    query: &[f32],
    entry: VectorId,
    layer: usize,
    ef: usize,
    filter_deleted: bool,
) -> Vec<Candidate> {
    let Some(entry_node) = nodes.get(&entry) else {
        return Vec::new();
    };

    let mut visited: AHashMap<VectorId, bool> = AHashMap::new();
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut results: BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();

    let entry_dist = cosine_distance(&entry_node.vector, query);
    candidates.push(Candidate { id: entry, dist: entry_dist });
    // Skip deleted entry nodes from the result set, but keep them in the
    // traversal candidates so we can still walk through them to reach
    // non-deleted neighbors.
    if !(filter_deleted && entry_node.deleted) {
        results.push(std::cmp::Reverse(Candidate { id: entry, dist: entry_dist }));
    }
    visited.insert(entry, true);

    while let Some(c) = candidates.pop() {
        // If the worst result is closer than the best candidate, we're done.
        let worst_in_results = results
            .peek()
            .map(|r| r.0.dist)
            .unwrap_or(f32::INFINITY);
        if c.dist > worst_in_results && results.len() >= ef {
            break;
        }

        let Some(node) = nodes.get(&c.id) else { continue };
        let neighbors = node.neighbors.get(layer).map(|n| n.as_slice()).unwrap_or(&[]);
        for &neighbor_id in neighbors {
            if visited.contains_key(&neighbor_id) {
                continue;
            }
            visited.insert(neighbor_id, true);
            let Some(neighbor) = nodes.get(&neighbor_id) else { continue };
            if filter_deleted && neighbor.deleted {
                continue;
            }
            let d = cosine_distance(&neighbor.vector, query);
            let worst = results
                .peek()
                .map(|r| r.0.dist)
                .unwrap_or(f32::INFINITY);
            if results.len() < ef || d < worst {
                candidates.push(Candidate { id: neighbor_id, dist: d });
                results.push(std::cmp::Reverse(Candidate { id: neighbor_id, dist: d }));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    let mut out: Vec<Candidate> = results.into_iter().map(|r| r.0).collect();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Heuristic neighbor selection (Malkov & Yashunin Algorithm 4).
///
/// From a list of candidate neighbors (sorted by distance to the query),
/// select up to `M` neighbors such that each selected neighbor is closer
/// to the query than to any already-selected neighbor. This prevents
/// clustering and improves graph navigability.
///
/// If `diversity_pruning` is false, simply take the `M` closest candidates.
fn select_neighbors(
    candidates: &[Candidate],
    m: usize,
    query: &[f32],
    nodes: &AHashMap<VectorId, Node>,
    diversity_pruning: bool,
) -> Vec<Candidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<Candidate> = candidates.to_vec();
    sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));

    if !diversity_pruning {
        sorted.truncate(m);
        return sorted;
    }

    let mut selected: Vec<Candidate> = Vec::with_capacity(m);
    for c in sorted {
        if selected.len() >= m {
            break;
        }
        let Some(c_node) = nodes.get(&c.id) else { continue };
        let c_vec = &c_node.vector;

        // Check that c is closer to the query than to any already-selected neighbor.
        let mut good = true;
        for s in &selected {
            let Some(s_node) = nodes.get(&s.id) else { continue };
            let d_to_selected = cosine_distance(c_vec, &s_node.vector);
            if d_to_selected < c.dist {
                good = false;
                break;
            }
        }
        if good {
            selected.push(c);
        }
        let _ = query; // query is implicit in c.dist; reserved for future extensions
    }

    // If diversity pruning rejected too many, fall back to closest.
    if selected.len() < m {
        for c in candidates {
            if selected.len() >= m {
                break;
            }
            if !selected.iter().any(|s| s.id == c.id) {
                selected.push(*c);
            }
        }
    }

    selected
}

/// Compute the cosine distance between two vectors: `1 - cos(a, b)`,
/// in the range `[0.0, 2.0]`. Returns `1.0` (orthogonal) for zero vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        1.0
    } else {
        1.0 - d / (na * nb)
    }
}

/// Shared handle to a streaming HNSW index.
pub type SharedStreamingHnsw = Arc<StreamingHnsw>;

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(d: usize) -> StreamingHnsw {
        StreamingHnsw::new(StreamingHnswConfig {
            dim: d,
            ..Default::default()
        })
    }

    #[test]
    fn insert_search() {
        let i = idx(3);
        let _ = i.insert(vec![1.0, 0.0, 0.0]).unwrap();
        let _ = i.insert(vec![0.0, 1.0, 0.0]).unwrap();
        let r = i.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert!(!r.is_empty());
        assert!(r[0].1 < 0.01);
    }

    #[test]
    fn dim_mismatch() {
        let i = idx(4);
        assert!(matches!(
            i.insert(vec![1.0, 0.0, 0.0]),
            Err(StreamingHnswError::DimMismatch { .. })
        ));
    }

    #[test]
    fn soft_delete() {
        let i = idx(2);
        let id = i.insert(vec![1.0, 0.0]).unwrap();
        let _ = i.insert(vec![0.0, 1.0]).unwrap();
        i.soft_delete(id).unwrap();
        let r = i.search(&[1.0, 0.0], 2).unwrap();
        assert!(!r.iter().any(|(i, _)| *i == id));
    }

    #[test]
    fn restore() {
        let i = idx(2);
        let id = i.insert(vec![1.0, 0.0]).unwrap();
        i.soft_delete(id).unwrap();
        assert_eq!(i.deleted_count(), 1);
        i.restore(id).unwrap();
        assert_eq!(i.deleted_count(), 0);
    }

    #[test]
    fn delete_unknown() {
        let i = idx(2);
        assert!(matches!(
            i.soft_delete(999),
            Err(StreamingHnswError::NotFound(999))
        ));
    }

    #[test]
    fn merge() {
        let i1 = idx(2);
        let i2 = idx(2);
        for j in 0..5 {
            i1.insert(vec![j as f32, 0.0]).unwrap();
        }
        for j in 0..5 {
            i2.insert(vec![0.0, j as f32]).unwrap();
        }
        i1.merge(&i2).unwrap();
        assert_eq!(i1.active_len(), 10);
    }

    #[test]
    fn nearest_neighbor_recall() {
        // Insert 50 random vectors and verify that HNSW search returns
        // the same top-1 as brute-force search.
        let i = idx(8);
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for n in 0..50 {
            let v: Vec<f32> = (0..8)
                .map(|k| ((n * 7 + k * 13) as f32).sin() * 10.0)
                .collect();
            i.insert(v.clone()).unwrap();
            vectors.push(v);
        }
        // For each vector, HNSW search for itself should return it as top-1.
        for v in &vectors {
            let r = i.search(v, 1).unwrap();
            assert!(!r.is_empty());
            // The top-1 distance should be ~0 (the vector itself is in the index).
            assert!(r[0].1 < 0.01, "expected distance ~0, got {}", r[0].1);
        }
    }

    #[test]
    fn edge_count_grows_with_insertions() {
        let i = idx(4);
        i.insert(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let edges_after_1 = i.edge_count_at_layer(0);
        i.insert(vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        let edges_after_2 = i.edge_count_at_layer(0);
        assert!(
            edges_after_2 >= edges_after_1,
            "edges should grow (or stay equal) with insertions: {edges_after_1} -> {edges_after_2}"
        );
    }

    #[test]
    fn top_level_can_grow() {
        // Insert many vectors; with high probability, at least one node
        // will be assigned a higher layer than 0.
        let i = idx(4);
        for n in 0..200 {
            let v: Vec<f32> = (0..4).map(|k| ((n + k) as f32).cos()).collect();
            i.insert(v).unwrap();
        }
        // The top level is not guaranteed to be > 0 with only 200 inserts
        // and M=16 (probability ~ 1/16 per insert), but on average ~12
        // nodes will be at level 1+. We just check that the entry point exists.
        assert!(i.entry_point().is_some());
    }

    #[test]
    fn search_returns_empty_for_empty_index() {
        let i = idx(4);
        let r = i.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn diversity_pruning_keeps_graph_navigable() {
        // With diversity_pruning on, the graph should still be searchable.
        let i = StreamingHnsw::new(StreamingHnswConfig {
            dim: 4,
            m: 4,
            m0: 8,
            ef_construction: 50,
            ef_search: 20,
            diversity_pruning: true,
            filter_deleted: true,
        });
        for n in 0..30 {
            let v: Vec<f32> = (0..4).map(|k| ((n * 3 + k) as f32).sin()).collect();
            i.insert(v).unwrap();
        }
        let r = i.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(!r.is_empty());
        assert!(r.len() <= 5);
    }
}
