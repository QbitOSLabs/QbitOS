//! HNSW - Hierarchical Navigable Small World graph index for ANN search.
#![allow(missing_docs)]
//!
//! Implements HNSW from Malkov & Yashunin (2016), refined by the 2026
//! ANN survey (OpenReview). Graph-based ANN with O(M*log N) search.

use ahash::AHashMap;
use std::cmp::Ordering;

/// A vector ID inside the HNSW index.
pub type NodeId = u64;

/// HNSW index, generic over distance function.
pub struct HnswIndex<F>
where F: Fn(&[f32], &[f32]) -> f32 + Send + Sync,
{
    distance: F,
    m: usize,
    m0: usize,
    ml: f64,
    ef_construction: usize,
    ef_search: usize,
    vectors: Vec<Vec<f32>>,
    layers: Vec<AHashMap<NodeId, Vec<NodeId>>>,
    max_level: usize,
    entry_point: Option<NodeId>,
    next_id: NodeId,
}

impl<F> HnswIndex<F>
where F: Fn(&[f32], &[f32]) -> f32 + Send + Sync,
{
    #[must_use]
    pub fn new(distance: F, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        Self { distance, m, m0: 2 * m, ml, ef_construction, ef_search,
            vectors: Vec::new(), layers: vec![AHashMap::new()], max_level: 0,
            entry_point: None, next_id: 0 }
    }

    pub fn insert(&mut self, vector: Vec<f32>) -> NodeId {
        let id = self.next_id; self.next_id += 1;
        self.vectors.push(vector.clone());
        let level = self.random_level();
        if level >= self.layers.len() { self.layers.resize_with(level + 1, AHashMap::new); }
        for l in 0..=level { self.layers[l].insert(id, Vec::new()); }
        match self.entry_point {
            None => { self.entry_point = Some(id); self.max_level = level; return id; }
            Some(entry) => {
                let mut current = entry;
                for l in (level + 1..=self.max_level).rev() { current = self.greedy_search(l, current, &vector); }
                let top = level.min(self.max_level);
                for l in (0..=top).rev() {
                    let candidates = self.search_layer(l, current, &vector, self.ef_construction);
                    let m_max = if l == 0 { self.m0 } else { self.m };
                    let neighbors: Vec<NodeId> = candidates.iter().take(m_max).map(|(n, _)| *n).collect();
                    if let Some(&nearest) = neighbors.first() { current = nearest; }
                    self.layers[l].insert(id, neighbors.clone());
                    for &n in &neighbors {
                        if let Some(adj) = self.layers[l].get_mut(&n) {
                            adj.push(id);
                            if adj.len() > m_max {
                                let n_vec = self.vectors[n as usize].clone();
                                let mut scored: Vec<(NodeId, f32)> = adj.iter()
                                    .filter_map(|&c| { if c == n { None } else {
                                        let cv = self.vectors[c as usize].clone();
                                        Some((c, (self.distance)(&n_vec, &cv))) } }).collect();
                                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                                adj.clear(); adj.extend(scored.into_iter().take(m_max).map(|(c, _)| c));
                            }
                        }
                    }
                }
            }
        }
        if level > self.max_level { self.max_level = level; self.entry_point = Some(id); }
        id
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        let entry = match self.entry_point { Some(e) => e, None => return Vec::new() };
        let mut current = entry;
        for l in (1..=self.max_level).rev() { current = self.greedy_search(l, current, query); }
        self.search_layer(0, current, query, self.ef_search.max(k)).into_iter().take(k).collect()
    }

    #[must_use] pub fn len(&self) -> usize { self.vectors.len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.vectors.is_empty() }

    fn random_level(&self) -> usize {
        // Standard HNSW level assignment: level = floor(-ln(uniform(0,1)) * ml)
        let u = next_random_f64().max(f64::MIN_POSITIVE); // avoid ln(0)
        let level = -(u.ln()) * self.ml;
        level.floor() as usize
    }

    fn greedy_search(&self, layer: usize, entry: NodeId, query: &[f32]) -> NodeId {
        let mut current = entry;
        let mut current_dist = (self.distance)(&self.vectors[current as usize], query);
        loop {
            let mut improved = false;
            if let Some(neighbors) = self.layers.get(layer).and_then(|l| l.get(&current)) {
                for &n in neighbors {
                    let d = (self.distance)(&self.vectors[n as usize], query);
                    if d < current_dist { current_dist = d; current = n; improved = true; }
                }
            }
            if !improved { break; }
        }
        current
    }

    fn search_layer(&self, layer: usize, entry: NodeId, query: &[f32], ef: usize) -> Vec<(NodeId, f32)> {
        use std::collections::{BinaryHeap, HashSet};
        let mut visited: HashSet<NodeId> = HashSet::new(); visited.insert(entry);
        let entry_dist = (self.distance)(&self.vectors[entry as usize], query);
        let mut candidates: BinaryHeap<(Reverse<OrdF>, NodeId)> = BinaryHeap::new();
        candidates.push((Reverse(OrdF(entry_dist)), entry));
        let mut results: BinaryHeap<(OrdF, NodeId)> = BinaryHeap::new();
        results.push((OrdF(entry_dist), entry));
        while let Some((_, curr)) = candidates.pop() {
            let curr_dist = (self.distance)(&self.vectors[curr as usize], query);
            let worst = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);
            if curr_dist > worst && results.len() >= ef { break; }
            if let Some(neighbors) = self.layers.get(layer).and_then(|l| l.get(&curr)) {
                for &n in neighbors {
                    if visited.contains(&n) { continue; }
                    visited.insert(n);
                    let d = (self.distance)(&self.vectors[n as usize], query);
                    let worst = results.peek().map(|(dd, _)| dd.0).unwrap_or(f32::INFINITY);
                    if d < worst || results.len() < ef {
                        candidates.push((Reverse(OrdF(d)), n));
                        results.push((OrdF(d), n));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }
        let mut out: Vec<(NodeId, f32)> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out
    }
}

#[derive(Clone, Copy, PartialEq)]
struct OrdF(f32);
// Eq is implemented manually because f32 doesn't implement Eq (due to NaN).
// Our Ord implementation treats NaN as equal to everything, which is safe
// for HNSW distance comparisons.
impl Eq for OrdF {}
impl PartialOrd for OrdF { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for OrdF { fn cmp(&self, o: &Self) -> Ordering { self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal) } }
use std::cmp::Reverse;

fn next_random_f64() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering as AO};
    static S: AtomicU64 = AtomicU64::new(0x9E3779B97F4A7C15);
    let mut s = S.load(AO::Relaxed); if s == 0 { s = 0x9E3779B97F4A7C15; }
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; S.store(s, AO::Relaxed);
    (s as f64) / (u64::MAX as f64)
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { 1.0 } else { 1.0 - dot / (na * nb) }
}

pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn hnsw_basic() {
        let mut idx = HnswIndex::new(cosine_distance, 8, 64, 32);
        idx.insert(vec![1.0, 0.0, 0.0]); idx.insert(vec![0.0, 1.0, 0.0]); idx.insert(vec![0.0, 0.0, 1.0]);
        let r = idx.search(&[1.0, 0.0, 0.0], 1); assert!(!r.is_empty()); assert!(r[0].1 < 0.01);
    }
    #[test] fn hnsw_many() {
        let mut idx = HnswIndex::new(l2_distance, 8, 64, 32);
        for i in 0..10 { for j in 0..10 { idx.insert(vec![i as f32, j as f32]); } }
        let r = idx.search(&[5.0, 5.0], 3); assert_eq!(r.len(), 3);
        for (_, d) in &r { assert!(*d < 2.0); }
    }
    #[test] fn hnsw_empty() { let idx: HnswIndex<_> = HnswIndex::new(cosine_distance, 8, 64, 32); assert!(idx.search(&[1.0], 5).is_empty()); assert!(idx.is_empty()); }
    #[test] fn cosine_orthogonal() { assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-5); }
    #[test] fn l2_basic() { assert!((l2_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-5); }
}
