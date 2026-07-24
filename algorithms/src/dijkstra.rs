//! Dijkstra's shortest-path algorithm.

use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::astar::{Graph, NodeId};

/// Result of a Dijkstra search.
#[derive(Debug, Clone)]
pub struct DijkstraResult {
    /// The path from start to goal.
    pub path: Vec<NodeId>,
    /// Total path cost.
    pub cost: f64,
    /// Number of nodes popped from the priority queue.
    pub nodes_expanded: usize,
    /// Full distance map (all reachable nodes → distance from start).
    pub distances: AHashMap<NodeId, f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct HeapEntry {
    cost: f64,
    node: NodeId,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Run Dijkstra's algorithm from `start` to `goal`.
///
/// Returns `None` if `goal` is unreachable from `start`.
/// The `distances` field in the result contains the full single-source
/// shortest-path tree from `start`.
pub fn dijkstra(graph: &Graph, start: NodeId, goal: NodeId) -> Option<DijkstraResult> {
    if start == goal {
        let mut distances = AHashMap::new();
        distances.insert(start, 0.0);
        return Some(DijkstraResult {
            path: vec![start],
            cost: 0.0,
            nodes_expanded: 0,
            distances,
        });
    }

    let mut open_set: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut distances: AHashMap<NodeId, f64> = AHashMap::new();
    let mut came_from: AHashMap<NodeId, NodeId> = AHashMap::new();
    let mut nodes_expanded = 0usize;

    distances.insert(start, 0.0);
    open_set.push(HeapEntry { cost: 0.0, node: start });

    while let Some(HeapEntry { node: current, cost: dist }) = open_set.pop() {
        // Skip stale entries.
        if let Some(&known) = distances.get(&current) {
            if dist > known {
                continue;
            }
        }
        nodes_expanded += 1;

        if current == goal {
            let mut path = vec![goal];
            let mut cursor = goal;
            while let Some(&prev) = came_from.get(&cursor) {
                path.push(prev);
                cursor = prev;
            }
            path.reverse();
            let cost = *distances.get(&goal).unwrap_or(&f64::INFINITY);
            return Some(DijkstraResult {
                path,
                cost,
                nodes_expanded,
                distances,
            });
        }

        let neighbors = match graph.get(&current) {
            Some(n) => n,
            None => continue,
        };
        for &(neighbor, weight) in neighbors {
            let alt = dist + weight;
            let known = distances.get(&neighbor).copied().unwrap_or(f64::INFINITY);
            if alt < known {
                distances.insert(neighbor, alt);
                came_from.insert(neighbor, current);
                open_set.push(HeapEntry {
                    cost: alt,
                    node: neighbor,
                });
            }
        }
    }

    None
}

/// Run Dijkstra to compute distances to *all* reachable nodes from `start`.
#[must_use]
pub fn dijkstra_all(graph: &Graph, start: NodeId) -> AHashMap<NodeId, f64> {
    let mut open_set: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut distances: AHashMap<NodeId, f64> = AHashMap::new();
    distances.insert(start, 0.0);
    open_set.push(HeapEntry { cost: 0.0, node: start });

    while let Some(HeapEntry { node: current, cost: dist }) = open_set.pop() {
        if let Some(&known) = distances.get(&current) {
            if dist > known {
                continue;
            }
        }
        let neighbors = match graph.get(&current) {
            Some(n) => n,
            None => continue,
        };
        for &(neighbor, weight) in neighbors {
            let alt = dist + weight;
            let known = distances.get(&neighbor).copied().unwrap_or(f64::INFINITY);
            if alt < known {
                distances.insert(neighbor, alt);
                open_set.push(HeapEntry {
                    cost: alt,
                    node: neighbor,
                });
            }
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dijkstra_finds_shortest_path() {
        let mut g: Graph = AHashMap::new();
        g.insert(0, vec![(1, 4.0), (2, 1.0)]);
        g.insert(1, vec![(3, 1.0)]);
        g.insert(2, vec![(1, 2.0), (3, 5.0)]);
        g.insert(3, vec![]);
        let r = dijkstra(&g, 0, 3).unwrap();
        // 0 → 2 → 1 → 3 = 1 + 2 + 1 = 4
        assert_eq!(r.path, vec![0, 2, 1, 3]);
        assert!((r.cost - 4.0).abs() < 1e-9);
    }

    #[test]
    fn dijkstra_all_computes_full_tree() {
        let mut g: Graph = AHashMap::new();
        g.insert(0, vec![(1, 1.0), (2, 4.0)]);
        g.insert(1, vec![(2, 2.0), (3, 5.0)]);
        g.insert(2, vec![(3, 1.0)]);
        g.insert(3, vec![]);
        let dist = dijkstra_all(&g, 0);
        assert!((dist[&0] - 0.0).abs() < 1e-9);
        assert!((dist[&1] - 1.0).abs() < 1e-9);
        assert!((dist[&2] - 3.0).abs() < 1e-9);
        assert!((dist[&3] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn dijkstra_unreachable_returns_none() {
        let mut g: Graph = AHashMap::new();
        g.insert(0, vec![(1, 1.0)]);
        g.insert(2, vec![]); // unreachable
        let r = dijkstra(&g, 0, 2);
        assert!(r.is_none());
    }
}
