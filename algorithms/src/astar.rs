//! A* search algorithm with configurable heuristics.

use ahash::AHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Identifier for a node in the graph. Generic over any hashable + eq type.
pub type NodeId = u64;

/// A node in the A* search result path.
#[derive(Debug, Clone)]
pub struct AStarNode {
    /// The node identifier.
    pub id: NodeId,
    /// The path cost from start to this node.
    pub g_score: f64,
    /// The estimated total cost (g + heuristic).
    pub f_score: f64,
}

/// Result of an A* search.
#[derive(Debug, Clone)]
pub struct AStarResult {
    /// The path from start to goal (inclusive).
    pub path: Vec<NodeId>,
    /// The total cost of the path.
    pub cost: f64,
    /// Number of nodes expanded.
    pub nodes_expanded: usize,
}

/// Edge in a weighted directed graph. Tuple of (destination, weight).
pub type Edge = (NodeId, f64);

/// Adjacency list representation of a graph.
pub type Graph = AHashMap<NodeId, Vec<Edge>>;

/// Heuristic function type. Takes (current_node, goal_node) and returns an estimate.
pub type Heuristic = Box<dyn Fn(NodeId, NodeId) -> f64 + Send + Sync>;

#[derive(Debug, Clone, PartialEq)]
struct HeapEntry {
    f_score: f64,
    node: NodeId,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so reverse the ordering for min-heap.
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Run A* search from `start` to `goal` on the given graph.
///
/// `heuristic` is admissible if it never overestimates the true cost.
/// An admissible heuristic guarantees optimality.
///
/// # Arguments
/// * `graph` — adjacency list
/// * `start` — starting node
/// * `goal` — goal node
/// * `heuristic` — function estimating distance from a node to the goal
///
/// # Returns
/// `Some(AStarResult)` if a path is found, `None` otherwise.
pub fn astar_search(
    graph: &Graph,
    start: NodeId,
    goal: NodeId,
    heuristic: Heuristic,
) -> Option<AStarResult> {
    if start == goal {
        return Some(AStarResult {
            path: vec![start],
            cost: 0.0,
            nodes_expanded: 0,
        });
    }

    let mut open_set: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut g_scores: AHashMap<NodeId, f64> = AHashMap::new();
    let mut came_from: AHashMap<NodeId, NodeId> = AHashMap::new();
    let mut closed: ahash::AHashSet<NodeId> = ahash::AHashSet::new();
    let mut nodes_expanded = 0usize;

    g_scores.insert(start, 0.0);
    open_set.push(HeapEntry {
        f_score: heuristic(start, goal),
        node: start,
    });

    while let Some(HeapEntry { node: current, .. }) = open_set.pop() {
        if current == goal {
            // Reconstruct path.
            let mut path = vec![goal];
            let mut cursor = goal;
            while let Some(&prev) = came_from.get(&cursor) {
                path.push(prev);
                cursor = prev;
            }
            path.reverse();
            let cost = g_scores.get(&goal).copied().unwrap_or(f64::INFINITY);
            return Some(AStarResult {
                path,
                cost,
                nodes_expanded,
            });
        }

        if closed.contains(&current) {
            continue;
        }
        closed.insert(current);
        nodes_expanded += 1;

        let current_g = g_scores.get(&current).copied().unwrap_or(f64::INFINITY);
        let neighbors = match graph.get(&current) {
            Some(n) => n,
            None => continue,
        };

        for &(neighbor, weight) in neighbors {
            if closed.contains(&neighbor) {
                continue;
            }
            let tentative_g = current_g + weight;
            let known_g = g_scores.get(&neighbor).copied().unwrap_or(f64::INFINITY);
            if tentative_g < known_g {
                came_from.insert(neighbor, current);
                g_scores.insert(neighbor, tentative_g);
                let f = tentative_g + heuristic(neighbor, goal);
                open_set.push(HeapEntry {
                    f_score: f,
                    node: neighbor,
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> Graph {
        let mut g: Graph = AHashMap::new();
        // Simple grid: 0-1-2-3, 0-4-5-3
        // 0 -- 1 (1), 1 -- 2 (1), 2 -- 3 (1)
        // 0 -- 4 (1), 4 -- 5 (1), 5 -- 3 (1)
        // 0 -- 5 (4) (a longer path)
        g.insert(0, vec![(1, 1.0), (4, 1.0), (5, 4.0)]);
        g.insert(1, vec![(2, 1.0)]);
        g.insert(2, vec![(3, 1.0)]);
        g.insert(4, vec![(5, 1.0)]);
        g.insert(5, vec![(3, 1.0)]);
        g
    }

    #[test]
    fn astar_finds_shortest_path() {
        let g = make_graph();
        // Zero heuristic → Dijkstra
        let result = astar_search(&g, 0, 3, Box::new(|_, _| 0.0)).unwrap();
        assert_eq!(result.path, vec![0, 1, 2, 3]);
        assert!((result.cost - 3.0).abs() < 1e-9);
    }

    #[test]
    fn astar_uses_heuristic_to_prune() {
        let g = make_graph();
        // Heuristic: estimate distance to 3 as (3 - node_id) as float.
        let h: Heuristic = Box::new(|node, goal| (goal as f64 - node as f64).abs());
        let result = astar_search(&g, 0, 3, h).unwrap();
        assert_eq!(result.path, vec![0, 1, 2, 3]);
        assert!((result.cost - 3.0).abs() < 1e-9);
    }

    #[test]
    fn astar_returns_none_when_unreachable() {
        let mut g: Graph = AHashMap::new();
        g.insert(0, vec![(1, 1.0)]);
        g.insert(2, vec![(3, 1.0)]); // 2-3 disconnected from 0-1
        let result = astar_search(&g, 0, 3, Box::new(|_, _| 0.0));
        assert!(result.is_none());
    }

    #[test]
    fn astar_start_equals_goal() {
        let g = make_graph();
        let result = astar_search(&g, 3, 3, Box::new(|_, _| 0.0)).unwrap();
        assert_eq!(result.path, vec![3]);
        assert!(result.cost.abs() < 1e-9);
    }
}
