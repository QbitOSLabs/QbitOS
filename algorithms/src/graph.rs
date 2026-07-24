//! Graph algorithms: connected components, topological sort, SCC (Tarjan).

use crate::astar::Graph;
use ahash::{AHashMap, AHashSet};
use std::collections::VecDeque;

/// Find connected components in an undirected graph.
///
/// `graph` is treated as undirected: if (u, v) is an edge, (v, u) is implicitly too.
/// Nodes that appear only as destinations are still included.
#[must_use]
pub fn connected_components(graph: &Graph) -> Vec<Vec<crate::astar::NodeId>> {
    // Build an undirected adjacency map on the fly.
    let mut undirected: AHashMap<crate::astar::NodeId, Vec<crate::astar::NodeId>> = AHashMap::new();
    for (&u, edges) in graph.iter() {
        undirected.entry(u).or_default();
        for &(v, _) in edges {
            undirected.entry(u).or_default().push(v);
            undirected.entry(v).or_default().push(u);
        }
    }
    let mut visited = AHashSet::new();
    let mut components = Vec::new();
    let nodes: Vec<_> = undirected.keys().copied().collect();
    for &start in &nodes {
        if visited.contains(&start) {
            continue;
        }
        let mut comp = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(node) = queue.pop_front() {
            comp.push(node);
            if let Some(neighbors) = undirected.get(&node) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        components.push(comp);
    }
    components
}

/// Topological sort using Kahn's algorithm (BFS).
///
/// `graph` is treated as a directed adjacency list. Returns `None` if the
/// graph contains a cycle.
#[must_use]
pub fn topological_sort(graph: &Graph) -> Option<Vec<crate::astar::NodeId>> {
    let mut in_degree: AHashMap<crate::astar::NodeId, usize> = AHashMap::new();
    for node in graph.keys() {
        in_degree.entry(*node).or_insert(0);
    }
    for edges in graph.values() {
        for &(dst, _) in edges {
            *in_degree.entry(dst).or_insert(0) += 1;
        }
    }
    let mut queue: VecDeque<crate::astar::NodeId> = in_degree
        .iter()
        .filter(|&(_, &d)| d == 0)
        .map(|(&n, _)| n)
        .collect();
    let mut order = Vec::new();
    while let Some(node) = queue.pop_front() {
        order.push(node);
        if let Some(neighbors) = graph.get(&node) {
            for &(neighbor, _) in neighbors {
                if let Some(d) = in_degree.get_mut(&neighbor) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    if order.len() == in_degree.len() {
        Some(order)
    } else {
        None
    }
}

/// Strongly connected components using Tarjan's algorithm.
#[must_use]
pub fn scc(graph: &Graph) -> Vec<Vec<crate::astar::NodeId>> {
    let mut state = TarjanState {
        index: 0,
        indices: AHashMap::new(),
        lowlink: AHashMap::new(),
        on_stack: AHashSet::new(),
        stack: Vec::new(),
        sccs: Vec::new(),
    };
    let nodes: Vec<_> = graph.keys().copied().collect();
    for v in nodes {
        if !state.indices.contains_key(&v) {
            strongconnect(graph, v, &mut state);
        }
    }
    state.sccs
}

struct TarjanState {
    index: usize,
    indices: AHashMap<crate::astar::NodeId, usize>,
    lowlink: AHashMap<crate::astar::NodeId, usize>,
    on_stack: AHashSet<crate::astar::NodeId>,
    stack: Vec<crate::astar::NodeId>,
    sccs: Vec<Vec<crate::astar::NodeId>>,
}

fn strongconnect(graph: &Graph, v: crate::astar::NodeId, state: &mut TarjanState) {
    state.indices.insert(v, state.index);
    state.lowlink.insert(v, state.index);
    state.index += 1;
    state.stack.push(v);
    state.on_stack.insert(v);

    if let Some(neighbors) = graph.get(&v) {
        for &(w, _) in neighbors {
            if !state.indices.contains_key(&w) {
                strongconnect(graph, w, state);
                let wl = *state.lowlink.get(&w).unwrap();
                let vl = *state.lowlink.get(&v).unwrap();
                state.lowlink.insert(v, vl.min(wl));
            } else if state.on_stack.contains(&w) {
                let wi = *state.indices.get(&w).unwrap();
                let vl = *state.lowlink.get(&v).unwrap();
                state.lowlink.insert(v, vl.min(wi));
            }
        }
    }

    if state.lowlink.get(&v) == state.indices.get(&v) {
        let mut component = Vec::new();
        loop {
            let w = state.stack.pop().unwrap();
            state.on_stack.remove(&w);
            component.push(w);
            if w == v {
                break;
            }
        }
        state.sccs.push(component);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astar::NodeId;

    fn build_graph(edges: &[(NodeId, NodeId)]) -> Graph {
        let mut g: Graph = AHashMap::new();
        for &(u, v) in edges {
            g.entry(u).or_default().push((v, 1.0));
            g.entry(v).or_default();
        }
        g
    }

    #[test]
    fn connected_components_finds_two() {
        let g = build_graph(&[(0, 1), (1, 2), (3, 4)]);
        let mut comps = connected_components(&g);
        comps.iter_mut().for_each(|c| c.sort_unstable());
        comps.sort_unstable();
        assert_eq!(comps, vec![vec![0, 1, 2], vec![3, 4]]);
    }

    #[test]
    fn topological_sort_dag() {
        let g = build_graph(&[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let order = topological_sort(&g).unwrap();
        let pos: AHashMap<_, _> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        assert!(pos[&0] < pos[&1]);
        assert!(pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&3]);
        assert!(pos[&2] < pos[&3]);
    }

    #[test]
    fn topological_sort_cycle_returns_none() {
        let g = build_graph(&[(0, 1), (1, 2), (2, 0)]);
        assert!(topological_sort(&g).is_none());
    }

    #[test]
    fn scc_finds_cycle() {
        let g = build_graph(&[(0, 1), (1, 2), (2, 0), (2, 3)]);
        let sccs = scc(&g);
        // The cycle 0→1→2→0 should form one SCC.
        assert!(sccs.iter().any(|s| s.contains(&0) && s.contains(&1) && s.contains(&2)));
        // Node 3 should be its own SCC.
        assert!(sccs.iter().any(|s| s == &vec![3]));
    }
}
