//! Causal engine.
//!
//! Reasoning about cause and effect. Maintains a causal graph: nodes are
//! events, edges are causal links (with strength and direction). Supports:
//!
//! - Forward inference: "if X happens, what will follow?"
//! - Backward inference: "what caused Y?"
//! - Counterfactual reasoning: "if X had not happened, would Y have happened?"

use ahash::AHashMap;
use parking_lot::RwLock;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// A causal link between two events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// Cause event id.
    pub cause: Uuid,
    /// Effect event id.
    pub effect: Uuid,
    /// Strength in `[0.0, 1.0]`.
    pub strength: f32,
    /// Description.
    pub description: String,
}

/// A causal rule: a template that, when matched, infers a new causal link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRule {
    /// Rule name.
    pub name: String,
    /// Cause pattern (a JSON expression).
    pub cause_pattern: serde_json::Value,
    /// Effect pattern.
    pub effect_pattern: serde_json::Value,
    /// Default strength.
    pub strength: f32,
}

/// The causal graph.
pub struct CausalGraph {
    graph: RwLock<DiGraph<Uuid, CausalLink>>,
    node_index: RwLock<AHashMap<Uuid, NodeIndex>>,
}

impl CausalGraph {
    /// Create a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(DiGraph::new()),
            node_index: RwLock::new(AHashMap::new()),
        }
    }

    /// Add an event node.
    pub fn add_event(&self, event_id: Uuid) {
        let mut graph = self.graph.write();
        let idx = graph.add_node(event_id);
        self.node_index.write().insert(event_id, idx);
    }

    /// Add a causal link.
    pub fn add_link(&self, link: CausalLink) -> bool {
        let mut graph = self.graph.write();
        let mut node_index = self.node_index.write();
        // Ensure both events exist
        let cause_idx = *node_index.entry(link.cause).or_insert_with(|| graph.add_node(link.cause));
        let effect_idx = *node_index.entry(link.effect).or_insert_with(|| graph.add_node(link.effect));
        graph.add_edge(cause_idx, effect_idx, link);
        true
    }

    /// Find all direct effects of an event.
    #[must_use]
    pub fn effects(&self, cause: Uuid) -> Vec<CausalLink> {
        let graph = self.graph.read();
        let node_index = self.node_index.read();
        let Some(&idx) = node_index.get(&cause) else { return Vec::new() };
        graph.edges(idx).map(|e| e.weight().clone()).collect()
    }

    /// Find all direct causes of an event.
    #[must_use]
    pub fn causes(&self, effect: Uuid) -> Vec<CausalLink> {
        let graph = self.graph.read();
        let node_index = self.node_index.read();
        let Some(&idx) = node_index.get(&effect) else { return Vec::new() };
        graph.edges_directed(idx, petgraph::Direction::Incoming).map(|e| e.weight().clone()).collect()
    }

    /// Returns the total number of events (nodes) in the graph.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.node_index.read().len()
    }

    /// Returns the total number of causal links (edges) in the graph.
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.graph.read().edge_count()
    }

    /// Returns `true` if the graph has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_index.read().is_empty()
    }

    /// Returns `true` if `event_id` is registered in the graph.
    #[must_use]
    pub fn contains_event(&self, event_id: Uuid) -> bool {
        self.node_index.read().contains_key(&event_id)
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// The causal reasoner.
pub struct CausalReasoner {
    graph: Arc<CausalGraph>,
    rules: RwLock<Vec<CausalRule>>,
}

impl CausalReasoner {
    /// Create a new reasoner with an empty rule set.
    #[must_use]
    pub fn new(graph: Arc<CausalGraph>) -> Self {
        Self { graph, rules: RwLock::new(Vec::new()) }
    }

    /// Add a rule.
    pub fn add_rule(&self, rule: CausalRule) {
        self.rules.write().push(rule);
    }

    /// Predict the effects of an event (forward inference).
    ///
    /// Walks the causal graph forward up to `depth` hops, accumulating
    /// probability (treating edge strengths as conditional probabilities).
    #[must_use]
    pub fn predict_effects(&self, cause: Uuid, depth: usize) -> Vec<(Uuid, f32)> {
        let mut visited: AHashMap<Uuid, f32> = AHashMap::new();
        visited.insert(cause, 1.0);
        let mut frontier = vec![cause];
        for _ in 0..depth {
            let mut next = Vec::new();
            for node in frontier {
                let prob = visited[&node];
                for link in self.graph.effects(node) {
                    let new_prob = prob * link.strength;
                    let entry = visited.entry(link.effect).or_insert(0.0);
                    if new_prob > *entry {
                        *entry = new_prob;
                        next.push(link.effect);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        visited.remove(&cause);
        let mut results: Vec<_> = visited.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Explain what caused an event (backward inference).
    #[must_use]
    pub fn explain_causes(&self, effect: Uuid, depth: usize) -> Vec<(Uuid, f32)> {
        let mut visited: AHashMap<Uuid, f32> = AHashMap::new();
        visited.insert(effect, 1.0);
        let mut frontier = vec![effect];
        for _ in 0..depth {
            let mut next = Vec::new();
            for node in frontier {
                let prob = visited[&node];
                for link in self.graph.causes(node) {
                    let new_prob = prob * link.strength;
                    let entry = visited.entry(link.cause).or_insert(0.0);
                    if new_prob > *entry {
                        *entry = new_prob;
                        next.push(link.cause);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        visited.remove(&effect);
        let mut results: Vec<_> = visited.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_effects() {
        let graph = Arc::new(CausalGraph::new());
        let reasoner = CausalReasoner::new(Arc::clone(&graph));
        let rain = Uuid::now_v7();
        let wet = Uuid::now_v7();
        let slip = Uuid::now_v7();
        graph.add_link(CausalLink { cause: rain, effect: wet, strength: 0.9, description: "rain wets ground".into() });
        graph.add_link(CausalLink { cause: wet, effect: slip, strength: 0.5, description: "wet ground causes slipping".into() });
        let effects = reasoner.predict_effects(rain, 2);
        // wet should have prob 0.9, slip should have prob 0.9 * 0.5 = 0.45
        let wet_prob = effects.iter().find(|(id, _)| *id == wet).map(|(_, p)| *p).unwrap();
        let slip_prob = effects.iter().find(|(id, _)| *id == slip).map(|(_, p)| *p).unwrap();
        assert!((wet_prob - 0.9).abs() < 1e-6);
        assert!((slip_prob - 0.45).abs() < 1e-6);
    }
}
