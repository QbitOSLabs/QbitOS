//! Knowledge graph.
//!
//! A directed graph of (subject, predicate, object) triples. Backed by
//! `petgraph` for in-process use; can also be mirrored to PostgreSQL +
//! Apache AGE for persistence and SPARQL-style queries.

use ahash::AHashMap;
use parking_lot::RwLock;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// A unique identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    /// Mint a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// A knowledge graph triple: (subject, predicate, object).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    /// Subject node id.
    pub subject: NodeId,
    /// Predicate (e.g. "is_a", "has_a", "related_to").
    pub predicate: String,
    /// Object node id.
    pub object: NodeId,
    /// Weight in `[0.0, 1.0]`.
    pub weight: f32,
}

/// The knowledge graph.
pub struct KnowledgeGraph {
    graph: RwLock<DiGraph<String, Triple>>,
    node_index: RwLock<AHashMap<NodeId, NodeIndex>>,
    label_index: RwLock<AHashMap<String, NodeId>>,
}

impl KnowledgeGraph {
    /// Create a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: RwLock::new(DiGraph::new()),
            node_index: RwLock::new(AHashMap::new()),
            label_index: RwLock::new(AHashMap::new()),
        }
    }

    /// Add a node with a label.
    pub fn add_node(&self, label: impl Into<String>) -> NodeId {
        let label = label.into();
        // Check if a node with this label already exists
        if let Some(id) = self.label_index.read().get(&label).copied() {
            return id;
        }
        let id = NodeId::new();
        let idx = self.graph.write().add_node(label.clone());
        self.node_index.write().insert(id, idx);
        self.label_index.write().insert(label, id);
        id
    }

    /// Add a triple (edge).
    pub fn add_triple(&self, triple: Triple) -> bool {
        let mut graph = self.graph.write();
        let node_index = self.node_index.read();
        let Some(&s_idx) = node_index.get(&triple.subject) else { return false };
        let Some(&o_idx) = node_index.get(&triple.object) else { return false };
        drop(node_index);
        graph.add_edge(s_idx, o_idx, triple);
        true
    }

    /// Convenience: add (subject, predicate, object) by label.
    pub fn add(&self, subject: impl Into<String>, predicate: impl Into<String>, object: impl Into<String>) -> Triple {
        let s = self.add_node(subject);
        let o = self.add_node(object);
        let triple = Triple { subject: s, predicate: predicate.into(), object: o, weight: 1.0 };
        self.add_triple(triple.clone());
        triple
    }

    /// Find all triples with the given subject.
    #[must_use]
    pub fn outgoing(&self, subject: NodeId) -> Vec<Triple> {
        let graph = self.graph.read();
        let node_index = self.node_index.read();
        let Some(&idx) = node_index.get(&subject) else { return Vec::new() };
        graph.edges(idx).map(|e| e.weight().clone()).collect()
    }

    /// Find all triples with the given object.
    #[must_use]
    pub fn incoming(&self, object: NodeId) -> Vec<Triple> {
        let graph = self.graph.read();
        let node_index = self.node_index.read();
        let Some(&idx) = node_index.get(&object) else { return Vec::new() };
        graph.edges_directed(idx, petgraph::Direction::Incoming).map(|e| e.weight().clone()).collect()
    }

    /// Look up a node by label.
    #[must_use]
    pub fn lookup(&self, label: &str) -> Option<NodeId> {
        self.label_index.read().get(label).copied()
    }

    /// Get the label of a node.
    #[must_use]
    pub fn label(&self, id: NodeId) -> Option<String> {
        let graph = self.graph.read();
        let node_index = self.node_index.read();
        let idx = node_index.get(&id)?;
        graph.node_weight(*idx).cloned()
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.read().node_count()
    }

    /// Number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.read().edge_count()
    }

    /// Returns all triples in the graph, with labels resolved.
    /// Used by the persistence layer to snapshot the full knowledge graph.
    #[must_use]
    pub fn all_triples(&self) -> Vec<(String, String, String)> {
        let graph = self.graph.read();
        let node_index = self.node_index.read();
        // Build a reverse map: NodeIndex → NodeId
        let mut idx_to_id: std::collections::HashMap<NodeIndex, NodeId> = std::collections::HashMap::new();
        for (id, &idx) in node_index.iter() {
            idx_to_id.insert(idx, *id);
        }
        let mut triples = Vec::new();
        for edge_ref in graph.edge_references() {
            let triple = edge_ref.weight();
            let subject_id = idx_to_id.get(&edge_ref.source()).copied();
            let object_id = idx_to_id.get(&edge_ref.target()).copied();
            let subject_label = subject_id
                .and_then(|id| self.label(id))
                .unwrap_or_default();
            let object_label = object_id
                .and_then(|id| self.label(id))
                .unwrap_or_default();
            triples.push((subject_label, triple.predicate.clone(), object_label));
        }
        triples
    }

    /// Returns all node labels in the graph.
    #[must_use]
    pub fn all_labels(&self) -> Vec<String> {
        self.label_index.read().keys().cloned().collect()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A shared knowledge graph.
pub type SharedKnowledgeGraph = Arc<KnowledgeGraph>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_query() {
        let kg = KnowledgeGraph::new();
        kg.add("dog", "is_a", "mammal");
        kg.add("cat", "is_a", "mammal");
        kg.add("mammal", "has_a", "spine");
        let dog = kg.lookup("dog").unwrap();
        let outgoing = kg.outgoing(dog);
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].predicate, "is_a");
        let mammal = kg.lookup("mammal").unwrap();
        let incoming = kg.incoming(mammal);
        assert_eq!(incoming.len(), 2);
    }
}
