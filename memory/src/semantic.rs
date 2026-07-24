//! L3 Semantic Memory.
//!
//! Stores facts, concepts, and their relationships. Backed by an HNSW index
//! for similarity search and a graph for relationships. In production this
//! layer is also mirrored to PostgreSQL + pgvector for durability.

use parking_lot::RwLock;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::embedding::SharedEmbeddingModel;
use crate::hnsw::HnswIndex;
use crate::object::{Importance, MemoryId, MemoryObject};

/// A semantic fact or concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Memory envelope.
    pub object: MemoryObject<serde_json::Value>,
    /// Human-readable label.
    pub label: String,
    /// Optional aliases.
    pub aliases: Vec<String>,
}

/// A typed edge between two concepts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEdge {
    /// Edge kind: "is_a", "has_a", "related_to", "synonym_of", etc.
    pub kind: String,
    /// Weight in `[0.0, 1.0]`.
    pub weight: f32,
}

/// L3 semantic memory: a concept graph plus a similarity index.
pub struct SemanticMemory {
    index: Arc<HnswIndex>,
    graph: RwLock<DiGraph<Concept, ConceptEdge>>,
    id_to_node: RwLock<HashMap<MemoryId, NodeIndex>>,
    embedder: SharedEmbeddingModel,
}

impl SemanticMemory {
    /// Create a new semantic memory.
    #[must_use]
    pub fn new(embedder: SharedEmbeddingModel) -> Self {
        Self {
            index: Arc::new(HnswIndex::default_config()),
            graph: RwLock::new(DiGraph::new()),
            id_to_node: RwLock::new(HashMap::new()),
            embedder,
        }
    }

    /// Add a concept to semantic memory.
    pub fn add_concept(&self, label: impl Into<String>, content: serde_json::Value) -> MemoryId {
        let label = label.into();
        let embedding = self.embedder.embed_text(&label);
        let mut object = MemoryObject::new(crate::object::MemoryType::Semantic, content, embedding.clone());
        object.importance = Importance(0.8);
        let id = object.id;
        let concept = Concept { object, label, aliases: Vec::new() };

        let mut graph = self.graph.write();
        let node = graph.add_node(concept);
        self.id_to_node.write().insert(id, node);
        let _ = self.index.insert(id, embedding);
        id
    }

    /// Add an alias to an existing concept.
    pub fn add_alias(&self, id: MemoryId, alias: impl Into<String>) -> bool {
        let id_map = self.id_to_node.read();
        let Some(node) = id_map.get(&id).copied() else { return false };
        drop(id_map);
        let mut graph = self.graph.write();
        if let Some(c) = graph.node_weight_mut(node) {
            c.aliases.push(alias.into());
            true
        } else {
            false
        }
    }

    /// Link two concepts with an edge.
    pub fn link(&self, from: MemoryId, to: MemoryId, edge: ConceptEdge) -> bool {
        let id_map = self.id_to_node.read();
        let Some(from_node) = id_map.get(&from).copied() else { return false };
        let Some(to_node) = id_map.get(&to).copied() else { return false };
        drop(id_map);
        let mut graph = self.graph.write();
        graph.add_edge(from_node, to_node, edge);
        true
    }

    /// Find concepts similar to `query`.
    #[must_use]
    pub fn find_similar(&self, query: &str, k: usize) -> Vec<Concept> {
        let embedding = self.embedder.embed_text(query);
        match self.index.search(&embedding, k) {
            Ok(results) => {
                let graph = self.graph.read();
                let id_map = self.id_to_node.read();
                results.into_iter()
                    .filter_map(|(id, _)| id_map.get(&id).and_then(|n| graph.node_weight(*n).cloned()))
                    .collect()
            }
            Err(_) => Vec::new(),
        }
    }

    /// Get all neighbors of a concept along edges of `kind`.
    #[must_use]
    pub fn neighbors(&self, id: MemoryId, kind: &str) -> Vec<Concept> {
        let id_map = self.id_to_node.read();
        let Some(node) = id_map.get(&id).copied() else { return Vec::new() };
        drop(id_map);
        let graph = self.graph.read();
        graph.edges(node)
            .filter(|e| e.weight().kind == kind)
            .filter_map(|e| graph.node_weight(e.target()).cloned())
            .collect()
    }

    /// Number of concepts in semantic memory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.read().node_count()
    }

    /// Returns `true` if no concepts are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.read().node_count() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingModelStub;

    #[test]
    fn add_and_query_concepts() {
        let emb: SharedEmbeddingModel = Arc::new(EmbeddingModelStub::new(64));
        let mem = SemanticMemory::new(emb);
        let dog = mem.add_concept("dog", serde_json::json!({"type": "mammal"}));
        let cat = mem.add_concept("cat", serde_json::json!({"type": "mammal"}));
        let mammal = mem.add_concept("mammal", serde_json::json!({"type": "class"}));
        mem.link(dog, mammal, ConceptEdge { kind: "is_a".into(), weight: 1.0 });
        mem.link(cat, mammal, ConceptEdge { kind: "is_a".into(), weight: 1.0 });

        let results = mem.find_similar("dog", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].label, "dog");

        let mammal_neighbors = mem.neighbors(mammal, "is_a");
        // Edges point from dog/cat to mammal, so mammal has no outgoing is_a edges.
        assert_eq!(mammal_neighbors.len(), 0);

        let dog_parents = mem.neighbors(dog, "is_a");
        assert_eq!(dog_parents.len(), 1);
        assert_eq!(dog_parents[0].label, "mammal");
    }
}
