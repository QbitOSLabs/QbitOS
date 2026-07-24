//! Memory object envelope.
//!
//! Every memory in QbitOS — regardless of layer — is represented as a
//! [`MemoryObject`]. This uniform envelope is what allows the consolidation
//! engine to move memories between layers without re-encoding them.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

use crate::embedding::Embedding;

/// Unique identifier for a memory object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// Mint a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Which memory layer an object lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// L0 sensory memory.
    Sensory,
    /// L1 working memory.
    Working,
    /// L2 episodic memory.
    Episodic,
    /// L3 semantic memory.
    Semantic,
    /// L4 procedural memory (skills).
    Procedural,
    /// L5 meta memory.
    Meta,
}

/// Importance in `[0.0, 1.0]`. Higher values survive consolidation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Importance(pub f32);

impl Default for Importance {
    fn default() -> Self {
        Self(0.5)
    }
}

impl Importance {
    /// Decay importance by a factor in `[0.0, 1.0]`.
    #[must_use]
    pub fn decay(self, factor: f32) -> Self {
        Self((self.0 * factor).clamp(0.0, 1.0))
    }

    /// Boost importance by a delta, clamped to 1.0.
    #[must_use]
    pub fn boost(self, delta: f32) -> Self {
        Self((self.0 + delta).clamp(0.0, 1.0))
    }
}

/// A typed relationship between two memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRelationship {
    /// Causal: this memory caused another.
    Caused(MemoryId),
    /// Temporal: this memory happened before another.
    Before(MemoryId),
    /// Taxonomic: this memory is a subtype of another.
    SubtypeOf(MemoryId),
    /// Associative: this memory is associated with another.
    Associated(MemoryId, f32),
    /// Derived: this memory was derived from another.
    DerivedFrom(MemoryId),
    /// Contradicts: this memory contradicts another.
    Contradicts(MemoryId),
}

/// A single causal link in a memory's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    /// The cause event id.
    pub cause: Uuid,
    /// The effect event id.
    pub effect: Uuid,
    /// Strength of the causal link in `[0.0, 1.0]`.
    pub strength: f32,
    /// Free-form description.
    pub description: String,
}

/// The uniform envelope for every memory object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryObject<T = serde_json::Value> {
    /// Unique id.
    pub id: MemoryId,
    /// Which layer this object lives in.
    pub layer: MemoryType,
    /// Arbitrary content.
    pub content: T,
    /// Vector embedding of `content`.
    pub embedding: Embedding,
    /// Importance in `[0.0, 1.0]`.
    pub importance: Importance,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// When the memory was created.
    pub created_at: DateTime<Utc>,
    /// When the memory was last accessed.
    pub last_accessed_at: DateTime<Utc>,
    /// Number of times this memory has been retrieved.
    pub usage_count: u32,
    /// Relationships to other memories.
    pub relationships: Vec<MemoryRelationship>,
    /// Causal links.
    pub causal_links: Vec<CausalLink>,
    /// Free-form metadata.
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl<T> MemoryObject<T> {
    /// Create a new memory object with default metadata.
    #[must_use]
    pub fn new(layer: MemoryType, content: T, embedding: Embedding) -> Self {
        let now = Utc::now();
        Self {
            id: MemoryId::new(),
            layer,
            content,
            embedding,
            importance: Importance::default(),
            confidence: 1.0,
            created_at: now,
            last_accessed_at: now,
            usage_count: 0,
            relationships: Vec::new(),
            causal_links: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Mark this memory as accessed now.
    pub fn touch(&mut self) {
        self.last_accessed_at = Utc::now();
        self.usage_count = self.usage_count.saturating_add(1);
    }

    /// Convert the content type using a closure.
    #[must_use]
    pub fn map_content<U, F>(self, f: F) -> MemoryObject<U>
    where
        F: FnOnce(T) -> U,
    {
        MemoryObject {
            id: self.id,
            layer: self.layer,
            content: f(self.content),
            embedding: self.embedding,
            importance: self.importance,
            confidence: self.confidence,
            created_at: self.created_at,
            last_accessed_at: self.last_accessed_at,
            usage_count: self.usage_count,
            relationships: self.relationships,
            causal_links: self.causal_links,
            metadata: self.metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_object_round_trip() {
        let emb = Embedding::new(vec![0.1, 0.2, 0.3]);
        let mut m = MemoryObject::new(MemoryType::Episodic, json_content(), emb);
        m.importance = Importance(0.9);
        m.touch();
        let serialized = serde_json::to_string(&m).unwrap();
        let _: MemoryObject = serde_json::from_str(&serialized).unwrap();
    }

    fn json_content() -> serde_json::Value {
        serde_json::json!({"event": "user_login", "user": "alice"})
    }
}
