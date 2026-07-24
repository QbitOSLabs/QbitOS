//! Visual memory.
//!
//! Stores image embeddings and their metadata so the cognitive kernel can
//! recall what it has seen before. Each entry is indexed by an HNSW vector
//! index (provided by the memory fabric) for fast similarity search.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use qbitos_memory::embedding::Embedding;

/// A visual memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualMemoryEntry {
    /// Unique id.
    pub id: Uuid,
    /// When the image was seen.
    pub seen_at: DateTime<Utc>,
    /// Source: file path, URL, or "camera".
    pub source: String,
    /// Image caption (natural-language description).
    pub caption: String,
    /// Detected objects.
    pub objects: Vec<String>,
    /// Embedding vector.
    pub embedding: Embedding,
    /// Image dimensions.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Importance in `[0.0, 1.0]`.
    pub importance: f32,
}

/// Visual memory: a store of images the system has seen.
pub struct VisualMemory {
    entries: Arc<RwLock<HashMap<Uuid, VisualMemoryEntry>>>,
    embedder: Arc<dyn qbitos_memory::embedding::EmbeddingModel + Send + Sync>,
}

impl VisualMemory {
    /// Create a new visual memory.
    #[must_use]
    pub fn new(embedder: Arc<dyn qbitos_memory::embedding::EmbeddingModel + Send + Sync>) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            embedder,
        }
    }

    /// Record a new visual memory.
    pub fn record(
        &self,
        source: impl Into<String>,
        caption: impl Into<String>,
        objects: Vec<String>,
        width: u32,
        height: u32,
    ) -> Uuid {
        let caption = caption.into();
        let embedding = self.embedder.embed_text(&caption);
        let id = Uuid::now_v7();
        let entry = VisualMemoryEntry {
            id,
            seen_at: Utc::now(),
            source: source.into(),
            caption,
            objects,
            embedding,
            width,
            height,
            importance: 0.5,
        };
        self.entries.write().insert(id, entry);
        id
    }

    /// Look up an entry by id.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<VisualMemoryEntry> {
        self.entries.read().get(&id).cloned()
    }

    /// Find entries similar to the query caption.
    #[must_use]
    pub fn find_similar(&self, query: &str, k: usize) -> Vec<VisualMemoryEntry> {
        let query_emb = self.embedder.embed_text(query);
        let entries = self.entries.read();
        let mut scored: Vec<(f32, VisualMemoryEntry)> = entries.values()
            .map(|e| (e.embedding.cosine(&query_emb), e.clone()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, e)| e).collect()
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Returns `true` if no entries are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qbitos_memory::embedding::EmbeddingModelStub;

    #[test]
    fn record_and_find() {
        let emb: Arc<dyn qbitos_memory::embedding::EmbeddingModel + Send + Sync> = Arc::new(EmbeddingModelStub::new(64));
        let mem = VisualMemory::new(emb);
        mem.record("/tmp/a.jpg", "a cat on a couch", vec!["cat".into()], 640, 480);
        mem.record("/tmp/b.jpg", "a dog in a park", vec!["dog".into()], 640, 480);
        let results = mem.find_similar("a feline", 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].caption.contains("cat"));
    }
}
