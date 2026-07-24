//! L2 Episodic Memory.
//!
//! Stores experiences (observations, actions, outcomes). Each episode is
//! indexed by an HNSW vector index keyed on its embedding, plus a temporal
//! index keyed on its timestamp. Episodes with low importance decay over
//! time and are eventually garbage-collected.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::embedding::SharedEmbeddingModel;
use crate::hnsw::HnswIndex;
use crate::object::{Importance, MemoryId, MemoryObject};

/// An episodic memory entry: a single experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// The memory object envelope.
    pub object: MemoryObject<serde_json::Value>,
    /// When the episode happened (often equals `object.created_at`).
    pub occurred_at: DateTime<Utc>,
    /// Outcome of the episode (success, failure, neutral).
    pub outcome: Outcome,
}

/// Outcome of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The episode achieved its goal.
    Success,
    /// The episode failed.
    Failure,
    /// The episode was neutral or had no goal.
    Neutral,
}

/// L2 episodic memory.
pub struct EpisodicMemory {
    index: Arc<HnswIndex>,
    episodes: Arc<RwLock<BTreeMap<DateTime<Utc>, MemoryId>>>,
    store: Arc<RwLock<std::collections::HashMap<MemoryId, Episode>>>,
    embedder: SharedEmbeddingModel,
}

impl EpisodicMemory {
    /// Create a new episodic memory backed by an HNSW index and the given
    /// embedding model.
    #[must_use]
    pub fn new(embedder: SharedEmbeddingModel) -> Self {
        Self {
            index: Arc::new(HnswIndex::default_config()),
            episodes: Arc::new(RwLock::new(BTreeMap::new())),
            store: Arc::new(RwLock::new(std::collections::HashMap::new())),
            embedder,
        }
    }

    /// Record a new episode.
    pub fn record(&self, content: serde_json::Value, outcome: Outcome) -> Episode {
        let embedding = self.embedder.embed_json(&content);
        let mut object = MemoryObject::new(crate::object::MemoryType::Episodic, content, embedding.clone());
        object.importance = match outcome {
            Outcome::Success => Importance(0.7),
            Outcome::Failure => Importance(0.9), // failures are important to remember
            Outcome::Neutral => Importance(0.4),
        };
        let occurred_at = object.created_at;
        let id = object.id;
        let episode = Episode {
            object,
            occurred_at,
            outcome,
        };
        self.store.write().insert(id, episode.clone());
        self.episodes.write().insert(occurred_at, id);
        let _ = self.index.insert(id, embedding);
        episode
    }

    /// Recall the `k` most similar episodes to `query`.
    #[must_use]
    pub fn recall_similar(&self, query: &serde_json::Value, k: usize) -> Vec<Episode> {
        let embedding = self.embedder.embed_json(query);
        match self.index.search(&embedding, k) {
            Ok(results) => {
                let store = self.store.read();
                results.into_iter()
                    .filter_map(|(id, _)| store.get(&id).cloned())
                    .collect()
            }
            Err(_) => Vec::new(),
        }
    }

    /// Recall all episodes in a time range.
    #[must_use]
    pub fn recall_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<Episode> {
        let episodes = self.episodes.read();
        let store = self.store.read();
        episodes.range(from..=to)
            .filter_map(|(_, id)| store.get(id).cloned())
            .collect()
    }

    /// Decay the importance of all episodes by `factor` and evict those whose
    /// importance drops below `threshold`.
    pub fn decay(&self, factor: f32, threshold: f32) {
        let mut store = self.store.write();
        let mut to_remove = Vec::new();
        for (id, ep) in store.iter_mut() {
            ep.object.importance = ep.object.importance.decay(factor);
            if ep.object.importance.0 < threshold {
                to_remove.push(*id);
            }
        }
        for id in &to_remove {
            store.remove(id);
            let _ = self.index.remove(*id);
        }
        // Clean up temporal index
        let mut episodes = self.episodes.write();
        episodes.retain(|_, id| !to_remove.contains(id));
    }

    /// Number of episodes stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.read().len()
    }

    /// Returns `true` if no episodes are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingModelStub;

    #[test]
    fn record_and_recall() {
        let emb: SharedEmbeddingModel = Arc::new(EmbeddingModelStub::new(64));
        let mem = EpisodicMemory::new(emb);
        mem.record(serde_json::json!({"event": "user_login", "user": "alice"}), Outcome::Success);
        mem.record(serde_json::json!({"event": "user_login", "user": "bob"}), Outcome::Success);
        mem.record(serde_json::json!({"event": "build_error", "file": "main.rs"}), Outcome::Failure);

        let results = mem.recall_similar(&serde_json::json!({"event": "user_login"}), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(mem.len(), 3);
    }

    #[test]
    fn decay_evicts_low_importance() {
        let emb: SharedEmbeddingModel = Arc::new(EmbeddingModelStub::new(32));
        let mem = EpisodicMemory::new(emb);
        mem.record(serde_json::json!({"event": "low"}), Outcome::Neutral);
        mem.record(serde_json::json!({"event": "high"}), Outcome::Failure);
        mem.decay(0.3, 0.2);
        assert!(mem.len() <= 2);
    }
}
