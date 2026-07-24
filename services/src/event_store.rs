//! Persistent event store — backs the in-memory event buffer with the memory
//! fabric's semantic layer so events survive across restarts.
//!
//! When a `PersistentEventStore` is attached to the orchestrator, every
//! published event is:
//!
//! 1. Written to the in-memory ring buffer (for fast WebSocket fan-out).
//! 2. Serialized to JSON and stored as a semantic-memory concept labeled
//!    `qbitos:event:<event_id>`. The concept's content payload carries the
//!    full event envelope, allowing later replay via semantic search.
//!
//! On startup, call `replay_into_buffer` to load recent events from the
//! semantic layer back into the in-memory buffer, restoring the event
//! history for any subscribers that connect after a restart.

use crate::events::{EnvelopedEvent, EventId};
use qbitos_memory::MemoryFabric;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Configuration for the persistent event store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStoreConfig {
    /// Maximum number of events to keep in the in-memory ring buffer.
    pub buffer_capacity: usize,
    /// Maximum number of events to replay from semantic memory on startup.
    pub replay_limit: usize,
    /// Label prefix used when storing events as semantic concepts.
    pub label_prefix: String,
    /// Whether persistence is enabled (if false, the store is a no-op).
    pub enabled: bool,
}

impl Default for EventStoreConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 10_000,
            replay_limit: 1_000,
            label_prefix: "qbitos:event".into(),
            enabled: true,
        }
    }
}

/// A persistent event store backed by the memory fabric's semantic layer.
pub struct PersistentEventStore {
    /// The memory fabric used for persistence.
    pub memory: Arc<MemoryFabric>,
    /// Configuration.
    pub config: EventStoreConfig,
    /// In-memory ring buffer (mirrors the broadcast channel for HTTP polling).
    buffer: Arc<Mutex<Vec<EnvelopedEvent>>>,
    /// Set of event IDs we've already persisted (to avoid duplicates on replay).
    persisted_ids: Arc<Mutex<std::collections::HashSet<u64>>>,
}

impl PersistentEventStore {
    /// Construct a new persistent event store.
    #[must_use]
    pub fn new(memory: Arc<MemoryFabric>, config: EventStoreConfig) -> Self {
        Self {
            memory,
            config,
            buffer: Arc::new(Mutex::new(Vec::new())),
            persisted_ids: Arc::new(Mutex::new(std::collections::HashSet::new())),
        }
    }

    /// Persist an event to both the in-memory buffer and the semantic layer.
    pub async fn persist(&self, enveloped: &EnvelopedEvent) {
        // Always push to the in-memory buffer.
        {
            let mut buf = self.buffer.lock().await;
            buf.push(enveloped.clone());
            if buf.len() > self.config.buffer_capacity {
                let drop_n = buf.len() - self.config.buffer_capacity;
                buf.drain(0..drop_n);
            }
        }

        if !self.config.enabled {
            return;
        }

        // Skip if we've already persisted this event id (e.g. during replay).
        {
            let mut seen = self.persisted_ids.lock().await;
            if !seen.insert(enveloped.id.0) {
                return;
            }
        }

        // Store as a semantic concept. The label is unique per event id.
        let label = format!("{}:{}", self.config.label_prefix, enveloped.id.0);
        let content = serde_json::to_value(enveloped).unwrap_or(serde_json::Value::Null);
        self.memory.semantic.add_concept(label, content);
    }

    /// Read up to `max` events from the in-memory buffer, starting from the
    /// given cursor (exclusive). Returns the events and the new cursor.
    pub async fn read_since(&self, cursor: u64, max: usize) -> (Vec<EnvelopedEvent>, u64) {
        let buf = self.buffer.lock().await;
        let mut out = Vec::new();
        let mut new_cursor = cursor;
        for evt in buf.iter() {
            if evt.id.0 > cursor {
                out.push(evt.clone());
                if out.len() >= max {
                    break;
                }
            }
            if evt.id.0 > new_cursor {
                new_cursor = evt.id.0;
            }
        }
        (out, new_cursor)
    }

    /// Total events in the in-memory buffer.
    pub async fn buffer_len(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Replay recent events from the semantic layer into the in-memory buffer.
    /// This is typically called once on startup.
    ///
    /// Returns the number of events replayed.
    pub async fn replay_into_buffer(&self) -> usize {
        if !self.config.enabled {
            return 0;
        }
        // Search for all concepts labeled with our prefix. The semantic
        // memory's `find_similar` uses vector search, but we want all events
        // — so we search for the prefix string itself and request a large k.
        let limit = self.config.replay_limit;
        let concepts = self
            .memory
            .semantic
            .find_similar(&self.config.label_prefix, limit);

        let mut replayed = 0;
        let mut events: Vec<EnvelopedEvent> = Vec::new();
        let mut seen = self.persisted_ids.lock().await;
        for concept in concepts {
            // Skip events we've already persisted (e.g. from this session).
            if let Some(id_val) = concept
                .object
                .content
                .get("id")
                .and_then(|v| v.get("0"))
                .and_then(|v| v.as_u64())
            {
                if !seen.insert(id_val) {
                    continue;
                }
            }
            if let Ok(env) =
                serde_json::from_value::<EnvelopedEvent>(concept.object.content.clone())
            {
                events.push(env);
                replayed += 1;
            }
        }
        drop(seen);

        // Sort by event id to restore chronological order.
        events.sort_by_key(|e| e.id.0);

        // Push into buffer (respecting capacity).
        let mut buf = self.buffer.lock().await;
        for env in events {
            buf.push(env);
        }
        if buf.len() > self.config.buffer_capacity {
            let drop_n = buf.len() - self.config.buffer_capacity;
            buf.drain(0..drop_n);
        }

        replayed
    }

    /// Look up an event by id from the semantic layer.
    pub async fn lookup(&self, id: EventId) -> Option<EnvelopedEvent> {
        if !self.config.enabled {
            return None;
        }
        let label = format!("{}:{}", self.config.label_prefix, id.0);
        // Search the semantic memory for the label.
        let concepts = self.memory.semantic.find_similar(&label, 1);
        for concept in concepts {
            if concept.label == label {
                if let Ok(env) =
                    serde_json::from_value::<EnvelopedEvent>(concept.object.content.clone())
                {
                    return Some(env);
                }
            }
        }
        None
    }

    /// Clear the in-memory buffer (does not affect the semantic layer).
    pub async fn clear_buffer(&self) {
        self.buffer.lock().await.clear();
    }

    /// The memory fabric backing this store.
    #[must_use]
    pub fn memory(&self) -> &Arc<MemoryFabric> {
        &self.memory
    }
}

impl std::fmt::Debug for PersistentEventStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentEventStore")
            .field("enabled", &self.config.enabled)
            .field("buffer_capacity", &self.config.buffer_capacity)
            .field("label_prefix", &self.config.label_prefix)
            .finish_non_exhaustive()
    }
}

/// Snapshot metadata about the persistent event store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStoreStats {
    /// Whether persistence is enabled.
    pub enabled: bool,
    /// Current in-memory buffer length.
    pub buffer_len: usize,
    /// Buffer capacity.
    pub buffer_capacity: usize,
    /// Total events persisted to the semantic layer (best-effort count).
    pub semantic_concepts: usize,
}

impl PersistentEventStore {
    /// Get stats about the store.
    pub async fn stats(&self) -> EventStoreStats {
        EventStoreStats {
            enabled: self.config.enabled,
            buffer_len: self.buffer_len().await,
            buffer_capacity: self.config.buffer_capacity,
            semantic_concepts: 0, // The semantic memory doesn't expose a count.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ServiceEvent;
    use chrono::Utc;

    fn make_store() -> PersistentEventStore {
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        PersistentEventStore::new(memory, EventStoreConfig::default())
    }

    fn make_event(id: u64, kind: &str) -> EnvelopedEvent {
        EnvelopedEvent {
            id: EventId(id),
            timestamp: Utc::now(),
            event: ServiceEvent::Log {
                level: "info".into(),
                message: kind.into(),
            },
        }
    }

    #[tokio::test]
    async fn persist_adds_to_buffer() {
        let store = make_store();
        store.persist(&make_event(1, "a")).await;
        store.persist(&make_event(2, "b")).await;
        assert_eq!(store.buffer_len().await, 2);
    }

    #[tokio::test]
    async fn read_since_returns_events_after_cursor() {
        let store = make_store();
        store.persist(&make_event(1, "a")).await;
        store.persist(&make_event(2, "b")).await;
        store.persist(&make_event(3, "c")).await;
        let (events, cursor) = store.read_since(1, 100).await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id.0, 2);
        assert_eq!(events[1].id.0, 3);
        assert_eq!(cursor, 3);
    }

    #[tokio::test]
    async fn buffer_capacity_evicts_oldest() {
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        let config = EventStoreConfig {
            buffer_capacity: 3,
            ..EventStoreConfig::default()
        };
        let store = PersistentEventStore::new(memory, config);
        for i in 1..=5 {
            store.persist(&make_event(i, "x")).await;
        }
        assert_eq!(store.buffer_len().await, 3);
        let (events, _) = store.read_since(0, 100).await;
        assert_eq!(events[0].id.0, 3); // oldest two were evicted
    }

    #[tokio::test]
    async fn disabled_store_does_not_persist_to_semantic() {
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        let config = EventStoreConfig {
            enabled: false,
            ..EventStoreConfig::default()
        };
        let store = PersistentEventStore::new(memory, config);
        store.persist(&make_event(1, "a")).await;
        // Buffer still receives the event.
        assert_eq!(store.buffer_len().await, 1);
        // Lookup returns None since persistence is disabled.
        let found = store.lookup(EventId(1)).await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn clear_buffer_empties_in_memory_state() {
        let store = make_store();
        store.persist(&make_event(1, "a")).await;
        store.clear_buffer().await;
        assert_eq!(store.buffer_len().await, 0);
    }
}
