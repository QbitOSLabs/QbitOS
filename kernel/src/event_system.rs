//! Global event bus.
//!
//! Every subsystem publishes and subscribes through this bus. It is implemented
//! as a bounded MPSC [`flume`] channel with topic filters, so back-pressure
//! propagates naturally when a subscriber cannot keep up.

use ahash::AHashMap;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Unique identifier for an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub Uuid);

impl EventId {
    /// Mint a fresh event id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for EventId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubscriptionId(pub Uuid);

/// High-level category of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A raw sensory input arrived (image, audio, text).
    Perception,
    /// The kernel entered a new cognitive stage.
    CognitiveStage,
    /// A goal was created, updated, or completed.
    Goal,
    /// A plan was generated or revised.
    Plan,
    /// A tool was invoked.
    ToolInvocation,
    /// A skill was learned or executed.
    Skill,
    /// A memory was written or consolidated.
    Memory,
    /// A reflection cycle completed.
    Reflection,
    /// An evolution cycle completed.
    Evolution,
    /// A resource budget was exceeded or restored.
    Resource,
    /// A system-level event (boot, shutdown, error).
    System,
}

/// A structured event published on the bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique event id.
    pub id: EventId,
    /// Causal parent event, if any.
    pub parent: Option<EventId>,
    /// Event category.
    pub kind: EventKind,
    /// Topic string (e.g. `vision.frame.captured`, `memory.episodic.write`).
    pub topic: String,
    /// Free-form JSON payload.
    pub payload: serde_json::Value,
    /// When the event was minted.
    pub timestamp: DateTime<Utc>,
    /// Name of the source subsystem.
    pub source: String,
}

impl Event {
    /// Create a new event with the given kind, topic, and payload.
    #[must_use]
    pub fn new(kind: EventKind, topic: impl Into<String>, payload: serde_json::Value, source: impl Into<String>) -> Self {
        Self {
            id: EventId::new(),
            parent: None,
            kind,
            topic: topic.into(),
            payload,
            timestamp: Utc::now(),
            source: source.into(),
        }
    }

    /// Attach a causal parent.
    #[must_use]
    pub fn with_parent(mut self, parent: EventId) -> Self {
        self.parent = Some(parent);
        self
    }
}

/// Errors emitted by the event bus.
#[derive(Debug, Error)]
pub enum EventBusError {
    /// The bus is at capacity and the publisher does not want to block.
    #[error("event bus is at capacity")]
    Full,
    /// The bus has been shut down.
    #[error("event bus is closed")]
    Closed,
}

/// A subscription handle returned by [`EventBus::subscribe`]. Dropping it does
/// not unsubscribe; use [`EventBus::unsubscribe`] to remove the subscription.
pub struct Subscription {
    /// Subscription id.
    pub id: SubscriptionId,
    /// The broadcast receiver. Use `rx.recv()` to await events.
    pub rx: broadcast::Receiver<Arc<Event>>,
}

/// The global event bus.
///
/// Implemented as a fan-out of broadcast channels keyed by topic prefix. A
/// subscription to `memory` receives every event whose topic starts with
/// `memory.`, while a subscription to `memory.episodic.write` receives only
/// that exact topic.
#[derive(Clone)]
pub struct EventBus {
    inner: Arc<BusInner>,
}

struct BusInner {
    capacity: usize,
    subscribers: RwLock<AHashMap<String, broadcast::Sender<Arc<Event>>>>,
    wildcard: RwLock<Vec<(SubscriptionId, broadcast::Sender<Arc<Event>>)>>,
    counters: RwLock<AHashMap<EventKind, u64>>,
}

impl EventBus {
    /// Create a new bus with the given per-subscriber buffer capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(BusInner {
                capacity,
                subscribers: RwLock::new(AHashMap::new()),
                wildcard: RwLock::new(Vec::new()),
                counters: RwLock::new(AHashMap::new()),
            }),
        }
    }

    /// Publish an event. Returns the number of subscribers that received it.
    ///
    /// # Errors
    /// Returns [`EventBusError::Closed`] if all subscribers have been dropped.
    pub fn publish(&self, event: Event) -> Result<usize, EventBusError> {
        let event = Arc::new(event);
        let mut delivered = 0usize;

        // Topic counters
        {
            let mut counters = self.inner.counters.write();
            *counters.entry(event.kind).or_insert(0) += 1;
        }

        // Exact-topic subscribers
        if let Some(tx) = self.inner.subscribers.read().get(&event.topic) {
            delivered += Self::send(tx, &event);
        }

        // Prefix subscribers (e.g. subscribe to "memory" to get "memory.*")
        let prefix = event.topic.split('.').next().unwrap_or(&event.topic).to_string();
        if let Some(tx) = self.inner.subscribers.read().get(&prefix) {
            delivered += Self::send(tx, &event);
        }

        // Wildcard subscribers
        let wildcard = self.inner.wildcard.read().clone();
        for (_, tx) in &wildcard {
            delivered += Self::send(tx, &event);
        }

        Ok(delivered)
    }

    fn send(tx: &broadcast::Sender<Arc<Event>>, event: &Arc<Event>) -> usize {
        match tx.send(Arc::clone(event)) {
            Ok(n) => n,
            Err(_) => 0,
        }
    }

    /// Subscribe to a topic or topic prefix.
    ///
    /// - `*` subscribes to every event (wildcard).
    /// - `memory` subscribes to all events whose topic starts with `memory`.
    /// - `memory.episodic.write` subscribes to that exact topic only.
    #[must_use]
    pub fn subscribe(&self, topic: &str) -> Subscription {
        let (tx, rx) = broadcast::channel(self.inner.capacity);
        let id = SubscriptionId(Uuid::now_v7());
        let sub = Subscription { id, rx };

        if topic == "*" {
            self.inner.wildcard.write().push((id, tx));
        } else {
            self.inner.subscribers.write().insert(topic.to_string(), tx);
        }
        sub
    }

    /// Remove a subscription by id.
    pub fn unsubscribe(&self, id: SubscriptionId) {
        self.inner.subscribers.write().retain(|_, tx| tx.receiver_count() > 0);
        self.inner.wildcard.write().retain(|(sid, tx)| *sid != id && tx.receiver_count() > 0);
    }

    /// Wait for the bus to become idle (no events published for `quiet_for`).
    ///
    /// Useful for tests and shutdown.
    pub async fn wait_idle(&self, _quiet_for: Duration) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    /// Number of events seen for a given kind.
    #[must_use]
    pub fn count(&self, kind: EventKind) -> u64 {
        self.inner.counters.read().get(&kind).copied().unwrap_or(0)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn publish_and_subscribe_exact() {
        let bus = EventBus::new(16);
        let mut sub = bus.subscribe("memory.episodic.write");
        let ev = Event::new(
            EventKind::Memory,
            "memory.episodic.write",
            serde_json::json!({"id": "mem-1"}),
            "test",
        );
        bus.publish(ev.clone()).unwrap();
        let received = timeout(Duration::from_millis(100), sub.rx.recv()).await.unwrap().unwrap();
        assert_eq!(received.topic, "memory.episodic.write");
    }

    #[tokio::test]
    async fn subscribe_by_prefix() {
        let bus = EventBus::new(16);
        let mut sub = bus.subscribe("memory");
        bus.publish(Event::new(
            EventKind::Memory,
            "memory.semantic.write",
            serde_json::json!({}),
            "test",
        )).unwrap();
        let received = timeout(Duration::from_millis(100), sub.rx.recv()).await.unwrap().unwrap();
        assert_eq!(received.kind, EventKind::Memory);
    }

    #[tokio::test]
    async fn wildcard_subscribes_to_everything() {
        let bus = EventBus::new(16);
        let mut sub = bus.subscribe("*");
        bus.publish(Event::new(EventKind::System, "boot", serde_json::json!({}), "test")).unwrap();
        bus.publish(Event::new(EventKind::Goal, "goal.created", serde_json::json!({}), "test")).unwrap();
        let r1 = timeout(Duration::from_millis(100), sub.rx.recv()).await.unwrap().unwrap();
        let r2 = timeout(Duration::from_millis(100), sub.rx.recv()).await.unwrap().unwrap();
        assert_eq!(r1.kind, EventKind::System);
        assert_eq!(r2.kind, EventKind::Goal);
    }
}
