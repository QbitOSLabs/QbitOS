//! Shared event bus and service event types.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::broadcast;
use uuid::Uuid;

/// Monotonically increasing event identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventId(pub u64);

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "evt-{}", self.0)
    }
}

/// Top-level service event type, broadcast to all subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ServiceEvent {
    /// A cognitive cycle completed.
    CognitiveCycleCompleted {
        /// Cycle ID.
        cycle_id: Uuid,
        /// Final phase name.
        phase: String,
        /// Duration in ms.
        duration_ms: u64,
        /// Human-readable summary.
        summary: String,
    },

    /// A new memory was stored.
    MemoryStored {
        /// Memory ID.
        memory_id: String,
        /// Memory layer.
        layer: String,
        /// Preview of content.
        content_preview: String,
    },

    /// A skill was executed.
    SkillExecuted {
        /// Skill ID.
        skill_id: String,
        /// Whether it succeeded.
        success: bool,
        /// Duration in ms.
        duration_ms: u64,
    },

    /// A tool was invoked.
    ToolInvoked {
        /// Tool name.
        tool_name: String,
        /// Whether it succeeded.
        success: bool,
    },

    /// A reasoning episode produced a conclusion.
    ReasoningConclusion {
        /// Engine name.
        engine: String,
        /// Conclusion text.
        conclusion: String,
        /// Confidence in [0, 1].
        confidence: f64,
    },

    /// A vision perception result was produced.
    VisionPerception {
        /// Number of objects detected.
        objects_detected: usize,
        /// Number of OCR characters.
        ocr_chars: usize,
    },

    /// A speech event (STT/TTS) occurred.
    SpeechEvent {
        /// Modality (stt or tts).
        modality: String,
        /// Duration in ms.
        duration_ms: u64,
    },

    /// A learning event (training step, evaluation).
    LearningEvent {
        /// Event type.
        event_type: String,
        /// Metric name.
        metric: String,
        /// Metric value.
        value: f64,
    },

    /// A safety event (filter triggered, audit log).
    SafetyEvent {
        /// Rule name.
        rule: String,
        /// Severity level.
        severity: String,
        /// Event message.
        message: String,
    },

    /// A subsystem started.
    SubsystemStarted {
        /// Subsystem name.
        name: String,
    },

    /// A subsystem stopped.
    SubsystemStopped {
        /// Subsystem name.
        name: String,
    },

    /// Generic service log.
    Log {
        /// Log level.
        level: String,
        /// Log message.
        message: String,
    },

    /// A subsystem emitted a custom event.
    Custom {
        /// Subsystem name.
        subsystem: String,
        /// Payload.
        payload: serde_json::Value,
    },
}

/// Event wrapped with metadata at publish time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopedEvent {
    /// Monotonic event identifier.
    pub id: EventId,
    /// Time the event was published.
    pub timestamp: DateTime<Utc>,
    /// The payload.
    pub event: ServiceEvent,
}

/// A subscriber handle that receives events.
pub struct EventSubscription {
    rx: broadcast::Receiver<ServiceEvent>,
    sub_id: u64,
    counter: Arc<AtomicU64>,
}

impl EventSubscription {
    /// Receive the next event (async).
    pub async fn recv(&mut self) -> Option<EnvelopedEvent> {
        loop {
            match self.rx.recv().await {
                Ok(event) => {
                    return Some(EnvelopedEvent {
                        id: EventId(self.counter.fetch_add(1, Ordering::Relaxed)),
                        timestamp: Utc::now(),
                        event,
                    });
                }
                Err(broadcast::error::RecvError::Closed) => return None,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(lagged = n, "event subscriber lagged");
                    // Retry on next iteration.
                }
            }
        }
    }

    /// The unique subscription identifier.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.sub_id
    }
}

/// Broadcast event bus shared across the service.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<ServiceEvent>,
    counter: Arc<AtomicU64>,
    subscribers_log: Arc<RwLock<ahash::AHashMap<u64, String>>>,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity.max(8));
        Self {
            tx,
            counter: Arc::new(AtomicU64::new(0)),
            subscribers_log: Arc::new(RwLock::new(ahash::AHashMap::new())),
        }
    }

    /// Publish an event. Returns the assigned event ID.
    pub fn publish(&self, event: ServiceEvent) -> EventId {
        let id = EventId(self.counter.fetch_add(1, Ordering::Relaxed));
        let _ = self.tx.send(event);
        id
    }

    /// Subscribe to events.
    #[must_use]
    pub fn subscribe(&self) -> EventSubscription {
        static SUB_ID: AtomicU64 = AtomicU64::new(1);
        let sub_id = SUB_ID.fetch_add(1, Ordering::Relaxed);
        self.subscribers_log
            .write()
            .insert(sub_id, format!("sub-{sub_id}"));
        EventSubscription {
            rx: self.tx.subscribe(),
            sub_id,
            counter: Arc::clone(&self.counter),
        }
    }

    /// Number of active subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Total events published so far.
    #[must_use]
    pub fn total_published(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
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

    #[tokio::test]
    async fn pub_sub_works() {
        let bus = EventBus::new(16);
        let mut sub = bus.subscribe();
        bus.publish(ServiceEvent::Log {
            level: "info".into(),
            message: "hello".into(),
        });
        let received = sub.recv().await;
        assert!(received.is_some());
    }

    #[test]
    fn subscriber_count_tracks_subscriptions() {
        let bus = EventBus::new(8);
        assert_eq!(bus.subscriber_count(), 0);
        let _s1 = bus.subscribe();
        let _s2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn total_published_increments() {
        let bus = EventBus::new(8);
        assert_eq!(bus.total_published(), 0);
        bus.publish(ServiceEvent::Log {
            level: "info".into(),
            message: "a".into(),
        });
        bus.publish(ServiceEvent::Log {
            level: "info".into(),
            message: "b".into(),
        });
        assert_eq!(bus.total_published(), 2);
    }
}
