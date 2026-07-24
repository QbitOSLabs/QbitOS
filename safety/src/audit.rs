//! Audit logger — records all safety-relevant events.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::VecDeque;

/// Severity level of an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditLevel {
    /// Informational.
    Info,
    /// Warning.
    Warning,
    /// Error.
    Error,
    /// Critical security event.
    Critical,
}

/// An audit event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique id.
    pub id: uuid::Uuid,
    /// Timestamp.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Severity level.
    pub level: AuditLevel,
    /// Event category (e.g., "injection_detected", "output_rejected").
    pub category: String,
    /// Human-readable message.
    pub message: String,
    /// Optional context data.
    pub context: Option<serde_json::Value>,
}

impl AuditEvent {
    /// Create a new event.
    #[must_use]
    pub fn new(level: AuditLevel, category: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::now_v7(),
            timestamp: chrono::Utc::now(),
            level,
            category: category.into(),
            message: message.into(),
            context: None,
        }
    }

    /// Add context.
    #[must_use]
    pub fn with_context(mut self, ctx: serde_json::Value) -> Self {
        self.context = Some(ctx);
        self
    }
}

/// Audit logger with a bounded ring buffer.
pub struct AuditLogger {
    events: Arc<RwLock<VecDeque<AuditEvent>>>,
    max_events: usize,
}

impl Default for AuditLogger {
    fn default() -> Self { Self::new(10_000) }
}

impl AuditLogger {
    /// Create a new logger with the given max events.
    #[must_use]
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(max_events))),
            max_events,
        }
    }

    /// Log an event.
    pub fn log(&self, event: AuditEvent) {
        let mut events = self.events.write();
        if events.len() >= self.max_events {
            events.pop_front();
        }
        events.push_back(event);
    }

    /// Log an info event.
    pub fn info(&self, category: &str, message: &str) {
        self.log(AuditEvent::new(AuditLevel::Info, category, message));
    }

    /// Log a warning event.
    pub fn warn(&self, category: &str, message: &str) {
        self.log(AuditEvent::new(AuditLevel::Warning, category, message));
    }

    /// Log an error event.
    pub fn error(&self, category: &str, message: &str) {
        self.log(AuditEvent::new(AuditLevel::Error, category, message));
    }

    /// Log a critical event.
    pub fn critical(&self, category: &str, message: &str) {
        self.log(AuditEvent::new(AuditLevel::Critical, category, message));
    }

    /// Get all events.
    #[must_use]
    pub fn events(&self) -> Vec<AuditEvent> {
        self.events.read().iter().cloned().collect()
    }

    /// Get events by level.
    #[must_use]
    pub fn events_by_level(&self, level: AuditLevel) -> Vec<AuditEvent> {
        self.events.read().iter().filter(|e| e.level == level).cloned().collect()
    }

    /// Get events by category.
    #[must_use]
    pub fn events_by_category(&self, category: &str) -> Vec<AuditEvent> {
        self.events.read().iter().filter(|e| e.category == category).cloned().collect()
    }

    /// Number of events.
    #[must_use]
    pub fn len(&self) -> usize { self.events.read().len() }

    /// Returns `true` if no events.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.events.read().is_empty() }

    /// Clear all events.
    pub fn clear(&self) { self.events.write().clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_and_retrieve() {
        let logger = AuditLogger::new(100);
        logger.info("test", "hello");
        logger.warn("security", "suspicious activity");
        logger.critical("breach", "injection detected");

        assert_eq!(logger.len(), 3);
        let warnings = logger.events_by_level(AuditLevel::Warning);
        assert_eq!(warnings.len(), 1);
        let security = logger.events_by_category("security");
        assert_eq!(security.len(), 1);
    }

    #[test]
    fn ring_buffer_evicts() {
        let logger = AuditLogger::new(3);
        logger.info("a", "1");
        logger.info("b", "2");
        logger.info("c", "3");
        logger.info("d", "4"); // should evict "1"
        assert_eq!(logger.len(), 3);
        let events = logger.events();
        assert_eq!(events[0].category, "b");
    }
}
