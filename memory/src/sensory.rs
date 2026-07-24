//! L0 Sensory Memory.
//!
//! Short-lived (<1s) ring buffer of raw perceptual input: video frames,
//! audio chunks, text tokens. The perception pipeline writes here; the
//! understand stage reads from here.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A single slot in the sensory ring buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorySlot {
    /// When the slot was captured.
    pub captured_at: chrono::DateTime<chrono::Utc>,
    /// Modality: vision, audio, text, etc.
    pub modality: String,
    /// Raw bytes (image data, audio samples, text).
    pub data: Vec<u8>,
    /// Optional mime type.
    pub mime: Option<String>,
}

impl SensorySlot {
    /// Create a new sensory slot.
    #[must_use]
    pub fn new(modality: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            captured_at: chrono::Utc::now(),
            modality: modality.into(),
            data,
            mime: None,
        }
    }

    /// Attach a mime type.
    #[must_use]
    pub fn with_mime(mut self, mime: impl Into<String>) -> Self {
        self.mime = Some(mime.into());
        self
    }
}

/// L0 sensory memory: a bounded ring buffer with time-based eviction.
pub struct SensoryMemory {
    buffer: Arc<RwLock<VecDeque<SensorySlot>>>,
    max_slots: usize,
    max_age: Duration,
    last_eviction: Arc<RwLock<Instant>>,
}

impl SensoryMemory {
    /// Create a new sensory memory that holds at most `max_slots` items and
    /// evicts anything older than `max_age`.
    #[must_use]
    pub fn new(max_slots: usize, max_age: Duration) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(VecDeque::with_capacity(max_slots))),
            max_slots,
            max_age,
            last_eviction: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Push a new slot. Evicts the oldest if at capacity.
    pub fn push(&self, slot: SensorySlot) {
        let mut buf = self.buffer.write();
        if buf.len() >= self.max_slots {
            buf.pop_front();
        }
        buf.push_back(slot);
    }

    /// Peek at the most recent slot.
    #[must_use]
    pub fn last(&self) -> Option<SensorySlot> {
        self.buffer.read().back().cloned()
    }

    /// Drain slots since the given instant.
    pub fn since(&self, since: Instant) -> Vec<SensorySlot> {
        let buf = self.buffer.read();
        buf.iter()
            .filter(|s| {
                let captured = chrono::Utc::now().timestamp() - s.captured_at.timestamp();
                captured >= 0 && (chrono::Utc::now().timestamp() - since.elapsed().as_secs() as i64 - s.captured_at.timestamp()).abs() < 60
            })
            .cloned()
            .collect()
    }

    /// Evict slots older than `max_age`.
    pub fn evict(&self) {
        let now = chrono::Utc::now();
        let cutoff = now - chrono::Duration::from_std(self.max_age).unwrap_or_else(|_| chrono::Duration::seconds(1));
        let mut buf = self.buffer.write();
        buf.retain(|s| s.captured_at >= cutoff);
        *self.last_eviction.write() = Instant::now();
    }

    /// Number of slots currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.read().len()
    }

    /// Returns `true` if no slots are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.read().is_empty()
    }
}

impl Default for SensoryMemory {
    fn default() -> Self {
        Self::new(256, Duration::from_secs(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_evicts() {
        let sm = SensoryMemory::new(3, Duration::from_secs(60));
        for i in 0..5 {
            sm.push(SensorySlot::new("text", vec![i]));
        }
        assert_eq!(sm.len(), 3);
        assert_eq!(sm.last().unwrap().data, vec![4]);
    }
}
