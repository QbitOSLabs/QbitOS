//! L1 Working Memory.
//!
//! Holds the current task, active reasoning state, and a small set of
//! "spotlight" memory objects. Capacity is bounded (default 7±2 items,
//! mirroring human working memory).

use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::object::{MemoryId, MemoryObject};

/// L1 working memory: a small bounded queue of "spotlight" objects plus
/// arbitrary scratch state.
pub struct WorkingMemory {
    spotlight: Arc<RwLock<VecDeque<MemoryObject>>>,
    scratch: Arc<RwLock<serde_json::Value>>,
    capacity: usize,
}

impl WorkingMemory {
    /// Create a new working memory with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            spotlight: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            scratch: Arc::new(RwLock::new(serde_json::Value::Null)),
            capacity,
        }
    }

    /// Push a memory object into the spotlight. Evicts the oldest if at capacity.
    pub fn push(&self, obj: MemoryObject) {
        let mut spot = self.spotlight.write();
        if spot.len() >= self.capacity {
            spot.pop_front();
        }
        spot.push_back(obj);
    }

    /// Read the spotlight as a vector (oldest first).
    #[must_use]
    pub fn spotlight(&self) -> Vec<MemoryObject> {
        self.spotlight.read().iter().cloned().collect()
    }

    /// Get the most recent spotlight object.
    #[must_use]
    pub fn last(&self) -> Option<MemoryObject> {
        self.spotlight.read().back().cloned()
    }

    /// Find a spotlight object by id.
    #[must_use]
    pub fn get(&self, id: MemoryId) -> Option<MemoryObject> {
        self.spotlight.read().iter().find(|o| o.id == id).cloned()
    }

    /// Replace the scratch state.
    pub fn set_scratch(&self, value: serde_json::Value) {
        *self.scratch.write() = value;
    }

    /// Read the scratch state.
    #[must_use]
    pub fn scratch(&self) -> serde_json::Value {
        self.scratch.read().clone()
    }

    /// Clear the spotlight and scratch.
    pub fn clear(&self) {
        self.spotlight.write().clear();
        *self.scratch.write() = serde_json::Value::Null;
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        // Miller's law: 7 ± 2 items.
        Self::new(7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Embedding;
    use crate::object::MemoryType;

    #[test]
    fn spotlight_evicts() {
        let wm = WorkingMemory::new(2);
        for i in 0..3 {
            let emb = Embedding::new(vec![i as f32]);
            wm.push(MemoryObject::new(MemoryType::Working, serde_json::Value::Number(i.into()), emb));
        }
        assert_eq!(wm.spotlight().len(), 2);
        assert_eq!(wm.last().unwrap().content, serde_json::Value::Number(2.into()));
    }
}
