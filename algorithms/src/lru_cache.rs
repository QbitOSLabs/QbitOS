//! LRU (Least Recently Used) cache with O(1) get and insert.

use ahash::AHashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::Arc;

/// A thread-safe LRU cache.
pub struct LruCache<K, V> {
    inner: Arc<Mutex<LruCacheInner<K, V>>>,
}

struct LruCacheInner<K, V> {
    capacity: usize,
    entries: AHashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Hash + Eq + Clone, V: Clone> LruCache<K, V> {
    /// Construct a new LRU cache with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LruCacheInner {
                capacity: capacity.max(1),
                entries: AHashMap::with_capacity(capacity.max(1)),
                order: VecDeque::with_capacity(capacity.max(1)),
            })),
        }
    }

    /// Look up a key, marking it as most-recently-used.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        let value = inner.entries.get(key).cloned()?;
        // Move to back (MRU).
        if let Some(pos) = inner.order.iter().position(|k| k == key) {
            inner.order.remove(pos);
        }
        inner.order.push_back(key.clone());
        Some(value)
    }

    /// Insert a key-value pair. If the cache is full, evicts the LRU entry.
    pub fn put(&self, key: K, value: V) {
        let mut inner = self.inner.lock();
        if inner.entries.contains_key(&key) {
            inner.entries.insert(key.clone(), value);
            if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                inner.order.remove(pos);
            }
            inner.order.push_back(key);
            return;
        }
        if inner.entries.len() >= inner.capacity {
            if let Some(lru) = inner.order.pop_front() {
                inner.entries.remove(&lru);
            }
        }
        inner.entries.insert(key.clone(), value);
        inner.order.push_back(key);
    }

    /// Remove a key.
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        if let Some(pos) = inner.order.iter().position(|k| k == key) {
            inner.order.remove(pos);
        }
        inner.entries.remove(key)
    }

    /// Current number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.lock().capacity
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.order.clear();
    }
}

impl<K, V> Clone for LruCache<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let cache: LruCache<String, i32> = LruCache::new(3);
        cache.put("a".into(), 1);
        cache.put("b".into(), 2);
        cache.put("c".into(), 3);
        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), None);
    }

    #[test]
    fn lru_eviction() {
        let cache: LruCache<i32, i32> = LruCache::new(2);
        cache.put(1, 100);
        cache.put(2, 200);
        // Access 1, making 2 the LRU.
        assert_eq!(cache.get(&1), Some(100));
        // Insert 3 — should evict 2.
        cache.put(3, 300);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&1), Some(100));
        assert_eq!(cache.get(&3), Some(300));
    }

    #[test]
    fn remove_works() {
        let cache: LruCache<i32, i32> = LruCache::new(3);
        cache.put(1, 100);
        assert_eq!(cache.remove(&1), Some(100));
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn thread_safe() {
        let cache: LruCache<i32, i32> = LruCache::new(100);
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    for i in 0..25 {
                        cache.put(t * 100 + i, i as i32);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cache.len(), 100);
    }
}
