//! LFU (Least Frequently Used) cache with O(log n) get and insert.

use ahash::AHashMap;
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::hash::Hash;
use std::sync::Arc;

/// A thread-safe LFU cache.
pub struct LfuCache<K, V> {
    inner: Arc<Mutex<LfuCacheInner<K, V>>>,
}

struct LfuCacheInner<K, V> {
    capacity: usize,
    entries: AHashMap<K, Entry<V>>,
    freq_index: BTreeSet<FreqKey<K>>,
    counter: u64,
}

#[derive(Clone)]
struct Entry<V> {
    value: V,
    freq: usize,
    counter: u64,
}

#[derive(Clone)]
struct FreqKey<K> {
    freq: usize,
    counter: u64,
    key: K,
}

impl<K: Eq> PartialEq for FreqKey<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K: Eq> Eq for FreqKey<K> {}

impl<K: Eq + Ord> PartialOrd for FreqKey<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Eq + Ord> Ord for FreqKey<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then(self.counter.cmp(&other.counter))
            .then(self.key.cmp(&other.key))
    }
}

impl<K: Hash + Eq + Clone + Ord, V: Clone> LfuCache<K, V> {
    /// Construct a new LFU cache with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LfuCacheInner {
                capacity: capacity.max(1),
                entries: AHashMap::new(),
                freq_index: BTreeSet::new(),
                counter: 0,
            })),
        }
    }

    /// Look up a key, incrementing its frequency.
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        let counter = inner.counter;
        inner.counter += 1;
        // Take the entry out to avoid double borrow.
        let mut entry = inner.entries.get(key).cloned()?;
        let old_fk = FreqKey {
            freq: entry.freq,
            counter: entry.counter,
            key: key.clone(),
        };
        inner.freq_index.remove(&old_fk);
        entry.freq += 1;
        entry.counter = counter;
        let new_fk = FreqKey {
            freq: entry.freq,
            counter,
            key: key.clone(),
        };
        inner.freq_index.insert(new_fk);
        // Re-insert with updated freq/counter.
        if let Some(stored) = inner.entries.get_mut(key) {
            stored.freq = entry.freq;
            stored.counter = entry.counter;
        }
        Some(entry.value)
    }

    /// Insert a key-value pair.
    pub fn put(&self, key: K, value: V) {
        let mut inner = self.inner.lock();
        let counter = inner.counter;
        inner.counter += 1;
        if inner.entries.contains_key(&key) {
            // Update existing — take the entry out, modify, re-insert.
            let mut entry = inner.entries.remove(&key).unwrap();
            let old_fk = FreqKey {
                freq: entry.freq,
                counter: entry.counter,
                key: key.clone(),
            };
            inner.freq_index.remove(&old_fk);
            entry.freq += 1;
            entry.counter = counter;
            entry.value = value;
            let new_fk = FreqKey {
                freq: entry.freq,
                counter,
                key: key.clone(),
            };
            inner.entries.insert(key, entry);
            inner.freq_index.insert(new_fk);
            return;
        }
        // Evict if at capacity.
        if inner.entries.len() >= inner.capacity {
            if let Some(lfu) = inner.freq_index.iter().next().cloned() {
                inner.freq_index.remove(&lfu);
                inner.entries.remove(&lfu.key);
            }
        }
        let fk = FreqKey {
            freq: 1,
            counter,
            key: key.clone(),
        };
        inner.entries.insert(
            key,
            Entry {
                value,
                freq: 1,
                counter,
            },
        );
        inner.freq_index.insert(fk);
    }

    /// Remove a key.
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.entries.remove(key) {
            let fk = FreqKey {
                freq: entry.freq,
                counter: entry.counter,
                key: key.clone(),
            };
            inner.freq_index.remove(&fk);
            Some(entry.value)
        } else {
            None
        }
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

    /// Clear all entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.freq_index.clear();
        inner.counter = 0;
    }
}

impl<K, V> Clone for LfuCache<K, V> {
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
        let cache: LfuCache<i32, i32> = LfuCache::new(3);
        cache.put(1, 10);
        cache.put(2, 20);
        cache.put(3, 30);
        assert_eq!(cache.get(&1), Some(10));
        assert_eq!(cache.get(&2), Some(20));
        assert_eq!(cache.get(&3), Some(30));
    }

    #[test]
    fn lfu_eviction_picks_least_frequent() {
        let cache: LfuCache<i32, &'static str> = LfuCache::new(3);
        cache.put(1, "a");
        cache.put(2, "b");
        cache.put(3, "c");
        // Access 1 and 3, leaving 2 with freq 1.
        cache.get(&1);
        cache.get(&3);
        // Insert 4 — should evict 2 (least frequent).
        cache.put(4, "d");
        assert_eq!(cache.get(&2), None);
        assert!(cache.get(&1).is_some());
        assert!(cache.get(&3).is_some());
        assert!(cache.get(&4).is_some());
    }

    #[test]
    fn remove_works() {
        let cache: LfuCache<i32, i32> = LfuCache::new(3);
        cache.put(1, 100);
        assert_eq!(cache.remove(&1), Some(100));
        assert_eq!(cache.get(&1), None);
    }
}
