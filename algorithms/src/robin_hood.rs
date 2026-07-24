#![allow(missing_docs)]

//! Robin Hood open-addressing hash map with backward-shift deletion.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
struct Slot<K, V> { hash: u64, key: K, value: V }

/// Robin Hood hash map: open addressing where the slot with the smallest
/// displacement is stolen from during insertion, reducing probe-length variance.
pub struct RobinHoodMap<K, V> {
    table: Vec<Option<Slot<K, V>>>,
    cap: usize,
    len: usize,
}

impl<K: Hash + Eq, V> RobinHoodMap<K, V> {
    /// Create a new empty map with default capacity.
    #[must_use]
    pub fn new() -> Self { Self::with_capacity(16) }

    /// Create a new empty map with at least `cap` slots.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(8);
        let mut table = Vec::with_capacity(cap);
        for _ in 0..cap { table.push(None); }
        Self { table, cap, len: 0 }
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize { self.len }
    /// Returns true if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Returns true if the key is present.
    pub fn contains(&self, key: &K) -> bool { self.get(key).is_some() }

    /// Look up a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = hash_key(key);
        let mut idx = self.ideal(hash);
        loop {
            match &self.table[idx] {
                None => return None,
                Some(slot) if slot.hash == hash && &slot.key == key => return Some(&slot.value),
                Some(_) => idx = (idx + 1) & (self.cap - 1),
            }
        }
    }

    /// Insert a key-value pair, replacing any existing value. Returns the
    /// previous value if the key already existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if (self.len + 1) * 4 > self.cap * 3 { self.resize(self.cap * 2); }
        let hash = hash_key(&key);
        let mut entry = Slot { hash, key, value };
        let mut idx = self.ideal(hash);
        loop {
            match &mut self.table[idx] {
                None => { self.table[idx] = Some(entry); self.len += 1; return None; }
                Some(slot) if slot.hash == entry.hash && slot.key == entry.key => {
                    let old = std::mem::replace(&mut slot.value, entry.value);
                    return Some(old);
                }
                Some(slot) => {
                    // Extract immutable values up-front to avoid simultaneous
                    // mutable + immutable borrows of `self`.
                    let slot_hash = slot.hash;
                    let existing_disp = self.displacement(slot_hash, idx);
                    let new_disp = self.displacement(entry.hash, idx);
                    if existing_disp < new_disp {
                        // Swap in-place: slot now holds the displaced entry,
                        // and `entry` becomes the next probe target.
                        let slot_ref = self.table[idx].as_mut().expect("slot just matched");
                        std::mem::swap(slot_ref, &mut entry);
                    }
                    idx = (idx + 1) & (self.cap - 1);
                }
            }
        }
    }

    /// Remove a key-value pair. Returns the removed value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let hash = hash_key(key);
        let mut idx = self.ideal(hash);
        let found = loop {
            match &self.table[idx] {
                None => return None,
                Some(slot) if slot.hash == hash && &slot.key == key => break idx,
                Some(_) => idx = (idx + 1) & (self.cap - 1),
            }
        };
        let removed = self.table[found].take().unwrap();
        self.len -= 1;
        // Backward-shift deletion: pull subsequent slots back by one as long
        // as their displacement > 0.
        let mut curr = found;
        let mut next = (curr + 1) & (self.cap - 1);
        while let Some(slot) = &self.table[next] {
            if self.displacement(slot.hash, next) == 0 { break; }
            let moved = self.table[next].take();
            self.table[curr] = moved;
            curr = next;
            next = (next + 1) & (self.cap - 1);
        }
        Some(removed.value)
    }

    fn ideal(&self, hash: u64) -> usize { (hash as usize) & (self.cap - 1) }
    fn displacement(&self, hash: u64, idx: usize) -> usize {
        (idx + self.cap - self.ideal(hash)) & (self.cap - 1)
    }

    fn resize(&mut self, new_cap: usize) {
        let new_cap = new_cap.next_power_of_two().max(8);
        let mut new_table = Vec::with_capacity(new_cap);
        for _ in 0..new_cap { new_table.push(None); }
        let old = std::mem::replace(&mut self.table, new_table);
        self.cap = new_cap;
        self.len = 0;
        for slot in old.into_iter().flatten() { self.insert(slot.key, slot.value); }
    }
}

impl<K: Hash + Eq, V> Default for RobinHoodMap<K, V> { fn default() -> Self { Self::new() } }

fn hash_key<K: Hash + ?Sized>(key: &K) -> u64 {
    let mut h = DefaultHasher::new();
    key.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut m: RobinHoodMap<&str, i32> = RobinHoodMap::new();
        assert!(m.is_empty());
        m.insert("alpha", 1); m.insert("beta", 2); m.insert("gamma", 3);
        assert_eq!(m.len(), 3);
        assert_eq!(m.get(&"beta"), Some(&2));
        assert_eq!(m.get(&"missing"), None);
        assert!(m.contains(&"alpha"));
    }

    #[test]
    fn insert_replaces_existing() {
        let mut m: RobinHoodMap<i32, &str> = RobinHoodMap::new();
        assert_eq!(m.insert(1, "a"), None);
        assert_eq!(m.insert(1, "b"), Some("a"));
        assert_eq!(m.get(&1), Some(&"b"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn remove_returns_value_and_preserves_others() {
        let mut m: RobinHoodMap<i32, i32> = RobinHoodMap::new();
        for i in 0..50 { m.insert(i, i * 10); }
        assert_eq!(m.remove(&25), Some(250));
        assert!(!m.contains(&25));
        assert_eq!(m.len(), 49);
        for i in 0..50 {
            if i == 25 { continue; }
            assert_eq!(m.get(&i), Some(&(i * 10)), "missing key {i}");
        }
    }

    #[test]
    fn grows_under_load() {
        let mut m: RobinHoodMap<u64, u64> = RobinHoodMap::new();
        for i in 0..1000u64 { m.insert(i, i.wrapping_mul(7)); }
        assert_eq!(m.len(), 1000);
        for i in 0..1000u64 { assert_eq!(m.get(&i), Some(&(i.wrapping_mul(7)))); }
    }

    #[test]
    fn default_is_empty() {
        let m: RobinHoodMap<u32, u32> = RobinHoodMap::default();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }
}
