//! Skip list - arena-based, no unsafe (Pugh 1990, Redis ZSET pattern).
#![allow(missing_docs)]

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

const DEFAULT_P: f64 = 0.5;
const DEFAULT_MAX_HEIGHT: usize = 32;
const NIL: usize = usize::MAX;
type NodeIdx = usize;

pub struct SkipList<K: Ord, V> {
    arena: Vec<Node2<K, V>>,
    head_next: Vec<NodeIdx>,
    max_height: usize, level: usize, p: f64, len: usize,
}
struct Node2<K, V> { key: K, value: V, next: Vec<NodeIdx> }

impl<K: Ord + Default, V: Default> SkipList<K, V> {
    #[must_use] pub fn new() -> Self {
        Self { arena: vec![Node2 { key: K::default(), value: V::default(), next: vec![] }],
            head_next: vec![NIL; DEFAULT_MAX_HEIGHT + 1], max_height: DEFAULT_MAX_HEIGHT,
            level: 0, p: DEFAULT_P, len: 0 }
    }
    pub fn insert(&mut self, key: K, value: V) where K: Clone {
        let mut update = vec![0usize; self.max_height + 1]; let mut current: NodeIdx = 0;
        for l in (0..=self.level).rev() {
            let mut next = self.next_at(current, l);
            while next != NIL && self.arena[next].key < key { current = next; next = self.next_at(current, l); }
            update[l] = current;
        }
        let next0 = self.next_at(current, 0);
        if next0 != NIL && self.arena[next0].key == key { self.arena[next0].value = value; return; }
        let height = self.random_height();
        if height > self.level { for l in (self.level + 1)..=height { update[l] = 0; } self.level = height; }
        let new_idx = self.arena.len();
        let mut new_node = Node2 { key: key.clone(), value, next: vec![NIL; height + 1] };
        for l in 0..=height {
            let prev = update[l];
            if prev == 0 { new_node.next[l] = self.head_next[l]; self.head_next[l] = new_idx; }
            else { new_node.next[l] = self.arena[prev].next[l]; self.arena[prev].next[l] = new_idx; }
        }
        self.arena.push(new_node); self.len += 1;
    }
    #[must_use] pub fn get(&self, key: &K) -> Option<&V> {
        let mut current: NodeIdx = 0;
        for l in (0..=self.level).rev() {
            let mut next = self.next_at(current, l);
            while next != NIL {
                match self.arena[next].key.cmp(key) {
                    Ordering::Less => { current = next; next = self.next_at(current, l); }
                    Ordering::Equal => return Some(&self.arena[next].value),
                    Ordering::Greater => break,
                }
            }
        }
        None
    }
    #[must_use] pub fn len(&self) -> usize { self.len }
    #[must_use] pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        let mut idx = self.head_next[0];
        std::iter::from_fn(move || {
            if idx == NIL { return None; }
            let node = &self.arena[idx];
            let result = (&node.key, &node.value);
            idx = if node.next.is_empty() { NIL } else { node.next[0] };
            Some(result)
        })
    }
    fn next_at(&self, node: NodeIdx, layer: usize) -> NodeIdx {
        if node == 0 { self.head_next[layer] }
        else if layer < self.arena[node].next.len() { self.arena[node].next[layer] }
        else { NIL }
    }
    fn random_height(&self) -> usize {
        let mut h = 0;
        while next_random_f64() < self.p && h < self.max_height { h += 1; }
        h
    }
}
impl<K: Ord + Default, V: Default> Default for SkipList<K, V> { fn default() -> Self { Self::new() } }

fn next_random_f64() -> f64 {
    static S: AtomicU64 = AtomicU64::new(0x123456789ABCDEF0);
    let mut s = S.load(AtomicOrdering::Relaxed); if s == 0 { s = 0x123456789ABCDEF0; }
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; S.store(s, AtomicOrdering::Relaxed);
    (s as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn insert_get() {
        let mut sl: SkipList<i32, &str> = SkipList::new();
        sl.insert(5, "five"); sl.insert(1, "one"); sl.insert(10, "ten"); sl.insert(3, "three");
        assert_eq!(sl.len(), 4); assert_eq!(sl.get(&1), Some(&"one")); assert_eq!(sl.get(&10), Some(&"ten")); assert_eq!(sl.get(&100), None);
    }
    #[test] fn update_existing() { let mut sl: SkipList<i32, &str> = SkipList::new(); sl.insert(1, "one"); sl.insert(1, "uno"); assert_eq!(sl.len(), 1); assert_eq!(sl.get(&1), Some(&"uno")); }
    #[test] fn empty() { let sl: SkipList<i32, i32> = SkipList::new(); assert!(sl.is_empty()); assert_eq!(sl.get(&1), None); }
    #[test] fn large() { let mut sl: SkipList<i32, i32> = SkipList::new(); for i in 0..1000 { sl.insert(i, i * 2); } assert_eq!(sl.len(), 1000); for i in 0..1000 { assert_eq!(sl.get(&i), Some(&(i * 2))); } }
    #[test] fn iter_ordered() { let mut sl: SkipList<i32, &str> = SkipList::new(); sl.insert(5, "five"); sl.insert(1, "one"); sl.insert(10, "ten"); sl.insert(3, "three"); let v: Vec<i32> = sl.iter().map(|(k, _)| *k).collect(); assert_eq!(v, vec![1, 3, 5, 10]); }
}
