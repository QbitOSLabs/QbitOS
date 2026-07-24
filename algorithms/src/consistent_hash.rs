//! Consistent hashing for distributed partitioning.

use ahash::AHasher;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// A consistent hash ring with virtual nodes.
pub struct ConsistentHashRing<N>
where
    N: Clone + Ord + Hash,
{
    ring: BTreeMap<u64, N>,
    nodes: Vec<N>,
    virtual_nodes: usize,
    _marker: PhantomData<N>,
}

// Manual Serialize/Deserialize to avoid `for<'de>` bound complexity.
impl<N> Serialize for ConsistentHashRing<N>
where
    N: Clone + Ord + Hash + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("ConsistentHashRing", 3)?;
        s.serialize_field("ring", &self.ring)?;
        s.serialize_field("nodes", &self.nodes)?;
        s.serialize_field("virtual_nodes", &self.virtual_nodes)?;
        s.end()
    }
}

impl<'de, N> Deserialize<'de> for ConsistentHashRing<N>
where
    N: Clone + Ord + Hash + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(bound = "N: Deserialize<'de>")]
        struct Helper<N> {
            ring: BTreeMap<u64, N>,
            nodes: Vec<N>,
            virtual_nodes: usize,
        }
        let h = Helper::<N>::deserialize(deserializer)?;
        Ok(Self {
            ring: h.ring,
            nodes: h.nodes,
            virtual_nodes: h.virtual_nodes,
            _marker: PhantomData,
        })
    }
}

impl<N> ConsistentHashRing<N>
where
    N: Clone + Ord + Hash,
{
    /// Construct a new ring with the given number of virtual nodes per physical node.
    #[must_use]
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            nodes: Vec::new(),
            virtual_nodes: virtual_nodes.max(1),
            _marker: PhantomData,
        }
    }

    /// Add a node to the ring.
    pub fn add_node(&mut self, node: N) {
        if self.nodes.contains(&node) {
            return;
        }
        for i in 0..self.virtual_nodes {
            let hash = hash_node(&node, i);
            self.ring.insert(hash, node.clone());
        }
        self.nodes.push(node);
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node: &N) -> bool {
        let pos = self.nodes.iter().position(|n| n == node);
        if let Some(p) = pos {
            self.nodes.remove(p);
            for i in 0..self.virtual_nodes {
                let hash = hash_node(node, i);
                self.ring.remove(&hash);
            }
            true
        } else {
            false
        }
    }

    /// Look up the node responsible for `key`.
    #[must_use]
    pub fn get_node(&self, key: &str) -> Option<&N> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = hash_key(key);
        if let Some((_, node)) = self.ring.range(hash..).next() {
            return Some(node);
        }
        self.ring.values().next()
    }

    /// Get the next `n` distinct physical nodes for replication.
    #[must_use]
    pub fn get_nodes(&self, key: &str, n: usize) -> Vec<&N> {
        if self.ring.is_empty() || n == 0 {
            return Vec::new();
        }
        let hash = hash_key(key);
        let mut out = Vec::with_capacity(n);
        let mut seen = std::collections::HashSet::new();
        let mut iter = self.ring.range(hash..).chain(self.ring.range(..hash));
        while let Some((_, node)) = iter.next() {
            if seen.insert(node) {
                out.push(node);
                if out.len() == n {
                    break;
                }
            }
        }
        out
    }

    /// Number of physical nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of virtual nodes.
    #[must_use]
    pub fn virtual_node_count(&self) -> usize {
        self.ring.len()
    }

    /// All physical nodes (unordered).
    #[must_use]
    pub fn nodes(&self) -> &[N] {
        &self.nodes
    }
}

fn hash_key(s: &str) -> u64 {
    let mut h = AHasher::default();
    s.hash(&mut h);
    h.finish()
}

fn hash_node<N: Hash>(node: &N, replica: usize) -> u64 {
    let mut h = AHasher::default();
    node.hash(&mut h);
    (replica as u64).hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_handles_all_keys() {
        let mut ring = ConsistentHashRing::<String>::new(64);
        ring.add_node("node1".into());
        for key in &["a", "b", "c", "d", "e"] {
            assert_eq!(ring.get_node(key), Some(&"node1".to_string()));
        }
    }

    #[test]
    fn keys_distribute_across_nodes() {
        let mut ring = ConsistentHashRing::<String>::new(128);
        ring.add_node("n1".into());
        ring.add_node("n2".into());
        ring.add_node("n3".into());
        let mut counts = ahash::AHashMap::new();
        for i in 0..1000 {
            let key = format!("key{i}");
            if let Some(node) = ring.get_node(&key) {
                *counts.entry(node.clone()).or_insert(0u32) += 1;
            }
        }
        assert_eq!(counts.len(), 3);
        for c in counts.values() {
            assert!(*c > 100, "node got too few keys: {c}");
        }
    }

    #[test]
    fn adding_node_minimizes_movement() {
        let mut ring = ConsistentHashRing::<String>::new(128);
        ring.add_node("n1".into());
        ring.add_node("n2".into());
        let mut before = ahash::AHashMap::new();
        for i in 0..1000 {
            let key = format!("key{i}");
            before.insert(key.clone(), ring.get_node(&key).cloned());
        }
        ring.add_node("n3".into());
        let mut moved = 0;
        for (key, prev) in &before {
            let now = ring.get_node(key);
            if now != prev.as_ref() {
                moved += 1;
            }
        }
        assert!(
            moved < 500,
            "too many keys moved when adding a node: {moved}/1000"
        );
    }

    #[test]
    fn get_nodes_returns_replicas() {
        let mut ring = ConsistentHashRing::<String>::new(64);
        ring.add_node("n1".into());
        ring.add_node("n2".into());
        ring.add_node("n3".into());
        let nodes = ring.get_nodes("key1", 2);
        assert_eq!(nodes.len(), 2);
        assert_ne!(nodes[0], nodes[1]);
    }

    #[test]
    fn remove_node_works() {
        let mut ring = ConsistentHashRing::<String>::new(32);
        ring.add_node("n1".into());
        ring.add_node("n2".into());
        assert_eq!(ring.node_count(), 2);
        assert!(ring.remove_node(&"n1".to_string()));
        assert_eq!(ring.node_count(), 1);
        assert!(!ring.remove_node(&"n3".to_string()));
    }
}
