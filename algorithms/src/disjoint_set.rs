//! Union-Find (Disjoint Set) with path compression + union by rank.

use serde::{Deserialize, Serialize};

/// Disjoint-set / union-find data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
    size: Vec<usize>,
    count: usize,
}

impl DisjointSet {
    /// Construct a new disjoint set with `n` elements (0..n), each in its own set.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            count: n,
        }
    }

    /// Find the representative of the set containing `x`, with path compression.
    ///
    /// # Panics
    /// Panics if `x >= n`.
    #[must_use]
    pub fn find(&mut self, x: usize) -> usize {
        assert!(x < self.parent.len(), "element out of range");
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression.
        let mut current = x;
        while self.parent[current] != root {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        root
    }

    /// Find the representative without path compression (immutable).
    #[must_use]
    pub fn find_immutable(&self, mut x: usize) -> usize {
        assert!(x < self.parent.len(), "element out of range");
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    /// Union the sets containing `x` and `y`. Returns `true` if the sets were
    /// actually merged (i.e. they were previously different).
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        // Union by rank.
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
            self.size[ry] += self.size[rx];
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
            self.size[rx] += self.size[ry];
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
            self.size[rx] += self.size[ry];
        }
        self.count -= 1;
        true
    }

    /// Whether `x` and `y` are in the same set.
    #[must_use]
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Number of disjoint sets.
    #[must_use]
    pub fn num_sets(&self) -> usize {
        self.count
    }

    /// Size of the set containing `x`.
    #[must_use]
    pub fn set_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    /// Reset to `n` singletons.
    pub fn reset(&mut self, n: usize) {
        self.parent = (0..n).collect();
        self.rank = vec![0; n];
        self.size = vec![1; n];
        self.count = n;
    }
}

impl Default for DisjointSet {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_set_each_element_alone() {
        let mut ds = DisjointSet::new(5);
        assert_eq!(ds.num_sets(), 5);
        for i in 0..5 {
            assert_eq!(ds.find(i), i);
        }
    }

    #[test]
    fn union_merges_sets() {
        let mut ds = DisjointSet::new(5);
        assert!(ds.union(0, 1));
        assert!(ds.union(2, 3));
        assert_eq!(ds.num_sets(), 3);
        assert!(ds.connected(0, 1));
        assert!(ds.connected(2, 3));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn union_already_connected_returns_false() {
        let mut ds = DisjointSet::new(3);
        assert!(ds.union(0, 1));
        assert!(!ds.union(0, 1));
        assert!(!ds.union(1, 0));
    }

    #[test]
    fn transitive_connection() {
        let mut ds = DisjointSet::new(6);
        ds.union(0, 1);
        ds.union(1, 2);
        ds.union(2, 3);
        ds.union(4, 5);
        assert!(ds.connected(0, 3));
        assert!(ds.connected(3, 0));
        assert!(!ds.connected(0, 4));
        assert_eq!(ds.num_sets(), 2);
        assert_eq!(ds.set_size(0), 4);
        assert_eq!(ds.set_size(4), 2);
    }

    #[test]
    fn path_compression_flattens_tree() {
        let mut ds = DisjointSet::new(10);
        // Build a long chain.
        for i in 0..9 {
            ds.union(i, i + 1);
        }
        // After find(0), the path should be compressed.
        let root = ds.find(0);
        for i in 0..10 {
            assert_eq!(ds.find(i), root);
        }
    }
}
