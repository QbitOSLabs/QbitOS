//! Top-K selection with a binary min-heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Maintain the top-K largest items seen so far.
pub struct TopK<T: Ord + Clone> {
    k: usize,
    heap: BinaryHeap<Reverse<T>>,
}

impl<T: Ord + Clone> TopK<T> {
    /// Construct a new TopK tracker for `k` items.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            heap: BinaryHeap::with_capacity(k.max(1) + 1),
        }
    }

    /// Add an item. If the heap is full and the item is larger than the
    /// current minimum, the minimum is evicted.
    pub fn add(&mut self, item: T) {
        if self.heap.len() < self.k {
            self.heap.push(Reverse(item));
        } else if let Some(&Reverse(ref min)) = self.heap.peek() {
            if &item > min {
                self.heap.pop();
                self.heap.push(Reverse(item));
            }
        }
    }

    /// Add multiple items.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.add(item);
        }
    }

    /// Return the top-K items sorted descending (largest first).
    #[must_use]
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(Reverse(item)) = self.heap.pop() {
            out.push(item);
        }
        out.reverse();
        out
    }

    /// Current number of items in the heap (≤ k).
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether no items have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// The configured K.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_finds_largest() {
        let mut t = TopK::new(3);
        t.extend(vec![5, 1, 9, 3, 7, 2, 8, 4, 6]);
        let result = t.into_sorted_vec();
        assert_eq!(result, vec![9, 8, 7]);
    }

    #[test]
    fn top_k_with_fewer_items_than_k() {
        let mut t = TopK::new(5);
        t.extend(vec![3, 1, 2]);
        let result = t.into_sorted_vec();
        assert_eq!(result, vec![3, 2, 1]);
    }

    #[test]
    fn top_k_empty() {
        let t: TopK<i32> = TopK::new(5);
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn top_k_with_duplicates() {
        let mut t = TopK::new(3);
        t.extend(vec![5, 5, 5, 5, 1]);
        let result = t.into_sorted_vec();
        assert_eq!(result, vec![5, 5, 5]);
    }
}
