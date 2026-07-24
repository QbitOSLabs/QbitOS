//! Count-Min Sketch — frequency estimation in sublinear space.

use ahash::AHasher;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// A Count-Min Sketch for approximate frequency counting.
///
/// Provides `(ε, δ)` guarantees: the estimated count is within `ε * N` of
/// the true count with probability at least `1 - δ`, where `N` is the
/// total count of all items.
#[derive(Debug, Serialize, Deserialize)]
pub struct CountMinSketch {
    /// `d` rows × `w` columns of counters.
    counters: Vec<Vec<u64>>,
    width: usize,
    depth: u32,
    total: u64,
    /// Optional per-row lock for concurrent updates (None = single-threaded).
    #[serde(skip)]
    locks: Vec<RwLock<()>>,
}

impl CountMinSketch {
    /// Construct a new sketch with `(ε, δ)` guarantees.
    ///
    /// - `epsilon`: error bound (e.g., 0.01 means estimate is within 1% of total)
    /// - `delta`: failure probability (e.g., 0.01 means 99% confidence)
    #[must_use]
    pub fn new(epsilon: f64, delta: f64) -> Self {
        let w = ((std::f64::consts::E / epsilon).ceil()) as usize;
        let d = ((1.0 / delta).ln().ceil()) as u32;
        let w = w.max(8);
        let d = d.max(2);
        let counters = vec![vec![0u64; w]; d as usize];
        let locks = (0..d).map(|_| RwLock::new(())).collect();
        Self {
            counters,
            width: w,
            depth: d,
            total: 0,
            locks,
        }
    }

    /// Construct with explicit dimensions.
    #[must_use]
    pub fn with_dims(width: usize, depth: u32) -> Self {
        let width = width.max(8);
        let depth = depth.max(2);
        let counters = vec![vec![0u64; width]; depth as usize];
        let locks = (0..depth).map(|_| RwLock::new(())).collect();
        Self {
            counters,
            width,
            depth,
            total: 0,
            locks,
        }
    }

    /// Increment the count for `item` by 1.
    pub fn increment<T: Hash>(&mut self, item: &T) {
        self.add(item, 1);
    }

    /// Increment the count for `item` by `n`.
    pub fn add<T: Hash>(&mut self, item: &T, n: u64) {
        let hashes = self.hashes(item);
        for (row, col) in hashes.iter().enumerate() {
            self.counters[row][*col] = self.counters[row][*col].saturating_add(n);
        }
        self.total = self.total.saturating_add(n);
    }

    /// Thread-safe increment (acquires per-row write locks for counting).
    pub fn increment_concurrent<T: Hash>(&self, item: &T) {
        self.add_concurrent(item, 1);
    }

    /// Thread-safe add.
    pub fn add_concurrent<T: Hash>(&self, item: &T, n: u64) {
        let hashes = self.hashes(item);
        for (row, col) in hashes.iter().enumerate() {
            let _guard = self.locks[row].write();
            // SAFETY: we hold the per-row lock; this is the only writer for this row.
            unsafe {
                let row_ptr = self.counters.as_ptr().add(row) as *mut Vec<u64>;
                let cell = (*row_ptr).as_mut_ptr().add(*col);
                *cell = (*cell).saturating_add(n);
            }
        }
    }

    /// Estimate the count for `item`. Always an over-estimate (never under).
    #[must_use]
    pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
        let hashes = self.hashes(item);
        let mut min = u64::MAX;
        for (row, col) in hashes.iter().enumerate() {
            min = min.min(self.counters[row][*col]);
        }
        if min == u64::MAX {
            0
        } else {
            min
        }
    }

    /// Total count across all items.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Number of rows.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Number of columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Reset all counters to zero.
    pub fn clear(&mut self) {
        for row in &mut self.counters {
            for c in row.iter_mut() {
                *c = 0;
            }
        }
        self.total = 0;
    }

    /// Merge another sketch into this one (must have the same dimensions).
    ///
    /// # Errors
    /// Returns an error if dimensions differ.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        if self.width != other.width || self.depth != other.depth {
            return Err("incompatible count-min sketch dimensions".into());
        }
        for (a_row, b_row) in self.counters.iter_mut().zip(other.counters.iter()) {
            for (a, b) in a_row.iter_mut().zip(b_row.iter()) {
                *a = a.saturating_add(*b);
            }
        }
        self.total = self.total.saturating_add(other.total);
        Ok(())
    }

    fn hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.depth as usize);
        let mut h1 = AHasher::default();
        item.hash(&mut h1);
        let h1v = h1.finish();
        let mut h2 = AHasher::default();
        item.hash(&mut h2);
        0x517c_c1b7_2722_0a95_u64.hash(&mut h2);
        let h2v = h2.finish() | 1;
        for i in 0..self.depth {
            let combined = h1v.wrapping_add((i as u64).wrapping_mul(h2v));
            out.push((combined % (self.width as u64)) as usize);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_counts_for_distinct_items() {
        let mut cms = CountMinSketch::new(0.01, 0.01);
        cms.add(&"a", 5);
        cms.add(&"b", 3);
        cms.add(&"c", 7);
        assert_eq!(cms.estimate(&"a"), 5);
        assert_eq!(cms.estimate(&"b"), 3);
        assert_eq!(cms.estimate(&"c"), 7);
        assert_eq!(cms.total(), 15);
    }

    #[test]
    fn estimates_never_undercount() {
        let mut cms = CountMinSketch::new(0.1, 0.1);
        for _ in 0..1000 {
            cms.increment(&"popular");
        }
        for _ in 0..10 {
            cms.increment(&"rare");
        }
        let pop = cms.estimate(&"popular");
        let rare = cms.estimate(&"rare");
        assert!(pop >= 1000, "undercount: {pop}");
        assert!(rare >= 10, "undercount: {rare}");
    }

    #[test]
    fn merge_combines_sketches() {
        let mut a = CountMinSketch::with_dims(1024, 4);
        let mut b = CountMinSketch::with_dims(1024, 4);
        a.add(&1_i64, 3);
        b.add(&1_i64, 5);
        a.merge(&b).unwrap();
        assert_eq!(a.estimate(&1_i64), 8);
        assert_eq!(a.total(), 8);
    }

    #[test]
    fn clear_resets_state() {
        let mut cms = CountMinSketch::with_dims(1024, 4);
        cms.add(&1_i64, 10);
        assert_eq!(cms.estimate(&1_i64), 10);
        cms.clear();
        assert_eq!(cms.estimate(&1_i64), 0);
        assert_eq!(cms.total(), 0);
    }
}
