//! Bloom filter — a probabilistic data structure for set membership testing.
//!
//! False positives are possible; false negatives are not.

use ahash::AHasher;
use serde::{Deserialize, Serialize};
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

/// A Bloom filter with `m` bits and `k` hash functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
    num_items: u64,
}

impl BloomFilter {
    /// Construct a new Bloom filter sized for `expected_items` with a target
    /// false-positive rate of `fp_rate` (e.g. `0.01` for 1%).
    ///
    /// Uses the standard formulas:
    ///   `m = -n * ln(p) / (ln(2)^2)`
    ///   `k = (m / n) * ln(2)`
    #[must_use]
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = fp_range_check(fp_rate);
        let m = (-(n * p.ln()) / (std::f64::consts::LN_2 * std::f64::consts::LN_2)).ceil() as usize;
        let m = m.max(64);
        let k = (((m as f64) / n) * std::f64::consts::LN_2).ceil() as u32;
        let k = k.max(1);
        let words = (m + 63) / 64;
        Self {
            bits: vec![0u64; words],
            num_bits: words * 64,
            num_hashes: k,
            num_items: 0,
        }
    }

    /// Construct a filter from a known number of bits and hashes.
    #[must_use]
    pub fn with_params(num_bits: usize, num_hashes: u32) -> Self {
        let words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; words],
            num_bits: words * 64,
            num_hashes,
            num_items: 0,
        }
    }

    /// Insert an item into the filter.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let (h1, h2) = double_hash(item);
        for i in 0..self.num_hashes {
            let bit = bit_index(h1, h2, i, self.num_bits);
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
        self.num_items += 1;
    }

    /// Test whether an item is *possibly* in the set.
    ///
    /// Returns `false` → definitely not in set. Returns `true` → probably in set
    /// (subject to the false-positive rate).
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (h1, h2) = double_hash(item);
        for i in 0..self.num_hashes {
            let bit = bit_index(h1, h2, i, self.num_bits);
            if (self.bits[bit / 64] >> (bit % 64)) & 1 == 0 {
                return false;
            }
        }
        true
    }

    /// Number of items inserted.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.num_items
    }

    /// Whether the filter is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_items == 0
    }

    /// Theoretical false-positive rate given the current load.
    #[must_use]
    pub fn estimated_fp_rate(&self) -> f64 {
        let m = self.num_bits as f64;
        let n = self.num_items as f64;
        let k = self.num_hashes as f64;
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Clear all bits.
    pub fn clear(&mut self) {
        self.bits.iter_mut().for_each(|w| *w = 0);
        self.num_items = 0;
    }

    /// Merge another filter into this one (must have the same dimensions).
    ///
    /// # Errors
    /// Returns an error if the filters have different sizes or hash counts.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err("incompatible bloom filter dimensions".into());
        }
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= b;
        }
        self.num_items += other.num_items;
        Ok(())
    }

    /// Memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

fn fp_range_check(p: f64) -> f64 {
    if p <= 0.0 {
        return 1e-9;
    }
    if p >= 1.0 {
        return 0.999_999;
    }
    p
}

fn double_hash<T: Hash>(item: &T) -> (u64, u64) {
    let bs1 = BuildHasherDefault::<AHasher>::default();
    let h1v = bs1.hash_one(item);
    // Use a second hasher with different seed by XORing the item with a constant.
    let mut h2 = AHasher::default();
    item.hash(&mut h2);
    0x9e37_79b9_7f4a_7c15_u64.hash(&mut h2);
    let h2v = h2.finish();
    (h1v, h2v | 1)
}

fn bit_index(h1: u64, h2: u64, i: u32, num_bits: usize) -> usize {
    let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
    (combined % (num_bits as u64)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_are_found() {
        let mut bf = BloomFilter::new(100, 0.01);
        for i in 0..100_i64 {
            bf.insert(&i);
        }
        for i in 0..100_i64 {
            assert!(bf.contains(&i), "missing item {i}");
        }
    }

    #[test]
    fn absent_items_usually_not_found() {
        let mut bf = BloomFilter::new(100, 0.01);
        for i in 0..100_i64 {
            bf.insert(&i);
        }
        // Items 1000..1100 were never inserted. FP rate is 1% so we should
        // rarely see hits — assert that at least 90% of them return false.
        let mut false_positives = 0;
        for i in 1000..1100_i64 {
            if bf.contains(&i) {
                false_positives += 1;
            }
        }
        assert!(
            false_positives < 10,
            "too many false positives: {false_positives}/100"
        );
    }

    #[test]
    fn clear_resets_state() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(&"hello");
        bf.insert(&"world");
        assert_eq!(bf.len(), 2);
        bf.clear();
        assert_eq!(bf.len(), 0);
        assert!(!bf.contains(&"hello"));
    }

    #[test]
    fn merge_combines_filters() {
        let mut a = BloomFilter::new(100, 0.01);
        let mut b = BloomFilter::new(100, 0.01);
        a.insert(&1_i64);
        a.insert(&2_i64);
        b.insert(&3_i64);
        b.insert(&4_i64);
        a.merge(&b).unwrap();
        for i in 1..=4_i64 {
            assert!(a.contains(&i));
        }
    }

    #[test]
    fn merge_rejects_incompatible_filters() {
        let a = BloomFilter::with_params(128, 3);
        let b = BloomFilter::with_params(256, 3);
        let mut a = a;
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn estimated_fp_rate_grows_with_load() {
        let mut bf = BloomFilter::new(100, 0.01);
        let r0 = bf.estimated_fp_rate();
        for i in 0..1000_i64 {
            bf.insert(&i);
        }
        let r1 = bf.estimated_fp_rate();
        assert!(r1 > r0);
    }
}
