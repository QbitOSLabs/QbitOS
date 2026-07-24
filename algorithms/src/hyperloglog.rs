//! HyperLogLog — distinct-element counting in constant space.

use ahash::AHasher;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// HyperLogLog cardinality estimator.
///
/// Uses `m = 2^precision` registers (default precision 14 → 16384 registers,
/// ~16 KB, standard error ~0.81%).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: u32,
    m: usize,
}

impl HyperLogLog {
    /// Construct with a given precision (4 ≤ p ≤ 18).
    ///
    /// `precision = 14` gives standard error ≈ 0.81%.
    #[must_use]
    pub fn new(precision: u32) -> Self {
        let p = precision.clamp(4, 18);
        let m = 1usize << p;
        Self {
            registers: vec![0u8; m],
            precision: p,
            m,
        }
    }

    /// Default precision 14.
    #[must_use]
    pub fn default_precision() -> Self {
        Self::new(14)
    }

    /// Add an item to the estimator.
    pub fn add<T: Hash>(&mut self, item: &T) {
        let mut hasher = AHasher::default();
        item.hash(&mut hasher);
        let h = hasher.finish();
        // Use the top `precision` bits as the register index.
        let idx = (h >> (64 - self.precision)) as usize;
        // Use the remaining bits for the rank of the leftmost 1-bit.
        let w = h << self.precision | (1u64 << (self.precision - 1));
        let rank = (leading_zeros_64(w) + 1) as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Estimate the cardinality (number of distinct items seen).
    #[must_use]
    pub fn estimate(&self) -> u64 {
        let alpha = self.alpha();
        let m = self.m as f64;
        let mut sum = 0.0_f64;
        for &r in &self.registers {
            sum += 2.0_f64.powi(-(i32::from(r)));
        }
        let raw = alpha * m * m / sum;

        // Small-range correction.
        if raw <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count() as f64;
            if zeros > 0.0 {
                return (m * (m / zeros).ln()) as u64;
            }
        }
        // Large-range correction.
        let two_to_64 = 2.0_f64.powi(64);
        if raw > two_to_64 / 30.0 {
            return (-two_to_64 * (1.0 - raw / two_to_64).ln()) as u64;
        }
        raw as u64
    }

    /// Merge another HLL into this one (must have the same precision).
    ///
    /// # Errors
    /// Returns an error if precisions differ.
    pub fn merge(&mut self, other: &Self) -> Result<(), String> {
        if self.precision != other.precision {
            return Err("incompatible hyperloglog precision".into());
        }
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *b > *a {
                *a = *b;
            }
        }
        Ok(())
    }

    /// Reset all registers.
    pub fn clear(&mut self) {
        self.registers.iter_mut().for_each(|r| *r = 0);
    }

    /// Memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.registers.len()
    }

    fn alpha(&self) -> f64 {
        let m = self.m as f64;
        match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        }
    }
}

fn leading_zeros_64(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_distinct_items() {
        let mut hll = HyperLogLog::new(12); // 4096 registers
        for i in 0..10_000_i64 {
            hll.add(&i);
        }
        let est = hll.estimate();
        // Standard error for p=12 is ~1.625%. 10000 ± 5% should pass.
        let err = ((est as f64) - 10_000.0).abs() / 10_000.0;
        assert!(err < 0.10, "estimate {est} too far from 10000 (err {err:.2}%)");
    }

    #[test]
    fn duplicates_do_not_inflate_count() {
        let mut hll = HyperLogLog::new(12);
        for _ in 0..1000 {
            hll.add(&"the_same_item");
        }
        let est = hll.estimate();
        // Should be close to 1.
        assert!(est <= 5, "expected ~1, got {est}");
    }

    #[test]
    fn merge_combines_estimators() {
        let mut a = HyperLogLog::new(12);
        let mut b = HyperLogLog::new(12);
        for i in 0..1000_i64 {
            a.add(&i);
        }
        for i in 500..1500_i64 {
            b.add(&i);
        }
        a.merge(&b).unwrap();
        let est = a.estimate();
        // Union of [0..1000) and [500..1500) is [0..1500) → 1500 distinct items.
        let err = ((est as f64) - 1500.0).abs() / 1500.0;
        assert!(err < 0.10, "estimate {est} too far from 1500");
    }

    #[test]
    fn clear_resets_state() {
        let mut hll = HyperLogLog::new(12);
        hll.add(&1_i64);
        hll.add(&2_i64);
        assert!(hll.estimate() > 0);
        hll.clear();
        assert_eq!(hll.estimate(), 0);
    }
}
