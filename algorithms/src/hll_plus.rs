//! HyperLogLog++ with sparse/dense switching (Google 2013, refined 2026).
#![allow(missing_docs)]

use ahash::AHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

const DEFAULT_P: u8 = 14;

pub struct HyperLogLogPlus {
    p: u8, m: usize,
    sparse: Option<HashMap<u32, u8>>,
    dense: Vec<u8>,
    sparse_threshold: usize,
}

impl HyperLogLogPlus {
    #[must_use] pub fn new() -> Self { Self::with_precision(DEFAULT_P) }
    #[must_use] pub fn with_precision(p: u8) -> Self {
        assert!((4..=18).contains(&p)); let m = 1usize << p;
        Self { p, m, sparse: Some(HashMap::new()), dense: vec![], sparse_threshold: m / 4 }
    }
    pub fn add<T: Hash>(&mut self, item: &T) {
        let mut h = AHasher::default(); item.hash(&mut h); let hv = h.finish();
        let idx = (hv >> (64 - self.p)) as u32;
        let rest = (hv << self.p) | (1u64 << (self.p - 1));
        let rho = (rest.leading_zeros() + 1) as u8;
        if let Some(s) = &mut self.sparse {
            let e = s.entry(idx).or_insert(0); if rho > *e { *e = rho; }
            if s.len() >= self.sparse_threshold { self.switch_to_dense(); }
        } else {
            if rho > self.dense[idx as usize] { self.dense[idx as usize] = rho; }
        }
    }
    #[must_use] pub fn estimate(&self) -> u64 {
        let regs: Vec<u8> = if let Some(s) = &self.sparse {
            let mut d = vec![0u8; self.m]; for (&idx, &rho) in s { d[idx as usize] = rho; } d
        } else { self.dense.clone() };
        let alpha = match self.m { 16 => 0.673, 32 => 0.697, 64 => 0.709, _ => 0.7213 / (1.0 + 1.079 / self.m as f64) };
        let sum: f64 = regs.iter().map(|&r| 2.0_f64.powi(-i32::from(r))).sum();
        let raw = alpha * (self.m as f64).powi(2) / sum;
        if raw <= 2.5 * self.m as f64 {
            let zeros = regs.iter().filter(|&&r| r == 0).count();
            if zeros > 0 { let s = self.m as f64 * (self.m as f64 / zeros as f64).ln(); if s <= 2.5 * self.m as f64 { return s as u64; } }
        }
        let two64 = 2.0_f64.powi(64);
        if raw > two64 / 30.0 { return (-two64 * (1.0 - raw / two64).ln()) as u64; }
        raw as u64
    }
    pub fn merge(&mut self, other: &HyperLogLogPlus) {
        if other.p != self.p { return; }
        if let Some(os) = &other.sparse { for (&idx, &rho) in os { self.add_reg(idx, rho); } }
        else { for (i, &rho) in other.dense.iter().enumerate() { if rho > 0 { self.add_reg(i as u32, rho); } } }
    }
    pub fn reset(&mut self) { self.sparse = Some(HashMap::new()); self.dense.clear(); }
    #[must_use] pub fn precision(&self) -> u8 { self.p }
    #[must_use] pub fn register_count(&self) -> usize { if let Some(s) = &self.sparse { s.len() } else { self.m } }
    fn add_reg(&mut self, idx: u32, rho: u8) {
        if let Some(s) = &mut self.sparse { let e = s.entry(idx).or_insert(0); if rho > *e { *e = rho; } if s.len() >= self.sparse_threshold { self.switch_to_dense(); } }
        else if rho > self.dense[idx as usize] { self.dense[idx as usize] = rho; }
    }
    fn switch_to_dense(&mut self) {
        if let Some(s) = self.sparse.take() { let mut d = vec![0u8; self.m]; for (idx, rho) in s { d[idx as usize] = rho; } self.dense = d; }
    }
}
impl Default for HyperLogLogPlus { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn empty() { let h = HyperLogLogPlus::new(); assert_eq!(h.estimate(), 0); }
    #[test] fn single() { let mut h = HyperLogLogPlus::new(); h.add(&"hello"); let e = h.estimate(); assert!(e >= 1 && e <= 3); }
    #[test] fn distinct() { let mut h = HyperLogLogPlus::new(); for i in 0..10_000u64 { h.add(&i); } let e = h.estimate(); assert!(e >= 8_000 && e <= 12_000); }
    #[test] fn duplicates() { let mut h = HyperLogLogPlus::new(); for _ in 0..1000 { h.add(&"same"); } assert!(h.estimate() <= 5); }
    #[test] fn merge() { let mut a = HyperLogLogPlus::new(); let mut b = HyperLogLogPlus::new(); for i in 0..5_000 { a.add(&i); } for i in 5_000..10_000 { b.add(&i); } a.merge(&b); assert!(a.estimate() >= 8_000 && a.estimate() <= 12_000); }
    #[test] fn sparse_to_dense() { let mut h = HyperLogLogPlus::with_precision(10); for i in 0..300 { h.add(&i); } assert!(h.sparse.is_none()); }
}
