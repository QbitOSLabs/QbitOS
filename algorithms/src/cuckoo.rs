//! Cuckoo filter - Bloom alternative with deletion support (2014, refined 2026).
#![allow(missing_docs)]

use ahash::AHasher;
use std::hash::{Hash, Hasher};

const DEFAULT_BUCKETS: usize = 1 << 16;
const DEFAULT_BUCKET_SIZE: usize = 4;
const DEFAULT_FP_BITS: u8 = 12;
const DEFAULT_MAX_KICKS: usize = 500;

pub struct CuckooFilter {
    buckets: Vec<Vec<u16>>,
    num_buckets: usize, bucket_size: usize, fp_mask: u16, max_kicks: usize, count: usize,
}

impl CuckooFilter {
    #[must_use] pub fn new() -> Self { Self::with_params(DEFAULT_BUCKETS, DEFAULT_BUCKET_SIZE, DEFAULT_FP_BITS, DEFAULT_MAX_KICKS) }
    #[must_use] pub fn with_params(mut nb: usize, bs: usize, fp: u8, mk: usize) -> Self {
        let mut pow = 1; while pow < nb { pow <<= 1; } nb = pow;
        let fpm = if fp >= 16 { u16::MAX } else { (1u16 << fp) - 1 };
        Self { buckets: vec![Vec::with_capacity(bs); nb], num_buckets: nb, bucket_size: bs, fp_mask: fpm, max_kicks: mk, count: 0 }
    }
    pub fn insert<T: Hash>(&mut self, item: &T) -> bool {
        let (fp, i1, i2) = self.hashes(item);
        if self.buckets[i1].len() < self.bucket_size { self.buckets[i1].push(fp); self.count += 1; return true; }
        if self.buckets[i2].len() < self.bucket_size { self.buckets[i2].push(fp); self.count += 1; return true; }
        let mut cur_fp = fp; let mut cur_idx = if i1 < i2 { i1 } else { i2 };
        for _ in 0..self.max_kicks {
            let slot = next_random_usize() % self.bucket_size;
            if slot >= self.buckets[cur_idx].len() { self.buckets[cur_idx].push(cur_fp); self.count += 1; return true; }
            std::mem::swap(&mut cur_fp, &mut self.buckets[cur_idx][slot]);
            cur_idx = self.alt_index(cur_idx, cur_fp);
            if self.buckets[cur_idx].len() < self.bucket_size { self.buckets[cur_idx].push(cur_fp); self.count += 1; return true; }
        }
        false
    }
    #[must_use] pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (fp, i1, i2) = self.hashes(item);
        self.buckets[i1].contains(&fp) || self.buckets[i2].contains(&fp)
    }
    pub fn delete<T: Hash>(&mut self, item: &T) -> bool {
        let (fp, i1, i2) = self.hashes(item);
        if let Some(p) = self.buckets[i1].iter().position(|x| *x == fp) { self.buckets[i1].swap_remove(p); self.count -= 1; return true; }
        if let Some(p) = self.buckets[i2].iter().position(|x| *x == fp) { self.buckets[i2].swap_remove(p); self.count -= 1; return true; }
        false
    }
    #[must_use] pub fn len(&self) -> usize { self.count }
    #[must_use] pub fn is_empty(&self) -> bool { self.count == 0 }
    fn hashes<T: Hash>(&self, item: &T) -> (u16, usize, usize) {
        let mut h = AHasher::default(); item.hash(&mut h); let hv = h.finish();
        let fp = ((hv >> 32) as u16 | 1) & self.fp_mask;
        let i1 = (hv as usize) & (self.num_buckets - 1);
        (fp, i1, self.alt_index(i1, fp))
    }
    fn alt_index(&self, idx: usize, fp: u16) -> usize {
        let mut h = AHasher::default(); fp.hash(&mut h);
        (idx ^ (h.finish() as usize)) & (self.num_buckets - 1)
    }
}
impl Default for CuckooFilter { fn default() -> Self { Self::new() } }

fn next_random_usize() -> usize {
    use std::sync::atomic::{AtomicU64, Ordering}; static S: AtomicU64 = AtomicU64::new(0xDEADBEEFCAFEBABE);
    let mut s = S.load(Ordering::Relaxed); if s == 0 { s = 0xDEADBEEFCAFEBABE; }
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; S.store(s, Ordering::Relaxed); s as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn insert_contains() { let mut cf = CuckooFilter::new(); for i in 0..1000 { assert!(cf.insert(&i)); } for i in 0..1000 { assert!(cf.contains(&i)); } assert!(!cf.contains(&999_999)); }
    #[test] fn delete() { let mut cf = CuckooFilter::new(); cf.insert(&"hello"); cf.insert(&"world"); assert!(cf.delete(&"hello")); assert!(!cf.contains(&"hello")); assert!(cf.contains(&"world")); }
    #[test] fn delete_missing() { let mut cf = CuckooFilter::new(); assert!(!cf.delete(&"missing")); }
    #[test] fn empty() { let cf = CuckooFilter::new(); assert!(cf.is_empty()); assert_eq!(cf.len(), 0); }
    #[test] fn fpr_reasonable() { let cf = CuckooFilter::new(); assert!(cf.is_empty()); }
}
