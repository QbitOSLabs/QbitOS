//! Quickselect — find the k-th smallest element in O(n) average time.

/// Find the k-th smallest element (0-indexed) in `data` (in-place partition).
///
/// # Panics
/// Panics if `k >= data.len()`.
pub fn quickselect<T: Ord + Clone>(data: &mut [T], k: usize) -> T {
    assert!(k < data.len(), "k out of range");
    let n = data.len();
    if n == 1 {
        return data[0].clone();
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    loop {
        if lo == hi {
            return data[lo].clone();
        }
        let pivot_idx = partition(data, lo, hi);
        if pivot_idx == k {
            return data[pivot_idx].clone();
        } else if pivot_idx < k {
            lo = pivot_idx + 1;
        } else {
            hi = pivot_idx.saturating_sub(1);
        }
    }
}

fn partition<T: Ord>(data: &mut [T], lo: usize, hi: usize) -> usize {
    // Median-of-three pivot selection.
    let mid = lo + (hi - lo) / 2;
    if data[lo] > data[mid] {
        data.swap(lo, mid);
    }
    if data[lo] > data[hi] {
        data.swap(lo, hi);
    }
    if data[mid] > data[hi] {
        data.swap(mid, hi);
    }
    let pivot = mid;
    data.swap(pivot, hi);
    let mut store = lo;
    for i in lo..hi {
        if data[i] <= data[hi] {
            data.swap(i, store);
            store += 1;
        }
    }
    data.swap(store, hi);
    store
}

/// Find the median element.
pub fn median<T: Ord + Clone>(data: &mut [T]) -> Option<T> {
    if data.is_empty() {
        return None;
    }
    let k = data.len() / 2;
    Some(quickselect(data, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_minimum() {
        let mut v = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
        assert_eq!(quickselect(&mut v, 0), 1);
    }

    #[test]
    fn finds_maximum() {
        let mut v = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
        assert_eq!(quickselect(&mut v, 8), 9);
    }

    #[test]
    fn finds_median_odd_length() {
        let mut v = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
        assert_eq!(median(&mut v), Some(5));
    }

    #[test]
    fn finds_median_even_length() {
        let mut v = vec![5, 3, 8, 1, 9, 2, 7, 4];
        // Sorted: [1,2,3,4,5,7,8,9]. length 8, k = 8/2 = 4 → v[4] = 5.
        assert_eq!(median(&mut v), Some(5));
    }

    #[test]
    fn handles_single_element() {
        let mut v = vec![42];
        assert_eq!(quickselect(&mut v, 0), 42);
    }

    #[test]
    fn handles_duplicates() {
        let mut v = vec![3, 3, 3, 1, 1, 1, 2, 2, 2];
        assert_eq!(quickselect(&mut v, 4), 2);
    }
}
