//! Levenshtein edit distance with operation tracking.

/// An edit operation in the Levenshtein algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevenshteinOp {
    /// No change (the character matched).
    Match,
    /// Substitute one character for another.
    Substitute,
    /// Insert a character.
    Insert,
    /// Delete a character.
    Delete,
}

/// Compute the Levenshtein edit distance between two strings.
#[must_use]
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Compute the Levenshtein distance and return the sequence of edit operations.
#[must_use]
pub fn levenshtein_distance_with_ops(a: &str, b: &str) -> (usize, Vec<LevenshteinOp>) {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![(0usize, LevenshteinOp::Match); n + 1]; m + 1];
    for i in 1..=m {
        dp[i][0] = (i, LevenshteinOp::Delete);
    }
    for j in 1..=n {
        dp[0][j] = (j, LevenshteinOp::Insert);
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = dp[i - 1][j].0 + 1;
            let ins = dp[i][j - 1].0 + 1;
            let sub = dp[i - 1][j - 1].0 + cost;
            let (val, op) = if sub <= del && sub <= ins {
                if cost == 0 {
                    (sub, LevenshteinOp::Match)
                } else {
                    (sub, LevenshteinOp::Substitute)
                }
            } else if del <= ins {
                (del, LevenshteinOp::Delete)
            } else {
                (ins, LevenshteinOp::Insert)
            };
            dp[i][j] = (val, op);
        }
    }
    // Backtrack.
    let mut ops = Vec::new();
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        let (_, op) = dp[i][j];
        ops.push(op);
        match op {
            LevenshteinOp::Match | LevenshteinOp::Substitute => {
                i -= 1;
                j -= 1;
            }
            LevenshteinOp::Delete => {
                i -= 1;
            }
            LevenshteinOp::Insert => {
                j -= 1;
            }
        }
    }
    ops.reverse();
    (dp[m][n].0, ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings_have_distance_zero() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn known_distances() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("saturday", "sunday"), 3);
        assert_eq!(levenshtein_distance("flaw", "lawn"), 2);
    }

    #[test]
    fn operations_track_correctly() {
        let (dist, ops) = levenshtein_distance_with_ops("kitten", "sitting");
        assert_eq!(dist, 3);
        // kitten → sitten → sittin → sitting
        let matches = ops.iter().filter(|&&o| o == LevenshteinOp::Match).count();
        let subs = ops.iter().filter(|&&o| o == LevenshteinOp::Substitute).count();
        let ins = ops.iter().filter(|&&o| o == LevenshteinOp::Insert).count();
        let del = ops.iter().filter(|&&o| o == LevenshteinOp::Delete).count();
        assert_eq!(matches + subs + ins + del, ops.len());
        assert_eq!(subs, 2); // k→s, e→i
        assert_eq!(ins, 1); // +g at end
    }
}
