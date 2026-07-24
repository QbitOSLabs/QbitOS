//! Unicode normalization using the `unicode-normalization` crate.

use serde::{Deserialize, Serialize};
use unicode_normalization::{UnicodeNormalization};

/// The normalization form to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NormalizationForm {
    /// NFC — Canonical Decomposition + Canonical Composition.
    Nfc,
    /// NFD — Canonical Decomposition.
    Nfd,
    /// NFKC — Compatibility Decomposition + Canonical Composition.
    Nfkc,
    /// NFKD — Compatibility Decomposition.
    Nfkd,
}

/// Normalize a string to NFC form.
#[must_use]
pub fn normalize_nfc(s: &str) -> String {
    s.nfc().collect()
}

/// Normalize a string to NFD form.
#[must_use]
pub fn normalize_nfd(s: &str) -> String {
    s.nfd().collect()
}

/// Normalize a string to NFKC form.
#[must_use]
pub fn normalize_nfkc(s: &str) -> String {
    s.nfkc().collect()
}

/// Normalize a string to NFKD form.
#[must_use]
pub fn normalize_nfkd(s: &str) -> String {
    s.nfkd().collect()
}

/// Strip diacritics from a string (useful for accent-insensitive search).
#[must_use]
pub fn strip_diacritics(s: &str) -> String {
    s.nfd()
        .filter(|c| !is_combining_mark(*c))
        .collect()
}

fn is_combining_mark(c: char) -> bool {
    let code = c as u32;
    // Combining Diacritical Marks block (U+0300 – U+036F) and a few others.
    (0x0300..=0x036F).contains(&code)
        || (0x1AB0..=0x1AFF).contains(&code)
        || (0x1DC0..=0x1DFF).contains(&code)
        || (0x20D0..=0x20FF).contains(&code)
        || (0xFE20..=0xFE2F).contains(&code)
}

/// Case-fold a string for case-insensitive comparison (Unicode-aware).
#[must_use]
pub fn casefold(s: &str) -> String {
    // Lower-case mapping for casefold; full Unicode casefold is more
    // aggressive, but lowercase covers most use cases.
    s.to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_composes_decomposed_chars() {
        // "café" with decomposed é (e + combining acute).
        let decomposed = "cafe\u{0301}";
        let composed = normalize_nfc(decomposed);
        assert_eq!(composed, "café");
    }

    #[test]
    fn nfd_decomposes_composed_chars() {
        let composed = "café";
        let decomposed = normalize_nfd(composed);
        assert_eq!(decomposed, "cafe\u{0301}");
    }

    #[test]
    fn strip_diacritics_removes_accents() {
        assert_eq!(strip_diacritics("café"), "cafe");
        assert_eq!(strip_diacritics("naïve"), "naive");
        assert_eq!(strip_diacritics("über"), "uber");
    }

    #[test]
    fn casefold_works() {
        assert_eq!(casefold("HELLO"), "hello");
        assert_eq!(casefold("Straße"), "straße");
    }
}
