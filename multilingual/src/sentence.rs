//! Sentence segmentation using Unicode UAX #29-style boundary detection.

/// A sentence segmenter that handles most common cases.
pub struct SentenceSegmenter {
    /// Whether to keep trailing whitespace with each sentence.
    pub keep_whitespace: bool,
}

impl Default for SentenceSegmenter {
    fn default() -> Self {
        Self {
            keep_whitespace: false,
        }
    }
}

impl SentenceSegmenter {
    /// Construct a new segmenter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Split text into sentences.
    #[must_use]
    pub fn split(&self, text: &str) -> Vec<String> {
        split_sentences(text)
    }
}

/// Split text into sentences.
///
/// Uses a heuristic algorithm:
/// 1. Split on `.`, `!`, `?`, `。`, `！`, `？` followed by whitespace or end.
/// 2. Skip common abbreviations (Mr., Dr., etc.).
/// 3. Skip numbers with decimal points (3.14 stays together).
/// 4. Skip single-letter initials (J. Smith).
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    crate::tokenize::sentence_tokenize(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmenter_splits_sentences() {
        let seg = SentenceSegmenter::new();
        let result = seg.split("Hello world. How are you?");
        assert_eq!(result, vec!["Hello world.", "How are you?"]);
    }

    #[test]
    fn split_returns_vec_of_sentences() {
        let result = split_sentences("One. Two! Three?");
        assert_eq!(result, vec!["One.", "Two!", "Three?"]);
    }
}
