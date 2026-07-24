//! QbitOS Multilingual — text processing across 12+ languages.
//!
//! Implements:
//!
//! 1. **Language detection** — n-gram-based identification for 12+ languages.
//! 2. **Unicode normalization** — NFC, NFD, NFKC, NFKD.
//! 3. **Tokenization** — word, sentence, BPE-style (CJK-aware).
//! 4. **Stop words** — curated lists for 12+ languages.
//! 5. **Transliteration** — Latin<->Cyrillic, Latin<->CJK pinyin.
//! 6. **Stemming** — Porter stemmer for English; light stemmers for others.
//! 7. **Sentence segmentation** — Unicode UAX #29-style boundary detection.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod detection;
pub mod low_resource_translation;
pub mod normalize;
pub mod tokenize;
pub mod stopwords;
pub mod transliteration;
pub mod stemming;
pub mod sentence;

pub use detection::{detect_language, detect_language_with_scores, Language, LanguageScore};
pub use normalize::{normalize_nfc, normalize_nfd, normalize_nfkc, normalize_nfkd, NormalizationForm};
pub use tokenize::{word_tokenize, sentence_tokenize, word_tokenize_with_lang, Token};
pub use stopwords::{is_stopword, stopword_set, stopword_list};
pub use transliteration::{cyrillic_to_latin, latin_to_cyrillic, pinyin_to_chinese_hint};
pub use stemming::{stem_english, stem_porter};
pub use sentence::{split_sentences, SentenceSegmenter};

/// ISO 639-1 language codes supported.
pub const SUPPORTED_LANGUAGES: &[Language] = &[
    Language::English,
    Language::Spanish,
    Language::French,
    Language::German,
    Language::Italian,
    Language::Portuguese,
    Language::Dutch,
    Language::Russian,
    Language::Chinese,
    Language::Japanese,
    Language::Korean,
    Language::Arabic,
];
