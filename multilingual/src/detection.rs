//! Language detection using character n-gram profiles.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

/// ISO 639-1 style language codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// English
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Italian
    Italian,
    /// Portuguese
    Portuguese,
    /// Dutch
    Dutch,
    /// Russian
    Russian,
    /// Chinese
    Chinese,
    /// Japanese
    Japanese,
    /// Korean
    Korean,
    /// Arabic
    Arabic,
    /// Unknown / unable to detect
    Unknown,
}

impl Language {
    /// ISO 639-1 code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Spanish => "es",
            Self::French => "fr",
            Self::German => "de",
            Self::Italian => "it",
            Self::Portuguese => "pt",
            Self::Dutch => "nl",
            Self::Russian => "ru",
            Self::Chinese => "zh",
            Self::Japanese => "ja",
            Self::Korean => "ko",
            Self::Arabic => "ar",
            Self::Unknown => "und",
        }
    }

    /// Display name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::English => "English",
            Self::Spanish => "Spanish",
            Self::French => "French",
            Self::German => "German",
            Self::Italian => "Italian",
            Self::Portuguese => "Portuguese",
            Self::Dutch => "Dutch",
            Self::Russian => "Russian",
            Self::Chinese => "Chinese",
            Self::Japanese => "Japanese",
            Self::Korean => "Korean",
            Self::Arabic => "Arabic",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A language-detection score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageScore {
    /// The language.
    pub language: Language,
    /// Confidence score in [0, 1].
    pub score: f64,
}

/// Detect the most likely language of `text`.
///
/// Uses a combination of:
/// 1. Script-based detection (CJK, Cyrillic, Arabic) — deterministic.
/// 2. Character n-gram frequency matching for Latin-script languages.
///
/// Returns `Language::Unknown` if no language scores above 0.1.
#[must_use]
pub fn detect_language(text: &str) -> Language {
    detect_language_with_scores(text)
        .into_iter()
        .next()
        .map(|s| s.language)
        .unwrap_or(Language::Unknown)
}

/// Detect language with full score breakdown (sorted descending).
#[must_use]
pub fn detect_language_with_scores(text: &str) -> Vec<LanguageScore> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return vec![LanguageScore {
            language: Language::Unknown,
            score: 0.0,
        }];
    }

    // Script-based detection (deterministic, high confidence).
    if let Some(script_lang) = detect_script(&chars) {
        return vec![LanguageScore {
            language: script_lang,
            score: 0.95,
        }];
    }

    // N-gram based detection for Latin-script languages.
    let ngrams = compute_ngrams(&chars, 3);
    let mut scores: AHashMap<Language, f64> = AHashMap::new();
    let languages = [
        Language::English,
        Language::Spanish,
        Language::French,
        Language::German,
        Language::Italian,
        Language::Portuguese,
        Language::Dutch,
    ];
    for lang in languages {
        let profile = ngram_profile(lang);
        let score = cosine_similarity(&ngrams, profile);
        scores.insert(lang, score);
    }
    let mut out: Vec<LanguageScore> = scores
        .into_iter()
        .map(|(language, score)| LanguageScore { language, score })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Filter out very low scores.
    out.retain(|s| s.score > 0.05);
    if out.is_empty() {
        out.push(LanguageScore {
            language: Language::Unknown,
            score: 0.0,
        });
    }
    out
}

fn detect_script(chars: &[char]) -> Option<Language> {
    let mut cjk = 0;
    let mut hiragana = 0;
    let mut katakana = 0;
    let mut hangul = 0;
    let mut cyrillic = 0;
    let mut arabic = 0;
    let mut latin = 0;
    for &c in chars {
        let code = c as u32;
        if (0x4E00..=0x9FFF).contains(&code) || (0x3400..=0x4DBF).contains(&code) {
            cjk += 1;
        } else if (0x3040..=0x309F).contains(&code) {
            hiragana += 1;
        } else if (0x30A0..=0x30FF).contains(&code) {
            katakana += 1;
        } else if (0xAC00..=0xD7AF).contains(&code) {
            hangul += 1;
        } else if (0x0400..=0x04FF).contains(&code) {
            cyrillic += 1;
        } else if (0x0600..=0x06FF).contains(&code) {
            arabic += 1;
        } else if c.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    if hiragana > 0 || katakana > 0 {
        return Some(Language::Japanese);
    }
    if hangul > 0 {
        return Some(Language::Korean);
    }
    if cjk > 0 {
        return Some(Language::Chinese);
    }
    if cyrillic > latin && cyrillic > 0 {
        return Some(Language::Russian);
    }
    if arabic > latin && arabic > 0 {
        return Some(Language::Arabic);
    }
    None
}

fn compute_ngrams(chars: &[char], n: usize) -> AHashMap<String, f64> {
    let mut counts: AHashMap<String, u32> = AHashMap::new();
    let mut total = 0u32;
    if chars.len() < n {
        return AHashMap::new();
    }
    for window in chars.windows(n) {
        let gram: String = window.iter().collect::<String>().to_lowercase();
        *counts.entry(gram).or_insert(0) += 1;
        total += 1;
    }
    if total == 0 {
        return AHashMap::new();
    }
    counts
        .into_iter()
        .map(|(k, v)| (k, v as f64 / total as f64))
        .collect()
}

fn cosine_similarity(a: &AHashMap<String, f64>, b: &AHashMap<String, f64>) -> f64 {
    let dot: f64 = a
        .iter()
        .filter_map(|(k, v)| b.get(k).map(|w| v * w))
        .sum();
    let norm_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let norm_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Hard-coded n-gram frequency profiles (truncated for compactness).
/// These are top-20 trigrams per language with approximate frequencies.
fn ngram_profile(lang: Language) -> &'static AHashMap<String, f64> {
    use std::sync::OnceLock;
    static PROFILES: OnceLock<AHashMap<Language, AHashMap<String, f64>>> = OnceLock::new();
    let profiles = PROFILES.get_or_init(|| {
        let mut m = AHashMap::new();
        m.insert(Language::English, profile_from_list(&ENGLISH_TRIGRAMS));
        m.insert(Language::Spanish, profile_from_list(&SPANISH_TRIGRAMS));
        m.insert(Language::French, profile_from_list(&FRENCH_TRIGRAMS));
        m.insert(Language::German, profile_from_list(&GERMAN_TRIGRAMS));
        m.insert(Language::Italian, profile_from_list(&ITALIAN_TRIGRAMS));
        m.insert(Language::Portuguese, profile_from_list(&PORTUGUESE_TRIGRAMS));
        m.insert(Language::Dutch, profile_from_list(&DUTCH_TRIGRAMS));
        m
    });
    profiles.get(&lang).unwrap_or_else(|| panic!("missing profile for {lang:?}"))
}

fn profile_from_list(trigrams: &[(&str, f64)]) -> AHashMap<String, f64> {
    trigrams.iter().map(|(s, w)| ((*s).to_string(), *w)).collect()
}

const ENGLISH_TRIGRAMS: &[(&str, f64)] = &[
    ("the", 0.035), ("and", 0.018), ("ing", 0.015), ("her", 0.009), ("hat", 0.008),
    ("his", 0.008), ("tha", 0.007), ("ere", 0.007), ("for", 0.007), ("ent", 0.006),
    ("ion", 0.006), ("ter", 0.006), ("was", 0.005), ("you", 0.005), ("ith", 0.005),
    ("ver", 0.005), ("all", 0.005), ("wit", 0.004), ("thi", 0.004), ("tio", 0.004),
];

const SPANISH_TRIGRAMS: &[(&str, f64)] = &[
    ("de", 0.030), ("la", 0.022), ("que", 0.015), ("el", 0.014), ("en", 0.013),
    ("los", 0.009), ("se", 0.009), ("las", 0.008), ("por", 0.007), ("del", 0.007),
    ("una", 0.007), ("su", 0.007), ("con", 0.006), ("no", 0.006), ("para", 0.005),
    ("es", 0.005), ("al", 0.005), ("lo", 0.005), ("como", 0.004), ("más", 0.004),
];

const FRENCH_TRIGRAMS: &[(&str, f64)] = &[
    ("de", 0.030), ("le", 0.018), ("et", 0.014), ("les", 0.012), ("des", 0.011),
    ("en", 0.010), ("un", 0.009), ("une", 0.008), ("à", 0.008), ("est", 0.008),
    ("que", 0.007), ("qui", 0.006), ("dans", 0.006), ("pour", 0.005), ("pas", 0.005),
    ("que", 0.005), ("ce", 0.005), ("la", 0.005), ("sur", 0.004), ("ne", 0.004),
];

const GERMAN_TRIGRAMS: &[(&str, f64)] = &[
    ("der", 0.022), ("die", 0.020), ("und", 0.018), ("in", 0.014), ("den", 0.012),
    ("von", 0.010), ("zu", 0.009), ("das", 0.009), ("mit", 0.008), ("ist", 0.007),
    ("im", 0.007), ("des", 0.007), ("nicht", 0.006), ("ein", 0.006), ("auf", 0.005),
    ("eine", 0.005), ("er", 0.005), ("auch", 0.004), ("sich", 0.004), ("an", 0.004),
];

const ITALIAN_TRIGRAMS: &[(&str, f64)] = &[
    ("di", 0.025), ("il", 0.015), ("che", 0.013), ("la", 0.012), ("in", 0.010),
    ("un", 0.009), ("per", 0.008), ("non", 0.008), ("una", 0.007), ("come", 0.006),
    ("si", 0.006), ("è", 0.006), ("ma", 0.005), ("da", 0.005), ("le", 0.005),
    ("con", 0.005), ("i", 0.004), ("lo", 0.004), ("gli", 0.004), ("anche", 0.004),
];

const PORTUGUESE_TRIGRAMS: &[(&str, f64)] = &[
    ("de", 0.028), ("que", 0.015), ("a", 0.014), ("do", 0.012), ("da", 0.011),
    ("em", 0.010), ("um", 0.008), ("para", 0.007), ("com", 0.007), ("não", 0.006),
    ("uma", 0.006), ("os", 0.006), ("no", 0.005), ("se", 0.005), ("na", 0.005),
    ("por", 0.005), ("mais", 0.004), ("as", 0.004), ("dos", 0.004), ("como", 0.004),
];

const DUTCH_TRIGRAMS: &[(&str, f64)] = &[
    ("de", 0.030), ("een", 0.015), ("het", 0.012), ("en", 0.011), ("van", 0.010),
    ("in", 0.008), ("te", 0.007), ("is", 0.006), ("die", 0.006), ("niet", 0.005),
    ("met", 0.005), ("het", 0.005), ("voor", 0.004), ("op", 0.004), ("aan", 0.004),
    ("ook", 0.004), ("dat", 0.004), ("maar", 0.003), ("zijn", 0.003), ("als", 0.003),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_english() {
        let text = "The quick brown fox jumps over the lazy dog. The dog was not happy.";
        let lang = detect_language(text);
        assert_eq!(lang, Language::English);
    }

    #[test]
    fn detect_french() {
        let text = "Le chat noir dort sur le tapis rouge. Les oiseaux chantent dans les arbres.";
        let lang = detect_language(text);
        assert_eq!(lang, Language::French);
    }

    #[test]
    fn detect_german() {
        let text = "Der Hund läuft im Park. Die Sonne scheint und die Vögel singen in den Bäumen.";
        let lang = detect_language(text);
        assert_eq!(lang, Language::German);
    }

    #[test]
    fn detect_russian_by_script() {
        let text = "Привет мир, как дела?";
        let lang = detect_language(text);
        assert_eq!(lang, Language::Russian);
    }

    #[test]
    fn detect_japanese_by_script() {
        let text = "こんにちは、世界！今日はいい天気ですね。";
        let lang = detect_language(text);
        assert_eq!(lang, Language::Japanese);
    }

    #[test]
    fn detect_chinese_by_script() {
        let text = "你好世界。今天天气很好。";
        let lang = detect_language(text);
        assert_eq!(lang, Language::Chinese);
    }

    #[test]
    fn detect_korean_by_script() {
        let text = "안녕하세요 세계!";
        let lang = detect_language(text);
        assert_eq!(lang, Language::Korean);
    }

    #[test]
    fn detect_arabic_by_script() {
        let text = "مرحبا بالعالم";
        let lang = detect_language(text);
        assert_eq!(lang, Language::Arabic);
    }

    #[test]
    fn empty_text_returns_unknown() {
        assert_eq!(detect_language(""), Language::Unknown);
    }

    #[test]
    fn language_codes_are_correct() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::French.code(), "fr");
        assert_eq!(Language::Russian.code(), "ru");
    }
}
