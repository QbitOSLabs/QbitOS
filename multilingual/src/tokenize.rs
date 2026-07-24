//! Tokenization: word and sentence splitting, CJK-aware.

use crate::detection::{detect_language, Language};

/// A token with its position in the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token text.
    pub text: String,
    /// Byte offset of the start of the token in the source.
    pub start: usize,
    /// Byte offset of the end of the token in the source.
    pub end: usize,
    /// The detected language (best guess).
    pub language: Language,
}

/// Word-tokenize a string.
///
/// Splits on whitespace and punctuation, keeping CJK characters as
/// individual tokens (one per character). Handles apostrophes in
/// contractions (e.g. "don't" stays as one token).
#[must_use]
pub fn word_tokenize(text: &str) -> Vec<Token> {
    word_tokenize_with_lang(text, detect_language(text))
}

/// Word-tokenize with an explicit language hint (for CJK splitting).
#[must_use]
pub fn word_tokenize_with_lang(text: &str, lang: Language) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut start_byte = 0usize;
    let mut current_start = 0usize;

    for (i, ch) in text.char_indices() {
        let is_cjk = is_cjk_char(ch);
        let is_whitespace = ch.is_whitespace();
        let is_punct = is_punctuation(ch);
        let is_apostrophe = ch == '\'' || ch == '\u{2019}' || ch == '\u{2018}';

        if is_cjk {
            // Flush any pending Latin token.
            if !current.is_empty() {
                tokens.push(Token {
                    text: std::mem::take(&mut current),
                    start: current_start,
                    end: i,
                    language: lang,
                });
            }
            // Emit the CJK character as its own token.
            tokens.push(Token {
                text: ch.to_string(),
                start: i,
                end: i + ch.len_utf8(),
                language: lang,
            });
            start_byte = i + ch.len_utf8();
        } else if is_whitespace {
            if !current.is_empty() {
                tokens.push(Token {
                    text: std::mem::take(&mut current),
                    start: current_start,
                    end: i,
                    language: lang,
                });
            }
            start_byte = i + ch.len_utf8();
        } else if is_punct && !is_apostrophe {
            if !current.is_empty() {
                tokens.push(Token {
                    text: std::mem::take(&mut current),
                    start: current_start,
                    end: i,
                    language: lang,
                });
            }
            tokens.push(Token {
                text: ch.to_string(),
                start: i,
                end: i + ch.len_utf8(),
                language: lang,
            });
            start_byte = i + ch.len_utf8();
        } else {
            if current.is_empty() {
                current_start = i;
            }
            // Normalize apostrophes to ASCII.
            let normalized = if is_apostrophe { '\'' } else { ch };
            current.push(normalized);
            start_byte = i + ch.len_utf8();
        }
    }
    let _ = start_byte;
    if !current.is_empty() {
        tokens.push(Token {
            text: current,
            start: current_start,
            end: text.len(),
            language: lang,
        });
    }
    tokens
}

/// Sentence-tokenize a string.
///
/// Algorithm: split on sentence terminators: split on `.`, `!`, `?`, `。`, `！`, `？`
/// followed by whitespace or end of string. Handles common abbreviations
/// (Mr., Dr., etc.) for English.
#[must_use]
pub fn sentence_tokenize(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let abbreviations = [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
        "etc", "vs", "fig", "vol", "no", "pp", "inc", "ltd", "co",
    ];

    for i in 0..chars.len() {
        let (_, ch) = chars[i];
        current.push(ch);
        let is_terminator = matches!(ch, '.' | '!' | '?' | '。' | '！' | '？');
        if is_terminator {
            // For CJK terminators, split regardless of following whitespace.
            let is_cjk_terminator = matches!(ch, '。' | '！' | '？');
            // Look ahead for whitespace or end.
            let next_is_space = chars
                .get(i + 1)
                .map(|(_, c)| c.is_whitespace())
                .unwrap_or(true);
            if is_cjk_terminator || next_is_space || i == chars.len() - 1 {
                // Check for abbreviations (English heuristic).
                let lower = current.to_lowercase();
                let last_word: String = lower
                    .rsplit(|c: char| c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .trim_end_matches('.')
                    .to_string();
                if abbreviations.contains(&last_word.as_str()) && ch == '.' {
                    // Not a real sentence break.
                    continue;
                }
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

fn is_cjk_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
        || (0x2F800..=0x2FA1F).contains(&code)
        || (0x3000..=0x303F).contains(&code) // CJK symbols & punctuation
        || (0x3040..=0x309F).contains(&code) // Hiragana
        || (0x30A0..=0x30FF).contains(&code) // Katakana
        || (0xAC00..=0xD7AF).contains(&code) // Hangul
}

fn is_punctuation(c: char) -> bool {
    if c.is_ascii_punctuation() {
        return true;
    }
    let code = c as u32;
    // General punctuation, CJK punctuation, etc.
    (0x2000..=0x206F).contains(&code)
        || (0x3000..=0x303F).contains(&code)
        || (0xFF00..=0xFFEF).contains(&code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_tokenize_english() {
        let tokens = word_tokenize("Hello, world! How are you?");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            vec!["Hello", ",", "world", "!", "How", "are", "you", "?"]
        );
    }

    #[test]
    fn word_tokenize_preserves_contractions() {
        let tokens = word_tokenize("don't stop believing");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["don't", "stop", "believing"]);
    }

    #[test]
    fn word_tokenize_splits_cjk() {
        let tokens = word_tokenize("Hello世界");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["Hello", "世", "界"]);
    }

    #[test]
    fn sentence_tokenize_english() {
        let s = sentence_tokenize("Hello world. How are you? I'm fine!");
        assert_eq!(s, vec!["Hello world.", "How are you?", "I'm fine!"]);
    }

    #[test]
    fn sentence_tokenize_handles_abbreviations() {
        let s = sentence_tokenize("Dr. Smith went to the store. He bought milk.");
        assert_eq!(s.len(), 2);
        assert!(s[0].contains("Dr. Smith"));
    }

    #[test]
    fn sentence_tokenize_chinese() {
        let s = sentence_tokenize("你好。世界！今天怎么样？");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn token_positions_are_correct() {
        let tokens = word_tokenize("ab cd");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 2);
        assert_eq!(tokens[1].start, 3);
        assert_eq!(tokens[1].end, 5);
    }
}
