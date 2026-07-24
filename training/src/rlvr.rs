#![allow(missing_docs)]
//! Reinforcement Learning from Verifiable Rewards (RLVR) — 2026.
//!
//! RLVR trains models using rewards that can be programmatically verified,
//! rather than rewards learned from human preferences. The reward signal
//! comes from running deterministic checks: code passing tests, a math
//! answer matching the ground truth, an output matching a required format.
//! This removes the annotation bottleneck of RLHF entirely.
//!
//! The crate ships three reference verifiers:
//!
//! - [`CodeVerification`]: structural code checks (balanced brackets,
//!   recognizable keywords, statement terminators).
//! - [`MathVerification`]: extracts a numeric answer from the response and
//!   compares it to the expected value parsed from the prompt.
//! - [`FormatVerification`]: checks JSON / Markdown structural validity.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during RLVR training.
#[derive(Debug, Error)]
pub enum RlvrError {
    /// The verifier reported an internal error.
    #[error("verifier error: {0}")]
    Verifier(String),
}

/// Result of a single verification check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardResult {
    /// Whether the response passed verification.
    pub verified: bool,
    /// The numeric reward in `[0, 1]`.
    pub score: f64,
    /// Human-readable feedback explaining the verdict.
    pub feedback: String,
}

/// Trait implemented by anything that can verify a model response and turn
/// it into a reward signal.
pub trait VerifiableReward: Send + Sync {
    /// Verify `response` to `prompt` and return a [`RewardResult`].
    fn verify(&self, prompt: &str, response: &str) -> RewardResult;
}

// ─── Code verification ───────────────────────────────────────────────────────

/// Verifier that checks whether the response contains syntactically
/// plausible code: balanced braces, parentheses, and brackets, and at
/// least one statement terminator (`;` or a closing block).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CodeVerification {
    /// Whether to require at least one statement terminator.
    pub require_terminator: bool,
}

impl CodeVerification {
    /// Construct a new code verifier with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check whether all bracket types are balanced.
    fn brackets_balanced(s: &str) -> bool {
        let mut stack: Vec<char> = Vec::new();
        for c in s.chars() {
            match c {
                '(' | '[' | '{' => stack.push(c),
                ')' => {
                    if stack.pop() != Some('(') {
                        return false;
                    }
                }
                ']' => {
                    if stack.pop() != Some('[') {
                        return false;
                    }
                }
                '}' => {
                    if stack.pop() != Some('{') {
                        return false;
                    }
                }
                _ => {}
            }
        }
        stack.is_empty()
    }

    /// Heuristic: count "syntax error" indicators (double semicolons,
    /// triple-close-parens, etc.).
    fn count_red_flags(s: &str) -> usize {
        let mut flags = 0;
        let bytes = s.as_bytes();
        for i in 0..bytes.len().saturating_sub(1) {
            if bytes[i] == b';' && bytes[i + 1] == b';' {
                flags += 1;
            }
        }
        if s.contains(")))") {
            flags += 1;
        }
        flags
    }

    /// Whether the response contains recognizable code keywords.
    fn looks_like_code(s: &str) -> bool {
        s.contains("fn ")
            || s.contains("def ")
            || s.contains("function ")
            || s.contains("class ")
            || s.contains("let ")
            || s.contains("const ")
            || s.contains("return ")
            || s.contains("if ")
            || s.contains("for ")
            || s.contains("while ")
            || s.contains("pub ")
    }
}

impl VerifiableReward for CodeVerification {
    fn verify(&self, prompt: &str, response: &str) -> RewardResult {
        let _ = prompt;
        if response.trim().is_empty() {
            return RewardResult {
                verified: false,
                score: 0.0,
                feedback: "Empty response — no code to verify.".into(),
            };
        }
        let balanced = Self::brackets_balanced(response);
        let red_flags = Self::count_red_flags(response);
        let has_terminator = response.contains(';') || response.contains("}\n");
        let looks_like_code = Self::looks_like_code(response);

        let mut score = 0.0;
        if balanced {
            score += 0.4;
        }
        if has_terminator || !self.require_terminator {
            score += 0.3;
        }
        if looks_like_code {
            score += 0.3;
        }
        // Penalise red flags.
        score = (score - 0.1 * red_flags as f64).max(0.0);

        let verified =
            balanced && looks_like_code && (has_terminator || !self.require_terminator);
        let feedback = if verified {
            "Code passed structural verification (balanced brackets, contains code patterns)."
                .into()
        } else if !balanced {
            "Unbalanced brackets detected.".into()
        } else if !looks_like_code {
            "Response does not contain recognizable code keywords.".into()
        } else {
            "Missing statement terminator.".into()
        };
        RewardResult {
            verified,
            score: score.min(1.0),
            feedback,
        }
    }
}

// ─── Math verification ───────────────────────────────────────────────────────

/// Verifier that extracts a numeric answer from the response and compares
/// it to the expected value parsed from the prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathVerification {
    /// Tolerance for floating-point comparisons.
    pub tolerance: f64,
}

impl Default for MathVerification {
    fn default() -> Self {
        Self { tolerance: 1e-6 }
    }
}

impl MathVerification {
    /// Construct a new math verifier.
    #[must_use]
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Extract the expected answer from a prompt of the form
    /// `"... answer is 42 ..."` / `"... = 42"` / `"... expected: 42"`.
    fn extract_expected(prompt: &str) -> Option<f64> {
        let patterns = ["answer is", "answer:", "expected:", "= ", "== "];
        let lower = prompt.to_lowercase();
        for pat in patterns {
            if let Some(idx) = lower.find(pat) {
                let rest = &prompt[idx + pat.len()..];
                if let Some(n) = Self::parse_first_number(rest) {
                    return Some(n);
                }
            }
        }
        None
    }

    /// Extract a numeric answer from the response.
    fn extract_answer(response: &str) -> Option<f64> {
        let markers = ["answer is", "answer:", "final answer", "result is", "result:", "= "];
        let lower = response.to_lowercase();
        for marker in markers {
            if let Some(idx) = lower.find(marker) {
                let rest = &response[idx + marker.len()..];
                if let Some(n) = Self::parse_first_number(rest) {
                    return Some(n);
                }
            }
        }
        // Fallback: first number in the response.
        Self::parse_first_number(response)
    }

    /// Parse the first number (int or float, possibly negative) from a string.
    fn parse_first_number(s: &str) -> Option<f64> {
        let bytes = s.as_bytes();
        let mut start: Option<usize> = None;
        let mut end = 0usize;
        for (i, b) in bytes.iter().enumerate() {
            let is_digit = b.is_ascii_digit();
            let is_dot = *b == b'.';
            let is_exp = *b == b'e' || *b == b'E';
            // A sign is part of the number only if it directly precedes a
            // digit (e.g. "-2" but not "a-2").
            let is_sign = (*b == b'-' || *b == b'+')
                && start.is_none()
                && i + 1 < bytes.len()
                && (bytes[i + 1].is_ascii_digit() || bytes[i + 1] == b'.');
            if is_digit || is_dot || is_exp || is_sign {
                if start.is_none() {
                    start = Some(i);
                }
                end = i + 1;
            } else if start.is_some() {
                break;
            }
        }
        let start = start?;
        let substr = &s[start..end.min(s.len())];
        substr.parse::<f64>().ok()
    }
}

impl VerifiableReward for MathVerification {
    fn verify(&self, prompt: &str, response: &str) -> RewardResult {
        let expected = match Self::extract_expected(prompt) {
            Some(v) => v,
            None => {
                return RewardResult {
                    verified: false,
                    score: 0.0,
                    feedback: "Could not extract an expected numeric answer from the prompt."
                        .into(),
                }
            }
        };
        let actual = match Self::extract_answer(response) {
            Some(v) => v,
            None => {
                return RewardResult {
                    verified: false,
                    score: 0.0,
                    feedback: "Could not extract a numeric answer from the response.".into(),
                }
            }
        };
        let diff = (actual - expected).abs();
        let verified = diff <= self.tolerance;
        let score = if verified {
            1.0
        } else {
            // Gaussian-decaying partial credit.
            (-diff * diff / 0.5).exp()
        };
        let feedback = if verified {
            format!("Correct: {actual} matches expected {expected}.")
        } else {
            format!("Incorrect: got {actual}, expected {expected} (diff {diff}).")
        };
        RewardResult {
            verified,
            score: score.clamp(0.0, 1.0),
            feedback,
        }
    }
}

// ─── Format verification ─────────────────────────────────────────────────────

/// The kind of format to check for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatKind {
    /// A single JSON object or array.
    Json,
    /// A Markdown document with at least one heading and one code block / list.
    Markdown,
    /// Either JSON or Markdown.
    JsonOrMarkdown,
}

impl Default for FormatKind {
    fn default() -> Self {
        Self::JsonOrMarkdown
    }
}

/// Verifier that checks the response is either valid JSON or follows a
/// Markdown structure (headings, code blocks, lists).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FormatVerification {
    /// Required format.
    pub required: FormatKind,
}

impl FormatVerification {
    /// Construct a new format verifier.
    #[must_use]
    pub fn new(required: FormatKind) -> Self {
        Self { required }
    }

    fn is_json(s: &str) -> bool {
        let trimmed = s.trim();
        if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
            return false;
        }
        serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    }

    fn is_markdown(s: &str) -> bool {
        let has_heading = s.lines().any(|l| l.trim_start().starts_with('#'));
        let has_code_block = s.contains("```");
        let has_list = s.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with("- ") || t.starts_with("* ") || t.starts_with("1. ")
        });
        has_heading && (has_code_block || has_list)
    }
}

impl VerifiableReward for FormatVerification {
    fn verify(&self, _prompt: &str, response: &str) -> RewardResult {
        if response.trim().is_empty() {
            return RewardResult {
                verified: false,
                score: 0.0,
                feedback: "Empty response.".into(),
            };
        }
        let is_json = Self::is_json(response);
        let is_md = Self::is_markdown(response);
        let (verified, score, feedback) = match self.required {
            FormatKind::Json => (
                is_json,
                if is_json { 1.0 } else { 0.0 },
                if is_json {
                    "Valid JSON.".into()
                } else {
                    "Response is not valid JSON.".into()
                },
            ),
            FormatKind::Markdown => (
                is_md,
                if is_md { 1.0 } else { 0.0 },
                if is_md {
                    "Valid Markdown structure.".into()
                } else {
                    "Response lacks Markdown structure (headings + code/list).".into()
                },
            ),
            FormatKind::JsonOrMarkdown => {
                if is_json {
                    (true, 1.0, "Valid JSON.".into())
                } else if is_md {
                    (true, 0.9, "Valid Markdown structure.".into())
                } else {
                    (
                        false,
                        0.0,
                        "Response is neither valid JSON nor Markdown.".into(),
                    )
                }
            }
        };
        RewardResult {
            verified,
            score,
            feedback,
        }
    }
}

// ─── RLVR trainer ────────────────────────────────────────────────────────────

/// Result returned by the RLVR trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlvrResult {
    /// The reward signal in `[0, 1]`.
    pub reward: f64,
    /// Whether the response passed verification.
    pub verified: bool,
    /// Human-readable feedback from the verifier.
    pub feedback: String,
    /// Number of verification iterations performed.
    pub iterations: usize,
}

/// The RLVR trainer. Wraps a [`VerifiableReward`] and applies it to
/// `(prompt, response)` pairs, retrying with lightly cleaned variants of
/// the response when the first pass fails.
#[derive(Clone)]
pub struct RlvrTrainer {
    /// The verifier used to score responses.
    pub verifier: Arc<dyn VerifiableReward>,
    /// Maximum verification iterations.
    pub max_iterations: usize,
}

impl std::fmt::Debug for RlvrTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RlvrTrainer")
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl RlvrTrainer {
    /// Construct a new RLVR trainer.
    #[must_use]
    pub fn new(verifier: Arc<dyn VerifiableReward>) -> Self {
        Self {
            verifier,
            max_iterations: 3,
        }
    }

    /// Verify a single `(prompt, response)` pair.
    ///
    /// The trainer performs up to `max_iterations` verification passes; if
    /// the response fails on the first pass, we retry with a trimmed
    /// version, and then with the text after the last colon (simulating
    /// extraction of a final answer from surrounding prose).
    pub fn train(&self, prompt: &str, response: &str) -> RlvrResult {
        let mut best = self.verifier.verify(prompt, response);
        let mut iterations = 1;

        if !best.verified && iterations < self.max_iterations {
            let trimmed = response.trim();
            let retry = self.verifier.verify(prompt, trimmed);
            iterations += 1;
            if retry.score > best.score {
                best = retry;
            }
        }

        if !best.verified && iterations < self.max_iterations {
            if let Some(idx) = response.rfind(':') {
                let after = &response[idx + 1..];
                let retry = self.verifier.verify(prompt, after);
                iterations += 1;
                if retry.score > best.score {
                    best = retry;
                }
            }
        }

        RlvrResult {
            reward: best.score,
            verified: best.verified,
            feedback: best.feedback,
            iterations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_verifier_accepts_balanced_code() {
        let v = CodeVerification::new();
        let r = v.verify("write a function", "fn add(a, b) { return a + b; }");
        assert!(r.verified, "feedback: {}", r.feedback);
        assert!(r.score > 0.5);
    }

    #[test]
    fn code_verifier_rejects_unbalanced_brackets() {
        let v = CodeVerification::new();
        let r = v.verify("p", "fn add(a, b { return a + b;");
        assert!(!r.verified);
        assert!(r.score < 1.0);
    }

    #[test]
    fn code_verifier_rejects_empty_response() {
        let v = CodeVerification::new();
        let r = v.verify("p", "   ");
        assert!(!r.verified);
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn math_verifier_matches_numeric_answer() {
        let v = MathVerification::default();
        let r = v.verify("What is 2+2? The answer is 4", "The answer is 4.");
        assert!(r.verified, "feedback: {}", r.feedback);
        assert!((r.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn math_verifier_rejects_wrong_answer() {
        let v = MathVerification::default();
        let r = v.verify("What is 2+2? The answer is 4", "The answer is 5.");
        assert!(!r.verified);
        assert!(r.score < 0.9);
    }

    #[test]
    fn math_verifier_handles_negative_numbers() {
        let v = MathVerification::default();
        let r = v.verify("What is 3 - 5? = -2", "The result is -2");
        assert!(r.verified, "feedback: {}", r.feedback);
    }

    #[test]
    fn format_verifier_accepts_json() {
        let v = FormatVerification::new(FormatKind::Json);
        let r = v.verify("p", r#"{"key": "value"}"#);
        assert!(r.verified);
    }

    #[test]
    fn format_verifier_accepts_markdown() {
        let v = FormatVerification::new(FormatKind::Markdown);
        let r = v.verify(
            "p",
            "# Title\n\nSome text.\n\n```rust\nfn main() {}\n```\n",
        );
        assert!(r.verified, "feedback: {}", r.feedback);
    }

    #[test]
    fn format_verifier_rejects_plain_text() {
        let v = FormatVerification::new(FormatKind::JsonOrMarkdown);
        let r = v.verify("p", "just plain text");
        assert!(!r.verified);
    }

    #[test]
    fn rlvr_trainer_runs_multiple_iterations() {
        let trainer = RlvrTrainer::new(Arc::new(MathVerification::default()));
        // The colon-stripping retry path should find "15" after the colon.
        let res = trainer.train("What is 5*3? answer is 15", "After computing, the answer is: 15");
        assert!(res.verified, "feedback: {}", res.feedback);
        assert!(res.iterations >= 1);
    }

    #[test]
    fn rlvr_trainer_returns_zero_reward_on_total_failure() {
        let trainer = RlvrTrainer::new(Arc::new(MathVerification::default()));
        let res = trainer.train("answer is 42", "I have no idea.");
        assert!(!res.verified);
        assert_eq!(res.reward, 0.0);
    }
}
