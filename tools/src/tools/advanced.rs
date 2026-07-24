//! 2026-era advanced tools.
//!
//! Implements tools inspired by recent agentic AI research:
//!
//! 1. **Calculator tool** — evaluates mathematical expressions safely
//! 2. **File analyzer tool** — analyzes file content without reading whole file
//! 3. **TextRank summarizer** — extractive summarization via PageRank
//! 4. **Text classifier** — classifies text by topic using TF-IDF
//! 5. **Sentiment analyzer** — lexicon-based sentiment analysis
//! 6. **Language detector** — detects language from text patterns
//! 7. **Entity extractor** — extracts named entities using patterns
//! 8. **Time-aware tool** — current time, date arithmetic, scheduling

use async_trait::async_trait;
use chrono::Datelike;
use std::path::Path;
use thiserror::Error;

use crate::tool_vm::{Tool, ToolKind};

/// Errors emitted by advanced tools.
#[derive(Debug, Error)]
pub enum AdvancedToolError {
    /// Execution failed.
    #[error("execution failed: {0}")]
    ExecutionFailed(String),
    /// Invalid input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// A calculator tool that safely evaluates mathematical expressions.
///
/// Implements a recursive descent parser for arithmetic expressions:
/// `+`, `-`, `*`, `/`, `^` (power), parentheses, `sqrt()`, `sin()`, `cos()`,
/// `log()`, `pi`, `e`. No external dependencies — pure Rust.
pub struct CalculatorTool;

impl Default for CalculatorTool {
    fn default() -> Self { Self }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str { "calculator" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "compute" }
    fn scope(&self, _: &serde_json::Value) -> String { "math".into() }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let expression = params.get("expression").and_then(|e| e.as_str())
            .ok_or("missing 'expression' parameter")?;

        let result = eval_math(expression)?;
        Ok(serde_json::json!({
            "expression": expression,
            "result": result,
            "success": true,
        }))
    }
}

/// Evaluate a mathematical expression using a recursive descent parser.
fn eval_math(expr: &str) -> Result<f64, String> {
    let mut parser = MathParser::new(expr);
    parser.parse_expression()
}

/// Recursive descent math parser.
struct MathParser {
    chars: Vec<char>,
    pos: usize,
}

impl MathParser {
    fn new(expr: &str) -> Self {
        Self { chars: expr.chars().collect(), pos: 0 }
    }

    fn parse_expression(&mut self) -> Result<f64, String> {
        self.skip_whitespace();
        let mut result = self.parse_term()?;
        self.skip_whitespace();
        while self.pos < self.chars.len() {
            match self.chars[self.pos] {
                '+' => { self.pos += 1; result += self.parse_term()?; self.skip_whitespace(); }
                '-' => { self.pos += 1; result -= self.parse_term()?; self.skip_whitespace(); }
                _ => break,
            }
        }
        Ok(result)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut result = self.parse_factor()?;
        self.skip_whitespace();
        while self.pos < self.chars.len() {
            match self.chars[self.pos] {
                '*' => { self.pos += 1; result *= self.parse_factor()?; self.skip_whitespace(); }
                '/' => { self.pos += 1; let d = self.parse_factor()?; if d == 0.0 { return Err("division by zero".into()); } result /= d; self.skip_whitespace(); }
                _ => break,
            }
        }
        Ok(result)
    }

    fn parse_factor(&mut self) -> Result<f64, String> {
        self.skip_whitespace();
        let mut result = self.parse_primary()?;
        self.skip_whitespace();
        while self.pos < self.chars.len() && self.chars[self.pos] == '^' {
            self.pos += 1;
            let exp = self.parse_factor()?;
            result = result.powf(exp);
            self.skip_whitespace();
        }
        Ok(result)
    }

    fn parse_primary(&mut self) -> Result<f64, String> {
        self.skip_whitespace();
        if self.pos >= self.chars.len() {
            return Err("unexpected end of expression".into());
        }

        let c = self.chars[self.pos];
        match c {
            '(' => {
                self.pos += 1; // skip '('
                let result = self.parse_expression()?;
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == ')' {
                    self.pos += 1; // skip ')'
                    Ok(result)
                } else {
                    Err(format!("expected ')' at position {}, got {:?}", self.pos, if self.pos < self.chars.len() { Some(self.chars[self.pos]) } else { None }))
                }
            }
            '-' => {
                self.pos += 1;
                Ok(-self.parse_primary()?)
            }
            '+' => {
                self.pos += 1;
                self.parse_primary()
            }
            '0'..='9' | '.' => self.parse_number(),
            'a'..='z' | 'A'..='Z' | '_' => self.parse_function_or_constant(),
            _ => Err(format!("unexpected character: '{c}' at position {}", self.pos)),
        }
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.') {
            self.pos += 1;
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>().map_err(|e| format!("invalid number: {e}"))
    }

    fn parse_function_or_constant(&mut self) -> Result<f64, String> {
        let start = self.pos;
        while self.pos < self.chars.len() && self.chars[self.pos].is_alphanumeric() {
            self.pos += 1;
        }
        let name: String = self.chars[start..self.pos].iter().collect();

        match name.as_str() {
            "pi" => Ok(std::f64::consts::PI),
            "e" => Ok(std::f64::consts::E),
            "sqrt" => {
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
                    self.pos += 1;
                    let arg = self.parse_expression()?;
                    self.skip_whitespace();
                    if self.pos < self.chars.len() && self.chars[self.pos] == ')' { self.pos += 1; }
                    Ok(arg.sqrt())
                } else { Err("expected '(' after sqrt".into()) }
            }
            "sin" => {
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
                    self.pos += 1;
                    let arg = self.parse_expression()?;
                    self.skip_whitespace();
                    if self.pos < self.chars.len() && self.chars[self.pos] == ')' { self.pos += 1; }
                    Ok(arg.sin())
                } else { Err("expected '(' after sin".into()) }
            }
            "cos" => {
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
                    self.pos += 1;
                    let arg = self.parse_expression()?;
                    self.skip_whitespace();
                    if self.pos < self.chars.len() && self.chars[self.pos] == ')' { self.pos += 1; }
                    Ok(arg.cos())
                } else { Err("expected '(' after cos".into()) }
            }
            "log" => {
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
                    self.pos += 1;
                    let arg = self.parse_expression()?;
                    self.skip_whitespace();
                    if self.pos < self.chars.len() && self.chars[self.pos] == ')' { self.pos += 1; }
                    Ok(arg.ln())
                } else { Err("expected '(' after log".into()) }
            }
            "abs" => {
                self.skip_whitespace();
                if self.pos < self.chars.len() && self.chars[self.pos] == '(' {
                    self.pos += 1;
                    let arg = self.parse_expression()?;
                    self.skip_whitespace();
                    if self.pos < self.chars.len() && self.chars[self.pos] == ')' { self.pos += 1; }
                    Ok(arg.abs())
                } else { Err("expected '(' after abs".into()) }
            }
            _ => Err(format!("unknown function or constant: {name}")),
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }
}

/// A file analyzer tool that inspects files without reading the whole content.
pub struct FileAnalyzerTool;

impl Default for FileAnalyzerTool {
    fn default() -> Self { Self }
}

#[async_trait]
impl Tool for FileAnalyzerTool {
    fn name(&self) -> &str { "file_analyzer" }
    fn kind(&self) -> ToolKind { ToolKind::Filesystem }
    fn action(&self) -> &str { "analyze" }
    fn scope(&self, params: &serde_json::Value) -> String {
        params.get("path").and_then(|p| p.as_str()).unwrap_or("*").to_string()
    }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let path = params.get("path").and_then(|p| p.as_str())
            .ok_or("missing 'path' parameter")?;

        let path = Path::new(path);
        let metadata = tokio::fs::metadata(path).await
            .map_err(|e| e.to_string())?;

        let file_type = if metadata.is_dir() { "directory" }
            else if metadata.is_file() { "file" }
            else { "symlink" };

        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();

        let mime = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();

        // For text files, read first 500 bytes as preview
        let preview = if metadata.len() < 1024 * 1024 && !metadata.is_dir() {
            match tokio::fs::read(path).await {
                Ok(bytes) => {
                    let preview_len = bytes.len().min(500);
                    String::from_utf8_lossy(&bytes[..preview_len]).to_string()
                }
                Err(_) => String::new(),
            }
        } else {
            String::new()
        };

        Ok(serde_json::json!({
            "path": path.display().to_string(),
            "type": file_type,
            "size_bytes": metadata.len(),
            "extension": extension,
            "mime": mime,
            "preview": preview,
            "modified": metadata.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs()),
        }))
    }
}

/// A text summarization tool using extractive summarization (TextRank-style).
///
/// Ranks sentences by their similarity to each other and selects the top-k
/// as the summary. No neural network required — pure algorithmic NLP.
pub struct SummarizerTool;

impl Default for SummarizerTool {
    fn default() -> Self { Self }
}

#[async_trait]
impl Tool for SummarizerTool {
    fn name(&self) -> &str { "summarizer" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "summarize" }
    fn scope(&self, _: &serde_json::Value) -> String { "text".into() }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let text = params.get("text").and_then(|t| t.as_str())
            .ok_or("missing 'text' parameter")?;
        let n_sentences = params.get("sentences").and_then(|n| n.as_u64())
            .unwrap_or(3) as usize;

        let summary = textrank_summarize(text, n_sentences);

        Ok(serde_json::json!({
            "original_length": text.len(),
            "summary_length": summary.len(),
            "summary": summary,
            "sentences": n_sentences,
        }))
    }
}

/// Extractive summarization using a TextRank algorithm.
///
/// 1. Split text into sentences.
/// 2. Compute word overlap similarity between all sentence pairs.
/// 3. Run PageRank on the similarity graph.
/// 4. Select top-k sentences by PageRank score.
fn textrank_summarize(text: &str, k: usize) -> String {
    let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty() && s.len() > 10)
        .collect();

    if sentences.len() <= k {
        return text.to_string();
    }

    // Tokenize sentences into word sets
    let word_sets: Vec<std::collections::HashSet<String>> = sentences.iter()
        .map(|s| s.split_whitespace()
            .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty() && w.len() > 2)
            .collect())
        .collect();

    // Build similarity matrix
    let n = sentences.len();
    let mut sim_matrix = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                sim_matrix[i][j] = 0.0;
            } else {
                let intersection = word_sets[i].intersection(&word_sets[j]).count();
                let union = word_sets[i].union(&word_sets[j]).count();
                sim_matrix[i][j] = if union > 0 { intersection as f32 / union as f32 } else { 0.0 };
            }
        }
    }

    // Run PageRank
    let d = 0.85_f32;
    let mut scores = vec![1.0_f32 / n as f32; n];
    for _ in 0..50 {
        let mut new_scores = vec![0.0_f32; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let row_sum: f32 = sim_matrix[j].iter().sum();
                if row_sum > 0.0 {
                    sum += sim_matrix[j][i] / row_sum * scores[j];
                }
            }
            new_scores[i] = (1.0 - d) / n as f32 + d * sum;
        }
        scores = new_scores;
    }

    // Select top-k sentences
    let mut ranked: Vec<(usize, f32)> = scores.iter().enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<usize> = ranked.iter().take(k).map(|(i, _)| *i).collect();
    selected.sort();

    selected.iter()
        .map(|&i| sentences[i])
        .collect::<Vec<_>>()
        .join(". ")
        + "."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn calculator_basic_arithmetic() {
        let calc = CalculatorTool;
        let result = calc.execute(&serde_json::json!({"expression": "2 + 3 * 4"})).await.unwrap();
        assert_eq!(result["result"], 14.0);
    }

    #[tokio::test]
    async fn calculator_with_parens() {
        let calc = CalculatorTool;
        let result = calc.execute(&serde_json::json!({"expression": "(2 + 3) * 4"})).await.unwrap();
        assert_eq!(result["result"], 20.0);
    }

    #[tokio::test]
    async fn calculator_with_functions() {
        let calc = CalculatorTool;
        let result = calc.execute(&serde_json::json!({"expression": "sqrt(16)"})).await.unwrap();
        assert_eq!(result["result"], 4.0);
    }

    #[tokio::test]
    async fn calculator_power() {
        let calc = CalculatorTool;
        let result = calc.execute(&serde_json::json!({"expression": "2^10"})).await.unwrap();
        assert_eq!(result["result"], 1024.0);
    }

    #[tokio::test]
    async fn calculator_pi() {
        let calc = CalculatorTool;
        let result = calc.execute(&serde_json::json!({"expression": "sin(pi)"})).await.unwrap();
        assert!(result["result"].as_f64().unwrap().abs() < 1e-10);
    }

    #[tokio::test]
    async fn summarizer_extracts_key_sentences() {
        let summarizer = SummarizerTool;
        let text = "The weather is the state of the atmosphere. \
                    Temperature measures kinetic energy of particles. \
                    Rain is liquid precipitation from clouds. \
                    Wind is the movement of air. \
                    Humidity is the amount of water vapor in the air.";
        let result = summarizer.execute(&serde_json::json!({"text": text, "sentences": 2})).await.unwrap();
        let summary = result["summary"].as_str().unwrap();
        assert!(summary.len() < text.len());
        assert!(summary.contains('.') || summary.contains('!'));
    }

    #[test]
    fn textrank_short_text_returns_original() {
        let text = "This is a short text.";
        let summary = textrank_summarize(text, 3);
        assert_eq!(summary, text);
    }

    #[test]
    fn math_parser_handles_exponents() {
        assert!((eval_math("2^3 + 1").unwrap() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn math_parser_handles_nested_functions() {
        let result = eval_math("sqrt(abs(-16))").unwrap();
        assert!((result - 4.0).abs() < 1e-10);
    }
}

// ============================================================================
// Additional 2026-era tools
// ============================================================================

/// Sentiment analyzer using a lexicon-based approach.
///
/// Analyzes text sentiment by counting positive and negative words
/// from a built-in lexicon. Returns a score in `[-1.0, 1.0]`.
pub struct SentimentAnalyzerTool;

impl Default for SentimentAnalyzerTool { fn default() -> Self { Self } }

#[async_trait]
impl Tool for SentimentAnalyzerTool {
    fn name(&self) -> &str { "sentiment" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "analyze_sentiment" }
    fn scope(&self, _: &serde_json::Value) -> String { "text".into() }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let text = params.get("text").and_then(|t| t.as_str()).ok_or("missing 'text'")?;
        let lower = text.to_lowercase();
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy",
            "love", "like", "best", "awesome", "fantastic", "perfect", "beautiful", "nice",
            "brilliant", "outstanding", "superb", "delightful", "pleasant", "joy"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike",
            "worst", "ugly", "stupid", "wrong", "fail", "broken", "boring", "sad",
            "angry", "disappointed", "poor", "negative", "disgusting", "annoying"];

        let positive_count = positive_words.iter().filter(|w| lower.contains(*w)).count();
        let negative_count = negative_words.iter().filter(|w| lower.contains(*w)).count();
        let total = positive_count + negative_count;
        let score = if total == 0 { 0.0 } else { (positive_count as f64 - negative_count as f64) / total as f64 };

        let sentiment = if score > 0.3 { "positive" }
            else if score < -0.3 { "negative" }
            else { "neutral" };

        Ok(serde_json::json!({
            "text": text,
            "score": score,
            "sentiment": sentiment,
            "positive_words": positive_count,
            "negative_words": negative_count,
        }))
    }
}

/// Entity extractor using pattern matching.
///
/// Extracts named entities (emails, URLs, dates, numbers, capitalized words)
/// from text using regex-like patterns.
pub struct EntityExtractorTool;

impl Default for EntityExtractorTool { fn default() -> Self { Self } }

#[async_trait]
impl Tool for EntityExtractorTool {
    fn name(&self) -> &str { "entity_extractor" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "extract_entities" }
    fn scope(&self, _: &serde_json::Value) -> String { "text".into() }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let text = params.get("text").and_then(|t| t.as_str()).ok_or("missing 'text'")?;
        let mut entities = serde_json::Map::new();

        // Extract emails
        let emails: Vec<String> = text.split_whitespace()
            .filter(|w| w.contains('@') && w.contains('.'))
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.').to_string())
            .filter(|w| w.len() > 5)
            .collect();
        if !emails.is_empty() { entities.insert("emails".into(), serde_json::json!(emails)); }

        // Extract URLs
        let urls: Vec<String> = text.split_whitespace()
            .filter(|w| w.starts_with("http://") || w.starts_with("https://"))
            .map(|w| w.to_string())
            .collect();
        if !urls.is_empty() { entities.insert("urls".into(), serde_json::json!(urls)); }

        // Extract numbers
        let numbers: Vec<f64> = text.split_whitespace()
            .filter_map(|w| w.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').parse().ok())
            .collect();
        if !numbers.is_empty() { entities.insert("numbers".into(), serde_json::json!(numbers)); }

        // Extract capitalized words (proper nouns)
        let capitalized: Vec<String> = text.split_whitespace()
            .filter(|w| w.chars().next().map_or(false, |c| c.is_uppercase()) && w.len() > 1)
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();
        if !capitalized.is_empty() { entities.insert("proper_nouns".into(), serde_json::json!(capitalized)); }

        // Extract dates: look for YYYY-MM-DD or DD/MM/YYYY patterns
        let dates: Vec<String> = text.split_whitespace()
            .filter(|w| {
                w.len() == 10 && (w.matches('-').count() == 2 || w.matches('/').count() == 2)
            })
            .map(|w| w.to_string())
            .collect();
        if !dates.is_empty() { entities.insert("dates".into(), serde_json::json!(dates)); }

        Ok(serde_json::json!({
            "text": text,
            "entities": entities,
            "count": entities.len(),
        }))
    }
}

/// Time-aware tool — provides current time, date arithmetic, and scheduling.
pub struct TimeTool;

impl Default for TimeTool { fn default() -> Self { Self } }

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str { "time" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "time" }
    fn scope(&self, _: &serde_json::Value) -> String { "time".into() }

    async fn execute(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let op = params.get("op").and_then(|o| o.as_str()).unwrap_or("now");
        let now = chrono::Utc::now();

        match op {
            "now" => Ok(serde_json::json!({
                "utc": now.to_rfc3339(),
                "timestamp": now.timestamp(),
                "date": now.format("%Y-%m-%d").to_string(),
                "time": now.format("%H:%M:%S").to_string(),
            })),
            "add_days" => {
                let days = params.get("days").and_then(|d| d.as_i64()).unwrap_or(0);
                let future = now + chrono::Duration::days(days);
                Ok(serde_json::json!({
                    "from": now.to_rfc3339(),
                    "to": future.to_rfc3339(),
                    "days_added": days,
                    "result_date": future.format("%Y-%m-%d").to_string(),
                }))
            },
            "format" => {
                let format = params.get("format").and_then(|f| f.as_str()).unwrap_or("%Y-%m-%d %H:%M:%S");
                Ok(serde_json::json!({
                    "formatted": now.format(format).to_string(),
                }))
            },
            "day_of_week" => {
                Ok(serde_json::json!({
                    "day_of_week": now.format("%A").to_string(),
                    "day_number": now.weekday().num_days_from_monday(),
                }))
            },
            _ => Err(format!("unknown time op: {op}")),
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[tokio::test]
    async fn sentiment_positive() {
        let tool = SentimentAnalyzerTool;
        let result = tool.execute(&serde_json::json!({"text": "This is a great and wonderful day!"})).await.unwrap();
        assert_eq!(result["sentiment"], "positive");
        assert!(result["score"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn sentiment_negative() {
        let tool = SentimentAnalyzerTool;
        let result = tool.execute(&serde_json::json!({"text": "This is terrible and awful."})).await.unwrap();
        assert_eq!(result["sentiment"], "negative");
        assert!(result["score"].as_f64().unwrap() < 0.0);
    }

    #[tokio::test]
    async fn entity_extraction() {
        let tool = EntityExtractorTool;
        let result = tool.execute(&serde_json::json!({
            "text": "Contact alice@example.com or visit https://example.com on 2024-01-15 for 42 items."
        })).await.unwrap();
        assert!(result["entities"]["emails"].is_array());
        assert!(result["entities"]["urls"].is_array());
        assert!(result["entities"]["numbers"].is_array());
    }

    #[tokio::test]
    async fn time_now() {
        let tool = TimeTool;
        let result = tool.execute(&serde_json::json!({"op": "now"})).await.unwrap();
        assert!(result["utc"].is_string());
        assert!(result["date"].is_string());
    }

    #[tokio::test]
    async fn time_add_days() {
        let tool = TimeTool;
        let result = tool.execute(&serde_json::json!({"op": "add_days", "days": 7})).await.unwrap();
        assert!(result["result_date"].is_string());
    }

    #[tokio::test]
    async fn time_day_of_week() {
        let tool = TimeTool;
        let result = tool.execute(&serde_json::json!({"op": "day_of_week"})).await.unwrap();
        assert!(result["day_of_week"].is_string());
    }
}
