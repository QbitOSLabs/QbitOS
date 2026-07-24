#![allow(missing_docs)]

//! Code analyzer tool: cyclomatic complexity, security, and style checks.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

use crate::tool_vm::{Tool, ToolKind};

/// Errors emitted by the code analyzer.
#[derive(Debug, Error)]
pub enum CodeAnalyzerError {
    /// Missing input.
    #[error("missing 'code' parameter")]
    MissingCode,
}

/// A detected security issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    /// Issue kind (e.g. `eval_usage`).
    pub kind: String,
    /// Severity: "high", "medium", or "low".
    pub severity: String,
    /// Line number (1-based).
    pub line: usize,
    /// Snippet of the offending code.
    pub snippet: String,
}

/// A style issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleIssue {
    /// Issue kind.
    pub kind: String,
    /// Line number (1-based).
    pub line: usize,
    /// Message.
    pub message: String,
}

/// Code analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Cyclomatic complexity.
    pub cyclomatic_complexity: u32,
    /// Number of lines.
    pub lines: u32,
    /// Non-blank, non-comment lines.
    pub sloc: u32,
    /// Detected security issues.
    pub security_issues: Vec<SecurityIssue>,
    /// Detected style issues.
    pub style_issues: Vec<StyleIssue>,
    /// Overall risk score in [0, 1].
    pub risk_score: f32,
}

/// Tool that analyzes code for complexity, security, and style.
pub struct CodeAnalyzerTool;

impl Default for CodeAnalyzerTool { fn default() -> Self { Self } }

#[async_trait]
impl Tool for CodeAnalyzerTool {
    fn name(&self) -> &str { "code_analyzer" }
    fn kind(&self) -> ToolKind { ToolKind::Custom }
    fn action(&self) -> &str { "analyze" }
    fn scope(&self, _: &Value) -> String { "code".into() }
    fn schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object", "required": ["code"],
            "properties": {
                "code": { "type": "string" },
                "language": { "type": "string" }
            }
        }))
    }
    async fn execute(&self, params: &Value) -> Result<Value, String> {
        let code = params.get("code").and_then(|v| v.as_str())
            .ok_or(CodeAnalyzerError::MissingCode.to_string())?;
        serde_json::to_value(analyze(code)).map_err(|e| e.to_string())
    }
}

/// Analyze a code string and produce a report.
#[must_use]
pub fn analyze(code: &str) -> AnalysisReport {
    let lines: Vec<&str> = code.lines().collect();
    let lines_count = lines.len() as u32;
    let sloc = lines.iter().filter(|l| {
        let t = l.trim();
        !t.is_empty() && !t.starts_with("//") && !t.starts_with('#') && !t.starts_with("/*")
    }).count() as u32;
    let complexity = cyclomatic_complexity(code);
    let security_issues = detect_security(&lines);
    let style_issues = detect_style(&lines);
    let high = security_issues.iter().filter(|i| i.severity == "high").count() as f32;
    let med = security_issues.iter().filter(|i| i.severity == "medium").count() as f32;
    let low = security_issues.iter().filter(|i| i.severity == "low").count() as f32;
    let risk = (high * 0.5 + med * 0.25 + low * 0.1).clamp(0.0, 1.0);
    AnalysisReport { cyclomatic_complexity: complexity, lines: lines_count, sloc, security_issues, style_issues, risk_score: risk }
}

fn cyclomatic_complexity(code: &str) -> u32 {
    let lower = code.to_lowercase();
    let mut cc = 1u32;
    for kw in ["if ", "if(", "else if", "elif ", "for ", "for(", "while ", "while(",
        "loop ", "match ", "switch ", "case ", "catch ", "?", "&&", "||", "do {", "do{"] {
        cc += lower.matches(kw).count() as u32;
    }
    cc
}

fn detect_security(lines: &[&str]) -> Vec<SecurityIssue> {
    const PATTERNS: &[(&str, &str, &str)] = &[
        ("eval(", "eval_usage", "high"), ("exec(", "exec_usage", "high"),
        ("exec (", "exec_usage", "high"), ("os.system(", "os_system", "high"),
        ("subprocess.call", "subprocess", "high"), ("innerHTML", "innerHTML", "medium"),
        ("dangerouslySetInnerHTML", "dangerous_html", "high"),
        ("document.write", "document_write", "medium"), ("shell=True", "shell_true", "high"),
        ("password =", "hardcoded_password", "medium"),
        ("api_key =", "hardcoded_secret", "medium"),
        ("secret_key =", "hardcoded_secret", "medium"),
        ("BEGIN RSA PRIVATE KEY", "embedded_private_key", "high"),
        ("SELECT * FROM", "raw_sql", "medium"), ("new Function(", "dynamic_function", "medium"),
    ];
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        for (pat, kind, sev) in PATTERNS {
            if line.contains(pat) {
                out.push(SecurityIssue {
                    kind: (*kind).into(), severity: (*sev).into(), line: i + 1,
                    snippet: line.trim().chars().take(80).collect(),
                });
            }
        }
    }
    out
}

fn detect_style(lines: &[&str]) -> Vec<StyleIssue> {
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.len() > 120 {
            out.push(StyleIssue { kind: "long_line".into(), line: i + 1,
                message: format!("line exceeds 120 chars ({} chars)", line.len()) });
        }
        if !line.is_empty() && line.ends_with(' ') {
            out.push(StyleIssue { kind: "trailing_whitespace".into(), line: i + 1,
                message: "trailing whitespace".into() });
        }
        if line.contains('\t') && line.contains(' ') && !line.starts_with('\t') {
            out.push(StyleIssue { kind: "mixed_indent".into(), line: i + 1,
                message: "mixed tabs and spaces".into() });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_complexity() {
        let code = "fn main() {\n  if a { for i in 0..10 { while x { y(); } } }\n  match z { 1 => {}, _ => {} }\n}";
        let r = analyze(code);
        assert!(r.cyclomatic_complexity >= 4);
    }

    #[test]
    fn detects_eval_and_exec() {
        let code = "let r = eval('1+1');\nexec('rm -rf /');\nos.system('ls');";
        let r = analyze(code);
        assert!(r.security_issues.iter().any(|i| i.kind == "eval_usage"));
        assert!(r.security_issues.iter().any(|i| i.kind == "exec_usage"));
        assert!(r.security_issues.iter().any(|i| i.kind == "os_system"));
        assert!(r.risk_score > 0.0);
    }

    #[test]
    fn detects_long_lines() {
        let code = format!("let x = {};\n", "a".repeat(200));
        assert!(analyze(&code).style_issues.iter().any(|s| s.kind == "long_line"));
    }

    #[test]
    fn sloc_excludes_comments_and_blanks() {
        assert_eq!(analyze("// comment\n\nlet x = 1;\nlet y = 2;\n").sloc, 2);
    }
}
