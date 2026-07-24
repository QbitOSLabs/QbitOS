//! Production guardrails stack (2026 AI Guardrails Production Guide).
#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GuardrailsError {
    #[error("input exceeds max length: {0} > {1}")] InputTooLong(usize, usize),
    #[error("output exceeds max length: {0} > {1}")] OutputTooLong(usize, usize),
    #[error("banned content: {0}")] BannedContent(String),
    #[error("rate limit exceeded: {0}")] RateLimitExceeded(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailsConfig {
    pub max_input_length: usize, pub max_output_length: usize, pub redact_pii: bool, pub scan_outputs: bool,
    pub banned_substrings: Vec<String>, pub allowed_url_domains: Vec<String>, pub audit_enabled: bool,
}
impl Default for GuardrailsConfig { fn default() -> Self { Self { max_input_length: 8192, max_output_length: 4096, redact_pii: true, scan_outputs: true, banned_substrings: vec!["ssn:".into(), "password:".into(), "api_key:".into(), "secret:".into()], allowed_url_domains: vec![], audit_enabled: true } } }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PiiKind { Email, Phone, Ssn, CreditCard, IpAddress }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiMatch { pub kind: PiiKind, pub start: usize, pub end: usize, pub matched: String, pub replacement: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputCheckResult { pub passed: bool, pub redacted_text: String, pub pii_matches: Vec<PiiMatch>, pub banned_matches: Vec<String>, pub rejection_reason: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputCheckResult { pub passed: bool, pub filtered_output: String, pub banned_matches: Vec<String>, pub urls_found: Vec<String>, pub rejection_reason: Option<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailAuditEvent { pub timestamp_ms: u64, pub event_type: GuardrailEventType, pub passed: bool, pub pii_count: usize, pub banned_count: usize, pub text_preview: String }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailEventType { InputCheck, OutputCheck, RateLimit }

pub struct Guardrails {
    cfg: GuardrailsConfig,
    email_re: Regex, phone_re: Regex, ssn_re: Regex, cc_re: Regex, ip_re: Regex, url_re: Regex,
    audit_log: RwLock<Vec<GuardrailAuditEvent>>,
}

impl Guardrails {
    #[must_use] pub fn new(cfg: GuardrailsConfig) -> Self {
        Self { cfg, email_re: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap(),
            phone_re: Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap(), ssn_re: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            cc_re: Regex::new(r"\b(?:\d[ -]*?){13,16}\b").unwrap(), ip_re: Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap(),
            url_re: Regex::new(r"https?://[A-Za-z0-9.-]+").unwrap(), audit_log: RwLock::new(Vec::new()) }
    }
    pub fn check_input(&self, text: &str) -> Result<InputCheckResult, GuardrailsError> {
        if text.len() > self.cfg.max_input_length { return Err(GuardrailsError::InputTooLong(text.len(), self.cfg.max_input_length)); }
        let mut redacted = text.to_string(); let mut pii = Vec::new();
        if self.cfg.redact_pii { let (r, m) = self.redact(&redacted); redacted = r; pii = m; }
        let mut banned = Vec::new(); let lower = redacted.to_lowercase();
        for b in &self.cfg.banned_substrings { if lower.contains(&b.to_lowercase()) { banned.push(b.clone()); } }
        let passed = banned.is_empty();
        if self.cfg.audit_enabled { self.audit(GuardrailEventType::InputCheck, passed, pii.len(), banned.len(), text); }
        Ok(InputCheckResult { passed, redacted_text: redacted, pii_matches: pii, banned_matches: banned, rejection_reason: if !passed { Some("banned substrings".into()) } else { None } })
    }
    pub fn check_output(&self, output: &str) -> Result<OutputCheckResult, GuardrailsError> {
        if output.len() > self.cfg.max_output_length { return Err(GuardrailsError::OutputTooLong(output.len(), self.cfg.max_output_length)); }
        let mut filtered = output.to_string(); let mut banned = Vec::new();
        if self.cfg.scan_outputs { let lower = filtered.to_lowercase();
            for b in &self.cfg.banned_substrings { if lower.contains(&b.to_lowercase()) { banned.push(b.clone()); let p = regex::escape(b); let re = Regex::new(&p).unwrap(); filtered = re.replace_all(&filtered, "[REDACTED]").to_string(); } } }
        let urls: Vec<String> = self.url_re.find_iter(output).map(|m| m.as_str().to_string()).collect();
        let mut reason = None;
        if !self.cfg.allowed_url_domains.is_empty() { for u in &urls { if !self.cfg.allowed_url_domains.iter().any(|d| u.contains(d)) { reason = Some(format!("URL not allowed: {u}")); break; } } }
        if !banned.is_empty() && reason.is_none() { reason = Some("banned substrings".into()); }
        let passed = reason.is_none();
        if self.cfg.audit_enabled { self.audit(GuardrailEventType::OutputCheck, passed, 0, banned.len(), output); }
        Ok(OutputCheckResult { passed, filtered_output: filtered, banned_matches: banned, urls_found: urls, rejection_reason: reason })
    }
    fn redact(&self, text: &str) -> (String, Vec<PiiMatch>) {
        let mut result = text.to_string(); let mut matches = Vec::new(); let mut off = 0isize;
        for m in self.email_re.find_iter(text) { let mt = m.as_str().to_string(); let rp = "[EMAIL]".to_string(); let s = (m.start() as isize + off) as usize; let e = (m.end() as isize + off) as usize; result.replace_range(s..e, &rp); off += rp.len() as isize - mt.len() as isize; matches.push(PiiMatch{kind:PiiKind::Email,start:m.start(),end:m.end(),matched:mt,replacement:rp}); }
        for m in self.ssn_re.find_iter(text) { if matches.iter().any(|p: &PiiMatch| p.start <= m.start() && p.end >= m.end()) { continue; } let mt = m.as_str().to_string(); let rp = "[SSN]".to_string(); let s = (m.start() as isize + off) as usize; let e = (m.end() as isize + off) as usize; result.replace_range(s..e, &rp); off += rp.len() as isize - mt.len() as isize; matches.push(PiiMatch{kind:PiiKind::Ssn,start:m.start(),end:m.end(),matched:mt,replacement:rp}); }
        for m in self.phone_re.find_iter(text) { if matches.iter().any(|p: &PiiMatch| p.start <= m.start() && p.end >= m.end()) { continue; } let mt = m.as_str().to_string(); let rp = "[PHONE]".to_string(); let s = (m.start() as isize + off) as usize; let e = (m.end() as isize + off) as usize; result.replace_range(s..e, &rp); off += rp.len() as isize - mt.len() as isize; matches.push(PiiMatch{kind:PiiKind::Phone,start:m.start(),end:m.end(),matched:mt,replacement:rp}); }
        // Credit-card redaction: skip spans already covered by an earlier PII match
        // (phone numbers can be a false-positive for the Luhn-style 13–16 digit pattern).
        for m in self.cc_re.find_iter(text) {
            if matches.iter().any(|p: &PiiMatch| p.start <= m.start() && p.end >= m.end()) { continue; }
            let digits: String = m.as_str().chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() < 13 || digits.len() > 16 { continue; }
            if !luhn_valid(&digits) { continue; }
            let mt = m.as_str().to_string(); let rp = "[CC]".to_string();
            let s = (m.start() as isize + off) as usize; let e = (m.end() as isize + off) as usize;
            result.replace_range(s..e, &rp);
            off += rp.len() as isize - mt.len() as isize;
            matches.push(PiiMatch{kind:PiiKind::CreditCard,start:m.start(),end:m.end(),matched:mt,replacement:rp});
        }
        for m in self.ip_re.find_iter(text) { if matches.iter().any(|p: &PiiMatch| p.start <= m.start() && p.end >= m.end()) { continue; } let mt = m.as_str().to_string(); let rp = "[IP]".to_string(); let s = (m.start() as isize + off) as usize; let e = (m.end() as isize + off) as usize; result.replace_range(s..e, &rp); off += rp.len() as isize - mt.len() as isize; matches.push(PiiMatch{kind:PiiKind::IpAddress,start:m.start(),end:m.end(),matched:mt,replacement:rp}); }
        (result, matches)
    }
    fn audit(&self, et: GuardrailEventType, passed: bool, pii: usize, banned: usize, text: &str) {
        self.audit_log.write().push(GuardrailAuditEvent { timestamp_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0), event_type: et, passed, pii_count: pii, banned_count: banned, text_preview: text.chars().take(100).collect() });
    }
    #[must_use] pub fn audit_log(&self) -> Vec<GuardrailAuditEvent> { self.audit_log.read().clone() }
    pub fn clear_audit_log(&self) { self.audit_log.write().clear(); }
}

pub type SharedGuardrails = Arc<Guardrails>;

/// Luhn checksum validation — used to confirm that a 13–16 digit sequence
/// is actually a credit-card number before redacting it. Without this check,
/// the regex would also match phone numbers, tracking IDs, and other numeric
/// sequences of the same length.
fn luhn_valid(digits: &str) -> bool {
    let mut sum: u32 = 0;
    let mut alternate = false;
    for c in digits.chars().rev() {
        let mut d = match c.to_digit(10) { Some(d) => d, None => return false };
        if alternate {
            d *= 2;
            if d > 9 { d -= 9; }
        }
        sum += d;
        alternate = !alternate;
    }
    sum % 10 == 0
}

pub struct TokenBucketRateLimiter { buckets: RwLock<AHashMap<String, (f64, u64)>>, capacity: f64, refill_rate: f64 }
impl TokenBucketRateLimiter { #[must_use] pub fn new(cap: f64, rate: f64) -> Self { Self { buckets: RwLock::new(AHashMap::new()), capacity: cap, refill_rate: rate } } pub fn try_consume(&self, key: &str) -> bool { let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0); let mut b = self.buckets.write(); let e = b.entry(key.to_string()).or_insert((self.capacity, now)); let (tok, last) = *e; let elapsed = (now.saturating_sub(last)) as f64 / 1000.0; let nt = (tok + elapsed * self.refill_rate).min(self.capacity); if nt >= 1.0 { *e = (nt - 1.0, now); true } else { *e = (nt, now); false } } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn redact_email() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_input("Contact user@example.com").unwrap(); assert!(r.redacted_text.contains("[EMAIL]")); }
    #[test] fn redact_phone() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_input("Call 555-123-4567").unwrap(); assert!(r.redacted_text.contains("[PHONE]")); }
    #[test] fn redact_ssn() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_input("SSN 123-45-6789").unwrap(); assert!(r.redacted_text.contains("[SSN]")); }
    #[test] fn banned() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_input("api_key: sk-12345").unwrap(); assert!(!r.passed); }
    #[test] fn long_input() { let g = Guardrails::new(GuardrailsConfig { max_input_length: 10, ..Default::default() }); assert!(matches!(g.check_input("this is too long"), Err(GuardrailsError::InputTooLong(_, 10)))); }
    #[test] fn clean() { let g = Guardrails::new(GuardrailsConfig::default()); assert!(g.check_input("Hello world").unwrap().passed); }
    #[test] fn output_filter() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_output("api_key: secret").unwrap(); assert!(r.filtered_output.contains("[REDACTED]")); }
    #[test] fn urls() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_output("Visit https://example.com").unwrap(); assert_eq!(r.urls_found.len(), 1); }
    #[test] fn url_allowlist() { let g = Guardrails::new(GuardrailsConfig { allowed_url_domains: vec!["trusted.com".into()], ..Default::default() }); let r = g.check_output("https://evil.com").unwrap(); assert!(!r.passed); }
    #[test] fn audit() { let g = Guardrails::new(GuardrailsConfig::default()); g.check_input("test").unwrap(); g.check_output("out").unwrap(); assert_eq!(g.audit_log().len(), 2); }
    #[test] fn rate_limit() { let l = TokenBucketRateLimiter::new(3.0, 1.0); assert!(l.try_consume("u1")); assert!(l.try_consume("u1")); assert!(l.try_consume("u1")); assert!(!l.try_consume("u1")); }
    #[test] fn rate_limit_isolated() { let l = TokenBucketRateLimiter::new(1.0, 1.0); assert!(l.try_consume("u1")); assert!(!l.try_consume("u1")); assert!(l.try_consume("u2")); }
    #[test] fn multi_pii() { let g = Guardrails::new(GuardrailsConfig::default()); let r = g.check_input("Email a@b.com or call 555-123-4567. IP: 192.168.1.1").unwrap(); assert!(r.redacted_text.contains("[EMAIL]")); assert!(r.redacted_text.contains("[PHONE]")); assert!(r.redacted_text.contains("[IP]")); }
    #[test] fn clear_audit() { let g = Guardrails::new(GuardrailsConfig::default()); g.check_input("test").unwrap(); assert!(!g.audit_log().is_empty()); g.clear_audit_log(); assert!(g.audit_log().is_empty()); }
}
