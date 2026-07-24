//! QbitOS Safety — Content filtering, prompt injection defense, output validation.
//!
//! Implements:
//!
//! 1. **Content filter** — detects harmful, toxic, or inappropriate content
//! 2. **Prompt injection defense** — detects and neutralizes injection attempts
//! 3. **Output validator** — validates agent outputs against safety policies
//! 4. **Rate limiter** — prevents resource exhaustion
//! 5. **Audit logger** — records all safety-relevant events

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod audit;
pub mod guardrails;
pub mod content_filter;
pub mod hallucination_guard;
pub mod injection;
pub mod output_validator;
pub mod rate_limiter;

pub use audit::{AuditEvent, AuditLogger, AuditLevel};
pub use guardrails::{Guardrails, GuardrailsConfig, GuardrailsError, GuardrailAuditEvent, GuardrailEventType, InputCheckResult, OutputCheckResult, PiiKind, PiiMatch, SharedGuardrails, TokenBucketRateLimiter};
pub use content_filter::{ContentFilter, ContentCategory, FilterResult};
pub use injection::{InjectionDetector, InjectionType, InjectionResult};
pub use output_validator::{OutputValidator, ValidationRule, ValidationResult};
pub use rate_limiter::{RateLimiter, RateLimit};
pub use hallucination_guard::{HallucinationGuard, Source, ClaimCheck, ResponseCheck};
