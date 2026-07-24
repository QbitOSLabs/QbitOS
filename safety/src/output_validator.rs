//! Output validator — validates agent outputs against safety policies.

use serde::{Deserialize, Serialize};

/// A validation rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name.
    pub name: String,
    /// Rule description.
    pub description: String,
    /// The check function as a string predicate.
    pub check: ValidationCheck,
}

/// Types of validation checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ValidationCheck {
    /// Output must not exceed max length.
    MaxLength(usize),
    /// Output must be at least min length.
    MinLength(usize),
    /// Output must not contain any of these substrings.
    MustNotContain(Vec<String>),
    /// Output must contain at least one of these substrings.
    MustContainOne(Vec<String>),
    /// Output must match a regex pattern.
    MustMatchRegex(String),
    /// Custom predicate (always passes — for programmatic checks).
    Custom,
}

/// Result of output validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the output passed all rules.
    pub valid: bool,
    /// Rules that failed.
    pub failed_rules: Vec<String>,
    /// Rules that passed.
    pub passed_rules: Vec<String>,
    /// Error messages for failed rules.
    pub errors: Vec<String>,
}

/// Output validator with a set of rules.
pub struct OutputValidator {
    rules: Vec<ValidationRule>,
}

impl Default for OutputValidator {
    fn default() -> Self {
        Self {
            rules: vec![
                ValidationRule {
                    name: "max_length".into(),
                    description: "Output must not exceed 10000 characters".into(),
                    check: ValidationCheck::MaxLength(10000),
                },
                ValidationRule {
                    name: "min_length".into(),
                    description: "Output must be at least 1 character".into(),
                    check: ValidationCheck::MinLength(1),
                },
                ValidationRule {
                    name: "no_secrets".into(),
                    description: "Output must not contain API keys or secrets".into(),
                    check: ValidationCheck::MustNotContain(vec![
                        "api_key".into(), "secret_key".into(), "private_key".into(),
                        "BEGIN RSA PRIVATE KEY".into(), "aws_secret".into(),
                    ]),
                },
                ValidationRule {
                    name: "no_harmful_instructions".into(),
                    description: "Output must not contain instructions for harmful activities".into(),
                    check: ValidationCheck::MustNotContain(vec![
                        "how to make a bomb".into(), "how to hack".into(),
                        "how to poison".into(), "how to steal".into(),
                    ]),
                },
            ],
        }
    }
}

impl OutputValidator {
    /// Create a new validator with custom rules.
    #[must_use]
    pub fn new(rules: Vec<ValidationRule>) -> Self {
        Self { rules }
    }

    /// Validate an output string.
    #[must_use]
    pub fn validate(&self, output: &str) -> ValidationResult {
        let mut failed = Vec::new();
        let mut passed = Vec::new();
        let mut errors = Vec::new();

        for rule in &self.rules {
            let (ok, error_msg) = self.check_rule(rule, output);
            if ok {
                passed.push(rule.name.clone());
            } else {
                failed.push(rule.name.clone());
                errors.push(error_msg);
            }
        }

        ValidationResult {
            valid: failed.is_empty(),
            failed_rules: failed,
            passed_rules: passed,
            errors,
        }
    }

    fn check_rule(&self, rule: &ValidationRule, output: &str) -> (bool, String) {
        match &rule.check {
            ValidationCheck::MaxLength(max) => {
                if output.len() > *max {
                    (false, format!("Output length {} exceeds maximum {}", output.len(), max))
                } else {
                    (true, String::new())
                }
            }
            ValidationCheck::MinLength(min) => {
                if output.len() < *min {
                    (false, format!("Output length {} is below minimum {}", output.len(), min))
                } else {
                    (true, String::new())
                }
            }
            ValidationCheck::MustNotContain(forbidden) => {
                let lower = output.to_lowercase();
                for f in forbidden {
                    if lower.contains(&f.to_lowercase()) {
                        return (false, format!("Output contains forbidden substring: '{f}'"));
                    }
                }
                (true, String::new())
            }
            ValidationCheck::MustContainOne(required) => {
                let lower = output.to_lowercase();
                if required.iter().any(|r| lower.contains(&r.to_lowercase())) {
                    (true, String::new())
                } else {
                    (false, format!("Output must contain at least one of: {:?}", required))
                }
            }
            ValidationCheck::MustMatchRegex(pattern) => {
                match regex::Regex::new(pattern) {
                    Ok(re) => {
                        if re.is_match(output) { (true, String::new()) }
                        else { (false, format!("Output does not match pattern: {pattern}")) }
                    }
                    Err(_) => (false, format!("Invalid regex pattern: {pattern}")),
                }
            }
            ValidationCheck::Custom => (true, String::new()),
        }
    }

    /// Add a custom rule.
    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_safe_output() {
        let validator = OutputValidator::default();
        let result = validator.validate("The weather is sunny today.");
        assert!(result.valid);
    }

    #[test]
    fn rejects_empty_output() {
        let validator = OutputValidator::default();
        let result = validator.validate("");
        assert!(!result.valid);
        assert!(result.failed_rules.contains(&"min_length".to_string()));
    }

    #[test]
    fn rejects_secrets() {
        let validator = OutputValidator::default();
        let result = validator.validate("Your api_key is abc123");
        assert!(!result.valid);
        assert!(result.failed_rules.contains(&"no_secrets".to_string()));
    }

    #[test]
    fn rejects_harmful() {
        let validator = OutputValidator::default();
        let result = validator.validate("Here is how to hack a system");
        assert!(!result.valid);
        assert!(result.failed_rules.contains(&"no_harmful_instructions".to_string()));
    }
}
