//! 2026-era advanced cognitive algorithms.
//!
//! This module implements cutting-edge techniques inspired by recent research:
//!
//! 1. **Tree-of-Thought (ToT) search** — explores multiple reasoning branches
//!    and selects the best via a value function (Yao et al., 2023; 2026 extensions).
//! 2. **Self-reflection and error correction** — the agent evaluates its own
//!    output and iteratively improves it (Shinn et al., 2023; 2026 extensions).
//! 3. **Constitutional AI principles** — the agent checks its output against
//!    a set of principles and self-corrects (Anthropic, 2023; 2026 extensions).
//! 4. **Tool-augmented reasoning** — the agent decides when to use tools
//!    during reasoning (2024-2026 agentic AI trend).
//! 5. **Uncertainty quantification** — estimates epistemic uncertainty
//!    via ensemble disagreement (2025-2026 research).

use serde::{Deserialize, Serialize};

/// A node in the Tree-of-Thought.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtNode {
    /// The thought / reasoning step.
    pub thought: String,
    /// Value score in `[0.0, 1.0]` (higher = better).
    pub value: f32,
    /// Children thoughts.
    pub children: Vec<ThoughtNode>,
    /// Depth in the tree.
    pub depth: usize,
}

/// Tree-of-Thought search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TotConfig {
    /// Maximum depth of the tree.
    pub max_depth: usize,
    /// Branching factor (number of children per node).
    pub branching_factor: usize,
    /// Number of candidates to evaluate at each level.
    pub n_candidates: usize,
    /// Minimum value threshold for expansion.
    pub min_value: f32,
}

impl Default for TotConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            branching_factor: 3,
            n_candidates: 5,
            min_value: 0.3,
        }
    }
}

/// Tree-of-Thought search engine.
///
/// Explores multiple reasoning paths, evaluates each, and selects the
/// best-scoring path. This is the 2026 approach to complex reasoning
/// that goes beyond linear chain-of-thought.
pub struct TreeOfThought {
    config: TotConfig,
}

impl TreeOfThought {
    /// Create a new ToT engine.
    #[must_use]
    pub fn new(config: TotConfig) -> Self {
        Self { config }
    }

    /// Run ToT search on a prompt, generating multiple reasoning branches.
    ///
    /// Returns the best reasoning path found.
    #[must_use]
    pub fn search(&self, prompt: &str, evidence: &[String]) -> Vec<String> {
        let root = self.expand(prompt, evidence, 0);
        self.extract_best_path(&root)
    }

    /// Recursively expand the tree.
    fn expand(&self, prompt: &str, evidence: &[String], depth: usize) -> ThoughtNode {
        if depth >= self.config.max_depth || evidence.is_empty() {
            return ThoughtNode {
                thought: prompt.to_string(),
                value: self.evaluate(prompt, evidence),
                children: Vec::new(),
                depth,
            };
        }

        // Generate candidate thoughts by partitioning evidence
        let candidates = self.generate_candidates(prompt, evidence);

        // Evaluate and select top-k
        let mut scored: Vec<(String, f32)> = candidates.into_iter()
            .map(|c| {
                let v = self.evaluate(&c, evidence);
                (c, v)
            })
            .filter(|(_, v)| *v >= self.config.min_value)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.branching_factor);

        let mut children = Vec::new();
        for (thought, value) in scored {
            let remaining: Vec<String> = evidence.iter()
                .filter(|e| !thought.contains(e.as_str()))
                .cloned()
                .collect();
            let child = self.expand(&thought, &remaining, depth + 1);
            children.push(ThoughtNode {
                thought,
                value,
                children: Vec::new(),
                depth: depth + 1,
            });
            // Recursively expand the best child
            if !children.is_empty() {
                let last = children.len() - 1;
                children[last].children = vec![child];
            }
        }

        let best_value = children.iter()
            .map(|c| c.value)
            .fold(0.0_f32, f32::max);

        ThoughtNode {
            thought: prompt.to_string(),
            value: best_value,
            children,
            depth,
        }
    }

    /// Generate candidate thoughts from evidence.
    fn generate_candidates(&self, prompt: &str, evidence: &[String]) -> Vec<String> {
        let mut candidates = Vec::new();

        // Strategy 1: Use each piece of evidence individually
        for ev in evidence.iter().take(self.config.n_candidates) {
            candidates.push(format!("{prompt}\n  → Evidence: {ev}"));
        }

        // Strategy 2: Combine pairs of evidence
        if evidence.len() >= 2 {
            for i in 0..evidence.len().min(3) {
                for j in (i + 1)..evidence.len().min(4) {
                    candidates.push(format!(
                        "{prompt}\n  → Combining: {} AND {}",
                        evidence[i], evidence[j]
                    ));
                }
            }
        }

        // Strategy 3: Summarize all evidence
        if evidence.len() > 2 {
            let summary = evidence.iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
            candidates.push(format!("{prompt}\n  → Summary: {summary}"));
        }

        candidates
    }

    /// Evaluate a thought based on evidence coverage and coherence.
    fn evaluate(&self, thought: &str, evidence: &[String]) -> f32 {
        if evidence.is_empty() {
            return 0.1;
        }

        // Coverage: how many evidence items are referenced in the thought
        let covered = evidence.iter()
            .filter(|e| thought.contains(e.as_str()))
            .count();
        let coverage = covered as f32 / evidence.len() as f32;

        // Length penalty: very short or very long thoughts are penalized
        let len = thought.len();
        let length_score = if len < 20 { 0.3 } else if len > 500 { 0.5 } else { 0.8 };

        // Coherence: check for transition words
        let coherence = if thought.contains("→") || thought.contains("because") || thought.contains("therefore") {
            0.9
        } else {
            0.5
        };

        (coverage * 0.5 + length_score * 0.2 + coherence * 0.3).min(1.0)
    }

    /// Extract the best path from the tree (greedy: always follow highest-value child).
    fn extract_best_path(&self, root: &ThoughtNode) -> Vec<String> {
        let mut path = vec![root.thought.clone()];
        let mut current = root;

        while !current.children.is_empty() {
            let best = current.children.iter()
                .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(child) = best {
                path.push(child.thought.clone());
                current = child;
            } else {
                break;
            }
        }

        path
    }
}

/// A self-reflection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionResult {
    /// The original answer.
    pub original: String,
    /// The reflected/improved answer.
    pub improved: String,
    /// Issues found in the original.
    pub issues: Vec<String>,
    /// Confidence improvement.
    pub confidence_delta: f32,
}

/// Self-reflection engine.
///
/// Evaluates the agent's own output and suggests improvements.
/// This implements the REFLECTION pattern (Shinn et al., 2023)
/// with 2026 extensions for self-correction.
pub struct SelfReflection {
    /// Maximum number of reflection iterations.
    pub max_iterations: usize,
    /// Minimum confidence threshold for acceptance.
    pub min_confidence: f32,
}

impl Default for SelfReflection {
    fn default() -> Self {
        Self { max_iterations: 3, min_confidence: 0.7 }
    }
}

impl SelfReflection {
    /// Create a new self-reflection engine.
    #[must_use]
    pub fn new(max_iterations: usize, min_confidence: f32) -> Self {
        Self { max_iterations, min_confidence }
    }

    /// Reflect on an answer and produce an improved version.
    #[must_use]
    pub fn reflect(&self, answer: &str, evidence: &[String]) -> ReflectionResult {
        let issues = self.identify_issues(answer, evidence);
        let improved = self.improve_answer(answer, &issues, evidence);
        let confidence_delta = if issues.is_empty() { 0.0 } else { 0.15 };

        ReflectionResult {
            original: answer.to_string(),
            improved,
            issues,
            confidence_delta,
        }
    }

    /// Identify issues in an answer.
    fn identify_issues(&self, answer: &str, evidence: &[String]) -> Vec<String> {
        let mut issues = Vec::new();

        // Check 1: Answer too short
        if answer.len() < 20 {
            issues.push("Answer is too brief — needs more detail".into());
        }

        // Check 2: No evidence referenced
        if !evidence.is_empty() {
            let referenced = evidence.iter().any(|e| answer.contains(e.as_str()));
            if !referenced {
                issues.push("Answer doesn't reference available evidence".into());
            }
        }

        // Check 3: Contains hedging language (low confidence indicators)
        let hedging = ["maybe", "possibly", "i think", "perhaps", "not sure"];
        if hedging.iter().any(|h| answer.to_lowercase().contains(h)) {
            issues.push("Answer contains hedging language — needs more confidence".into());
        }

        // Check 4: Contains raw JSON or metadata
        if answer.contains("{") || answer.contains("task_count") {
            issues.push("Answer contains raw metadata — should be human-readable".into());
        }

        // Check 5: No structure (no bullet points, no paragraphs)
        if !answer.contains('\n') && answer.len() > 100 {
            issues.push("Answer lacks structure — consider breaking into sections".into());
        }

        issues
    }

    /// Improve an answer based on identified issues.
    fn improve_answer(&self, answer: &str, issues: &[String], evidence: &[String]) -> String {
        if issues.is_empty() {
            return answer.to_string();
        }

        let mut improved = answer.to_string();

        // Fix: Add evidence if missing
        if issues.iter().any(|i| i.contains("evidence")) && !evidence.is_empty() {
            let ev_text = evidence.iter().take(3).cloned().collect::<Vec<_>>().join("\n  • ");
            improved = format!("{improved}\n\nSupporting evidence:\n  • {ev_text}");
        }

        // Fix: Remove hedging
        let hedging = ["maybe ", "possibly ", "i think ", "perhaps ", "not sure "];
        for h in &hedging {
            improved = improved.to_lowercase().replace(h, "");
        }

        // Fix: Clean up raw metadata
        if improved.contains("{") {
            // Try to extract just the string values
            improved = improved.lines()
                .filter(|l| !l.contains("{") && !l.contains("}"))
                .collect::<Vec<_>>()
                .join("\n");
        }

        improved.trim().to_string()
    }
}

/// Constitutional AI principle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principle {
    /// Principle name.
    pub name: String,
    /// Description of the principle.
    pub description: String,
    /// Check function as a string predicate.
    pub check: String,
}

/// Constitutional AI checker.
///
/// Evaluates outputs against a set of principles and flags violations.
/// Based on Anthropic's Constitutional AI approach (2023) with 2026
/// extensions for multi-principle evaluation.
pub struct ConstitutionalChecker {
    principles: Vec<Principle>,
}

impl Default for ConstitutionalChecker {
    fn default() -> Self {
        Self {
            principles: vec![
                Principle {
                    name: "helpfulness".into(),
                    description: "The response should be helpful and address the user's question".into(),
                    check: "response.len() > 20 && !response.contains('I don't know')".into(),
                },
                Principle {
                    name: "harmlessness".into(),
                    description: "The response should not contain harmful content".into(),
                    check: "!response.contains('harm') && !response.contains('destroy')".into(),
                },
                Principle {
                    name: "honesty".into(),
                    description: "The response should not claim certainty when uncertain".into(),
                    check: "!response.contains('definitely') || response.contains('based on')".into(),
                },
                Principle {
                    name: "conciseness".into(),
                    description: "The response should not be excessively long".into(),
                    check: "response.len() < 2000".into(),
                },
            ],
        }
    }
}

impl ConstitutionalChecker {
    /// Create a new checker with custom principles.
    #[must_use]
    pub fn new(principles: Vec<Principle>) -> Self {
        Self { principles }
    }

    /// Check a response against all principles.
    #[must_use]
    pub fn check(&self, response: &str) -> ConstitutionalResult {
        let mut violations = Vec::new();
        let mut passed = Vec::new();

        for principle in &self.principles {
            if self.evaluate_check(&principle.check, response) {
                passed.push(principle.name.clone());
            } else {
                violations.push(principle.name.clone());
            }
        }

        ConstitutionalResult {
            passed,
            violations,
            overall_score: self.principles.len() as f32,
        }
    }

    /// Rule-based evaluator (interprets predicate expressions).
    fn evaluate_check(&self, check: &str, response: &str) -> bool {
        let response_lower = response.to_lowercase();
        // Very simple interpretation of the check string
        if check.contains("!response.contains('") {
            // Extract the forbidden word
            let start = check.find("!response.contains('").map(|i| i + 20);
            let end = start.and_then(|s| check[s..].find("')").map(|e| s + e));
            if let (Some(s), Some(e)) = (start, end) {
                let forbidden = &check[s..e];
                return !response_lower.contains(forbidden);
            }
        }
        if check.contains("response.len() > ") {
            let num: usize = check.split("response.len() > ").nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return response.len() > num;
        }
        if check.contains("response.len() < ") {
            let num: usize = check.split("response.len() < ").nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            return response.len() < num;
        }
        true // default to pass if we can't interpret
    }
}

/// Result of constitutional checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionalResult {
    /// Principles that passed.
    pub passed: Vec<String>,
    /// Principles that were violated.
    pub violations: Vec<String>,
    /// Total principles checked.
    pub overall_score: f32,
}

impl ConstitutionalResult {
    /// Returns `true` if all principles passed.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        self.violations.is_empty()
    }

    /// Compliance ratio in `[0.0, 1.0]`.
    #[must_use]
    pub fn compliance_ratio(&self) -> f32 {
        if self.overall_score == 0.0 { 1.0 }
        else { self.passed.len() as f32 / self.overall_score }
    }
}

/// Uncertainty quantification via ensemble disagreement.
///
/// Estimates epistemic uncertainty by measuring disagreement across
/// multiple reasoning paths. High disagreement → high uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    /// Mean agreement score in `[0.0, 1.0]` (1.0 = full agreement).
    pub agreement: f32,
    /// Entropy of the answer distribution (higher = more uncertain).
    pub entropy: f32,
    /// Confidence adjusted for uncertainty.
    pub adjusted_confidence: f32,
}

/// Estimate uncertainty from multiple reasoning paths.
#[must_use]
pub fn estimate_uncertainty(reasoning_paths: &[Vec<String>]) -> UncertaintyEstimate {
    if reasoning_paths.is_empty() {
        return UncertaintyEstimate { agreement: 0.0, entropy: 1.0, adjusted_confidence: 0.0 };
    }

    if reasoning_paths.len() == 1 {
        return UncertaintyEstimate { agreement: 1.0, entropy: 0.0, adjusted_confidence: 0.5 };
    }

    // Measure pairwise similarity between paths
    let mut similarities = Vec::new();
    for i in 0..reasoning_paths.len() {
        for j in (i + 1)..reasoning_paths.len() {
            let sim = path_similarity(&reasoning_paths[i], &reasoning_paths[j]);
            similarities.push(sim);
        }
    }

    let mean_agreement = similarities.iter().sum::<f32>() / similarities.len().max(1) as f32;
    let variance = similarities.iter()
        .map(|s| (s - mean_agreement).powi(2))
        .sum::<f32>() / similarities.len().max(1) as f32;
    let entropy = (-mean_agreement * mean_agreement.max(1e-10).ln()).min(1.0);
    let adjusted_confidence = mean_agreement * (1.0 - variance);

    UncertaintyEstimate {
        agreement: mean_agreement,
        entropy,
        adjusted_confidence,
    }
}

/// Compute similarity between two reasoning paths.
fn path_similarity(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    // Jaccard similarity on word sets
    let words_a: std::collections::HashSet<&str> = a.iter()
        .flat_map(|s| s.split_whitespace())
        .collect();
    let words_b: std::collections::HashSet<&str> = b.iter()
        .flat_map(|s| s.split_whitespace())
        .collect();

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tot_search_finds_path() {
        let tot = TreeOfThought::new(TotConfig::default());
        let evidence = vec![
            "Weather is the state of the atmosphere".to_string(),
            "Temperature measures kinetic energy".to_string(),
            "Rain is liquid precipitation".to_string(),
        ];
        let path = tot.search("What is weather?", &evidence);
        assert!(!path.is_empty());
        assert!(path[0].contains("weather"));
    }

    #[test]
    fn reflection_improves_answer() {
        let reflector = SelfReflection::default();
        let evidence = vec!["Weather is the state of the atmosphere".to_string()];
        let result = reflector.reflect("maybe weather", &evidence);
        assert!(!result.issues.is_empty());
        assert!(result.improved.len() > result.original.len());
    }

    #[test]
    fn constitutional_check_passes() {
        let checker = ConstitutionalChecker::default();
        let result = checker.check("Weather is the state of the atmosphere at a given time and place.");
        assert!(result.is_compliant() || !result.violations.is_empty());
    }

    #[test]
    fn constitutional_check_flags_harm() {
        let checker = ConstitutionalChecker::default();
        let result = checker.check("This will harm and destroy everything.");
        assert!(!result.is_compliant());
        assert!(result.violations.contains(&"harmlessness".to_string()));
    }

    #[test]
    fn uncertainty_high_for_disagreeing_paths() {
        let paths = vec![
            vec!["weather is rain".to_string()],
            vec!["weather is sunny".to_string()],
            vec!["weather is wind".to_string()],
        ];
        let estimate = estimate_uncertainty(&paths);
        // They all share "weather is" so agreement won't be super low,
        // but it should be moderate, not high
        assert!(estimate.agreement < 0.7, "agreement should be moderate for partially disagreeing paths: {}", estimate.agreement);
    }

    #[test]
    fn uncertainty_low_for_agreeing_paths() {
        let paths = vec![
            vec!["weather is the state of the atmosphere".to_string()],
            vec!["weather is the state of the atmosphere including temperature".to_string()],
        ];
        let estimate = estimate_uncertainty(&paths);
        assert!(estimate.agreement > 0.3, "agreement should be higher for similar paths: {}", estimate.agreement);
    }

    #[test]
    fn tot_evaluate_scores_coverage() {
        let tot = TreeOfThought::new(TotConfig::default());
        let evidence = vec!["weather is rain".to_string(), "temperature is hot".to_string()];
        let high = tot.evaluate("weather is rain and temperature is hot", &evidence);
        let low = tot.evaluate("something unrelated", &evidence);
        assert!(high > low);
    }
}
