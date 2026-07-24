//! Self-improvement cycle: detect weaknesses → propose fixes → test → accept/reject.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A weakness detected by the self-improvement cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaknessReport {
    /// Unique id.
    pub id: String,
    /// The subsystem or skill affected (e.g. "skill:summarize", "memory:recall").
    pub target: String,
    /// A human-readable description.
    pub description: String,
    /// Severity (0..1, higher = worse).
    pub severity: f64,
    /// Suggested fix.
    pub suggested_fix: String,
    /// Metric value that triggered the detection.
    pub metric_value: f64,
    /// Metric name.
    pub metric_name: String,
}

/// A proposed improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementProposal {
    /// Unique id.
    pub id: String,
    /// The weakness being addressed.
    pub weakness_id: String,
    /// The target subsystem.
    pub target: String,
    /// The proposed change (free-form JSON).
    pub change: serde_json::Value,
    /// Expected improvement.
    pub expected_improvement: f64,
    /// Actual measured improvement (filled after testing).
    #[serde(default)]
    pub actual_improvement: Option<f64>,
    /// Whether the proposal was accepted.
    #[serde(default)]
    pub accepted: bool,
    /// Reason for acceptance/rejection.
    #[serde(default)]
    pub reason: String,
}

/// Performance metrics used for weakness detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Per-skill success rates.
    pub skill_success_rates: AHashMap<String, f64>,
    /// Per-skill latencies (ms).
    pub skill_latencies_ms: AHashMap<String, f64>,
    /// Memory recall precision.
    pub memory_recall_precision: f64,
    /// Reasoning accuracy.
    pub reasoning_accuracy: f64,
    /// Vision perception accuracy.
    pub vision_accuracy: f64,
    /// Speech accuracy.
    pub speech_accuracy: f64,
    /// Average response latency (ms).
    pub avg_response_latency_ms: f64,
    /// Total requests handled.
    pub total_requests: u64,
}

/// The self-improvement cycle.
pub struct SelfImprovementCycle {
    /// History of detected weaknesses.
    pub weaknesses: Arc<RwLock<Vec<WeaknessReport>>>,
    /// History of proposals.
    pub proposals: Arc<RwLock<Vec<ImprovementProposal>>>,
    /// Threshold below which a skill is considered weak.
    pub weak_skill_threshold: f64,
    /// Threshold above which a skill is considered slow.
    pub slow_skill_threshold_ms: f64,
    /// Minimum improvement required to accept a proposal.
    pub min_acceptable_improvement: f64,
    /// Regression guard: maximum allowed regression on other metrics.
    pub max_regression: f64,
    /// Whether self-improvement is enabled.
    pub enabled: bool,
}

impl Default for SelfImprovementCycle {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfImprovementCycle {
    /// Construct a new self-improvement cycle.
    #[must_use]
    pub fn new() -> Self {
        Self {
            weaknesses: Arc::new(RwLock::new(Vec::new())),
            proposals: Arc::new(RwLock::new(Vec::new())),
            weak_skill_threshold: 0.5,
            slow_skill_threshold_ms: 5_000.0,
            min_acceptable_improvement: 0.05,
            max_regression: 0.02,
            enabled: true,
        }
    }

    /// Detect weaknesses from current performance metrics.
    ///
    /// Returns the list of newly detected weaknesses (also appends to history).
    pub fn detect_weaknesses(&self, metrics: &PerformanceMetrics) -> Vec<WeaknessReport> {
        let mut detected = Vec::new();

        // Weak skills.
        for (skill, rate) in &metrics.skill_success_rates {
            if *rate < self.weak_skill_threshold {
                let severity = 1.0 - *rate;
                detected.push(WeaknessReport {
                    id: format!("weakness_skill_{skill}"),
                    target: format!("skill:{skill}"),
                    description: format!(
                        "Skill '{skill}' has low success rate ({:.2})",
                        rate
                    ),
                    severity,
                    suggested_fix: format!(
                        "Rephrase the skill prompt for '{skill}' or add a verification step."
                    ),
                    metric_value: *rate,
                    metric_name: "success_rate".into(),
                });
            }
        }

        // Slow skills.
        for (skill, latency) in &metrics.skill_latencies_ms {
            if *latency > self.slow_skill_threshold_ms {
                let severity = ((*latency - self.slow_skill_threshold_ms)
                    / self.slow_skill_threshold_ms)
                    .min(1.0);
                detected.push(WeaknessReport {
                    id: format!("slow_skill_{skill}"),
                    target: format!("skill:{skill}"),
                    description: format!(
                        "Skill '{skill}' is slow ({:.0}ms avg)",
                        latency
                    ),
                    severity,
                    suggested_fix: format!(
                        "Cache intermediate results for '{skill}' or split it into smaller steps."
                    ),
                    metric_value: *latency,
                    metric_name: "latency_ms".into(),
                });
            }
        }

        // Low memory recall.
        if metrics.memory_recall_precision < 0.6 {
            detected.push(WeaknessReport {
                id: "weak_memory_recall".into(),
                target: "memory:recall".into(),
                description: format!(
                    "Memory recall precision is low ({:.2})",
                    metrics.memory_recall_precision
                ),
                severity: 1.0 - metrics.memory_recall_precision,
                suggested_fix: "Increase HNSW ef parameter or improve embedding model.".into(),
                metric_value: metrics.memory_recall_precision,
                metric_name: "recall_precision".into(),
            });
        }

        // Low reasoning accuracy.
        if metrics.reasoning_accuracy < 0.7 {
            detected.push(WeaknessReport {
                id: "weak_reasoning".into(),
                target: "reasoning:engine".into(),
                description: format!(
                    "Reasoning accuracy is low ({:.2})",
                    metrics.reasoning_accuracy
                ),
                severity: 1.0 - metrics.reasoning_accuracy,
                suggested_fix: "Add more reasoning rules or improve the inference backend.".into(),
                metric_value: metrics.reasoning_accuracy,
                metric_name: "accuracy".into(),
            });
        }

        let mut weaknesses = self.weaknesses.write();
        weaknesses.extend(detected.clone());
        detected
    }

    /// Propose an improvement for a weakness.
    pub fn propose_improvement(&self, weakness: &WeaknessReport) -> ImprovementProposal {
        let change = serde_json::json!({
            "target": weakness.target,
            "fix": weakness.suggested_fix,
            "metric": weakness.metric_name,
            "current_value": weakness.metric_value,
        });
        let expected = weakness.severity * 0.5; // Expect to fix half the gap.
        let proposal = ImprovementProposal {
            id: format!("proposal_{}", weakness.id),
            weakness_id: weakness.id.clone(),
            target: weakness.target.clone(),
            change,
            expected_improvement: expected,
            actual_improvement: None,
            accepted: false,
            reason: String::new(),
        };
        self.proposals.write().push(proposal.clone());
        proposal
    }

    /// Run a complete cycle: detect weaknesses, propose fixes, test, accept/reject.
    ///
    /// Returns the list of accepted proposals.
    pub fn run_cycle(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Vec<ImprovementProposal> {
        if !self.enabled {
            return Vec::new();
        }
        let weaknesses = self.detect_weaknesses(metrics);
        let mut accepted = Vec::new();
        for weakness in &weaknesses {
            let mut proposal = self.propose_improvement(weakness);
            // Simulate testing the proposal: in a real implementation, this
            // would apply the change and re-run the evaluation suite.
            // Here we accept the proposal if its expected improvement exceeds
            // the threshold.
            if proposal.expected_improvement >= self.min_acceptable_improvement {
                proposal.actual_improvement = Some(proposal.expected_improvement);
                proposal.accepted = true;
                proposal.reason = "Expected improvement exceeds threshold.".into();
                accepted.push(proposal.clone());
            } else {
                proposal.actual_improvement = Some(0.0);
                proposal.accepted = false;
                proposal.reason = "Expected improvement below threshold.".into();
            }
            // Update the proposal in history.
            if let Some(last) = self.proposals.write().last_mut() {
                *last = proposal;
            }
        }
        accepted
    }

    /// Total number of weaknesses ever detected.
    #[must_use]
    pub fn total_weaknesses(&self) -> usize {
        self.weaknesses.read().len()
    }

    /// Total number of proposals ever made.
    #[must_use]
    pub fn total_proposals(&self) -> usize {
        self.proposals.read().len()
    }

    /// Number of accepted proposals.
    #[must_use]
    pub fn accepted_count(&self) -> usize {
        self.proposals.read().iter().filter(|p| p.accepted).count()
    }

    /// Acceptance rate.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.total_proposals();
        if total == 0 {
            return 0.0;
        }
        self.accepted_count() as f64 / total as f64
    }
}

impl std::fmt::Debug for SelfImprovementCycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfImprovementCycle")
            .field("weaknesses", &self.total_weaknesses())
            .field("proposals", &self.total_proposals())
            .field("accepted", &self.accepted_count())
            .field("enabled", &self.enabled)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metrics() -> PerformanceMetrics {
        let mut m = PerformanceMetrics::default();
        m.skill_success_rates.insert("good".into(), 0.95);
        m.skill_success_rates.insert("bad".into(), 0.20);
        m.skill_latencies_ms.insert("fast".into(), 50.0);
        m.skill_latencies_ms.insert("slow".into(), 8_000.0);
        m.memory_recall_precision = 0.45;
        m.reasoning_accuracy = 0.60;
        m
    }

    #[test]
    fn detect_weaknesses_finds_weak_and_slow_skills() {
        let cycle = SelfImprovementCycle::new();
        let metrics = make_metrics();
        let weaknesses = cycle.detect_weaknesses(&metrics);
        // bad skill, slow skill, weak memory, weak reasoning = 4
        assert_eq!(weaknesses.len(), 4);
    }

    #[test]
    fn propose_improvement_returns_proposal() {
        let cycle = SelfImprovementCycle::new();
        let metrics = make_metrics();
        let weaknesses = cycle.detect_weaknesses(&metrics);
        let proposal = cycle.propose_improvement(&weaknesses[0]);
        assert!(!proposal.id.is_empty());
        assert!(!proposal.accepted);
    }

    #[test]
    fn run_cycle_accepts_high_impact_proposals() {
        let cycle = SelfImprovementCycle::new();
        let metrics = make_metrics();
        let accepted = cycle.run_cycle(&metrics);
        // Should accept some proposals.
        assert!(!accepted.is_empty());
        assert!(cycle.acceptance_rate() > 0.0);
    }

    #[test]
    fn disabled_cycle_returns_no_acceptances() {
        let mut cycle = SelfImprovementCycle::new();
        cycle.enabled = false;
        let metrics = make_metrics();
        let accepted = cycle.run_cycle(&metrics);
        assert!(accepted.is_empty());
    }

    #[test]
    fn history_grows_with_each_cycle() {
        let cycle = SelfImprovementCycle::new();
        let metrics = make_metrics();
        let _ = cycle.run_cycle(&metrics);
        assert_eq!(cycle.total_weaknesses(), 4);
        assert_eq!(cycle.total_proposals(), 4);
    }
}
