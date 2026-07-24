#![allow(missing_docs)]
//! Planning replay: replays plans with outcome scores to learn which planning
//! patterns succeed for which kinds of goals. Plans are stored along with
//! their outcome (Success/PartialSuccess/Failure/Timeout), reward, and
//! duration. The replay engine exposes pattern-level success rates and can
//! recommend the best historical plan for a similar goal.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PlanningReplayError { #[error("no plans recorded")] NoPlans }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlanOutcome { Success, PartialSuccess, Failure, Timeout }

impl PlanOutcome {
    #[must_use]
    pub fn score(&self) -> f64 {
        match self { Self::Success => 1.0, Self::PartialSuccess => 0.5, Self::Failure => 0.0, Self::Timeout => 0.2 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRecord {
    pub id: String,
    pub goal: String,
    pub plan_steps: Vec<String>,
    pub tools_used: Vec<String>,
    pub outcome: PlanOutcome,
    pub reward: f64,
    pub duration_ms: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PlanRecord {
    #[must_use]
    pub fn new(
        goal: impl Into<String>, plan_steps: Vec<String>, tools_used: Vec<String>,
        outcome: PlanOutcome, reward: f64, duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(), goal: goal.into(), plan_steps, tools_used,
            outcome, reward, duration_ms, timestamp: chrono::Utc::now(),
        }
    }
    fn pattern_key(&self) -> String { self.plan_steps.join(" → ") }
    fn goal_tokens(&self) -> HashSet<String> {
        self.goal.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty()).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningStats {
    pub total_plans: u64,
    pub success_rate: f64,
    pub mean_reward: f64,
    pub mean_duration_ms: f64,
}

pub struct PlanningReplay {
    records: Arc<RwLock<Vec<PlanRecord>>>,
    pattern_scores: Arc<RwLock<AHashMap<String, (f64, u64)>>>,
}

impl Default for PlanningReplay { fn default() -> Self { Self::new() } }

impl PlanningReplay {
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
            pattern_scores: Arc::new(RwLock::new(AHashMap::new())),
        }
    }

    pub fn record(&self, plan: PlanRecord) {
        let key = plan.pattern_key();
        let reward = plan.reward;
        let mut scores = self.pattern_scores.write();
        let entry = scores.entry(key).or_insert((0.0, 0));
        entry.0 += reward;
        entry.1 += 1;
        self.records.write().push(plan);
    }

    #[must_use]
    pub fn get_best_pattern(&self, goal: &str) -> Option<Vec<String>> {
        let records = self.records.read();
        if records.is_empty() { return None; }
        let q: HashSet<String> = goal.split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|s| !s.is_empty()).collect();
        let mut candidates: Vec<(f64, &PlanRecord)> = records.iter()
            .filter(|r| matches!(r.outcome, PlanOutcome::Success | PlanOutcome::PartialSuccess))
            .map(|r| {
                if q.is_empty() { return (0.0, r); }
                let rt = r.goal_tokens();
                let inter = q.intersection(&rt).count() as f64;
                let union = q.union(&rt).count() as f64;
                ((if union > 0.0 { inter / union } else { 0.0 }) * r.reward, r)
            })
            .filter(|(s, _)| *s > 0.0).collect();
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.first().map(|(_, r)| r.plan_steps.clone())
    }

    #[must_use]
    pub fn pattern_success_rate(&self, pattern: &str) -> f64 {
        let records = self.records.read();
        let matching: Vec<&PlanRecord> = records.iter().filter(|r| r.pattern_key() == pattern).collect();
        if matching.is_empty() { 0.0 } else {
            matching.iter().filter(|r| matches!(r.outcome, PlanOutcome::Success)).count() as f64 / matching.len() as f64
        }
    }

    #[must_use]
    pub fn stats(&self) -> PlanningStats {
        let records = self.records.read();
        let n = records.len() as u64;
        if n == 0 { return PlanningStats { total_plans: 0, success_rate: 0.0, mean_reward: 0.0, mean_duration_ms: 0.0 }; }
        let successes = records.iter().filter(|r| matches!(r.outcome, PlanOutcome::Success)).count() as u64;
        let total_reward: f64 = records.iter().map(|r| r.reward).sum();
        let total_dur: f64 = records.iter().map(|r| r.duration_ms as f64).sum();
        PlanningStats {
            total_plans: n,
            success_rate: successes as f64 / n as f64,
            mean_reward: total_reward / n as f64,
            mean_duration_ms: total_dur / n as f64,
        }
    }
    #[must_use] pub fn len(&self) -> usize { self.records.read().len() }
    #[must_use] pub fn is_empty(&self) -> bool { self.records.read().is_empty() }
}

impl std::fmt::Debug for PlanningReplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanningReplay").field("plans", &self.len())
            .field("patterns", &self.pattern_scores.read().len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_increments_count() {
        let p = PlanningReplay::new();
        assert_eq!(p.len(), 0);
        p.record(PlanRecord::new("goal", vec!["step1".into()], vec!["tool1".into()], PlanOutcome::Success, 1.0, 100));
        assert_eq!(p.len(), 1);
    }
    #[test]
    fn stats_computes_correctly() {
        let p = PlanningReplay::new();
        p.record(PlanRecord::new("g", vec!["a".into()], vec![], PlanOutcome::Success, 1.0, 100));
        p.record(PlanRecord::new("g", vec!["b".into()], vec![], PlanOutcome::Failure, 0.0, 200));
        let s = p.stats();
        assert_eq!(s.total_plans, 2);
        assert!((s.success_rate - 0.5).abs() < 1e-9);
        assert!((s.mean_reward - 0.5).abs() < 1e-9);
        assert!((s.mean_duration_ms - 150.0).abs() < 1e-9);
    }
    #[test]
    fn get_best_pattern_finds_similar_success() {
        let p = PlanningReplay::new();
        p.record(PlanRecord::new("debug rust code", vec!["analyze".into(), "fix".into()], vec![], PlanOutcome::Success, 1.0, 100));
        p.record(PlanRecord::new("cook pasta", vec!["boil".into()], vec![], PlanOutcome::Success, 1.0, 50));
        let best = p.get_best_pattern("debug rust issue");
        assert!(best.is_some());
        assert_eq!(best.unwrap()[0], "analyze");
    }
    #[test]
    fn pattern_success_rate_returns_zero_for_unknown() {
        let p = PlanningReplay::new();
        assert!((p.pattern_success_rate("unknown") - 0.0).abs() < 1e-9);
    }
    #[test]
    fn pattern_success_rate_computes_correctly() {
        let p = PlanningReplay::new();
        let steps = vec!["a".into(), "b".into()];
        p.record(PlanRecord::new("g", steps.clone(), vec![], PlanOutcome::Success, 1.0, 100));
        p.record(PlanRecord::new("g", steps.clone(), vec![], PlanOutcome::Failure, 0.0, 100));
        p.record(PlanRecord::new("g", steps, vec![], PlanOutcome::Success, 1.0, 100));
        assert!((p.pattern_success_rate("a → b") - 2.0 / 3.0).abs() < 1e-6);
    }
    #[test]
    fn outcome_score_reflects_severity() {
        assert_eq!(PlanOutcome::Success.score(), 1.0);
        assert_eq!(PlanOutcome::Failure.score(), 0.0);
        assert!(PlanOutcome::Timeout.score() < PlanOutcome::PartialSuccess.score());
    }
}
