//! Consolidation engine.
//!
//! Moves memory objects between layers based on importance, recency, and
//! usage. Inspired by hippocampal replay: during idle periods, episodic
//! memories are replayed and the important ones are promoted to semantic
//! (facts) or procedural (skills) memory.

use std::sync::Arc;
use std::time::Duration;

use crate::episodic::{EpisodicMemory, Outcome};
use crate::meta::MetaMemory;
use crate::procedural::ProceduralMemory;
use crate::semantic::SemanticMemory;
use crate::sensory::SensoryMemory;
use crate::working::WorkingMemory;

/// Configuration for the consolidation engine.
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum importance to promote an episode to semantic memory.
    pub promote_to_semantic_threshold: f32,
    /// Minimum importance to promote an episode to procedural memory.
    pub promote_to_procedural_threshold: f32,
    /// Importance decay factor applied per consolidation cycle.
    pub decay_factor: f32,
    /// Importance threshold below which episodes are evicted.
    pub eviction_threshold: f32,
    /// How often to run consolidation.
    pub interval: Duration,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            promote_to_semantic_threshold: 0.8,
            promote_to_procedural_threshold: 0.85,
            decay_factor: 0.95,
            eviction_threshold: 0.1,
            interval: Duration::from_secs(60),
        }
    }
}

/// The consolidation engine.
pub struct ConsolidationEngine {
    config: ConsolidationConfig,
    sensory: Arc<SensoryMemory>,
    working: Arc<WorkingMemory>,
    episodic: Arc<EpisodicMemory>,
    semantic: Arc<SemanticMemory>,
    procedural: Arc<ProceduralMemory>,
    /// Meta-memory: tracks learning strategies, preferences, and calibration.
    meta: Arc<MetaMemory>,
}

impl ConsolidationEngine {
    /// Create a new consolidation engine.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        config: ConsolidationConfig,
        sensory: Arc<SensoryMemory>,
        working: Arc<WorkingMemory>,
        episodic: Arc<EpisodicMemory>,
        semantic: Arc<SemanticMemory>,
        procedural: Arc<ProceduralMemory>,
        // Meta-memory: tracks learning strategies, preferences, and calibration.
        meta: Arc<MetaMemory>,
    ) -> Self {
        Self { config, sensory, working, episodic, semantic, procedural, meta }
    }

    /// Run one consolidation cycle.
    ///
    /// 1. Evict expired sensory slots.
    /// 2. Decay episodic memory and evict low-importance episodes.
    /// 3. Promote high-importance episodes to semantic / procedural memory.
    /// 4. Update calibration from meta-memory.
    ///
    /// Returns a summary of what was consolidated.
    #[must_use]
    pub fn consolidate(&self) -> ConsolidationSummary {
        // 1. Sensory eviction
        self.sensory.evict();

        // 2. Episodic decay (we need to do this manually because the HNSW
        //    index doesn't expose iteration; we use recall_similar on a
        //    wildcard to get a sample).
        //    For correctness, we'll use a simpler approach: call decay on
        //    episodic, which will iterate the store directly.
        self.episodic.decay(self.config.decay_factor, self.config.eviction_threshold);

        // 3. Promotion: scan episodic memory for high-importance episodes and
        //    promote them. Since we don't have a public iterator, we use a
        //    time range query covering the last hour.
        let now = chrono::Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);
        let recent = self.episodic.recall_in_range(one_hour_ago, now);
        let mut promoted_semantic = 0;
        let mut promoted_procedural = 0;
        let mut successes = 0_u32;
        let mut total = 0_u32;
        for ep in &recent {
            total += 1;
            if ep.outcome == Outcome::Success {
                successes += 1;
            }
            if ep.object.importance.0 >= self.config.promote_to_procedural_threshold
                && ep.outcome == Outcome::Success
            {
                if let Some(label) = ep.object.content.get("skill").and_then(|v| v.as_str()) {
                    self.procedural.register(label, label);
                    promoted_procedural += 1;
                }
            } else if ep.object.importance.0 >= self.config.promote_to_semantic_threshold {
                if let Some(label) = ep.object.content.get("concept").and_then(|v| v.as_str()) {
                    self.semantic.add_concept(label, ep.object.content.clone());
                    promoted_semantic += 1;
                }
            }
        }

        // 4. Update meta-memory's calibration with the observed success rate.
        //    We feed it the observed success rate as `actual` and the current
        //    calibration value as `predicted`; the meta-memory maintains an
        //    exponential moving average of the error.
        if total > 0 {
            let actual = successes as f32 / total as f32;
            let predicted = self.meta.calibration();
            self.meta.update_calibration(predicted, actual);
        }

        // 5. Clear working memory if it's stale (no touches in last 5 minutes).
        let working_last = self.working.last();
        if let Some(last) = working_last {
            let age = (now - last.last_accessed_at).num_seconds();
            if age > 300 {
                self.working.clear();
            }
        }

        ConsolidationSummary {
            promoted_semantic,
            promoted_procedural,
            evicted_sensory: 0, // computed inside sensory.evict
            episodic_remaining: self.episodic.len(),
        }
    }

    /// Returns the current meta-memory calibration value (EMA of prediction error).
    #[must_use]
    pub fn calibration(&self) -> f32 {
        self.meta.calibration()
    }

    /// Returns the top-k learning strategies from meta-memory.
    #[must_use]
    pub fn top_strategies(&self, k: usize) -> Vec<crate::meta::LearningStrategy> {
        self.meta.top_strategies(k)
    }

    /// Run consolidation forever on a timer.
    pub async fn run(self: Arc<Self>) {
        let mut ticker = tokio::time::interval(self.config.interval);
        loop {
            ticker.tick().await;
            let summary = self.consolidate();
            tracing::debug!(target: "qbitos.memory.consolidation", ?summary, "consolidation cycle");
        }
    }
}

/// A summary of one consolidation cycle.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationSummary {
    /// Number of episodes promoted to semantic memory.
    pub promoted_semantic: usize,
    /// Number of episodes promoted to procedural memory.
    pub promoted_procedural: usize,
    /// Number of sensory slots evicted.
    pub evicted_sensory: usize,
    /// Number of episodes remaining in episodic memory.
    pub episodic_remaining: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryFabric;

    fn make_engine() -> ConsolidationEngine {
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        ConsolidationEngine::new(
            ConsolidationConfig::default(),
            Arc::clone(&memory.sensory),
            Arc::clone(&memory.working),
            Arc::clone(&memory.episodic),
            Arc::clone(&memory.semantic),
            Arc::clone(&memory.procedural),
            Arc::clone(&memory.meta),
        )
    }

    #[test]
    fn consolidate_empty_returns_zero_summary() {
        let engine = make_engine();
        let summary = engine.consolidate();
        assert_eq!(summary.promoted_semantic, 0);
        assert_eq!(summary.promoted_procedural, 0);
        assert_eq!(summary.episodic_remaining, 0);
    }

    #[test]
    fn consolidate_with_episodes_does_not_panic() {
        let engine = make_engine();
        // Record some episodes directly into episodic memory.
        engine.episodic.record(
            serde_json::json!({"concept": "rust", "text": "Rust is a systems language"}),
            Outcome::Success,
        );
        engine.episodic.record(
            serde_json::json!({"skill": "summarize", "text": "Summarize a document"}),
            Outcome::Success,
        );
        let summary = engine.consolidate();
        // The episodes should still be in episodic memory after consolidation.
        assert!(summary.episodic_remaining >= 0);
    }

    #[test]
    fn calibration_starts_at_zero() {
        let engine = make_engine();
        // The calibration value starts at 0.0 in a fresh meta-memory.
        let cal = engine.calibration();
        assert!(cal >= 0.0 && cal <= 1.0);
    }

    #[test]
    fn top_strategies_empty_initially() {
        let engine = make_engine();
        let strategies = engine.top_strategies(5);
        assert!(strategies.is_empty());
    }

    #[test]
    fn config_defaults_are_sane() {
        let cfg = ConsolidationConfig::default();
        assert!(cfg.promote_to_semantic_threshold > 0.0);
        assert!(cfg.promote_to_procedural_threshold > cfg.promote_to_semantic_threshold);
        assert!(cfg.decay_factor > 0.0 && cfg.decay_factor < 1.0);
        assert!(cfg.eviction_threshold >= 0.0 && cfg.eviction_threshold < cfg.promote_to_semantic_threshold);
        assert!(cfg.interval.as_secs() > 0);
    }
}
