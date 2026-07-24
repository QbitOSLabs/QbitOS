#![allow(missing_docs)]
//! Offline consolidation ("sleep"): a background process that compresses
//! memory, updates the world model, and consolidates learning. Sleep runs when
//! the cognitive loop has accumulated enough cycles since the last run. The
//! simulation computes how many items would be compressed / replayed /
//! consolidated based on configurable percentages.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

const SLEEP_THRESHOLD_CYCLES: u64 = 100;

#[derive(Debug, Error)]
pub enum SleepError { #[error("sleep cycle interrupted: {0}")] Interrupted(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    pub duration_secs: u64,
    pub compress_memories: bool,
    pub update_world_model: bool,
    pub consolidate_skills: bool,
    pub replay_experiences: bool,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            duration_secs: 60, compress_memories: true, update_world_model: true,
            consolidate_skills: true, replay_experiences: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepStats {
    pub memories_compressed: u64,
    pub world_model_updates: u64,
    pub skills_consolidated: u64,
    pub experiences_replayed: u64,
    pub total_duration_ms: u64,
    pub improvement_detected: bool,
}

impl Default for SleepStats {
    fn default() -> Self {
        Self {
            memories_compressed: 0, world_model_updates: 0, skills_consolidated: 0,
            experiences_replayed: 0, total_duration_ms: 0, improvement_detected: false,
        }
    }
}

/// A sleep cycle: runs offline consolidation when invoked.
pub struct SleepCycle {
    pub config: SleepConfig,
    stats: Arc<RwLock<SleepStats>>,
    cycles_since_last: Arc<RwLock<u64>>,
}

impl SleepCycle {
    #[must_use]
    pub fn new(config: SleepConfig) -> Self {
        Self {
            config, stats: Arc::new(RwLock::new(SleepStats::default())),
            cycles_since_last: Arc::new(RwLock::new(0)),
        }
    }

    /// Run an offline consolidation cycle over the current counts of memories,
    /// experiences, and skills. Returns the resulting stats.
    pub fn run(&self, memory_count: usize, experience_count: usize, skill_count: usize) -> SleepStats {
        let start = Instant::now();
        let mut s = SleepStats::default();
        if self.config.compress_memories { s.memories_compressed = pct(memory_count, 30); }
        if self.config.update_world_model { s.world_model_updates = pct(experience_count, 20); }
        if self.config.consolidate_skills { s.skills_consolidated = pct(skill_count, 10); }
        if self.config.replay_experiences { s.experiences_replayed = pct(experience_count, 50); }
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let configured_ms = self.config.duration_secs.saturating_mul(1000);
        s.total_duration_ms = elapsed_ms.saturating_add(configured_ms);
        s.improvement_detected = s.experiences_replayed > 0 && s.world_model_updates > 0;
        *self.stats.write() = s.clone();
        *self.cycles_since_last.write() = 0;
        s
    }

    /// Whether the system should sleep, given cycles elapsed since the last run.
    #[must_use]
    pub fn should_sleep(&self, cycles_since_last: u64) -> bool { cycles_since_last > SLEEP_THRESHOLD_CYCLES }

    /// Tick the cycle counter (call once per cognitive cycle).
    pub fn tick(&self) { *self.cycles_since_last.write() += 1; }

    #[must_use] pub fn cycles_since_last_sleep(&self) -> u64 { *self.cycles_since_last.read() }
    #[must_use] pub fn last_stats(&self) -> SleepStats { self.stats.read().clone() }
}

impl Default for SleepCycle { fn default() -> Self { Self::new(SleepConfig::default()) } }

#[inline]
fn pct(count: usize, percent: u64) -> u64 { ((count as u64) * percent) / 100 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_config_has_reasonable_values() {
        let c = SleepConfig::default();
        assert_eq!(c.duration_secs, 60);
        assert!(c.compress_memories && c.update_world_model && c.consolidate_skills && c.replay_experiences);
    }
    #[test]
    fn run_computes_stats_with_default_config() {
        let cycle = SleepCycle::new(SleepConfig::default());
        let stats = cycle.run(100, 50, 20);
        assert_eq!(stats.memories_compressed, 30);
        assert_eq!(stats.world_model_updates, 10);
        assert_eq!(stats.skills_consolidated, 2);
        assert_eq!(stats.experiences_replayed, 25);
        assert!(stats.improvement_detected);
        assert!(stats.total_duration_ms >= 60_000);
    }
    #[test]
    fn should_sleep_returns_true_after_threshold() {
        let cycle = SleepCycle::new(SleepConfig::default());
        assert!(!cycle.should_sleep(50));
        assert!(!cycle.should_sleep(100));
        assert!(cycle.should_sleep(101));
        assert!(cycle.should_sleep(200));
    }
    #[test]
    fn last_stats_returns_latest_run() {
        let cycle = SleepCycle::new(SleepConfig::default());
        let _ = cycle.run(10, 5, 2);
        let stats = cycle.last_stats();
        assert_eq!(stats.memories_compressed, 3);
        assert_eq!(stats.experiences_replayed, 2);
    }
    #[test]
    fn disabled_phases_produce_zero_counts() {
        let mut cfg = SleepConfig::default();
        cfg.compress_memories = false;
        cfg.replay_experiences = false;
        let cycle = SleepCycle::new(cfg);
        let stats = cycle.run(100, 100, 100);
        assert_eq!(stats.memories_compressed, 0);
        assert_eq!(stats.experiences_replayed, 0);
        assert!(!stats.improvement_detected);
    }
    #[test]
    fn tick_increments_cycle_counter_and_run_resets_it() {
        let cycle = SleepCycle::new(SleepConfig::default());
        assert_eq!(cycle.cycles_since_last_sleep(), 0);
        cycle.tick();
        cycle.tick();
        assert_eq!(cycle.cycles_since_last_sleep(), 2);
        cycle.run(0, 0, 0);
        assert_eq!(cycle.cycles_since_last_sleep(), 0);
    }
}
