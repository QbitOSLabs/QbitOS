//! Simulation engine.
//!
//! Runs forward simulations of the world model: given an initial state and
//! a set of events, predicts future states. Used for planning ("what happens
//! if I do X?") and counterfactual reasoning ("what if Y had not happened?").

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::causal_engine::{CausalGraph, CausalReasoner};

/// A single step in a simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationStep {
    /// Step number (0 = initial state).
    pub step: usize,
    /// Events that occurred at this step.
    pub events: Vec<Uuid>,
    /// Predicted probabilities for each event.
    pub probabilities: Vec<(Uuid, f32)>,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

/// A simulation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    /// Initial events.
    pub initial_events: Vec<Uuid>,
    /// Predicted steps.
    pub steps: Vec<SimulationStep>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// The simulation engine.
pub struct Simulation {
    causal_graph: Arc<CausalGraph>,
    reasoner: Arc<CausalReasoner>,
}

impl Simulation {
    /// Create a new simulation engine.
    #[must_use]
    pub fn new(causal_graph: Arc<CausalGraph>, reasoner: Arc<CausalReasoner>) -> Self {
        Self { causal_graph, reasoner }
    }

    /// Returns a reference to the underlying causal graph.
    #[must_use]
    pub fn graph(&self) -> &Arc<CausalGraph> {
        &self.causal_graph
    }

    /// Returns the total number of events in the causal graph.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.causal_graph.event_count()
    }

    /// Returns the total number of causal links in the graph.
    #[must_use]
    pub fn link_count(&self) -> usize {
        self.causal_graph.link_count()
    }

    /// Run a forward simulation: given `initial_events`, predict effects
    /// up to `depth` steps.
    #[must_use]
    pub fn forward(&self, initial_events: &[Uuid], depth: usize) -> SimulationResult {
        let start = std::time::Instant::now();
        let mut steps = Vec::with_capacity(depth);
        let mut current_events: Vec<Uuid> = initial_events.to_vec();
        for step in 0..depth {
            // For each event in the current step, predict effects
            let mut all_effects: ahash::AHashMap<Uuid, f32> = ahash::AHashMap::new();
            for event in &current_events {
                let effects = self.reasoner.predict_effects(*event, 1);
                for (effect_id, prob) in effects {
                    let entry = all_effects.entry(effect_id).or_insert(0.0);
                    *entry = (*entry).max(prob);
                }
            }
            let probabilities: Vec<(Uuid, f32)> = all_effects.into_iter()
                .filter(|(_, p)| *p > 0.01)
                .collect();
            let next_events: Vec<Uuid> = probabilities.iter().map(|(id, _)| *id).collect();
            steps.push(SimulationStep {
                step,
                events: current_events.clone(),
                probabilities: probabilities.clone(),
                timestamp: Utc::now(),
            });
            if next_events.is_empty() {
                break;
            }
            current_events = next_events;
        }
        SimulationResult {
            initial_events: initial_events.to_vec(),
            steps,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        }
    }

    /// Counterfactual: "if `counterfactual` had not happened, would `target`
    /// still have happened?"
    ///
    /// Returns the difference in probability for `target` with vs. without
    /// `counterfactual`. Positive values mean `counterfactual` increased the
    /// probability of `target`.
    #[must_use]
    pub fn counterfactual(&self, counterfactual: Uuid, target: Uuid, depth: usize) -> f32 {
        // Without counterfactual: compute base probability of target
        // (We can't easily "subtract" a cause from the graph, so we approximate
        // by computing the probability of target given all other causes.)
        let base = self.reasoner.explain_causes(target, depth)
            .into_iter()
            .filter(|(id, _)| *id != counterfactual)
            .map(|(_, p)| p)
            .fold(0.0_f32, f32::max);
        // With counterfactual: compute probability of target given counterfactual
        let with_cf = self.reasoner.predict_effects(counterfactual, depth)
            .into_iter()
            .find(|(id, _)| *id == target)
            .map(|(_, p)| p)
            .unwrap_or(0.0);
        with_cf - base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_engine::CausalLink;

    #[test]
    fn forward_simulation() {
        let graph = Arc::new(CausalGraph::new());
        let reasoner = Arc::new(CausalReasoner::new(Arc::clone(&graph)));
        let sim = Simulation::new(Arc::clone(&graph), Arc::clone(&reasoner));
        let rain = Uuid::now_v7();
        let wet = Uuid::now_v7();
        let slip = Uuid::now_v7();
        graph.add_link(CausalLink { cause: rain, effect: wet, strength: 0.9, description: "rain wets ground".into() });
        graph.add_link(CausalLink { cause: wet, effect: slip, strength: 0.5, description: "wet ground causes slipping".into() });
        let result = sim.forward(&[rain], 2);
        assert!(!result.steps.is_empty());
    }
}
