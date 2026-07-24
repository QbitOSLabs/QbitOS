//! Composite evolution module.

use std::sync::Arc;

use crate::evaluator::Evaluator;
use crate::improvement_engine::ImprovementEngine;
use crate::regression_guard::RegressionGuard;

/// The composite evolution module.
pub struct EvolutionModule {
    /// The evaluator.
    pub evaluator: Arc<Evaluator>,
    /// The improvement engine.
    pub improvement: Arc<ImprovementEngine>,
    /// The regression guard.
    pub guard: Arc<RegressionGuard>,
}

impl Default for EvolutionModule {
    fn default() -> Self {
        Self::new()
    }
}

impl EvolutionModule {
    /// Build a new evolution module.
    #[must_use]
    pub fn new() -> Self {
        let evaluator = Arc::new(Evaluator::new());
        let improvement = Arc::new(ImprovementEngine::new(Arc::clone(&evaluator)));
        let guard = Arc::new(RegressionGuard::new(Arc::clone(&evaluator)));
        Self { evaluator, improvement, guard }
    }

    /// Run one evolution cycle:
    /// 1. Snapshot the baseline.
    /// 2. Detect weaknesses.
    /// 3. For each proposal, check for regressions.
    /// 4. Accept safe improvements, reject unsafe ones.
    ///
    /// Returns the number of accepted improvements.
    pub fn run_cycle(&self) -> usize {
        self.guard.snapshot_baseline();
        let proposals = self.improvement.detect_weaknesses();
        let mut accepted = 0;
        for proposal in proposals {
            let report = self.guard.check(&proposal);
            if report.safe {
                self.improvement.accept(proposal.id);
                accepted += 1;
            } else {
                self.improvement.reject(proposal.id);
            }
        }
        accepted
    }
}
