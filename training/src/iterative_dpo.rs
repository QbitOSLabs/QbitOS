#![allow(missing_docs)]
//! Iterative Direct Preference Optimization (DPO) — 2026.
//!
//! Iterative DPO runs DPO in multiple rounds. Each round:
//!
//! 1. Samples completions from the current policy.
//! 2. Pairs them into new preference pairs (best vs worst).
//! 3. Runs a DPO gradient step on those pairs.
//! 4. Updates the policy.
//!
//! Because each round generates fresh preference pairs from the model's
//! own outputs, iterative DPO continuously refines the policy without
//! requiring additional human annotations.
//!
//! The DPO loss for a single preference pair `(x, y_w, y_l)` is:
//!
//! ```text
//! L = -log σ( β · ( log π_θ(y_w | x) - log π_θ(y_l | x)
//!                 - log π_ref(y_w | x) + log π_ref(y_l | x) ) )
//! ```

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during iterative DPO training.
#[derive(Debug, Error)]
pub enum IterativeDpoError {
    /// No preferences were provided for the round.
    #[error("no preferences provided for round {0}")]
    Empty(usize),
    /// The maximum number of rounds has been exceeded.
    #[error("max rounds ({0}) exceeded")]
    MaxRoundsExceeded(usize),
}

/// Configuration for iterative DPO.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterativeDpoConfig {
    /// Number of DPO rounds to run.
    pub num_rounds: usize,
    /// β parameter (controls deviation from the reference policy).
    pub beta: f64,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of preference pairs to sample per round.
    pub samples_per_round: usize,
}

impl Default for IterativeDpoConfig {
    fn default() -> Self {
        Self {
            num_rounds: 3,
            beta: 0.1,
            learning_rate: 1e-5,
            samples_per_round: 32,
        }
    }
}

/// A single preference pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoPreference {
    /// The prompt.
    pub prompt: String,
    /// The preferred (chosen) completion.
    pub chosen: String,
    /// The dispreferred (rejected) completion.
    pub rejected: String,
}

/// Result of a single DPO round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoRoundResult {
    /// The round index (0-based).
    pub round: usize,
    /// Final mean loss for the round.
    pub loss: f64,
    /// Fraction of preferences where the current policy already preferred
    /// `chosen` over `rejected` (accuracy).
    pub accuracy: f64,
    /// Whether the policy was updated this round.
    pub policy_updated: bool,
    /// Number of preferences used in this round.
    pub num_preferences: usize,
}

/// The iterative DPO trainer.
pub struct IterativeDpoTrainer {
    /// Trainer configuration.
    pub config: IterativeDpoConfig,
    /// Reference policy (frozen). Maps `prompt → log π_ref(chosen | prompt)`.
    pub reference_policy: AHashMap<String, f64>,
    /// Current policy. Maps `prompt → log π_θ(chosen | prompt)`.
    pub current_policy: AHashMap<String, f64>,
    /// Current round index.
    pub round: usize,
}

impl std::fmt::Debug for IterativeDpoTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IterativeDpoTrainer")
            .field("config", &self.config)
            .field("round", &self.round)
            .field("reference_policy_size", &self.reference_policy.len())
            .field("current_policy_size", &self.current_policy.len())
            .finish()
    }
}

impl IterativeDpoTrainer {
    /// Construct a new iterative DPO trainer with an empty reference policy.
    #[must_use]
    pub fn new(config: IterativeDpoConfig) -> Self {
        Self {
            config,
            reference_policy: AHashMap::new(),
            current_policy: AHashMap::new(),
            round: 0,
        }
    }

    /// Estimate a (toy) log-probability for a completion under a policy.
    ///
    /// In a real implementation this is the model's `log π(y | x)`. Here we
    /// use a deterministic length-based surrogate so the trainer is fully
    /// self-contained and testable without a neural-network backend.
    fn log_prob(prompt: &str, completion: &str) -> f64 {
        let p_len = prompt.len().max(1) as f64;
        let c_len = completion.len().max(1) as f64;
        // Log-prob ~ -log(length): longer completions have lower log-prob.
        -((c_len + 0.5 * p_len).ln())
    }

    /// Initialize the reference policy from a list of preferences (only on
    /// the first round).
    fn init_reference(&mut self, prefs: &[DpoPreference]) {
        if !self.reference_policy.is_empty() {
            return;
        }
        for p in prefs {
            let lp = Self::log_prob(&p.prompt, &p.chosen);
            self.reference_policy.insert(p.prompt.clone(), lp);
            self.current_policy.insert(p.prompt.clone(), lp);
        }
    }

    /// Compute the DPO loss, per-pair accuracy, and the gradient of the
    /// loss with respect to `log π_θ(chosen)` for a single preference.
    ///
    /// Returns `(loss, accuracy, grad_wrt_log_pi_chosen)` where:
    /// - `loss` is the DPO loss `-log σ(z)`.
    /// - `accuracy` is `1.0` if the current policy already prefers
    ///   `chosen` over `rejected` (i.e. `σ(z) ≥ 0.5`), else `0.0`.
    /// - `grad_wrt_log_pi_chosen` is `dL/d(log π_θ(chosen)) = -β·(1-σ(z))`.
    fn dpo_loss_and_grad(&self, pref: &DpoPreference) -> (f64, f64, f64) {
        let lp_chosen_theta = self
            .current_policy
            .get(&pref.prompt)
            .copied()
            .unwrap_or_else(|| Self::log_prob(&pref.prompt, &pref.chosen));
        let lp_rejected_theta = Self::log_prob(&pref.prompt, &pref.rejected);
        let lp_chosen_ref = self
            .reference_policy
            .get(&pref.prompt)
            .copied()
            .unwrap_or(lp_chosen_theta);
        let lp_rejected_ref = Self::log_prob(&pref.prompt, &pref.rejected);

        let delta_theta = lp_chosen_theta - lp_rejected_theta;
        let delta_ref = lp_chosen_ref - lp_rejected_ref;
        let z = self.config.beta * (delta_theta - delta_ref);
        let sigmoid = 1.0 / (1.0 + (-z).exp());
        let loss = -sigmoid.ln().max(-1e6);
        let accuracy = if sigmoid >= 0.5 { 1.0 } else { 0.0 };
        // dL/d(log π_θ(chosen)) = -β · (1 - σ(z))
        let grad = -self.config.beta * (1.0 - sigmoid);
        (loss, accuracy, grad)
    }

    /// Run a single DPO training round on the given preferences.
    ///
    /// # Errors
    /// Returns `IterativeDpoError::Empty` if `preferences` is empty, or
    /// `IterativeDpoError::MaxRoundsExceeded` if `round >= num_rounds`.
    pub fn train_round(
        &mut self,
        preferences: &[DpoPreference],
    ) -> Result<DpoRoundResult, IterativeDpoError> {
        if preferences.is_empty() {
            return Err(IterativeDpoError::Empty(self.round));
        }
        if self.round >= self.config.num_rounds {
            return Err(IterativeDpoError::MaxRoundsExceeded(
                self.config.num_rounds,
            ));
        }

        // Initialize the reference policy on the first round.
        self.init_reference(preferences);

        // Compute per-preference losses and apply gradient updates.
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        let mut policy_updated = false;
        let lr = self.config.learning_rate;
        for pref in preferences {
            let (loss, acc, grad) = self.dpo_loss_and_grad(pref);
            total_loss += loss;
            total_acc += acc;
            // Gradient descent on log π_θ(chosen):
            //   log π_θ(chosen) ← log π_θ(chosen) - lr · grad
            // Since grad = -β·(1-σ), this increases log π_θ(chosen) when σ < 1.
            let cur = self
                .current_policy
                .get(&pref.prompt)
                .copied()
                .unwrap_or_else(|| Self::log_prob(&pref.prompt, &pref.chosen));
            let new_lp = cur - lr * grad;
            self.current_policy.insert(pref.prompt.clone(), new_lp);
            policy_updated = true;
        }

        let n = preferences.len();
        let result = DpoRoundResult {
            round: self.round,
            loss: total_loss / n as f64,
            accuracy: total_acc / n as f64,
            policy_updated,
            num_preferences: n,
        };
        self.round += 1;
        Ok(result)
    }

    /// Borrow the current policy.
    #[must_use]
    pub fn get_policy(&self) -> &AHashMap<String, f64> {
        &self.current_policy
    }

    /// Whether the trainer has finished all rounds.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.round >= self.config.num_rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefs() -> Vec<DpoPreference> {
        vec![
            DpoPreference {
                prompt: "What is 2+2?".into(),
                chosen: "The answer is 4.".into(),
                rejected: "idk".into(),
            },
            DpoPreference {
                prompt: "Capital of France?".into(),
                chosen: "Paris is the capital of France.".into(),
                rejected: "London".into(),
            },
        ]
    }

    #[test]
    fn default_config_has_three_rounds() {
        let cfg = IterativeDpoConfig::default();
        assert_eq!(cfg.num_rounds, 3);
        assert!((cfg.beta - 0.1).abs() < 1e-9);
        assert!(cfg.learning_rate > 0.0);
    }

    #[test]
    fn train_round_updates_policy() {
        let mut trainer = IterativeDpoTrainer::new(IterativeDpoConfig::default());
        let r = trainer.train_round(&prefs()).unwrap();
        assert_eq!(r.round, 0);
        assert!(r.policy_updated);
        assert_eq!(r.num_preferences, 2);
        assert_eq!(trainer.round, 1);
        assert!(!trainer.get_policy().is_empty());
    }

    #[test]
    fn train_round_rejects_empty() {
        let mut trainer = IterativeDpoTrainer::new(IterativeDpoConfig::default());
        let r = trainer.train_round(&[]);
        assert!(matches!(r, Err(IterativeDpoError::Empty(0))));
    }

    #[test]
    fn train_round_rejects_after_max_rounds() {
        let cfg = IterativeDpoConfig {
            num_rounds: 1,
            ..IterativeDpoConfig::default()
        };
        let mut trainer = IterativeDpoTrainer::new(cfg);
        let _ = trainer.train_round(&prefs()).unwrap();
        let r = trainer.train_round(&prefs());
        assert!(matches!(r, Err(IterativeDpoError::MaxRoundsExceeded(1))));
    }

    #[test]
    fn multiple_rounds_complete_to_done() {
        let mut trainer = IterativeDpoTrainer::new(IterativeDpoConfig {
            num_rounds: 3,
            ..IterativeDpoConfig::default()
        });
        let p = prefs();
        let _ = trainer.train_round(&p).unwrap();
        assert!(!trainer.is_done());
        let _ = trainer.train_round(&p).unwrap();
        assert!(!trainer.is_done());
        let _ = trainer.train_round(&p).unwrap();
        assert!(trainer.is_done());
    }

    #[test]
    fn loss_is_finite_and_nonneg() {
        let mut trainer = IterativeDpoTrainer::new(IterativeDpoConfig::default());
        let r = trainer.train_round(&prefs()).unwrap();
        assert!(r.loss.is_finite());
        assert!(r.loss >= 0.0);
        assert!(r.accuracy >= 0.0 && r.accuracy <= 1.0);
    }

    #[test]
    fn reference_policy_initialized_on_first_round() {
        let mut trainer = IterativeDpoTrainer::new(IterativeDpoConfig::default());
        assert!(trainer.reference_policy.is_empty());
        let _ = trainer.train_round(&prefs()).unwrap();
        assert_eq!(trainer.reference_policy.len(), 2);
        // Second round should NOT re-initialize (reference stays frozen).
        let _ = trainer.train_round(&prefs()).unwrap();
        assert_eq!(trainer.reference_policy.len(), 2);
    }
}
