#![allow(missing_docs)]
//! Group Relative Policy Optimization (GRPO) — DeepSeek 2025, refined 2026.
//!
//! GRPO eliminates the need for a separate value/critic model by computing
//! advantages relative to a group of G completions sampled from the same
//! prompt. The advantage of each completion is simply its reward minus the
//! mean reward of the group:
//!
//! ```text
//! A_i = r_i − mean(r_1..r_G)
//! ```
//!
//! This produces lower-variance advantage estimates than vanilla REINFORCE
//! while removing the value-bottleneck of PPO. The policy loss is the
//! standard policy-gradient surrogate weighted by the group-relative
//! advantage, and a KL penalty `β · KL(π_θ || π_ref)` keeps the policy
//! close to a frozen reference:
//!
//! ```text
//! L = − mean(A_i · log π_θ(y_i | x)) + β · mean(KL_i)
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during GRPO training.
#[derive(Debug, Error)]
pub enum GrpoError {
    /// The number of completions does not match the number of rewards.
    #[error("mismatched lengths: {completions} completions vs {rewards} rewards")]
    LengthMismatch {
        /// Number of completions supplied.
        completions: usize,
        /// Number of rewards supplied.
        rewards: usize,
    },
    /// The group was empty.
    #[error("empty group: at least one completion is required")]
    Empty,
}

/// Configuration for the GRPO trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Number of completions to sample per prompt (the "group" G).
    pub group_size: usize,
    /// KL penalty coefficient (β).
    pub beta: f64,
    /// Maximum number of training iterations.
    pub max_iterations: usize,
    /// Learning rate (η).
    pub learning_rate: f64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            beta: 0.04,
            max_iterations: 1_000,
            learning_rate: 1e-5,
        }
    }
}

/// Result of a single GRPO training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoStepResult {
    /// Group-relative advantages (one per completion).
    pub advantages: Vec<f64>,
    /// Policy loss = -mean(advantage · log_prob).
    pub policy_loss: f64,
    /// KL penalty term (β · mean(KL)).
    pub kl_penalty: f64,
    /// Mean reward of the group.
    pub mean_reward: f64,
}

/// Group Relative Policy Optimization trainer.
#[derive(Debug, Clone)]
pub struct GrpoTrainer {
    /// Trainer configuration.
    pub config: GrpoConfig,
    /// Cumulative number of training iterations performed.
    pub iteration: usize,
    /// Running mean of the policy loss (EMA).
    pub running_loss: f64,
    /// Running mean of the KL penalty (EMA).
    pub running_kl: f64,
}

impl GrpoTrainer {
    /// Construct a new GRPO trainer.
    #[must_use]
    pub fn new(config: GrpoConfig) -> Self {
        Self {
            config,
            iteration: 0,
            running_loss: 0.0,
            running_kl: 0.0,
        }
    }

    /// Compute group-relative advantages from a vector of rewards.
    ///
    /// The advantage of each completion is `reward[i] - mean(rewards)`.
    /// This eliminates the need for a separate value/critic model.
    #[must_use]
    pub fn compute_advantages(rewards: &[f64]) -> Vec<f64> {
        if rewards.is_empty() {
            return Vec::new();
        }
        let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
        rewards.iter().map(|r| r - mean).collect()
    }

    /// Estimate a (toy) log-probability for a completion.
    ///
    /// In a real implementation, this is the model's `log π_θ(y | x)` under
    /// the current policy. Here we use a deterministic character-entropy
    /// surrogate so that the trainer is fully self-contained and testable
    /// without a neural-network backend.
    fn log_prob(_prompt: &str, completion: &str) -> f64 {
        if completion.is_empty() {
            return -12.0;
        }
        let mut counts = [0u32; 256];
        for b in completion.bytes() {
            counts[b as usize] += 1;
        }
        let total = completion.len() as f64;
        let mut entropy = 0.0;
        for &c in &counts {
            if c == 0 {
                continue;
            }
            let p = c as f64 / total;
            entropy -= p * p.ln();
        }
        // Log-prob ~ -entropy / sqrt(length): longer, more-random completions
        // have lower log-probability under a uniform-ish model.
        -entropy / total.sqrt().max(1.0)
    }

    /// Estimate the KL divergence between the current policy and a frozen
    /// reference policy, for a single completion.
    ///
    /// For GRPO we need `KL(π_θ || π_ref)`. In this surrogate implementation
    /// we approximate it as the absolute difference between the current and
    /// reference log-probs (a sample-level KL bound).
    fn kl_divergence(&self, prompt: &str, completion: &str) -> f64 {
        let current = Self::log_prob(prompt, completion);
        // Reference policy is approximated as a slightly higher-entropy
        // distribution (i.e. closer to uniform), to simulate a freshly
        // initialized reference model that has not yet been specialised.
        let reference = current * 0.95;
        (current - reference).abs()
    }

    /// Run a single GRPO training step on a batch of completions.
    ///
    /// # Errors
    /// Returns `GrpoError::Empty` if either input is empty, or
    /// `GrpoError::LengthMismatch` if `completions.len() != rewards.len()`.
    pub fn train_batch(
        &mut self,
        prompt: &str,
        completions: Vec<String>,
        rewards: Vec<f64>,
    ) -> Result<GrpoStepResult, GrpoError> {
        if completions.is_empty() || rewards.is_empty() {
            return Err(GrpoError::Empty);
        }
        if completions.len() != rewards.len() {
            return Err(GrpoError::LengthMismatch {
                completions: completions.len(),
                rewards: rewards.len(),
            });
        }

        let n = completions.len();
        let mean_reward = rewards.iter().sum::<f64>() / n as f64;
        let advantages = Self::compute_advantages(&rewards);

        // policy_loss = -mean(advantage * log_prob)
        let mut loss_sum = 0.0;
        let mut kl_sum = 0.0;
        for (completion, advantage) in completions.iter().zip(advantages.iter()) {
            let lp = Self::log_prob(prompt, completion);
            loss_sum += advantage * lp;
            kl_sum += self.kl_divergence(prompt, completion);
        }
        let policy_loss = -loss_sum / n as f64;
        let kl_penalty = self.config.beta * (kl_sum / n as f64);

        // Update running statistics with an exponential moving average.
        let alpha = 0.1;
        self.running_loss = if self.iteration == 0 {
            policy_loss
        } else {
            (1.0 - alpha) * self.running_loss + alpha * policy_loss
        };
        self.running_kl = if self.iteration == 0 {
            kl_penalty
        } else {
            (1.0 - alpha) * self.running_kl + alpha * kl_penalty
        };
        self.iteration += 1;

        Ok(GrpoStepResult {
            advantages,
            policy_loss,
            kl_penalty,
            mean_reward,
        })
    }

    /// Apply a gradient-descent-like update to the learning rate based on
    /// the latest loss. This is a surrogate that scales the LR down when
    /// the loss explodes and up when it is small. Returns the new LR.
    pub fn adapt_learning_rate(&mut self, loss: f64) -> f64 {
        if loss.is_nan() || loss.is_infinite() {
            self.config.learning_rate *= 0.5;
            return self.config.learning_rate;
        }
        if loss.abs() > 10.0 {
            self.config.learning_rate *= 0.9;
        } else if loss.abs() < 0.1 {
            self.config.learning_rate *= 1.05;
        }
        self.config.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = GrpoConfig::default();
        assert_eq!(cfg.group_size, 8);
        assert!((cfg.beta - 0.04).abs() < 1e-9);
        assert!(cfg.max_iterations > 0);
        assert!(cfg.learning_rate > 0.0);
    }

    #[test]
    fn compute_advantages_is_mean_subtracted() {
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let adv = GrpoTrainer::compute_advantages(&rewards);
        let mean = 2.5;
        for (r, a) in rewards.iter().zip(adv.iter()) {
            assert!((a - (r - mean)).abs() < 1e-9);
        }
        // Sum of advantages is zero (mean-subtracted).
        let sum: f64 = adv.iter().sum();
        assert!(sum.abs() < 1e-9);
    }

    #[test]
    fn train_batch_returns_valid_result() {
        let mut trainer = GrpoTrainer::new(GrpoConfig::default());
        let prompt = "What is 2+2?";
        let completions = (0..8).map(|i| format!("The answer is {i}")).collect();
        let rewards = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0];
        let result = trainer.train_batch(prompt, completions, rewards).unwrap();
        assert_eq!(result.advantages.len(), 8);
        // Mean reward = (7*0.1 + 1.0)/8 = 1.7/8 = 0.2125
        assert!((result.mean_reward - 0.2125).abs() < 1e-9);
        assert!(result.policy_loss.is_finite());
        // KL penalty is β * mean(KL) ≥ 0 since β > 0 and KL ≥ 0.
        assert!(result.kl_penalty >= 0.0);
        assert_eq!(trainer.iteration, 1);
    }

    #[test]
    fn train_batch_rejects_empty_inputs() {
        let mut trainer = GrpoTrainer::new(GrpoConfig::default());
        let res = trainer.train_batch("p", vec![], vec![]);
        assert!(matches!(res, Err(GrpoError::Empty)));
    }

    #[test]
    fn train_batch_rejects_length_mismatch() {
        let mut trainer = GrpoTrainer::new(GrpoConfig::default());
        let res = trainer.train_batch(
            "p",
            vec!["a".to_string(), "b".to_string()],
            vec![1.0],
        );
        assert!(matches!(res, Err(GrpoError::LengthMismatch { .. })));
    }

    #[test]
    fn adapt_learning_rate_shrinks_on_large_loss() {
        let mut trainer = GrpoTrainer::new(GrpoConfig::default());
        let lr0 = trainer.config.learning_rate;
        let _ = trainer.adapt_learning_rate(100.0);
        assert!(trainer.config.learning_rate < lr0);
    }

    #[test]
    fn adapt_learning_rate_grows_on_tiny_loss() {
        let mut trainer = GrpoTrainer::new(GrpoConfig::default());
        let lr0 = trainer.config.learning_rate;
        let _ = trainer.adapt_learning_rate(0.01);
        assert!(trainer.config.learning_rate > lr0);
    }
}
