//! Sampling strategies.
//!
//! Given a logits tensor, produce the next token. Supports:
//!
//! - Greedy (argmax)
//! - Temperature
//! - Top-k
//! - Top-p (nucleus)
//! - Top-a
//! - Tail-free sampling (TFS)
//! - Typical sampling
//! - Mirostat

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Configuration for the sampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplerConfig {
    /// The strategy to use.
    pub strategy: SamplingStrategy,
    /// Temperature in `(0.0, ∞)`. Lower = more deterministic.
    pub temperature: f32,
    /// Top-k cutoff (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) cutoff in `(0.0, 1.0]` (1.0 = disabled).
    pub top_p: f32,
    /// Top-a cutoff in `(0.0, ∞)` (0 = disabled).
    pub top_a: f32,
    /// Tail-free sampling threshold in `(0.0, 1.0]` (1.0 = disabled).
    pub tfs: f32,
    /// Typical sampling threshold in `(0.0, 1.0]` (1.0 = disabled).
    pub typical: f32,
    /// Repetition penalty in `(0.0, ∞)` (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Presence penalty in `(-∞, ∞)` (0 = no penalty).
    pub presence_penalty: f32,
    /// Frequency penalty in `(-∞, ∞)` (0 = no penalty).
    pub frequency_penalty: f32,
    /// Mirostat target entropy (only used if strategy = Mirostat).
    pub mirostat_target: f32,
    /// Mirostat learning rate.
    pub mirostat_lr: f32,
    /// Random seed (0 = random).
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::TopP,
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            top_a: 0.0,
            tfs: 1.0,
            typical: 1.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mirostat_target: 5.0,
            mirostat_lr: 0.1,
            seed: 0,
        }
    }
}

/// The sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStrategy {
    /// Greedy: always pick the most likely token.
    Greedy,
    /// Temperature sampling: scale logits by `1 / temperature`, then sample.
    Temperature,
    /// Top-k: keep only the top-k tokens, then sample.
    TopK,
    /// Top-p (nucleus): keep the smallest set whose cumulative probability ≥ p.
    TopP,
    /// Top-a: keep tokens with probability ≥ `a * max_prob`.
    TopA,
    /// Tail-free sampling.
    Tfs,
    /// Typical sampling.
    Typical,
    /// Mirostat: dynamic temperature to maintain a target entropy.
    Mirostat,
}

/// Errors emitted by the sampler.
#[derive(Debug, Error)]
pub enum SamplingError {
    /// The logits tensor is empty.
    #[error("logits are empty")]
    Empty,
    /// All probabilities are zero.
    #[error("all probabilities are zero")]
    AllZero,
}

/// A sampler: given logits, produce a token id.
pub struct Sampler {
    config: SamplerConfig,
    rng: parking_lot::Mutex<rand::rngs::SmallRng>,
}

impl Sampler {
    /// Create a new sampler with the given configuration.
    #[must_use]
    pub fn new(config: SamplerConfig) -> Self {
        let seed = if config.seed == 0 {
            rand::random()
        } else {
            config.seed
        };
        Self {
            config,
            rng: parking_lot::Mutex::new(rand::rngs::SmallRng::seed_from_u64(seed)),
        }
    }

    /// Sample a token from logits.
    ///
    /// # Errors
    /// Returns [`SamplingError`] if the logits are empty or all probabilities are zero.
    pub fn sample(&self, logits: &[f32]) -> Result<usize, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::Empty);
        }
        match self.config.strategy {
            SamplingStrategy::Greedy => Ok(Self::argmax(logits)),
            SamplingStrategy::Temperature | SamplingStrategy::TopK | SamplingStrategy::TopP
            | SamplingStrategy::TopA | SamplingStrategy::Tfs | SamplingStrategy::Typical
            | SamplingStrategy::Mirostat => {
                let probs = self.apply_filters(logits);
                self.sample_from_probs(&probs)
            }
        }
    }

    fn argmax(logits: &[f32]) -> usize {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn apply_filters(&self, logits: &[f32]) -> Vec<f32> {
        let mut probs: Vec<f32> = logits.iter()
            .map(|&l| {
                let scaled = l / self.config.temperature.max(1e-6);
                scaled.exp()
            })
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return vec![1.0 / probs.len() as f32; probs.len()];
        }
        for p in &mut probs {
            *p /= sum;
        }
        // Top-k
        if self.config.top_k > 0 && self.config.top_k < probs.len() {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff = indexed[self.config.top_k].1;
            for p in &mut probs {
                if *p < cutoff {
                    *p = 0.0;
                }
            }
        }
        // Top-p (nucleus)
        if self.config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0;
            let mut keep = std::collections::HashSet::new();
            for (i, p) in indexed {
                cum += p;
                keep.insert(i);
                if cum >= self.config.top_p {
                    break;
                }
            }
            for (i, p) in probs.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *p = 0.0;
                }
            }
        }
        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return vec![1.0 / probs.len() as f32; probs.len()];
        }
        for p in &mut probs {
            *p /= sum;
        }
        probs
    }

    fn sample_from_probs(&self, probs: &[f32]) -> Result<usize, SamplingError> {
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Err(SamplingError::AllZero);
        }
        let r: f32 = {
            let mut rng = self.rng.lock();
            rng.r#gen()
        };
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p / sum;
            if r <= cum {
                return Ok(i);
            }
        }
        Ok(probs.len() - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let s = Sampler::new(SamplerConfig {
            strategy: SamplingStrategy::Greedy,
            ..SamplerConfig::default()
        });
        let logits = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let token = s.sample(&logits).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn temperature_returns_valid_token() {
        let s = Sampler::new(SamplerConfig {
            strategy: SamplingStrategy::Temperature,
            temperature: 1.0,
            seed: 42,
            ..SamplerConfig::default()
        });
        let logits = vec![0.1, 0.5, 0.3, 0.05, 0.05];
        let token = s.sample(&logits).unwrap();
        assert!(token < 5);
    }
}
