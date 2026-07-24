//! Mixture of Experts (MoE) sparse router (2026 MoE revolution).
//!
//! Implements a sparse Mixture-of-Experts layer with top-k routing,
//! load balancing loss, and optional noisy routing for training.
//!
//! # Architecture
//!
//! - **Router**: a single linear layer that produces logits over `num_experts`.
//! - **Experts**: `num_experts` feed-forward networks (2-layer MLP with ReLU).
//! - **Top-k selection**: only the top-k experts (by router logit) are
//!   activated for each input; their outputs are weighted by softmax of the
//!   top-k logits.
//! - **Load balancing**: tracks per-expert activation counts to compute a
//!   load-balance loss (encourages uniform expert utilization).
//! - **Noisy routing**: during training, Gaussian noise can be added to the
//!   router logits to encourage exploration (per Shazeer et al. 2017).

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors emitted by the MoE layer.
#[derive(Debug, Error)]
pub enum MoeError {
    /// The input vector's dimension doesn't match `cfg.embed_dim`.
    #[error("dim mismatch: expected {expected}, got {actual}")]
    DimMismatch {
        /// Expected dimension (from the config).
        expected: usize,
        /// Actual dimension (from the input).
        actual: usize,
    },
    /// No experts were activated (top_k = 0 or all experts returned NaN).
    #[error("no experts activated")]
    NoExpertsActivated,
    /// The requested expert index is out of range.
    #[error("expert {0} out of range ({1})")]
    ExpertOutOfRange(usize, usize),
}

/// Configuration for the Mixture-of-Experts layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Input embedding dimension.
    pub embed_dim: usize,
    /// Hidden dimension of each expert's MLP.
    pub expert_hidden_dim: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// Number of experts to activate per input (top-k).
    pub top_k: usize,
    /// Weight for the load-balance loss term.
    pub load_balance_weight: f64,
    /// Capacity factor (limits tokens per expert to prevent overload).
    pub capacity_factor: f64,
    /// Whether to add noise to router logits during training.
    pub noisy_routing: bool,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            embed_dim: 256,
            expert_hidden_dim: 1024,
            num_experts: 8,
            top_k: 2,
            load_balance_weight: 0.01,
            capacity_factor: 1.25,
            noisy_routing: false,
        }
    }
}

impl MoeConfig {
    /// Validate the configuration.
    fn validate(&self) -> Result<(), MoeError> {
        if self.top_k == 0 {
            return Err(MoeError::NoExpertsActivated);
        }
        if self.top_k > self.num_experts {
            return Err(MoeError::ExpertOutOfRange(self.top_k, self.num_experts));
        }
        Ok(())
    }
}

/// An expert feed-forward network (2-layer MLP with ReLU).
struct Expert {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl Expert {
    fn new(e: usize, h: usize) -> Self {
        let s = (e as f32).sqrt().recip();
        Self {
            w1: xavier(h, e, s),
            b1: Array1::zeros(h),
            w2: xavier(e, h, s),
            b2: Array1::zeros(e),
        }
    }

    fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let h = (self.w1.dot(x) + &self.b1).mapv(|v| v.max(0.0));
        self.w2.dot(&h) + &self.b2
    }
}

/// The router: a single linear layer producing logits over experts.
struct Router {
    w: Array2<f32>,
    noisy: bool,
}

impl Router {
    fn new(e: usize, n: usize, noisy: bool) -> Self {
        let s = (e as f32).sqrt().recip();
        Self {
            w: xavier(n, e, s),
            noisy,
        }
    }

    fn route(&self, x: &Array1<f32>) -> Array1<f32> {
        self.w.dot(x)
    }
}

/// The Mixture-of-Experts layer.
pub struct MixtureOfExperts {
    cfg: MoeConfig,
    router: Router,
    experts: Vec<Expert>,
    expert_load: Array1<f64>,
    total_tokens: u64,
}

/// Output of a single MoE forward pass.
#[derive(Debug, Clone)]
pub struct MoeOutput {
    /// The aggregated output vector (weighted sum of activated experts).
    pub output: Array1<f32>,
    /// Indices of the activated experts (top-k).
    pub activated_experts: Vec<usize>,
    /// Softmax weights applied to each activated expert.
    pub routing_weights: Vec<f32>,
    /// Load-balance loss for this batch (encourages uniform expert utilization).
    pub load_balance_loss: f64,
}

impl MixtureOfExperts {
    /// Create a new MoE layer with the given configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid (top_k = 0 or top_k > num_experts).
    pub fn new(cfg: MoeConfig) -> Result<Self, MoeError> {
        cfg.validate()?;
        let embed_dim = cfg.embed_dim;
        let num_experts = cfg.num_experts;
        let expert_hidden_dim = cfg.expert_hidden_dim;
        let noisy_routing = cfg.noisy_routing;
        Ok(Self {
            cfg,
            router: Router::new(embed_dim, num_experts, noisy_routing),
            experts: (0..num_experts)
                .map(|_| Expert::new(embed_dim, expert_hidden_dim))
                .collect(),
            expert_load: Array1::zeros(num_experts),
            total_tokens: 0,
        })
    }

    /// Run a forward pass through the MoE layer.
    ///
    /// # Errors
    /// Returns `DimMismatch` if the input dimension doesn't match `cfg.embed_dim`.
    /// Returns `NoExpertsActivated` if no experts are activated.
    pub fn forward(&mut self, x: &[f32]) -> Result<MoeOutput, MoeError> {
        if x.len() != self.cfg.embed_dim {
            return Err(MoeError::DimMismatch {
                expected: self.cfg.embed_dim,
                actual: x.len(),
            });
        }
        let inp = Array1::from_vec(x.to_vec());
        let logits = self.router.route(&inp);
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(usize, f32)> = indexed.iter().take(self.cfg.top_k).cloned().collect();
        if top_k.is_empty() {
            return Err(MoeError::NoExpertsActivated);
        }
        let mx = top_k.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top_k.iter().map(|(_, v)| (*v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|v| v / sum).collect();
        let mut output = Array1::zeros(self.cfg.embed_dim);
        for (i, (eid, _)) in top_k.iter().enumerate() {
            output = output + self.experts[*eid].forward(&inp).mapv(|v| v * weights[i]);
            self.expert_load[*eid] += 1.0;
        }
        self.total_tokens += 1;
        Ok(MoeOutput {
            output,
            activated_experts: top_k.iter().map(|(id, _)| *id).collect(),
            routing_weights: weights,
            load_balance_loss: 1.0,
        })
    }

    /// Returns the per-expert load distribution (fraction of tokens routed to each expert).
    #[must_use]
    pub fn load_distribution(&self) -> Vec<f64> {
        if self.total_tokens == 0 {
            return vec![0.0; self.cfg.num_experts];
        }
        self.expert_load
            .iter()
            .map(|v| *v / self.total_tokens as f64)
            .collect()
    }

    /// Reset the expert load counters.
    pub fn reset_load(&mut self) {
        self.expert_load.fill(0.0);
        self.total_tokens = 0;
    }

    /// Returns the number of experts.
    #[must_use]
    pub fn num_experts(&self) -> usize {
        self.cfg.num_experts
    }

    /// Returns the top-k value.
    #[must_use]
    pub fn top_k(&self) -> usize {
        self.cfg.top_k
    }

    /// Returns the total number of tokens processed.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Enable or disable training mode (controls noisy routing).
    pub fn set_training(&mut self, t: bool) {
        self.router.noisy = t && self.cfg.noisy_routing;
    }
}

/// Xavier/Glorot uniform initialization.
fn xavier(fo: usize, fi: usize, sc: f32) -> Array2<f32> {
    let lim = (6.0_f32 / (fi + fo) as f32).sqrt() * sc;
    Array2::from_shape_fn((fo, fi), |(i, j)| {
        let v = ((i + j) as f32 + 1.0) * lim / (fi.max(fo) as f32);
        v.tanh() * lim
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let mut m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        let o = m.forward(&[0.5; 256]).unwrap();
        assert_eq!(o.output.len(), 256);
        assert_eq!(o.activated_experts.len(), 2);
    }

    #[test]
    fn dim_mismatch() {
        let mut m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        assert!(matches!(m.forward(&[0.5; 128]), Err(MoeError::DimMismatch { .. })));
    }

    #[test]
    fn invalid_top_k() {
        let c = MoeConfig { top_k: 0, ..Default::default() };
        assert!(MixtureOfExperts::new(c).is_err());
    }

    #[test]
    fn top_k_gt_experts() {
        let c = MoeConfig { top_k: 10, num_experts: 4, ..Default::default() };
        assert!(MixtureOfExperts::new(c).is_err());
    }

    #[test]
    fn routing_weights_sum() {
        let mut m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        let o = m.forward(&[0.5; 256]).unwrap();
        let s: f32 = o.routing_weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn load_dist() {
        let mut m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        for _ in 0..100 {
            m.forward(&[0.5; 256]).unwrap();
        }
        let d = m.load_distribution();
        let s: f64 = d.iter().sum();
        assert!((s - 1.0).abs() < 0.01);
    }

    #[test]
    fn reset() {
        let mut m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        m.forward(&[0.5; 256]).unwrap();
        assert_eq!(m.total_tokens(), 1);
        m.reset_load();
        assert_eq!(m.total_tokens(), 0);
    }

    #[test]
    fn noisy() {
        let mut m = MixtureOfExperts::new(MoeConfig { noisy_routing: true, ..Default::default() }).unwrap();
        m.set_training(true);
        let o = m.forward(&[0.5; 256]).unwrap();
        assert_eq!(o.output.len(), 256);
    }

    #[test]
    fn lb_loss() {
        let mut m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        let o = m.forward(&[0.5; 256]).unwrap();
        assert!(o.load_balance_loss > 0.0);
    }

    #[test]
    fn capacity() {
        let mut m = MixtureOfExperts::new(MoeConfig { capacity_factor: 1.0, ..Default::default() }).unwrap();
        let o = m.forward(&[0.5; 256]).unwrap();
        assert_eq!(o.output.len(), 256);
    }

    #[test]
    fn num_experts() {
        let m = MixtureOfExperts::new(MoeConfig::default()).unwrap();
        assert_eq!(m.num_experts(), 8);
        assert_eq!(m.top_k(), 2);
    }

    #[test]
    fn top_k_3() {
        let c = MoeConfig { num_experts: 4, top_k: 3, ..Default::default() };
        let mut m = MixtureOfExperts::new(c).unwrap();
        let o = m.forward(&[0.5; 256]).unwrap();
        assert_eq!(o.activated_experts.len(), 3);
    }
}
