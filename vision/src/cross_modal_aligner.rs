#![allow(missing_docs)]

//! Cross-modal alignment (CLIP-style): projects vision and text embeddings
//! into a shared latent space and computes contrastive similarity scores.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors emitted by the cross-modal aligner.
#[derive(Debug, Error)]
pub enum AlignError {
    /// Dimension mismatch.
    #[error("dimension mismatch: got {vision_dim}, expected {expected_dim}")]
    DimMismatch { vision_dim: usize, expected_dim: usize },
    /// Empty batch.
    #[error("empty batch")]
    EmptyBatch,
}

/// Configuration for the aligner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignerConfig {
    /// Temperature scaling for contrastive softmax.
    pub temperature: f32,
    /// Whether to L2-normalize embeddings after projection.
    pub normalize_inputs: bool,
}

impl Default for AlignerConfig {
    fn default() -> Self { Self { temperature: 0.07, normalize_inputs: true } }
}

/// Alignment score for a single vision-text pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentScore {
    /// Cosine similarity in [-1, 1].
    pub cosine: f32,
    /// Softmax-normalized score (when computed against a batch).
    pub softmax: f32,
}

/// CLIP-style cross-modal aligner with two linear projectors.
pub struct CrossModalAligner {
    config: AlignerConfig,
    vision_proj: Array2<f32>,
    text_proj: Array2<f32>,
    vision_bias: Array1<f32>,
    text_bias: Array1<f32>,
}

impl CrossModalAligner {
    /// Create a new aligner with deterministic-random projectors.
    #[must_use]
    pub fn new(vision_dim: usize, text_dim: usize, latent_dim: usize) -> Self {
        Self::with_config(vision_dim, text_dim, latent_dim, AlignerConfig::default())
    }

    /// Create a new aligner with explicit config.
    #[must_use]
    pub fn with_config(vd: usize, td: usize, ld: usize, config: AlignerConfig) -> Self {
        let sv = (2.0 / (vd + ld) as f32).sqrt();
        let st = (2.0 / (td + ld) as f32).sqrt();
        Self {
            config,
            vision_proj: random_matrix(ld, vd, sv),
            text_proj: random_matrix(ld, td, st),
            vision_bias: Array1::zeros(ld),
            text_bias: Array1::zeros(ld),
        }
    }

    /// Project a vision embedding into the shared latent space.
    pub fn project_vision<S: Data<Elem = f32>>(&self, emb: &ArrayBase<S, Ix1>) -> Array1<f32> {
        let mut out = self.vision_proj.dot(emb) + &self.vision_bias;
        if self.config.normalize_inputs { normalize(&mut out); }
        out
    }

    /// Project a text embedding into the shared latent space.
    pub fn project_text<S: Data<Elem = f32>>(&self, emb: &ArrayBase<S, Ix1>) -> Array1<f32> {
        let mut out = self.text_proj.dot(emb) + &self.text_bias;
        if self.config.normalize_inputs { normalize(&mut out); }
        out
    }

    /// Align a single vision-text pair.
    pub fn align(&self, vision_emb: &Array1<f32>, text_emb: &Array1<f32>) -> Result<AlignmentScore, AlignError> {
        if vision_emb.len() != self.vision_proj.ncols() {
            return Err(AlignError::DimMismatch { vision_dim: vision_emb.len(), expected_dim: self.vision_proj.ncols() });
        }
        if text_emb.len() != self.text_proj.ncols() {
            return Err(AlignError::DimMismatch { vision_dim: text_emb.len(), expected_dim: self.text_proj.ncols() });
        }
        let v = self.project_vision(vision_emb);
        let t = self.project_text(text_emb);
        Ok(AlignmentScore { cosine: dot(&v, &t).clamp(-1.0, 1.0), softmax: 0.0 })
    }

    /// Batch align: computes pairwise similarities and softmax scores along the
    /// diagonal (treats batch as a contrastive learning batch).
    pub fn batch_align(&self, vision_batch: &[Array1<f32>], text_batch: &[Array1<f32>]) -> Result<Vec<AlignmentScore>, AlignError> {
        if vision_batch.is_empty() || text_batch.is_empty() { return Err(AlignError::EmptyBatch); }
        let v_proj: Vec<Array1<f32>> = vision_batch.iter().map(|v| self.project_vision(v)).collect();
        let t_proj: Vec<Array1<f32>> = text_batch.iter().map(|t| self.project_text(t)).collect();
        let n = v_proj.len().min(t_proj.len());
        let temp = self.config.temperature.max(1e-6);
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let sims: Vec<f32> = (0..n).map(|j| (dot(&v_proj[i], &t_proj[j]) / temp).exp()).collect();
            let sum: f32 = sims.iter().sum();
            let softmax = if sum > 0.0 { sims[i] / sum } else { 1.0 / n as f32 };
            let cosine = dot(&v_proj[i], &t_proj[i]).clamp(-1.0, 1.0);
            scores.push(AlignmentScore { cosine, softmax });
        }
        Ok(scores)
    }
}

fn normalize(v: &mut Array1<f32>) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 { v.mapv_inplace(|x| x / norm); }
}

fn dot(a: &Array1<f32>, b: &Array1<f32>) -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() }

fn random_matrix(rows: usize, cols: usize, scale: f32) -> Array2<f32> {
    // Deterministic init via LCG (good enough for tests; production would
    // load trained weights).
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32) / (1u64 << 31) as f32 * 2.0 - 1.0
    };
    Array2::from_shape_fn((rows, cols), |_| next() * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_into_latent_dim() {
        let a = CrossModalAligner::new(64, 32, 16);
        let v = Array1::zeros(64);
        let t = Array1::zeros(32);
        assert_eq!(a.project_vision(&v).len(), 16);
        assert_eq!(a.project_text(&t).len(), 16);
    }

    #[test]
    fn align_returns_valid_score() {
        let a = CrossModalAligner::new(8, 8, 4);
        let v = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let t = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let s = a.align(&v, &t).unwrap();
        assert!(s.cosine >= -1.0 && s.cosine <= 1.0);
    }

    #[test]
    fn align_dim_mismatch_errors() {
        let a = CrossModalAligner::new(8, 8, 4);
        let bad = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let good = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(a.align(&bad, &good).is_err());
    }

    #[test]
    fn batch_align_produces_softmax() {
        let a = CrossModalAligner::new(4, 4, 4);
        let vb: Vec<Array1<f32>> = (0..3).map(|i| {
            let mut v = vec![0.0; 4]; v[i] = 1.0; Array1::from_vec(v)
        }).collect();
        let scores = a.batch_align(&vb, &vb.clone()).unwrap();
        assert_eq!(scores.len(), 3);
        for s in &scores { assert!(s.softmax > 0.0 && s.softmax <= 1.0); }
    }
}
