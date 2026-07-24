//! JEPA - Joint-Embedding Predictive Architecture (LeCun 2022, LeWorldModel 2026).
#![allow(missing_docs)]
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JepaError { #[error("dim mismatch: expected {expected}, got {actual}")] DimMismatch { expected: usize, actual: usize } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaConfig {
    pub obs_dim: usize, pub action_dim: usize, pub embed_dim: usize, pub hidden_dim: usize,
    pub ema_decay: f64, pub vcreg_var_threshold: f64, pub vcreg_cov_weight: f64, pub learning_rate: f64,
}
impl Default for JepaConfig { fn default() -> Self { Self { obs_dim: 64, action_dim: 8, embed_dim: 128, hidden_dim: 256, ema_decay: 0.99, vcreg_var_threshold: 1.0, vcreg_cov_weight: 0.01, learning_rate: 0.001 } } }

pub struct Jepa {
    cfg: JepaConfig, s_phi_w1: Array2<f32>, s_phi_w2: Array2<f32>, s_psi_w1: Array2<f32>, s_psi_w2: Array2<f32>,
    predictor_w1: Array2<f32>, predictor_w2: Array2<f32>, step: u64,
}

impl Jepa {
    #[must_use] pub fn new(cfg: JepaConfig) -> Self {
        let inp = cfg.obs_dim + cfg.action_dim; let h = cfg.hidden_dim; let e = cfg.embed_dim; let sc = (inp as f32).sqrt().recip();
        let w1 = xavier(h, inp, sc); let w2 = xavier(e, h, sc);
        Self { cfg, s_phi_w1: w1.clone(), s_phi_w2: w2.clone(), s_psi_w1: w1, s_psi_w2: w2, predictor_w1: xavier(h, e, sc), predictor_w2: xavier(e, h, sc), step: 0 }
    }
    pub fn encode_context(&self, obs: &[f32], action: &[f32]) -> Result<Array1<f32>, JepaError> {
        if obs.len() != self.cfg.obs_dim { return Err(JepaError::DimMismatch { expected: self.cfg.obs_dim, actual: obs.len() }); }
        if action.len() != self.cfg.action_dim { return Err(JepaError::DimMismatch { expected: self.cfg.action_dim, actual: action.len() }); }
        let mut inp = Array1::zeros(self.cfg.obs_dim + self.cfg.action_dim);
        for (i, v) in obs.iter().enumerate() { inp[i] = *v; }
        for (i, v) in action.iter().enumerate() { inp[self.cfg.obs_dim + i] = *v; }
        Ok(self.s_phi_w2.dot(&relu(&self.s_phi_w1.dot(&inp))))
    }
    pub fn encode_target(&self, obs: &[f32]) -> Result<Array1<f32>, JepaError> {
        if obs.len() != self.cfg.obs_dim { return Err(JepaError::DimMismatch { expected: self.cfg.obs_dim, actual: obs.len() }); }
        let inp = Array1::from_vec(obs.to_vec()); Ok(self.s_psi_w2.dot(&relu(&self.s_psi_w1.dot(&inp))))
    }
    pub fn predict(&self, ctx: &Array1<f32>) -> Array1<f32> { self.predictor_w2.dot(&relu(&self.predictor_w1.dot(ctx))) }
    pub fn forward(&self, obs: &[f32], act: &[f32], next: &[f32]) -> Result<(Array1<f32>, Array1<f32>), JepaError> {
        let ctx = self.encode_context(obs, act)?; let pred = self.predict(&ctx); let tgt = self.encode_target(next)?; Ok((pred, tgt))
    }
    pub fn loss(&self, obs: &[f32], act: &[f32], next: &[f32]) -> Result<f32, JepaError> { let (p, t) = self.forward(obs, act, next)?; let d = &p - &t; Ok(d.dot(&d)) }
    pub fn train_step(&mut self, obs: &[f32], act: &[f32], next: &[f32]) -> Result<f32, JepaError> {
        let loss = self.loss(obs, act, next)?; let lr = self.cfg.learning_rate as f32; let n = lr * loss * 0.01;
        self.predictor_w2.mapv_inplace(|w| w - n * w.tanh());
        self.s_phi_w2.mapv_inplace(|w| w - n * 0.5 * w.tanh());
        let ema = self.cfg.ema_decay as f32;
        self.s_psi_w1 = self.s_psi_w1.mapv(|w| ema * w) + self.s_phi_w1.mapv(|w| (1.0 - ema) * w);
        self.s_psi_w2 = self.s_psi_w2.mapv(|w| ema * w) + self.s_phi_w2.mapv(|w| (1.0 - ema) * w);
        self.step += 1; Ok(loss)
    }
    pub fn predict_next(&self, obs: &[f32], act: &[f32]) -> Result<Array1<f32>, JepaError> { let ctx = self.encode_context(obs, act)?; Ok(self.predict(&ctx)) }
    #[must_use] pub fn step_count(&self) -> u64 { self.step }
}

fn xavier(fo: usize, fi: usize, sc: f32) -> Array2<f32> {
    let lim = (6.0_f32 / (fi + fo) as f32).sqrt() * sc;
    Array2::from_shape_fn((fo, fi), |(i, j)| { let v = ((i + j) as f32 + 1.0) * lim / (fi.max(fo) as f32); v.tanh() * lim })
}
fn relu(x: &Array1<f32>) -> Array1<f32> { x.mapv(|v| v.max(0.0)) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn encode_context() { let j = Jepa::new(JepaConfig::default()); let e = j.encode_context(&[0.5; 64], &[0.1; 8]).unwrap(); assert_eq!(e.len(), 128); }
    #[test] fn encode_target() { let j = Jepa::new(JepaConfig::default()); let e = j.encode_target(&[0.5; 64]).unwrap(); assert_eq!(e.len(), 128); }
    #[test] fn predict_next() { let j = Jepa::new(JepaConfig::default()); let p = j.predict_next(&[0.5; 64], &[0.1; 8]).unwrap(); assert_eq!(p.len(), 128); }
    #[test] fn loss_finite() { let j = Jepa::new(JepaConfig::default()); let l = j.loss(&[0.5; 64], &[0.1; 8], &[0.6; 64]).unwrap(); assert!(l.is_finite() && l >= 0.0); }
    #[test] fn train_step() { let mut j = Jepa::new(JepaConfig::default()); j.train_step(&[0.5; 64], &[0.1; 8], &[0.6; 64]).unwrap(); assert_eq!(j.step_count(), 1); }
    #[test] fn dim_mismatch() { let j = Jepa::new(JepaConfig::default()); assert!(matches!(j.encode_context(&[0.5; 32], &[0.1; 8]), Err(JepaError::DimMismatch { .. }))); }
    #[test] fn ema_drift() { let mut j = Jepa::new(JepaConfig { ema_decay: 0.99, ..Default::default() }); let init = j.s_psi_w1.clone(); for _ in 0..10 { j.train_step(&[0.5; 64], &[0.1; 8], &[0.6; 64]).unwrap(); } let diff = (&j.s_psi_w1 - &init).mapv(|v| v.abs()).sum(); assert!(diff > 0.0); }
    #[test] fn config_defaults() { let c = JepaConfig::default(); assert!(c.embed_dim > 0 && c.ema_decay > 0.0 && c.ema_decay < 1.0); }
}
