#![allow(missing_docs)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    pub enabled: bool,
    pub initial_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
    pub min_scale: f32,
}
impl Default for PrecisionConfig {
    fn default() -> Self { Self { enabled: true, initial_scale: 65536.0, growth_factor: 2.0, backoff_factor: 0.5, growth_interval: 2000, min_scale: 1.0 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionStats {
    pub current_scale: f32,
    pub overflow_count: u64,
    pub steps_since_growth: usize,
    pub mean_gradient_magnitude: f32,
}

pub struct LossScaler {
    config: PrecisionConfig,
    current_scale: RwLock<f32>,
    steps_since_growth: RwLock<usize>,
    overflow_count: AtomicU64,
    grad_magnitude_sum: RwLock<f64>,
    grad_magnitude_count: RwLock<u64>,
}

impl LossScaler {
    pub fn new(config: PrecisionConfig) -> Self {
        let scale = config.initial_scale;
        Self { config, current_scale: RwLock::new(scale), steps_since_growth: RwLock::new(0), overflow_count: AtomicU64::new(0), grad_magnitude_sum: RwLock::new(0.0), grad_magnitude_count: RwLock::new(0) }
    }
    pub fn scale(&self, loss: f32) -> f32 { loss * *self.current_scale.read() }
    pub fn unscale(&self, gradients: &mut [f32]) {
        let scale = *self.current_scale.read();
        let inv = 1.0 / scale;
        for g in gradients.iter_mut() { *g *= inv; }
    }
    pub fn check_overflow(&self, gradients: &[f32]) -> bool {
        for g in gradients {
            if !g.is_finite() { return true; }
        }
        let mag: f64 = gradients.iter().map(|g| (*g as f64).abs()).sum::<f64>() / gradients.len().max(1) as f64;
        *self.grad_magnitude_sum.write() += mag;
        *self.grad_magnitude_count.write() += 1;
        false
    }
    pub fn update_scale(&self, overflow: bool) {
        if overflow {
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
            let mut scale = self.current_scale.write();
            *scale = (*scale * self.config.backoff_factor).max(self.config.min_scale);
            *self.steps_since_growth.write() = 0;
        } else {
            let mut steps = self.steps_since_growth.write();
            *steps += 1;
            if *steps >= self.config.growth_interval {
                let mut scale = self.current_scale.write();
                *scale *= self.config.growth_factor;
                *steps = 0;
            }
        }
    }
    pub fn current_scale(&self) -> f32 { *self.current_scale.read() }
    pub fn stats(&self) -> MixedPrecisionStats {
        let count = *self.grad_magnitude_count.read();
        let mean = if count == 0 { 0.0 } else { (*self.grad_magnitude_sum.read() / count as f64) as f32 };
        MixedPrecisionStats {
            current_scale: *self.current_scale.read(),
            overflow_count: self.overflow_count.load(Ordering::Relaxed),
            steps_since_growth: *self.steps_since_growth.read(),
            mean_gradient_magnitude: mean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn scale_unscale_roundtrip() {
        let ls = LossScaler::new(PrecisionConfig::default());
        let loss = 0.5;
        let scaled = ls.scale(loss);
        assert_eq!(scaled, 0.5 * 65536.0);
        let mut grads = vec![scaled; 4];
        ls.unscale(&mut grads);
        for g in &grads { assert!((g - 0.5).abs() < 1e-3); }
    }
    #[test] fn overflow_detection() {
        let ls = LossScaler::new(PrecisionConfig::default());
        assert!(!ls.check_overflow(&[1.0, 2.0, 3.0]));
        assert!(ls.check_overflow(&[1.0, f32::INFINITY, 3.0]));
        assert!(ls.check_overflow(&[1.0, f32::NAN, 3.0]));
    }
    #[test] fn scale_backoff_on_overflow() {
        let ls = LossScaler::new(PrecisionConfig::default());
        let initial = ls.current_scale();
        ls.update_scale(true);
        assert!(ls.current_scale() < initial);
    }
    #[test] fn scale_growth_on_success() {
        let ls = LossScaler::new(PrecisionConfig { growth_interval: 3, ..Default::default() });
        let initial = ls.current_scale();
        ls.update_scale(false); ls.update_scale(false); ls.update_scale(false);
        assert!(ls.current_scale() > initial);
    }
}
