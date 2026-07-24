#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdafactorConfig {
    pub lr: f32, pub beta1: f32, pub beta2: f32, pub epsilon1: f32, pub epsilon2: f32,
    pub clipping_threshold: f32, pub decay_rate: f32, pub weight_decay: f32,
}
impl Default for AdafactorConfig {
    fn default() -> Self { Self { lr: 1e-3, beta1: 0.9, beta2: 0.999, epsilon1: 1e-30, epsilon2: 1e-3, clipping_threshold: 1.0, decay_rate: -0.8, weight_decay: 0.0 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStats {
    pub optimizer_name: String,
    pub num_params: usize,
    pub state_memory_bytes: usize,
    pub equivalent_adam_memory_bytes: usize,
    pub compression_ratio: f32,
}

struct AdafactorState {
    step: u64,
    exp_avg: Vec<f32>,
    exp_avg_row: Option<Vec<f32>>,
    exp_avg_col: Option<Vec<f32>>,
}

pub struct Adafactor {
    config: AdafactorConfig,
    states: RwLock<AHashMap<String, AdafactorState>>,
}

impl Adafactor {
    pub fn new(config: AdafactorConfig) -> Self { Self { config, states: RwLock::new(AHashMap::new()) } }
    pub fn step(&self, name: &str, param: &mut [f32], grad: &[f32], rows: usize, cols: usize) {
        let mut states = self.states.write();
        let state = states.entry(name.to_string()).or_insert_with(|| AdafactorState {
            step: 0,
            exp_avg: vec![0.0; param.len()],
            exp_avg_row: if rows > 1 && cols > 1 { Some(vec![0.0; rows]) } else { None },
            exp_avg_col: if rows > 1 && cols > 1 { Some(vec![0.0; cols]) } else { None },
        });
        state.step += 1;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let bias_c1 = 1.0 - beta1.powi(state.step as i32);
        let bias_c2 = 1.0 - beta2.powi(state.step as i32);
        for (i, g) in grad.iter().enumerate() {
            state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;
        }
        if let (Some(row_stats), Some(col_stats)) = (&mut state.exp_avg_row, &mut state.exp_avg_col) {
            let rho_t = 1.0 - state.step as f32 / (state.step as f32 + 1.0);
            for r in 0..rows {
                let mut sum_sq = 0.0f32;
                for c in 0..cols { sum_sq += grad[r * cols + c].powi(2); }
                row_stats[r] = beta2 * row_stats[r] + (1.0 - beta2) * (sum_sq / cols as f32).max(self.config.epsilon1);
            }
            for c in 0..cols {
                let mut sum_sq = 0.0f32;
                for r in 0..rows { sum_sq += grad[r * cols + c].powi(2); }
                col_stats[c] = beta2 * col_stats[c] + (1.0 - beta2) * (sum_sq / rows as f32).max(self.config.epsilon1);
            }
            for i in 0..param.len() {
                let r = i / cols;
                let c = i % cols;
                let v_hat = (row_stats[r] * col_stats[c]).max(self.config.epsilon2);
                let m_hat = state.exp_avg[i] / bias_c1;
                let update = m_hat / (v_hat.sqrt() / bias_c2.sqrt());
                let clipped = update.min(self.config.clipping_threshold).max(-self.config.clipping_threshold);
                param[i] -= self.config.lr * (clipped + self.config.weight_decay * param[i]);
            }
        } else {
            for i in 0..param.len() {
                let m_hat = state.exp_avg[i] / bias_c1;
                param[i] -= self.config.lr * m_hat;
            }
        }
    }
    pub fn memory_usage(&self) -> usize {
        let states = self.states.read();
        let mut bytes = 0;
        for s in states.values() {
            bytes += s.exp_avg.len() * 4;
            if let Some(r) = &s.exp_avg_row { bytes += r.len() * 4; }
            if let Some(c) = &s.exp_avg_col { bytes += c.len() * 4; }
        }
        bytes
    }
    pub fn stats(&self) -> OptimizerStats {
        let states = self.states.read();
        let n = states.len();
        let mem = self.memory_usage();
        let adam_equiv: usize = states.values().map(|s| s.exp_avg.len() * 8).sum();
        OptimizerStats {
            optimizer_name: "Adafactor".into(),
            num_params: n,
            state_memory_bytes: mem,
            equivalent_adam_memory_bytes: adam_equiv,
            compression_ratio: if adam_equiv > 0 { adam_equiv as f32 / mem as f32 } else { 1.0 },
        }
    }
}

struct EightBitState { step: u64, m_quantized: Vec<u8>, v_quantized: Vec<u8>, m_scale: f32, v_scale: f32 }

pub struct EightBitAdam {
    lr: f32, beta1: f32, beta2: f32,
    states: RwLock<AHashMap<String, EightBitState>>,
}

impl EightBitAdam {
    pub fn new(lr: f32, beta1: f32, beta2: f32) -> Self { Self { lr, beta1, beta2, states: RwLock::new(AHashMap::new()) } }
    pub fn step(&self, name: &str, param: &mut [f32], grad: &[f32]) {
        let mut states = self.states.write();
        let n = param.len();
        let state = states.entry(name.to_string()).or_insert_with(|| EightBitState {
            step: 0, m_quantized: vec![0u8; n], v_quantized: vec![0u8; n], m_scale: 1.0, v_scale: 1.0,
        });
        state.step += 1;
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        for i in 0..n {
            m[i] = (state.m_quantized[i] as f32 - 128.0) * state.m_scale;
            v[i] = (state.v_quantized[i] as f32 - 128.0) * state.v_scale;
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
        }
        let m_max = m.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
        let v_max = v.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
        state.m_scale = m_max / 127.0;
        state.v_scale = v_max / 127.0;
        for i in 0..n {
            state.m_quantized[i] = ((m[i] / state.m_scale).round().clamp(-127.0, 127.0) + 128.0) as u8;
            state.v_quantized[i] = ((v[i] / state.v_scale).round().clamp(-127.0, 127.0) + 128.0) as u8;
        }
        let bias_c1 = 1.0 - self.beta1.powi(state.step as i32);
        let bias_c2 = 1.0 - self.beta2.powi(state.step as i32);
        for i in 0..n {
            let m_hat = m[i] / bias_c1;
            let v_hat = v[i] / bias_c2;
            param[i] -= self.lr * m_hat / (v_hat.sqrt() + 1e-8);
        }
    }
    pub fn memory_usage(&self) -> usize {
        let states = self.states.read();
        states.values().map(|s| s.m_quantized.len() * 2).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn adafactor_step() {
        let opt = Adafactor::new(AdafactorConfig::default());
        let mut param = vec![1.0, 2.0, 3.0, 4.0];
        let grad = vec![0.1, 0.2, 0.3, 0.4];
        opt.step("w", &mut param, &grad, 2, 2);
        assert!(param[0] < 1.0, "param should decrease");
    }
    #[test] fn adafactor_memory() {
        let opt = Adafactor::new(AdafactorConfig::default());
        let mut p = vec![1.0; 100];
        let g = vec![0.1; 100];
        opt.step("w", &mut p, &g, 10, 10);
        let mem = opt.memory_usage();
        assert!(mem > 0);
        let s = opt.stats();
        assert!(s.compression_ratio > 1.0, "Adafactor should use less memory than Adam");
    }
    #[test] fn eightbit_adam_step() {
        let opt = EightBitAdam::new(1e-3, 0.9, 0.999);
        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2, 0.3];
        opt.step("w", &mut param, &grad);
        assert!(param[0] < 1.0);
    }
    #[test] fn eightbit_memory() {
        let opt = EightBitAdam::new(1e-3, 0.9, 0.999);
        let mut p = vec![1.0; 1000];
        let g = vec![0.1; 1000];
        opt.step("w", &mut p, &g);
        let mem = opt.memory_usage();
        assert_eq!(mem, 2000); // 2 bytes per param (m + v, each 1 byte)
    }
}
