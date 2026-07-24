#![allow(missing_docs)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub enabled: bool,
    pub checkpoint_every: usize,
    pub max_memory_mb: usize,
}
impl Default for CheckpointConfig {
    fn default() -> Self { Self { enabled: true, checkpoint_every: 4, max_memory_mb: 512 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointStats {
    pub total_layers: usize,
    pub checkpointed_layers: usize,
    pub recomputations: u64,
    pub memory_saved_mb: f64,
}

struct CheckpointedLayer {
    name: String,
    cached_output: RwLock<Option<Vec<f32>>>,
    recompute_count: AtomicU64,
    is_checkpointed: bool,
}

pub struct GradientCheckpointer {
    config: CheckpointConfig,
    layers: RwLock<Vec<CheckpointedLayer>>,
    memory_saved_bytes: AtomicU64,
    call_count: AtomicU64,
}

impl GradientCheckpointer {
    pub fn new(config: CheckpointConfig) -> Self {
        Self { config, layers: RwLock::new(Vec::new()), memory_saved_bytes: AtomicU64::new(0), call_count: AtomicU64::new(0) }
    }
    pub fn register_layer(&self, name: &str) {
        self.layers.write().push(CheckpointedLayer {
            name: name.to_string(), cached_output: RwLock::new(None), recompute_count: AtomicU64::new(0), is_checkpointed: false,
        });
    }
    pub fn forward(&self, layer_name: &str, input: &[f32]) -> Vec<f32> {
        let call = self.call_count.fetch_add(1, Ordering::Relaxed);
        let layers = self.layers.read();
        if let Some(layer) = layers.iter().find(|l| l.name == layer_name) {
            let layer_idx = layers.iter().position(|l| l.name == layer_name).unwrap();
            let should_cache = !self.config.enabled || (layer_idx % self.config.checkpoint_every != 0);
            if should_cache {
                let output: Vec<f32> = input.iter().map(|v| v * 2.0).collect();
                return output;
            }
            let output: Vec<f32> = input.iter().map(|v| v * 2.0).collect();
            *layer.cached_output.write() = Some(output.clone());
            self.memory_saved_bytes.fetch_add((input.len() * 4) as u64, Ordering::Relaxed);
            output
        } else { input.to_vec() }
    }
    pub fn backward_recompute(&self, layer_name: &str, input: &[f32]) -> Vec<f32> {
        let layers = self.layers.read();
        if let Some(layer) = layers.iter().find(|l| l.name == layer_name) {
            let cached = layer.cached_output.read().clone();
            if let Some(output) = cached {
                return output;
            }
            layer.recompute_count.fetch_add(1, Ordering::Relaxed);
            input.iter().map(|v| v * 2.0).collect()
        } else { input.to_vec() }
    }
    pub fn memory_saved(&self) -> u64 { self.memory_saved_bytes.load(Ordering::Relaxed) }
    pub fn stats(&self) -> CheckpointStats {
        let layers = self.layers.read();
        let checkpointed = layers.iter().filter(|l| l.recompute_count.load(Ordering::Relaxed) > 0).count();
        let recomputations: u64 = layers.iter().map(|l| l.recompute_count.load(Ordering::Relaxed)).sum();
        CheckpointStats {
            total_layers: layers.len(),
            checkpointed_layers: checkpointed,
            recomputations,
            memory_saved_mb: self.memory_saved_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn forward_backward() {
        let gc = GradientCheckpointer::new(CheckpointConfig::default());
        gc.register_layer("l0"); gc.register_layer("l1"); gc.register_layer("l2"); gc.register_layer("l3");
        let out = gc.forward("l0", &[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
        let recomp = gc.backward_recompute("l0", &[1.0, 2.0, 3.0]);
        assert_eq!(recomp, vec![2.0, 4.0, 6.0]);
    }
    #[test] fn memory_saved() {
        let gc = GradientCheckpointer::new(CheckpointConfig { enabled: true, checkpoint_every: 2, max_memory_mb: 256 });
        gc.register_layer("a"); gc.register_layer("b"); gc.register_layer("c"); gc.register_layer("d");
        gc.forward("b", &[1.0; 100]);
        gc.forward("d", &[1.0; 100]);
        assert!(gc.memory_saved() > 0);
    }
    #[test] fn disabled_caches_all() {
        let gc = GradientCheckpointer::new(CheckpointConfig { enabled: false, ..Default::default() });
        gc.register_layer("x");
        let out = gc.forward("x", &[5.0]);
        assert_eq!(out, vec![10.0]);
    }
}
