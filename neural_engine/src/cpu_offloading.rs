#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadConfig {
    pub enabled: bool,
    pub gpu_memory_budget_mb: usize,
    pub access_threshold: u32,
    pub monitoring_window: usize,
}
impl Default for OffloadConfig {
    fn default() -> Self { Self { enabled: true, gpu_memory_budget_mb: 2048, access_threshold: 3, monitoring_window: 100 } }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LayerLocation { Gpu, Cpu, Offloading }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadStats {
    pub gpu_layers: usize,
    pub cpu_layers: usize,
    pub gpu_memory_mb: f64,
    pub cpu_memory_mb: f64,
    pub total_transfers: u64,
    pub offload_savings_mb: f64,
}

struct OffloadableLayer {
    name: String,
    size_bytes: usize,
    location: RwLock<LayerLocation>,
    access_count: AtomicU64,
    last_access_step: AtomicU64,
}

pub struct CpuOffloader {
    config: OffloadConfig,
    layers: RwLock<Vec<OffloadableLayer>>,
    gpu_used_bytes: AtomicU64,
    cpu_used_bytes: AtomicU64,
    total_transfers: AtomicU64,
}

impl CpuOffloader {
    pub fn new(config: OffloadConfig) -> Self {
        Self { config, layers: RwLock::new(Vec::new()), gpu_used_bytes: AtomicU64::new(0), cpu_used_bytes: AtomicU64::new(0), total_transfers: AtomicU64::new(0) }
    }
    pub fn register_layer(&self, name: &str, size_bytes: usize) {
        self.layers.write().push(OffloadableLayer {
            name: name.to_string(), size_bytes, location: RwLock::new(LayerLocation::Gpu),
            access_count: AtomicU64::new(0), last_access_step: AtomicU64::new(0),
        });
        self.gpu_used_bytes.fetch_add(size_bytes as u64, Ordering::Relaxed);
    }
    pub fn access(&self, name: &str, step: u64) {
        let layers = self.layers.read();
        if let Some(layer) = layers.iter().find(|l| l.name == name) {
            layer.access_count.fetch_add(1, Ordering::Relaxed);
            layer.last_access_step.store(step, Ordering::Relaxed);
            let mut loc = layer.location.write();
            if *loc == LayerLocation::Cpu {
                *loc = LayerLocation::Gpu;
                self.gpu_used_bytes.fetch_add(layer.size_bytes as u64, Ordering::Relaxed);
                self.cpu_used_bytes.fetch_sub(layer.size_bytes as u64, Ordering::Relaxed);
                self.total_transfers.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    pub fn maybe_offload(&self, current_step: u64) {
        let budget_bytes = (self.config.gpu_memory_budget_mb * 1024 * 1024) as u64;
        let gpu_used = self.gpu_used_bytes.load(Ordering::Relaxed);
        if gpu_used <= budget_bytes { return; }
        let layers = self.layers.read();
        let mut candidates: Vec<(usize, u64, u64)> = layers.iter().enumerate()
            .filter(|(_, l) { *l.location.read() == LayerLocation::Gpu })
            .map(|(i, l)| (i, l.access_count.load(Ordering::Relaxed), l.size_bytes as u64))
            .collect();
        candidates.sort_by_key(|(_, access, _)| *access);
        drop(layers);
        let mut layers = self.layers.write();
        for (idx, access, size) in candidates {
            if access < self.config.access_threshold as u64 {
                if let Some(layer) = layers.get_mut(idx) {
                    *layer.location.write() = LayerLocation::Cpu;
                    self.gpu_used_bytes.fetch_sub(size, Ordering::Relaxed);
                    self.cpu_used_bytes.fetch_add(size, Ordering::Relaxed);
                    self.total_transfers.fetch_add(1, Ordering::Relaxed);
                    if self.gpu_used_bytes.load(Ordering::Relaxed) <= budget_bytes { break; }
                }
            }
        }
    }
    pub fn stats(&self) -> OffloadStats {
        let layers = self.layers.read();
        let gpu = layers.iter().filter(|l| *l.location.read() == LayerLocation::Gpu).count();
        let cpu = layers.iter().filter(|l| *l.location.read() == LayerLocation::Cpu).count();
        OffloadStats {
            gpu_layers: gpu, cpu_layers: cpu,
            gpu_memory_mb: self.gpu_used_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            cpu_memory_mb: self.cpu_used_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            total_transfers: self.total_transfers.load(Ordering::Relaxed),
            offload_savings_mb: self.cpu_used_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn register_and_access() {
        let co = CpuOffloader::new(OffloadConfig::default());
        co.register_layer("l0", 1024 * 1024);
        co.access("l0", 1);
        let s = co.stats();
        assert_eq!(s.gpu_layers, 1);
    }
    #[test] fn offload_when_budget_exceeded() {
        let co = CpuOffloader::new(OffloadConfig { gpu_memory_budget_mb: 2, access_threshold: 5, ..Default::default() });
        co.register_layer("hot", 1024 * 1024);
        co.register_layer("cold", 1024 * 1024);
        co.access("hot", 1); co.access("hot", 2); co.access("hot", 3);
        co.maybe_offload(10);
        let s = co.stats();
        assert!(s.total_transfers > 0, "should have offloaded cold layer");
    }
    #[test] fn access_brings_back_to_gpu() {
        let co = CpuOffloader::new(OffloadConfig::default());
        co.register_layer("l", 1024);
        co.maybe_offload(1);
        co.access("l", 5);
        let s = co.stats();
        assert_eq!(s.gpu_layers, 1);
    }
}
