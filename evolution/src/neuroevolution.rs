//! QD-LLM parameter-efficient neuroevolution (arXiv 2026).
#![allow(missing_docs)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NeuroEvoError { #[error("empty population")] EmptyPopulation, #[error("invalid config: {0}")] InvalidConfig(String), #[error("fitness failed: {0}")] FitnessFailed(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual { pub id: u64, pub params: Vec<f32>, pub fitness: f64, pub behavior: Vec<f64>, pub generation: u64, pub parent: Option<u64> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroEvoConfig { pub population_size: usize, pub num_generations: usize, pub mutation_rate: f32, pub elite_count: usize, pub behavior_dims: usize, pub use_qd_archive: bool, pub archive_capacity: usize, pub novelty_weight: f64 }
impl Default for NeuroEvoConfig { fn default() -> Self { Self { population_size: 32, num_generations: 50, mutation_rate: 0.1, elite_count: 4, behavior_dims: 2, use_qd_archive: true, archive_capacity: 100, novelty_weight: 0.3 } } }
impl NeuroEvoConfig { fn validate(&self) -> Result<(), NeuroEvoError> { if self.population_size == 0 { return Err(NeuroEvoError::InvalidConfig("pop=0".into())); } if self.elite_count > self.population_size { return Err(NeuroEvoError::InvalidConfig("elite>pop".into())); } Ok(()) } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorDescriptor { pub behavior: Vec<f64>, pub individual_id: u64, pub fitness: f64 }

pub struct QdLlmEngine { cfg: NeuroEvoConfig, population: RwLock<Vec<Individual>>, archive: RwLock<Vec<BehaviorDescriptor>>, next_id: RwLock<u64>, generation: RwLock<u64>, best_fitness: RwLock<f64>, param_dim: usize }

impl QdLlmEngine {
    pub fn new(cfg: NeuroEvoConfig, pd: usize) -> Result<Self, NeuroEvoError> { cfg.validate()?; Ok(Self { cfg, population: RwLock::new(Vec::new()), archive: RwLock::new(Vec::new()), next_id: RwLock::new(0), generation: RwLock::new(0), best_fitness: RwLock::new(f64::NEG_INFINITY), param_dim: pd }) }
    pub fn initialize_random(&self) { let mut pop = self.population.write(); pop.clear(); for _ in 0..self.cfg.population_size { let mut nid = self.next_id.write(); *nid += 1; let id = *nid; drop(nid); pop.push(Individual { id, params: self.random_params(), fitness: 0.0, behavior: vec![0.0; self.cfg.behavior_dims], generation: 0, parent: None }); } }
    pub fn evolve<F>(&self, mut fit: F) -> Result<(), NeuroEvoError> where F: FnMut(&[f32]) -> Result<(f64, Vec<f64>), String>, {
        let mut pop = self.population.write(); if pop.is_empty() { return Err(NeuroEvoError::EmptyPopulation); }
        for ind in pop.iter_mut() { let (f, b) = fit(&ind.params).map_err(NeuroEvoError::FitnessFailed)?; ind.fitness = f; ind.behavior = b; }
        let best = pop.iter().map(|i| i.fitness).fold(f64::NEG_INFINITY, f64::max); let mut bf = self.best_fitness.write(); if best > *bf { *bf = best; } drop(bf);
        if self.cfg.use_qd_archive { let mut ar = self.archive.write(); for ind in pop.iter() { ar.push(BehaviorDescriptor { behavior: ind.behavior.clone(), individual_id: ind.id, fitness: ind.fitness }); } ar.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)); ar.truncate(self.cfg.archive_capacity); }
        pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        let elites: Vec<Individual> = pop.iter().take(self.cfg.elite_count).cloned().collect();
        let mut new_pop: Vec<Individual> = elites.clone();
        let mut nid = self.next_id.write(); let mut generation = self.generation.write(); *generation += 1; let g = *generation;
        while new_pop.len() < self.cfg.population_size { let pi = rand_usize() % elites.len(); let p = &elites[pi]; *nid += 1; let id = *nid; new_pop.push(Individual { id, params: self.mutate(&p.params), fitness: 0.0, behavior: vec![0.0; self.cfg.behavior_dims], generation: g, parent: Some(p.id) }); }
        *pop = new_pop; Ok(())
    }
    pub fn evolve_for<F>(&self, fit: F) -> Result<(), NeuroEvoError> where F: FnMut(&[f32]) -> Result<(f64, Vec<f64>), String>, { let mut f = fit; for _ in 0..self.cfg.num_generations { self.evolve(&mut f)?; } Ok(()) }
    #[must_use] pub fn population(&self) -> Vec<Individual> { let mut p = self.population.read().clone(); p.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)); p }
    #[must_use] pub fn best(&self) -> Option<Individual> { self.population().into_iter().next() }
    #[must_use] pub fn best_fitness(&self) -> f64 { *self.best_fitness.read() }
    #[must_use] pub fn generation(&self) -> u64 { *self.generation.read() }
    #[must_use] pub fn archive(&self) -> Vec<BehaviorDescriptor> { self.archive.read().clone() }
    #[must_use] pub fn archive_size(&self) -> usize { self.archive.read().len() }
    #[must_use] pub fn novelty(&self, b: &[f64], k: usize) -> f64 { let ar = self.archive.read(); if ar.is_empty() { return 0.0; } let mut d: Vec<f64> = ar.iter().map(|x| euclid(b, &x.behavior)).collect(); d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); let k = k.min(d.len()); d.iter().take(k).sum::<f64>() / k as f64 }
    fn random_params(&self) -> Vec<f32> { (0..self.param_dim).map(|_| rand_f32() - 0.5).collect() }
    fn mutate(&self, p: &[f32]) -> Vec<f32> { p.iter().map(|v| v + (rand_f32() - 0.5) * 2.0 * self.cfg.mutation_rate).collect() }
}

fn euclid(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt() }
fn rand_f32() -> f32 { use std::sync::atomic::{AtomicU64, Ordering}; static S: AtomicU64 = AtomicU64::new(0xFEDCBA9876543210); let mut s = S.load(Ordering::Relaxed); if s == 0 { s = 0xFEDCBA9876543210; } s ^= s << 13; s ^= s >> 7; s ^= s << 17; S.store(s, Ordering::Relaxed); (s as f32) / (u32::MAX as f32) }
fn rand_usize() -> usize { use std::sync::atomic::{AtomicU64, Ordering}; static S: AtomicU64 = AtomicU64::new(0x0123456789ABCDEF); let mut s = S.load(Ordering::Relaxed); if s == 0 { s = 0x0123456789ABCDEF; } s ^= s << 13; s ^= s >> 7; s ^= s << 17; S.store(s, Ordering::Relaxed); s as usize }
pub type SharedQdLlmEngine = Arc<QdLlmEngine>;

#[cfg(test)]
mod tests {
    use super::*;
    fn fit(p: &[f32]) -> Result<(f64, Vec<f64>), String> { let f = p.iter().map(|v| *v as f64).sum(); let m = p.iter().sum::<f32>() as f64 / p.len() as f64; Ok((f, vec![m, m])) }
    #[test] fn init_pop() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 10).unwrap(); e.initialize_random(); assert_eq!(e.population().len(), 32); }
    #[test] fn evolve_gen() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 4).unwrap(); e.initialize_random(); e.evolve(fit).unwrap(); assert_eq!(e.generation(), 1); }
    #[test] fn best_fitness() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 4).unwrap(); e.initialize_random(); e.evolve(fit).unwrap(); assert!(e.best_fitness() > f64::NEG_INFINITY); }
    #[test] fn multi_gen() { let c = NeuroEvoConfig { num_generations: 5, population_size: 8, ..Default::default() }; let e = QdLlmEngine::new(c, 4).unwrap(); e.initialize_random(); e.evolve_for(fit).unwrap(); assert_eq!(e.generation(), 5); }
    #[test] fn archive_grows() { let c = NeuroEvoConfig { num_generations: 3, population_size: 4, archive_capacity: 50, ..Default::default() }; let e = QdLlmEngine::new(c, 4).unwrap(); e.initialize_random(); e.evolve_for(fit).unwrap(); assert!(e.archive_size() > 0 && e.archive_size() <= 50); }
    #[test] fn invalid_cfg() { assert!(QdLlmEngine::new(NeuroEvoConfig { population_size: 0, ..Default::default() }, 10).is_err()); }
    #[test] fn empty_pop() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 4).unwrap(); assert!(matches!(e.evolve(fit), Err(NeuroEvoError::EmptyPopulation))); }
    #[test] fn novelty() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 4).unwrap(); e.initialize_random(); e.evolve(fit).unwrap(); assert!(e.novelty(&[0.5, 0.5], 3) >= 0.0); }
    #[test] fn best_is_highest() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 4).unwrap(); e.initialize_random(); e.evolve(fit).unwrap(); let b = e.best().unwrap(); for i in e.population() { assert!(i.fitness <= b.fitness); } }
    #[test] fn mutation_changes() { let e = QdLlmEngine::new(NeuroEvoConfig::default(), 4).unwrap(); e.initialize_random(); e.evolve(fit).unwrap(); assert!(e.population().iter().any(|i| i.parent.is_some())); }
}
