//! DPO - Direct Preference Optimization (Rafailov NeurIPS 2023, production 2026).
#![allow(missing_docs)]
use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DpoError { #[error("invalid config: {0}")] InvalidConfig(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoPair { pub prompt: String, pub chosen: String, pub rejected: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoConfig { pub beta: f64, pub learning_rate: f64, pub max_steps: usize, pub convergence_threshold: f64, pub use_kl_regularization: bool, pub kl_weight: f64 }
impl Default for DpoConfig { fn default() -> Self { Self { beta: 0.1, learning_rate: 0.01, max_steps: 1000, convergence_threshold: 0.01, use_kl_regularization: true, kl_weight: 0.05 } } }
impl DpoConfig { fn validate(&self) -> Result<(), DpoError> { if self.beta <= 0.0 { return Err(DpoError::InvalidConfig("beta must be positive".into())); } if self.learning_rate <= 0.0 { return Err(DpoError::InvalidConfig("lr must be positive".into())); } Ok(()) } }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DpoStats { pub loss: f64, pub accuracy: f64, pub kl_divergence: f64, pub num_pairs: usize, pub step: usize }

pub struct DpoTrainer { cfg: DpoConfig, policy: RwLock<AHashMap<String, f64>>, reference: RwLock<AHashMap<String, f64>>, step: RwLock<usize> }

impl DpoTrainer {
    pub fn new(cfg: DpoConfig) -> Result<Self, DpoError> { cfg.validate()?; Ok(Self { cfg, policy: RwLock::new(AHashMap::new()), reference: RwLock::new(AHashMap::new()), step: RwLock::new(0) }) }
    pub fn initialize_reference(&self, lp: AHashMap<String, f64>) { *self.reference.write() = lp.clone(); *self.policy.write() = lp; }
    #[must_use] pub fn policy_log_prob(&self, p: &str, r: &str) -> f64 { self.policy.read().get(&format!("{p}\x00{r}")).copied().unwrap_or(-10.0) }
    #[must_use] pub fn reference_log_prob(&self, p: &str, r: &str) -> f64 { self.reference.read().get(&format!("{p}\x00{r}")).copied().unwrap_or(-10.0) }
    pub fn set_policy_log_prob(&self, p: &str, r: &str, lp: f64) { self.policy.write().insert(format!("{p}\x00{r}"), lp); }
    pub fn train_step(&self, pairs: &[DpoPair]) -> Result<DpoStats, DpoError> {
        if pairs.is_empty() { return Ok(DpoStats::default()); }
        let mut tl = 0.0_f64; let mut correct = 0; let mut tkl = 0.0_f64; let beta = self.cfg.beta; let lr = self.cfg.learning_rate;
        let mut updates: AHashMap<String, f64> = AHashMap::new();
        for p in pairs {
            let ck = format!("{}\x00{}", p.prompt, p.chosen); let rk = format!("{}\x00{}", p.prompt, p.rejected);
            let pc = self.policy.read().get(&ck).copied().unwrap_or(-10.0); let pr = self.policy.read().get(&rk).copied().unwrap_or(-10.0);
            let rc = self.reference.read().get(&ck).copied().unwrap_or(-10.0); let rr = self.reference.read().get(&rk).copied().unwrap_or(-10.0);
            let diff = beta * ((pc - rc) - (pr - rr));
            let loss = -log_sigmoid(diff); tl += loss;
            // Gradient of loss = -log_sigmoid(diff) w.r.t. diff is (sigmoid(diff) - 1).
            // Through chain rule: d(loss)/d(pc) = beta * (sigmoid(diff) - 1),
            //                       d(loss)/d(pr) = -beta * (sigmoid(diff) - 1).
            // Gradient descent: chosen += lr * beta * (1 - sigmoid(diff)),
            //                   rejected -= lr * beta * (1 - sigmoid(diff)).
            let gs = beta * (1.0 - sigmoid(diff));
            *updates.entry(ck).or_insert(0.0) += gs;   // chosen: increase log-prob
            *updates.entry(rk).or_insert(0.0) -= gs;   // rejected: decrease log-prob
            if pc > pr { correct += 1; }
            tkl += (pc - rc).powi(2) + (pr - rr).powi(2);
        }
        let mut pol = self.policy.write();
        // Apply updates as gradient DESCENT: param -= lr * grad. The updates
        // map already holds the NEGATIVE gradient (i.e., the ascent direction
        // for chosen / descent for rejected), so we ADD lr * update.
        for (k, g) in &updates { let cur = pol.get(k).copied().unwrap_or(-10.0); pol.insert(k.clone(), cur + lr * g); }
        drop(pol);
        if self.cfg.use_kl_regularization { let mut pol = self.policy.write(); let rfr = self.reference.read(); let w = self.cfg.kl_weight * lr; for (k, v) in pol.iter_mut() { if let Some(r) = rfr.get(k) { *v += w * (r - *v); } } }
        let n = pairs.len();
        Ok(DpoStats { loss: tl / n as f64, accuracy: correct as f64 / n as f64, kl_divergence: tkl / n as f64, num_pairs: n, step: { let mut s = self.step.write(); *s += 1; *s } })
    }
    pub fn train(&self, pairs: &[DpoPair], cb: Option<&dyn Fn(&DpoStats)>) -> Result<DpoStats, DpoError> {
        let mut last = DpoStats::default();
        for _ in 0..self.cfg.max_steps { let s = self.train_step(pairs)?; if let Some(c) = cb { c(&s); } last = s; if last.loss < self.cfg.convergence_threshold { break; } }
        Ok(last)
    }
    #[must_use] pub fn current_step(&self) -> usize { *self.step.read() }
    pub fn reset_policy(&self) { *self.policy.write() = self.reference.read().clone(); *self.step.write() = 0; }
}

pub type SharedDpoTrainer = Arc<DpoTrainer>;

fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) } }
fn log_sigmoid(x: f64) -> f64 { if x >= 0.0 { -(1.0 + (-x).exp()).ln() } else { x - (1.0 + x.exp()).ln() } }

#[cfg(test)]
mod tests {
    use super::*;
    fn mk(p:&str,c:&str,r:&str)->DpoPair{DpoPair{prompt:p.into(),chosen:c.into(),rejected:r.into()}}
    #[test] fn init(){let t=DpoTrainer::new(DpoConfig::default()).unwrap();assert_eq!(t.current_step(),0);}
    #[test] fn invalid(){assert!(DpoTrainer::new(DpoConfig{beta:-1.0,..Default::default()}).is_err());}
    #[test] fn loss_reduces(){let t=DpoTrainer::new(DpoConfig{learning_rate:0.1,max_steps:1,..Default::default()}).unwrap();let mut i=AHashMap::new();i.insert("p\x00good".into(),-1.0);i.insert("p\x00bad".into(),-2.0);t.initialize_reference(i);let s1=t.train_step(&[mk("p","good","bad")]).unwrap();let s2=t.train_step(&[mk("p","good","bad")]).unwrap();assert!(s2.loss<=s1.loss+1e-6);}
    #[test] fn accuracy(){let t=DpoTrainer::new(DpoConfig{learning_rate:0.5,max_steps:50,convergence_threshold:0.001,..Default::default()}).unwrap();let mut i=AHashMap::new();i.insert("p\x00good".into(),-1.0);i.insert("p\x00bad".into(),-1.5);t.initialize_reference(i);let s=t.train(&[mk("p","good","bad")],None).unwrap();assert!(s.accuracy>0.5);}
    #[test] fn kl_reg(){let t=DpoTrainer::new(DpoConfig{learning_rate:0.5,kl_weight:0.5,use_kl_regularization:true,max_steps:1,..Default::default()}).unwrap();let mut i=AHashMap::new();i.insert("p\x00a".into(),-1.0);i.insert("p\x00b".into(),-2.0);t.initialize_reference(i);let s=t.train_step(&[mk("p","a","b")]).unwrap();assert!(s.kl_divergence<1.0);}
    #[test] fn reset(){let t=DpoTrainer::new(DpoConfig::default()).unwrap();let mut i=AHashMap::new();i.insert("p\x00a".into(),-1.0);t.initialize_reference(i);t.set_policy_log_prob("p","a",-0.5);assert_eq!(t.policy_log_prob("p","a"),-0.5);t.reset_policy();assert_eq!(t.policy_log_prob("p","a"),-1.0);assert_eq!(t.current_step(),0);}
    #[test] fn empty(){let t=DpoTrainer::new(DpoConfig::default()).unwrap();let s=t.train_step(&[]).unwrap();assert_eq!(s.num_pairs,0);}
    #[test] fn converge(){let t=DpoTrainer::new(DpoConfig{learning_rate:1.0,max_steps:2000,convergence_threshold:0.5,use_kl_regularization:false,..Default::default()}).unwrap();let mut i=AHashMap::new();i.insert("p\x00good".into(),-1.0);i.insert("p\x00bad".into(),-1.1);t.initialize_reference(i);let s=t.train(&[mk("p","good","bad")],None).unwrap();assert!(s.loss<0.5||s.step<2000);}
    #[test] fn sigmoid_consistent(){for &x in &[0.0,1.0,-1.0,5.0,-5.0]{assert!((log_sigmoid(x)-sigmoid(x).ln()).abs()<1e-6);}}
}
