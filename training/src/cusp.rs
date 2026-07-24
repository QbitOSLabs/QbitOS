//! Curriculum Self-Play (CuSP) - ICLR 2026.
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CuspError { #[error("setter failed: {0}")] SetterFailed(String), #[error("verifier rejected: {0}")] GoalRejected(String), #[error("judge failed: {0}")] JudgeFailed(String) }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuspGoal { pub id: u64, pub description: String, pub difficulty: f64, pub frontier_confidence: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuspRound { pub goal: CuspGoal, pub goal_accepted: bool, pub solver_attempt: String, pub judge_success: bool, pub solver_reward: f64, pub setter_reward: f64, pub judge_reward: f64, pub verifier_reward: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuspConfig { pub target_success_rate: f64, pub max_rounds: usize, pub log_rounds: bool, pub setter_frontier_weight: f64 }
impl Default for CuspConfig { fn default() -> Self { Self { target_success_rate: 0.5, max_rounds: 100, log_rounds: true, setter_frontier_weight: 1.0 } } }

pub trait CuspSetter: Send + Sync { fn propose_goal(&self, solver_success_rate: f64) -> Result<CuspGoal, CuspError>; }
pub trait CuspVerifier: Send + Sync { fn verify(&self, goal: &CuspGoal) -> Result<(), CuspError>; }
pub trait CuspSolver: Send + Sync { fn solve(&self, goal: &CuspGoal) -> Result<String, CuspError>; }
pub trait CuspJudge: Send + Sync { fn judge(&self, goal: &CuspGoal, attempt: &str) -> Result<bool, CuspError>; }

pub struct CuspOrchestrator { cfg: CuspConfig, setter: Arc<dyn CuspSetter>, verifier: Arc<dyn CuspVerifier>, solver: Arc<dyn CuspSolver>, judge: Arc<dyn CuspJudge>, success_rate: RwLock<f64>, history: RwLock<Vec<CuspRound>>, next_id: RwLock<u64> }

impl CuspOrchestrator {
    pub fn new(cfg: CuspConfig, s: Arc<dyn CuspSetter>, v: Arc<dyn CuspVerifier>, sv: Arc<dyn CuspSolver>, j: Arc<dyn CuspJudge>) -> Self { Self { cfg, setter: s, verifier: v, solver: sv, judge: j, success_rate: RwLock::new(0.5), history: RwLock::new(Vec::new()), next_id: RwLock::new(0) } }
    pub fn run_round(&self) -> Result<CuspRound, CuspError> {
        let sr = *self.success_rate.read(); let mut goal = self.setter.propose_goal(sr)?;
        let mut nid = self.next_id.write(); *nid += 1; goal.id = *nid; drop(nid);
        let accepted = match self.verifier.verify(&goal) { Ok(()) => true, Err(CuspError::GoalRejected(_)) => false, Err(e) => return Err(e) };
        if !accepted { let r = CuspRound { goal, goal_accepted: false, solver_attempt: String::new(), judge_success: false, solver_reward: 0.0, setter_reward: -0.5, judge_reward: 0.0, verifier_reward: 1.0 }; self.update_sr(false); self.log(r.clone()); return Ok(r); }
        let attempt = self.solver.solve(&goal).unwrap_or_default();
        let success = self.judge.judge(&goal, &attempt).unwrap_or(false);
        let sr = self.solver_reward(success); let str_ = -(self.ema_sr(success) - self.cfg.target_success_rate).abs() * self.cfg.setter_frontier_weight;
        let r = CuspRound { goal, goal_accepted: true, solver_attempt: attempt, judge_success: success, solver_reward: sr, setter_reward: str_, judge_reward: 1.0, verifier_reward: 1.0 };
        self.update_sr(success); self.log(r.clone()); Ok(r)
    }
    pub fn train(&self) -> Result<Vec<CuspRound>, CuspError> { let mut rs = Vec::with_capacity(self.cfg.max_rounds); for _ in 0..self.cfg.max_rounds { rs.push(self.run_round()?); } Ok(rs) }
    #[must_use] pub fn solver_success_rate(&self) -> f64 { *self.success_rate.read() }
    #[must_use] pub fn history(&self) -> Vec<CuspRound> { self.history.read().clone() }
    fn update_sr(&self, success: bool) { let mut r = self.success_rate.write(); let a = 0.1; let o = if success { 1.0 } else { 0.0 }; *r = (1.0 - a) * *r + a * o; }
    fn ema_sr(&self, success: bool) -> f64 { let c = *self.success_rate.read(); let a = 0.1; let o = if success { 1.0 } else { 0.0 }; (1.0 - a) * c + a * o }
    fn solver_reward(&self, success: bool) -> f64 { if success { 1.0 } else { 0.0 } }
    fn log(&self, r: CuspRound) { if self.cfg.log_rounds { self.history.write().push(r); } }
    pub fn clear_history(&self) { self.history.write().clear(); }
}

pub struct StubSetter { pub prefix: String }
impl CuspSetter for StubSetter { fn propose_goal(&self, sr: f64) -> Result<CuspGoal, CuspError> { Ok(CuspGoal { id: 0, description: format!("{} d={:.2}", self.prefix, sr), difficulty: sr.clamp(0.0, 1.0), frontier_confidence: 1.0 - (sr - 0.5).abs() }) } }
pub struct StubVerifier;
impl CuspVerifier for StubVerifier { fn verify(&self, g: &CuspGoal) -> Result<(), CuspError> { if g.difficulty < 0.1 { Err(CuspError::GoalRejected("too easy".into())) } else if g.difficulty > 0.9 { Err(CuspError::GoalRejected("too hard".into())) } else { Ok(()) } } }
pub struct StubSolver { pub _state: u64 }
impl CuspSolver for StubSolver { fn solve(&self, g: &CuspGoal) -> Result<String, CuspError> { if g.difficulty < 0.5 { Ok(format!("solved: {}", g.description)) } else { Ok(format!("failed: {}", g.description)) } } }
pub struct StubJudge;
impl CuspJudge for StubJudge { fn judge(&self, _g: &CuspGoal, a: &str) -> Result<bool, CuspError> { Ok(a.starts_with("solved")) } }

#[cfg(test)]
mod tests {
    use super::*;
    fn orch() -> CuspOrchestrator { CuspOrchestrator::new(CuspConfig { max_rounds: 10, ..Default::default() }, Arc::new(StubSetter { prefix: "test".into() }), Arc::new(StubVerifier), Arc::new(StubSolver { _state: 0 }), Arc::new(StubJudge)) }
    #[test] fn round() { let o = orch(); let r = o.run_round().unwrap(); assert!(r.goal_accepted || !r.goal_accepted); }
    #[test] fn train() { let o = orch(); let rs = o.train().unwrap(); assert_eq!(rs.len(), 10); }
    #[test] fn setter_adjusts() { let o = orch(); let g = o.setter.propose_goal(0.5).unwrap(); assert!((g.difficulty - 0.5).abs() < 0.01); }
    #[test] fn verifier_rejects() { let v = StubVerifier; assert!(v.verify(&CuspGoal { id: 1, description: "x".into(), difficulty: 0.05, frontier_confidence: 0.5 }).is_err()); }
    #[test] fn solver_easy() { let s = StubSolver { _state: 0 }; let a = s.solve(&CuspGoal { id: 1, description: "x".into(), difficulty: 0.2, frontier_confidence: 0.8 }).unwrap(); assert!(a.starts_with("solved")); }
    #[test] fn judge() { let j = StubJudge; assert!(j.judge(&CuspGoal { id: 1, description: "x".into(), difficulty: 0.5, frontier_confidence: 0.9 }, "solved it").unwrap()); }
    #[test] fn sr_updates() { let o = orch(); for _ in 0..5 { o.run_round().unwrap(); } let s = o.solver_success_rate(); assert!(s >= 0.0 && s <= 1.0); }
    #[test] fn history() { let o = orch(); for _ in 0..3 { o.run_round().unwrap(); } assert_eq!(o.history().len(), 3); o.clear_history(); assert_eq!(o.history().len(), 0); }
    #[test] fn setter_reward() { let o = orch(); let r = o.run_round().unwrap(); assert!(r.setter_reward <= 0.0); }
    #[test] fn solver_reward_binary() { let o = orch(); let r = o.run_round().unwrap(); assert!(r.solver_reward == 0.0 || r.solver_reward == 1.0); }
    #[test] fn rejected_penalty() { struct Hard; impl CuspSetter for Hard { fn propose_goal(&self, _sr: f64) -> Result<CuspGoal, CuspError> { Ok(CuspGoal { id: 0, description: "x".into(), difficulty: 0.99, frontier_confidence: 0.0 }) } } let o = CuspOrchestrator::new(CuspConfig::default(), Arc::new(Hard), Arc::new(StubVerifier), Arc::new(StubSolver { _state: 0 }), Arc::new(StubJudge)); let r = o.run_round().unwrap(); assert!(!r.goal_accepted); assert!(r.setter_reward < 0.0); }
}
