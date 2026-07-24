#![allow(missing_docs)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacitySchedule {
    pub initial_hidden_dim: usize,
    pub target_hidden_dim: usize,
    pub initial_layers: usize,
    pub target_layers: usize,
    pub growth_interval: usize,
    pub growth_factor: f32,
}
impl Default for CapacitySchedule {
    fn default() -> Self { Self { initial_hidden_dim: 128, target_hidden_dim: 512, initial_layers: 2, target_layers: 8, growth_interval: 1000, growth_factor: 1.5 } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityStage {
    pub stage: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub step_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityTransition {
    pub from_stage: usize,
    pub to_stage: usize,
    pub hidden_dim_before: usize,
    pub hidden_dim_after: usize,
    pub step: usize,
    pub improvement_observed: Option<f64>,
}

pub struct ProgressiveCapacity {
    schedule: CapacitySchedule,
    current_stage: RwLock<usize>,
    stages: Vec<CapacityStage>,
    transitions: RwLock<Vec<CapacityTransition>>,
    loss_history: RwLock<Vec<f64>>,
}

impl ProgressiveCapacity {
    pub fn new(schedule: CapacitySchedule) -> Self {
        let mut stages = Vec::new();
        let mut dim = schedule.initial_hidden_dim;
        let mut layers = schedule.initial_layers;
        let mut step = 0;
        loop {
            stages.push(CapacityStage { stage: stages.len(), hidden_dim: dim, num_layers: layers, num_experts: (stages.len() + 1).min(8), step_threshold: step });
            if dim >= schedule.target_hidden_dim && layers >= schedule.target_layers { break; }
            step += schedule.growth_interval;
            dim = ((dim as f32 * schedule.growth_factor) as usize).min(schedule.target_hidden_dim);
            layers = (layers + 1).min(schedule.target_layers);
            if stages.len() > 20 { break; }
        }
        Self { schedule, current_stage: RwLock::new(0), stages, transitions: RwLock::new(Vec::new()), loss_history: RwLock::new(Vec::new()) }
    }
    pub fn current(&self) -> CapacityStage { self.stages[*self.current_stage.read()].clone() }
    pub fn should_grow(&self, step: usize, current_loss: f64) -> bool {
        let stage = self.current();
        if stage.stage >= self.stages.len() - 1 { return false; }
        if step < stage.step_threshold + self.schedule.growth_interval { return false; }
        let mut history = self.loss_history.write();
        history.push(current_loss);
        if history.len() > self.schedule.growth_interval { history.remove(0); }
        if history.len() < 10 { return false; }
        let recent: f64 = history.iter().rev().take(10).sum::<f64>() / 10.0;
        let older: f64 = history.iter().rev().skip(10).take(10).sum::<f64>() / 10.0;
        let reduction = (older - recent) / older.max(1e-8);
        reduction < 0.01
    }
    pub fn grow(&self, step: usize) -> CapacityTransition {
        let mut current_idx = self.current_stage.write();
        let from = *current_idx;
        *current_idx = (*current_idx + 1).min(self.stages.len() - 1);
        let from_stage = &self.stages[from];
        let to_stage = &self.stages[*current_idx];
        let transition = CapacityTransition {
            from_stage: from, to_stage: *current_idx,
            hidden_dim_before: from_stage.hidden_dim, hidden_dim_after: to_stage.hidden_dim,
            step, improvement_observed: None,
        };
        self.transitions.write().push(transition.clone());
        transition
    }
    pub fn current_hidden_dim(&self) -> usize { self.current().hidden_dim }
    pub fn current_num_layers(&self) -> usize { self.current().num_layers }
    pub fn stages(&self) -> &[CapacityStage] { &self.stages }
    pub fn transitions(&self) -> Vec<CapacityTransition> { self.transitions.read().clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn schedule_creates_stages() {
        let pc = ProgressiveCapacity::new(CapacitySchedule::default());
        assert!(pc.stages().len() > 1);
        assert_eq!(pc.current_hidden_dim(), 128);
    }
    #[test] fn grow_increases_capacity() {
        let pc = ProgressiveCapacity::new(CapacitySchedule::default());
        let before = pc.current_hidden_dim();
        let t = pc.grow(1000);
        assert!(pc.current_hidden_dim() >= before);
        assert_eq!(t.from_stage, 0);
        assert_eq!(t.to_stage, 1);
    }
    #[test] fn should_grow_on_plateau() {
        let pc = ProgressiveCapacity::new(CapacitySchedule { growth_interval: 20, ..Default::default() });
        for i in 0..50 { pc.should_grow(i, 0.5); }
        assert!(pc.should_grow(50, 0.5));
    }
    #[test] fn reaches_target() {
        let pc = ProgressiveCapacity::new(CapacitySchedule::default());
        for _ in 0..20 { pc.grow(0); }
        assert_eq!(pc.current_hidden_dim(), 512);
    }
}
