//! Resource manager.
//!
//! Tracks and enforces per-agent resource budgets: CPU time, resident memory,
//! GPU memory, token throughput, and I/O. Budgets are soft: exceeding them
//! raises [`EventKind::Resource`] events so the scheduler can preempt the
//! offending task.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::System;
use thiserror::Error;
use uuid::Uuid;

/// The kind of resource being accounted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// CPU time, in nanoseconds.
    CpuTime,
    /// Resident memory, in bytes.
    Memory,
    /// GPU memory, in bytes.
    GpuMemory,
    /// Token throughput (inference tokens / sec).
    Tokens,
    /// Disk I/O, in bytes.
    DiskIo,
    /// Network I/O, in bytes.
    NetworkIo,
}

/// A budget for a single resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudget {
    /// Maximum allowed usage.
    pub limit: u64,
    /// Current usage.
    pub used: u64,
    /// Soft threshold (fraction of limit) at which to emit a warning event.
    pub soft_threshold: f64,
}

impl ResourceBudget {
    /// Create a new budget with the given limit and a default soft threshold of 0.8.
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self { limit, used: 0, soft_threshold: 0.8 }
    }

    /// Consume `amount` of the budget. Returns `false` if the limit would be
    /// exceeded (in which case `used` is not modified).
    pub fn consume(&mut self, amount: u64) -> bool {
        let new_used = self.used.saturating_add(amount);
        if new_used > self.limit {
            return false;
        }
        self.used = new_used;
        true
    }

    /// Release `amount` back to the budget.
    pub fn release(&mut self, amount: u64) {
        self.used = self.used.saturating_sub(amount);
    }

    /// Returns `true` if usage exceeds the soft threshold.
    #[must_use]
    pub fn is_soft_exceeded(&self) -> bool {
        if self.limit == 0 {
            return false;
        }
        (self.used as f64) / (self.limit as f64) >= self.soft_threshold
    }

    /// Utilization fraction in `[0.0, 1.0]`.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.limit == 0 { 0.0 } else { (self.used as f64) / (self.limit as f64) }
    }
}

/// Errors emitted by the resource manager.
#[derive(Debug, Error)]
pub enum ResourceError {
    /// The budget for a resource was exceeded.
    #[error("resource {kind:?} exceeded: used={used} amount={amount} limit={limit}")]
    Exceeded {
        /// Which resource was exceeded.
        kind: ResourceKind,
        /// Current usage before the attempt.
        used: u64,
        /// Amount that was requested.
        amount: u64,
        /// Hard limit.
        limit: u64,
    },
    /// No budget was registered for the given (agent, kind) pair.
    #[error("no budget registered for agent {agent} kind {kind:?}")]
    NotRegistered {
        /// Agent id.
        agent: Uuid,
        /// Resource kind.
        kind: ResourceKind,
    },
}

impl ResourceBudget {
    /// Try to consume `amount`, returning a typed error on failure.
    fn try_consume(&mut self, amount: u64, kind: ResourceKind) -> Result<(), ResourceError> {
        if self.consume(amount) {
            Ok(())
        } else {
            Err(ResourceError::Exceeded {
                kind,
                used: self.used,
                amount,
                limit: self.limit,
            })
        }
    }
}

/// Per-agent resource accounting.
#[derive(Default)]
struct AgentAccount {
    budgets: AHashMap<ResourceKind, ResourceBudget>,
    cpu_start: Option<Instant>,
}

/// The resource manager.
#[derive(Clone)]
pub struct ResourceManager {
    inner: Arc<RwLock<ResourceManagerInner>>,
}

struct ResourceManagerInner {
    accounts: AHashMap<Uuid, AgentAccount>,
    sys: System,
    last_poll: Instant,
}

impl ResourceManager {
    /// Create a new resource manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ResourceManagerInner {
                accounts: AHashMap::new(),
                sys: System::new_all(),
                last_poll: Instant::now(),
            })),
        }
    }

    /// Register a new agent with default budgets derived from the host.
    pub fn register_agent(&self, agent: Uuid, memory_budget_bytes: u64) {
        let mut inner = self.inner.write();
        let cpu_count = num_cpus::get() as u64;
        let mut budgets = AHashMap::new();
        budgets.insert(ResourceKind::CpuTime, ResourceBudget::new(u64::MAX));
        budgets.insert(ResourceKind::Memory, ResourceBudget::new(memory_budget_bytes));
        budgets.insert(ResourceKind::GpuMemory, ResourceBudget::new(memory_budget_bytes / 2));
        budgets.insert(ResourceKind::Tokens, ResourceBudget::new(8_192));
        budgets.insert(ResourceKind::DiskIo, ResourceBudget::new(1024 * 1024 * 1024));
        budgets.insert(ResourceKind::NetworkIo, ResourceBudget::new(512 * 1024 * 1024));
        let _ = cpu_count; // reserved for future per-core accounting
        inner.accounts.insert(agent, AgentAccount { budgets, cpu_start: None });
    }

    /// Begin CPU accounting for an agent.
    pub fn cpu_enter(&self, agent: Uuid) {
        let mut inner = self.inner.write();
        if let Some(acc) = inner.accounts.get_mut(&agent) {
            acc.cpu_start = Some(Instant::now());
        }
    }

    /// End CPU accounting for an agent and consume the elapsed time.
    pub fn cpu_exit(&self, agent: Uuid) -> Result<(), ResourceError> {
        let elapsed_ns = {
            let mut inner = self.inner.write();
            let acc = inner.accounts.get_mut(&agent).ok_or(ResourceError::NotRegistered { agent, kind: ResourceKind::CpuTime })?;
            let start = acc.cpu_start.take().ok_or(ResourceError::NotRegistered { agent, kind: ResourceKind::CpuTime })?;
            start.elapsed().as_nanos() as u64
        };
        self.consume(agent, ResourceKind::CpuTime, elapsed_ns)
    }

    /// Consume `amount` of `kind` for `agent`.
    ///
    /// # Errors
    /// See [`ResourceBudget::consume`].
    pub fn consume(&self, agent: Uuid, kind: ResourceKind, amount: u64) -> Result<(), ResourceError> {
        let mut inner = self.inner.write();
        let acc = inner.accounts.get_mut(&agent).ok_or(ResourceError::NotRegistered { agent, kind })?;
        let budget = acc.budgets.get_mut(&kind).ok_or(ResourceError::NotRegistered { agent, kind })?;
        budget.try_consume(amount, kind)
    }

    /// Release `amount` of `kind` for `agent`.
    pub fn release(&self, agent: Uuid, kind: ResourceKind, amount: u64) {
        let mut inner = self.inner.write();
        if let Some(acc) = inner.accounts.get_mut(&agent) {
            if let Some(budget) = acc.budgets.get_mut(&kind) {
                budget.release(amount);
            }
        }
    }

    /// Returns the host's total physical memory in bytes.
    #[must_use]
    pub fn total_memory(&self) -> u64 {
        let mut inner = self.inner.write();
        inner.sys.refresh_memory();
        inner.sys.total_memory()
    }

    /// Returns the host's currently used memory in bytes.
    #[must_use]
    pub fn used_memory(&self) -> u64 {
        let mut inner = self.inner.write();
        inner.sys.refresh_memory();
        inner.sys.used_memory()
    }

    /// Returns the host's CPU utilization averaged over all cores, in `[0.0, 1.0]`.
    #[must_use]
    pub fn cpu_utilization(&self) -> f32 {
        let mut inner = self.inner.write();
        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_poll).as_secs_f32();
        if elapsed < 0.1 {
            return 0.0;
        }
        inner.sys.refresh_cpu_usage();
        inner.last_poll = now;
        let cpus = inner.sys.cpus();
        if cpus.is_empty() {
            return 0.0;
        }
        cpus.iter().map(|c| c.cpu_usage() / 100.0).sum::<f32>() / cpus.len() as f32
    }

    /// Returns the current utilization fraction of an agent's budget for `kind`.
    #[must_use]
    pub fn utilization(&self, agent: Uuid, kind: ResourceKind) -> f64 {
        let inner = self.inner.read();
        inner.accounts.get(&agent)
            .and_then(|acc| acc.budgets.get(&kind))
            .map(|b| b.utilization())
            .unwrap_or(0.0)
    }

    /// Returns `true` if any of an agent's budgets has crossed its soft threshold.
    #[must_use]
    pub fn is_soft_exceeded(&self, agent: Uuid) -> Option<ResourceKind> {
        let inner = self.inner.read();
        let acc = inner.accounts.get(&agent)?;
        for (kind, budget) in &acc.budgets {
            if budget.is_soft_exceeded() {
                return Some(*kind);
            }
        }
        None
    }

    /// Poll the host (refresh sysinfo) — called periodically by the scheduler.
    pub fn poll(&self, _max_age: Duration) {
        let mut inner = self.inner.write();
        inner.sys.refresh_memory();
        inner.sys.refresh_cpu_usage();
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_consume_and_release() {
        let mut b = ResourceBudget::new(100);
        assert!(b.consume(40));
        assert_eq!(b.used, 40);
        assert!(!b.consume(70));
        b.release(20);
        assert_eq!(b.used, 20);
        assert!(b.consume(80));
    }

    #[test]
    fn soft_threshold() {
        let mut b = ResourceBudget::new(100);
        b.soft_threshold = 0.5;
        assert!(!b.is_soft_exceeded());
        assert!(b.consume(49));
        assert!(!b.is_soft_exceeded());
        assert!(b.consume(2));
        assert!(b.is_soft_exceeded());
    }

    #[test]
    fn agent_accounting() {
        let rm = ResourceManager::new();
        let agent = Uuid::now_v7();
        rm.register_agent(agent, 1024 * 1024);
        rm.consume(agent, ResourceKind::Memory, 100_000).unwrap();
        assert!((rm.utilization(agent, ResourceKind::Memory) - (100_000.0 / 1_048_576.0)).abs() < 1e-9);
    }
}
