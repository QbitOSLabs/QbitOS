//! AgentOS reasoning kernel scheduler (2026).
//!
//! Implements a multi-agent scheduling kernel that manages cognitive agent
//! contexts, resource quotas, and scheduling policies. Inspired by the
//! AgentOS reasoning kernel architecture (2026).
//!
//! # Scheduling Policies
//!
//! - **RoundRobin**: cycles through agents in order.
//! - **Priority**: picks the agent with the highest priority value.
//! - **FairShare**: picks the agent with the fewest completed rounds.
//! - **Deadline**: picks the agent with the earliest deadline.
//!
//! # Resource Quotas
//!
//! Each agent has a per-round resource quota:
//! - `max_tokens_per_round` — LLM token budget.
//! - `max_tool_calls_per_round` — tool invocation budget.
//! - `max_memory_writes_per_round` — memory write budget.
//! - `max_time_ms_per_round` — wall-clock time budget.
//!
//! Quota usage is tracked per round and reset when `complete_round` is called.
//! Exceeding any quota returns a `QuotaExceeded` error.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Errors emitted by the reasoning kernel scheduler.
#[derive(Debug, Error)]
pub enum KernelSchedError {
    /// The agent with the given id was not found.
    #[error("agent `{0}` not found")]
    AgentNotFound(String),
    /// The agent exceeded its resource quota.
    #[error("quota exceeded for `{agent}`: {resource}")]
    QuotaExceeded {
        /// The agent that exceeded the quota.
        agent: String,
        /// The resource that was exhausted (e.g. "tokens", "tool_calls").
        resource: String,
    },
}

/// Unique identifier for an agent.
pub type AgentId = String;

/// The scheduling policy used to pick the next agent.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingPolicy {
    /// Round-robin: cycle through agents in order.
    RoundRobin,
    /// Priority: pick the agent with the highest priority value.
    Priority,
    /// Fair-share: pick the agent with the fewest completed rounds.
    FairShare,
    /// Deadline: pick the agent with the earliest deadline.
    Deadline,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::Priority
    }
}

/// Per-round resource quota for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Maximum LLM tokens per round.
    pub max_tokens_per_round: u64,
    /// Maximum tool calls per round.
    pub max_tool_calls_per_round: u64,
    /// Maximum memory writes per round.
    pub max_memory_writes_per_round: u64,
    /// Maximum wall-clock time (ms) per round.
    pub max_time_ms_per_round: u64,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_tokens_per_round: 4096,
            max_tool_calls_per_round: 16,
            max_memory_writes_per_round: 32,
            max_time_ms_per_round: 5000,
        }
    }
}

/// Per-round resource usage tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Tokens used this round.
    pub tokens_used: u64,
    /// Tool calls made this round.
    pub tool_calls_used: u64,
    /// Memory writes this round.
    pub memory_writes_used: u64,
    /// Wall-clock time used this round (ms).
    pub time_used_ms: u64,
}

/// The runtime context for a registered agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    /// The agent's unique id.
    pub id: AgentId,
    /// Priority (higher = more important).
    pub priority: i32,
    /// Deadline in milliseconds (0 = no deadline).
    pub deadline_ms: u64,
    /// Resource quota for this agent.
    pub quota: ResourceQuota,
    /// Current round's resource usage.
    pub usage: ResourceUsage,
    /// Whether the agent is currently running.
    pub is_running: bool,
    /// Total rounds completed.
    pub rounds_completed: u64,
    /// Total time consumed across all rounds (ms).
    pub total_time_consumed_ms: u64,
}

/// An audit log entry for a scheduling decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingAuditEntry {
    /// Timestamp (ms since Unix epoch).
    pub timestamp_ms: u64,
    /// The agent that was scheduled.
    pub agent_id: AgentId,
    /// The scheduling policy used.
    pub policy: SchedulingPolicy,
    /// Human-readable reason for the scheduling decision.
    pub reason: String,
    /// The agent's resource usage at scheduling time.
    pub usage: ResourceUsage,
}

/// Configuration for the reasoning kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// The scheduling policy.
    pub policy: SchedulingPolicy,
    /// Default quota for new agents.
    pub default_quota: ResourceQuota,
    /// Maximum number of registered agents.
    pub max_agents: usize,
    /// Whether to record audit entries.
    pub audit_enabled: bool,
    /// Maximum number of audit entries to keep.
    pub max_audit_entries: usize,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Priority,
            default_quota: ResourceQuota::default(),
            max_agents: 32,
            audit_enabled: true,
            max_audit_entries: 10_000,
        }
    }
}

/// The reasoning kernel: manages agent registration, scheduling, and quotas.
pub struct ReasoningKernel {
    cfg: KernelConfig,
    agents: RwLock<AHashMap<AgentId, AgentContext>>,
    rr_cursor: RwLock<usize>,
    audit_log: RwLock<Vec<SchedulingAuditEntry>>,
}

impl ReasoningKernel {
    /// Create a new reasoning kernel with the given configuration.
    #[must_use]
    pub fn new(cfg: KernelConfig) -> Self {
        Self {
            cfg,
            agents: RwLock::new(AHashMap::new()),
            rr_cursor: RwLock::new(0),
            audit_log: RwLock::new(Vec::new()),
        }
    }

    /// Register a new agent with the default quota.
    ///
    /// # Errors
    /// Returns an error if `max_agents` is exceeded.
    pub fn register_agent(&self, id: AgentId, priority: i32) -> Result<(), KernelSchedError> {
        self.register_with_quota(id, priority, self.cfg.default_quota.clone())
    }

    /// Register a new agent with a custom quota.
    ///
    /// # Errors
    /// Returns an error if `max_agents` is exceeded.
    pub fn register_with_quota(
        &self,
        id: AgentId,
        priority: i32,
        quota: ResourceQuota,
    ) -> Result<(), KernelSchedError> {
        let mut a = self.agents.write();
        if a.len() >= self.cfg.max_agents && !a.contains_key(&id) {
            return Err(KernelSchedError::AgentNotFound("max_agents exceeded".into()));
        }
        a.insert(
            id.clone(),
            AgentContext {
                id,
                priority,
                deadline_ms: 0,
                quota,
                usage: ResourceUsage::default(),
                is_running: false,
                rounds_completed: 0,
                total_time_consumed_ms: 0,
            },
        );
        Ok(())
    }

    /// Unregister an agent.
    ///
    /// # Errors
    /// Returns an error if the agent doesn't exist.
    pub fn unregister(&self, id: &str) -> Result<(), KernelSchedError> {
        if self.agents.write().remove(id).is_none() {
            return Err(KernelSchedError::AgentNotFound(id.into()));
        }
        Ok(())
    }

    /// Pick the next agent to schedule, based on the configured policy.
    ///
    /// # Errors
    /// Returns an error if no agents are registered or (for Deadline policy) no
    /// agents have deadlines set.
    pub fn pick_next(&self) -> Result<(AgentId, String), KernelSchedError> {
        let a = self.agents.read();
        if a.is_empty() {
            return Err(KernelSchedError::AgentNotFound("none".into()));
        }
        let mut c: Vec<&AgentContext> = a.values().collect();
        let (id, reason) = match self.cfg.policy {
            SchedulingPolicy::RoundRobin => {
                let mut cur = self.rr_cursor.write();
                *cur = (*cur + 1).max(1) % c.len().max(1);
                let i = (*cur).saturating_sub(1) % c.len();
                (c[i].id.clone(), format!("rr turn {}", i + 1))
            }
            SchedulingPolicy::Priority => {
                c.sort_by(|a, b| b.priority.cmp(&a.priority));
                (c[0].id.clone(), format!("priority {}", c[0].priority))
            }
            SchedulingPolicy::FairShare => {
                c.sort_by(|a, b| a.rounds_completed.cmp(&b.rounds_completed));
                (
                    c[0].id.clone(),
                    format!("fairshare rounds={}", c[0].rounds_completed),
                )
            }
            SchedulingPolicy::Deadline => {
                c.retain(|a| a.deadline_ms > 0);
                if c.is_empty() {
                    return Err(KernelSchedError::AgentNotFound("no deadlines".into()));
                }
                c.sort_by(|a, b| a.deadline_ms.cmp(&b.deadline_ms));
                (c[0].id.clone(), format!("deadline {}", c[0].deadline_ms))
            }
        };
        drop(a);
        self.agents
            .write()
            .get_mut(&id)
            .map(|ag| ag.is_running = true);
        if self.cfg.audit_enabled {
            let u = self
                .agents
                .read()
                .get(&id)
                .map(|a| a.usage.clone())
                .unwrap_or_default();
            let mut log = self.audit_log.write();
            log.push(SchedulingAuditEntry {
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                agent_id: id.clone(),
                policy: self.cfg.policy,
                reason: reason.clone(),
                usage: u,
            });
            if log.len() > self.cfg.max_audit_entries {
                let ex = log.len() - self.cfg.max_audit_entries;
                log.drain(0..ex);
            }
        }
        Ok((id, reason))
    }

    /// Mark an agent's round as complete and reset its usage counters.
    ///
    /// # Errors
    /// Returns an error if the agent doesn't exist.
    pub fn complete_round(&self, id: &str, usage: ResourceUsage) -> Result<(), KernelSchedError> {
        let mut a = self.agents.write();
        let ag = a
            .get_mut(id)
            .ok_or_else(|| KernelSchedError::AgentNotFound(id.into()))?;
        ag.is_running = false;
        ag.rounds_completed += 1;
        ag.total_time_consumed_ms += usage.time_used_ms;
        ag.usage = ResourceUsage::default();
        Ok(())
    }

    /// Record token usage for an agent.
    ///
    /// # Errors
    /// Returns `QuotaExceeded` if the agent exceeds its token quota.
    pub fn record_tokens(&self, id: &str, tokens: u64) -> Result<(), KernelSchedError> {
        let mut a = self.agents.write();
        let ag = a
            .get_mut(id)
            .ok_or_else(|| KernelSchedError::AgentNotFound(id.into()))?;
        ag.usage.tokens_used += tokens;
        if ag.usage.tokens_used > ag.quota.max_tokens_per_round {
            return Err(KernelSchedError::QuotaExceeded {
                agent: id.into(),
                resource: "tokens".into(),
            });
        }
        Ok(())
    }

    /// Record a tool call for an agent.
    ///
    /// # Errors
    /// Returns `QuotaExceeded` if the agent exceeds its tool-call quota.
    pub fn record_tool_call(&self, id: &str) -> Result<(), KernelSchedError> {
        let mut a = self.agents.write();
        let ag = a
            .get_mut(id)
            .ok_or_else(|| KernelSchedError::AgentNotFound(id.into()))?;
        ag.usage.tool_calls_used += 1;
        if ag.usage.tool_calls_used > ag.quota.max_tool_calls_per_round {
            return Err(KernelSchedError::QuotaExceeded {
                agent: id.into(),
                resource: "tool_calls".into(),
            });
        }
        Ok(())
    }

    /// Set a deadline for an agent.
    ///
    /// # Errors
    /// Returns an error if the agent doesn't exist.
    pub fn set_deadline(&self, id: &str, ms: u64) -> Result<(), KernelSchedError> {
        self.agents
            .write()
            .get_mut(id)
            .ok_or_else(|| KernelSchedError::AgentNotFound(id.into()))?
            .deadline_ms = ms;
        Ok(())
    }

    /// Returns the context for the given agent.
    #[must_use]
    pub fn get_agent(&self, id: &str) -> Option<AgentContext> {
        self.agents.read().get(id).cloned()
    }

    /// Returns all registered agents.
    #[must_use]
    pub fn list_agents(&self) -> Vec<AgentContext> {
        self.agents.read().values().cloned().collect()
    }

    /// Returns the number of registered agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agents.read().len()
    }

    /// Returns the audit log.
    #[must_use]
    pub fn audit_log(&self) -> Vec<SchedulingAuditEntry> {
        self.audit_log.read().clone()
    }

    /// Clears the audit log.
    pub fn clear_audit_log(&self) {
        self.audit_log.write().clear();
    }

    /// Returns the configured scheduling policy.
    #[must_use]
    pub fn policy(&self) -> SchedulingPolicy {
        self.cfg.policy
    }
}

/// Shared handle to a reasoning kernel.
pub type SharedReasoningKernel = Arc<ReasoningKernel>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a1".into(), 5).unwrap();
        assert_eq!(k.agent_count(), 1);
    }

    #[test]
    fn unregister() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a1".into(), 5).unwrap();
        k.unregister("a1").unwrap();
        assert_eq!(k.agent_count(), 0);
    }

    #[test]
    fn priority() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("low".into(), 1).unwrap();
        k.register_agent("high".into(), 10).unwrap();
        let (id, _) = k.pick_next().unwrap();
        assert_eq!(id, "high");
    }

    #[test]
    fn round_robin() {
        let k = ReasoningKernel::new(KernelConfig {
            policy: SchedulingPolicy::RoundRobin,
            ..Default::default()
        });
        k.register_agent("a".into(), 1).unwrap();
        k.register_agent("b".into(), 1).unwrap();
        let (f, _) = k.pick_next().unwrap();
        let (s, _) = k.pick_next().unwrap();
        assert_ne!(f, s);
    }

    #[test]
    fn fair_share() {
        let k = ReasoningKernel::new(KernelConfig {
            policy: SchedulingPolicy::FairShare,
            ..Default::default()
        });
        k.register_agent("a".into(), 1).unwrap();
        k.register_agent("b".into(), 1).unwrap();
        {
            k.agents.write().get_mut("a").unwrap().rounds_completed = 5;
        }
        let (id, _) = k.pick_next().unwrap();
        assert_eq!(id, "b");
    }

    #[test]
    fn deadline() {
        let k = ReasoningKernel::new(KernelConfig {
            policy: SchedulingPolicy::Deadline,
            ..Default::default()
        });
        k.register_agent("a".into(), 1).unwrap();
        k.register_agent("b".into(), 1).unwrap();
        k.set_deadline("a", 2000).unwrap();
        k.set_deadline("b", 1000).unwrap();
        let (id, _) = k.pick_next().unwrap();
        assert_eq!(id, "b");
    }

    #[test]
    fn record_tokens() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a".into(), 1).unwrap();
        k.record_tokens("a", 100).unwrap();
        assert_eq!(k.get_agent("a").unwrap().usage.tokens_used, 100);
    }

    #[test]
    fn quota_exceeded() {
        let k = ReasoningKernel::new(KernelConfig {
            default_quota: ResourceQuota {
                max_tokens_per_round: 100,
                ..Default::default()
            },
            ..Default::default()
        });
        k.register_agent("a".into(), 1).unwrap();
        k.record_tokens("a", 50).unwrap();
        assert!(k.record_tokens("a", 100).is_err());
    }

    #[test]
    fn complete_round() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a".into(), 1).unwrap();
        let _ = k.pick_next().unwrap();
        k.complete_round(
            "a",
            ResourceUsage {
                time_used_ms: 100,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(k.get_agent("a").unwrap().rounds_completed, 1);
    }

    #[test]
    fn audit() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a".into(), 1).unwrap();
        let _ = k.pick_next().unwrap();
        assert_eq!(k.audit_log().len(), 1);
    }

    #[test]
    fn tool_calls() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a".into(), 1).unwrap();
        k.record_tool_call("a").unwrap();
        k.record_tool_call("a").unwrap();
        assert_eq!(k.get_agent("a").unwrap().usage.tool_calls_used, 2);
    }

    #[test]
    fn clear_audit() {
        let k = ReasoningKernel::new(KernelConfig::default());
        k.register_agent("a".into(), 1).unwrap();
        let _ = k.pick_next().unwrap();
        k.clear_audit_log();
        assert!(k.audit_log().is_empty());
    }

    #[test]
    fn unknown_agent() {
        let k = ReasoningKernel::new(KernelConfig::default());
        assert!(k.record_tokens("missing", 100).is_err());
    }
}
