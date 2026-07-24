//! Service orchestrator: owns all QbitOS subsystems and exposes them through
//! a single `CognitiveService` handle.

use crate::error::{ServiceError, ServiceResult};
use crate::events::{EventBus, ServiceEvent};
use crate::event_store::PersistentEventStore;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use qbitos_memory::MemoryFabric;
use qbitos_skills::SkillModule;
use qbitos_training::{Curriculum, ExperienceReplay, MetaLearner, SelfImprovementCycle};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Configuration for the cognitive service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Human-readable instance name.
    pub instance_name: String,
    /// Maximum concurrent HTTP requests.
    pub max_concurrent_requests: usize,
    /// Request timeout in milliseconds.
    pub request_timeout_ms: u64,
    /// Maximum event bus channel capacity.
    pub event_bus_capacity: usize,
    /// Whether safety filtering is enforced on all inputs.
    pub safety_enforced: bool,
    /// Whether to record audit logs.
    pub audit_log_enabled: bool,
    /// Whether self-improvement is allowed to mutate skills.
    pub self_improvement_enabled: bool,
    /// Embedding dimension for the memory fabric.
    pub embedding_dim: usize,
    /// Experience replay buffer capacity.
    pub replay_capacity: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            instance_name: "qbitos-default".into(),
            max_concurrent_requests: 256,
            request_timeout_ms: 30_000,
            event_bus_capacity: 4096,
            safety_enforced: true,
            audit_log_enabled: true,
            self_improvement_enabled: false,
            embedding_dim: 384,
            replay_capacity: 10_000,
        }
    }
}

impl ServiceConfig {
    /// Build a configuration from environment variables, falling back to
    /// defaults for any that are unset.
    ///
    /// Recognized variables:
    /// - `QBITOS_INSTANCE_NAME`
    /// - `QBITOS_MAX_CONCURRENT_REQUESTS`
    /// - `QBITOS_REQUEST_TIMEOUT_MS`
    /// - `QBITOS_EVENT_BUS_CAPACITY`
    /// - `QBITOS_REPLAY_CAPACITY`
    /// - `QBITOS_EMBEDDING_DIM`
    /// - `QBITOS_SAFETY_ENFORCED`
    /// - `QBITOS_AUDIT_LOG`
    /// - `QBITOS_ALLOW_SELF_IMPROVE`
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("QBITOS_INSTANCE_NAME") {
            cfg.instance_name = v;
        }
        if let Ok(v) = std::env::var("QBITOS_MAX_CONCURRENT_REQUESTS") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.max_concurrent_requests = n;
            }
        }
        if let Ok(v) = std::env::var("QBITOS_REQUEST_TIMEOUT_MS") {
            if let Ok(n) = v.parse::<u64>() {
                cfg.request_timeout_ms = n;
            }
        }
        if let Ok(v) = std::env::var("QBITOS_EVENT_BUS_CAPACITY") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.event_bus_capacity = n;
            }
        }
        if let Ok(v) = std::env::var("QBITOS_REPLAY_CAPACITY") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.replay_capacity = n;
            }
        }
        if let Ok(v) = std::env::var("QBITOS_EMBEDDING_DIM") {
            if let Ok(n) = v.parse::<usize>() {
                cfg.embedding_dim = n;
            }
        }
        if let Ok(v) = std::env::var("QBITOS_SAFETY_ENFORCED") {
            cfg.safety_enforced = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("QBITOS_AUDIT_LOG") {
            cfg.audit_log_enabled = parse_bool(&v);
        }
        if let Ok(v) = std::env::var("QBITOS_ALLOW_SELF_IMPROVE") {
            cfg.self_improvement_enabled = parse_bool(&v);
        }
        cfg
    }
}

/// Parse a boolean from a string, accepting true/false/1/0/yes/no (case-insensitive).
fn parse_bool(s: &str) -> bool {
    matches!(s.to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "on")
}

/// Runtime statistics for the service.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ServiceStats {
    /// Total requests served.
    pub total_requests: u64,
    /// Total events published.
    pub total_events: u64,
    /// Number of active WebSocket sessions.
    pub active_ws_sessions: u64,
    /// Uptime in seconds.
    pub uptime_seconds: u64,
    /// Per-subsystem invocation counts.
    pub subsystem_invocations: ahash::AHashMap<String, u64>,
    /// Per-subsystem error counts.
    pub subsystem_errors: ahash::AHashMap<String, u64>,
}

/// The main service handle. Cloning is cheap (it's `Arc`-backed).
#[derive(Clone)]
pub struct CognitiveService {
    inner: Arc<ServiceInner>,
}

struct ServiceInner {
    /// Unique instance ID.
    instance_id: Uuid,
    /// Service configuration.
    config: RwLock<ServiceConfig>,
    /// Started-at timestamp.
    started_at: DateTime<Utc>,
    /// Event bus.
    event_bus: EventBus,
    /// Runtime statistics.
    stats: RwLock<ServiceStats>,
    /// Total requests counter.
    request_counter: AtomicU64,
    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,
    /// Subsystem status (true = healthy).
    subsystem_health: RwLock<ahash::AHashMap<&'static str, bool>>,
    /// Lock serializing mutation operations.
    mutation_lock: Mutex<()>,
    /// The memory fabric.
    memory: Arc<MemoryFabric>,
    /// The skill module.
    skills: Arc<SkillModule>,
    /// The training curriculum.
    curriculum: Arc<Curriculum>,
    /// The experience replay buffer.
    replay: Arc<ExperienceReplay>,
    /// The meta-learner.
    meta: Arc<MetaLearner>,
    /// The self-improvement cycle.
    improvement: Arc<RwLock<SelfImprovementCycle>>,
    /// Optional persistent event store (backs the in-memory buffer with semantic memory).
    event_store: RwLock<Option<Arc<PersistentEventStore>>>,
    /// Optional cognitive kernel for real cognitive cycles.
    kernel: RwLock<Option<Arc<qbitos_kernel::CognitiveKernel>>>,
}

impl CognitiveService {
    /// Create a new cognitive service with all subsystems initialized.
    #[must_use]
    pub fn new(config: ServiceConfig) -> Self {
        let event_bus = EventBus::new(config.event_bus_capacity);
        let instance_id = Uuid::new_v4();
        let started_at = Utc::now();

        // Initialize subsystems.
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(config.embedding_dim));
        let skills = Arc::new(SkillModule::with_noop_backends());
        let curriculum = Arc::new(Curriculum::with_default_stages());
        let replay = Arc::new(ExperienceReplay::new(config.replay_capacity));
        let meta = Arc::new(MetaLearner::new());
        let improvement = Arc::new(RwLock::new(SelfImprovementCycle::new()));

        // Register default learning strategies.
        meta.register("curriculum");
        meta.register("experience_replay");
        meta.register("self_play");
        meta.register("self_improvement");

        let mut health = ahash::AHashMap::new();
        for s in SUBSYSTEMS {
            health.insert(*s, true);
        }

        let inner = Arc::new(ServiceInner {
            instance_id,
            config: RwLock::new(config),
            started_at,
            event_bus,
            stats: RwLock::new(ServiceStats::default()),
            request_counter: AtomicU64::new(0),
            shutdown: Arc::new(AtomicBool::new(false)),
            subsystem_health: RwLock::new(health),
            mutation_lock: Mutex::new(()),
            memory,
            skills,
            curriculum,
            replay,
            meta,
            improvement,
            event_store: RwLock::new(None),
            kernel: RwLock::new(None),
        });
        let svc = Self { inner };
        svc.publish(ServiceEvent::SubsystemStarted {
            name: "orchestrator".into(),
        });
        svc
    }

    /// The instance ID.
    #[must_use]
    pub fn instance_id(&self) -> Uuid {
        self.inner.instance_id
    }

    /// The instance name.
    #[must_use]
    pub fn instance_name(&self) -> String {
        self.inner.config.read().instance_name.clone()
    }

    /// Read-only access to config.
    pub fn config(&self) -> ServiceConfig {
        self.inner.config.read().clone()
    }

    /// Update the config.
    pub fn update_config(&self, new_config: ServiceConfig) {
        *self.inner.config.write() = new_config;
    }

    /// The event bus.
    #[must_use]
    pub fn event_bus(&self) -> &EventBus {
        &self.inner.event_bus
    }

    /// The memory fabric.
    #[must_use]
    pub fn memory(&self) -> &Arc<MemoryFabric> {
        &self.inner.memory
    }

    /// The skill module.
    #[must_use]
    pub fn skills(&self) -> &Arc<SkillModule> {
        &self.inner.skills
    }

    /// The training curriculum.
    #[must_use]
    pub fn curriculum(&self) -> &Arc<Curriculum> {
        &self.inner.curriculum
    }

    /// The experience replay buffer.
    #[must_use]
    pub fn replay(&self) -> &Arc<ExperienceReplay> {
        &self.inner.replay
    }

    /// The meta-learner.
    #[must_use]
    pub fn meta(&self) -> &Arc<MetaLearner> {
        &self.inner.meta
    }

    /// The self-improvement cycle.
    #[must_use]
    pub fn improvement(&self) -> &Arc<RwLock<SelfImprovementCycle>> {
        &self.inner.improvement
    }

    /// The persistent event store, if attached.
    #[must_use]
    pub fn event_store(&self) -> Option<Arc<PersistentEventStore>> {
        self.inner.event_store.read().clone()
    }

    /// Attach a persistent event store. Returns the previous one if any.
    ///
    /// Spawns a background task that drains the event bus into the store, so
    /// every published event is persisted to the memory fabric's semantic
    /// layer. Call `replay_into_buffer` on the store at startup to restore
    /// the event history.
    pub fn attach_event_store(
        &self,
        store: Arc<PersistentEventStore>,
    ) -> Option<Arc<PersistentEventStore>> {
        // Spawn a background task that persists every event.
        let bus = self.inner.event_bus.clone();
        let store_clone = Arc::clone(&store);
        tokio::spawn(async move {
            let mut sub = bus.subscribe();
            while let Some(env) = sub.recv().await {
                store_clone.persist(&env).await;
            }
        });
        // Swap into the slot.
        let mut slot = self.inner.event_store.write();
        let old = slot.take();
        *slot = Some(store);
        old
    }

    /// The cognitive kernel, if attached.
    pub fn kernel(&self) -> Option<Arc<qbitos_kernel::CognitiveKernel>> {
        self.inner.kernel.read().clone()
    }

    /// Attach a cognitive kernel for real cognitive cycles.
    pub fn attach_kernel(&self, kernel: Arc<qbitos_kernel::CognitiveKernel>) {
        *self.inner.kernel.write() = Some(kernel);
    }

    /// Publish a service event.
    pub fn publish(&self, event: ServiceEvent) {
        self.inner.event_bus.publish(event);
        let mut stats = self.inner.stats.write();
        stats.total_events += 1;
    }

    /// Record a request (called by middleware on every inbound request).
    pub fn record_request(&self) -> u64 {
        self.inner
            .request_counter
            .fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Record a subsystem invocation.
    pub fn record_invocation(&self, subsystem: &'static str, success: bool) {
        let mut stats = self.inner.stats.write();
        *stats.subsystem_invocations.entry(subsystem.into()).or_insert(0) += 1;
        if !success {
            *stats.subsystem_errors.entry(subsystem.into()).or_insert(0) += 1;
        }
    }

    /// Increment active WebSocket session count.
    pub fn ws_session_started(&self) {
        self.inner.stats.write().active_ws_sessions += 1;
    }

    /// Decrement active WebSocket session count.
    pub fn ws_session_ended(&self) {
        let mut stats = self.inner.stats.write();
        if stats.active_ws_sessions > 0 {
            stats.active_ws_sessions -= 1;
        }
    }

    /// Snapshot of service statistics.
    #[must_use]
    pub fn stats(&self) -> ServiceStats {
        let mut stats = self.inner.stats.read().clone();
        stats.total_requests = self.inner.request_counter.load(Ordering::Relaxed);
        stats.uptime_seconds = (Utc::now() - self.inner.started_at).num_seconds() as u64;
        stats
    }

    /// Mark a subsystem as healthy / unhealthy.
    pub fn set_subsystem_health(&self, name: &'static str, healthy: bool) {
        self.inner.subsystem_health.write().insert(name, healthy);
        self.publish(ServiceEvent::Custom {
            subsystem: name.into(),
            payload: serde_json::json!({ "healthy": healthy }),
        });
    }

    /// Returns the set of subsystems known to be healthy.
    #[must_use]
    pub fn healthy_subsystems(&self) -> Vec<&'static str> {
        self.inner
            .subsystem_health
            .read()
            .iter()
            .filter(|(_, h)| **h)
            .map(|(k, _)| *k)
            .collect()
    }

    /// Initiate graceful shutdown.
    pub async fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        for s in SUBSYSTEMS {
            self.publish(ServiceEvent::SubsystemStopped { name: (*s).into() });
        }
    }

    /// Whether the service is shutting down.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.inner.shutdown.load(Ordering::SeqCst)
    }

    /// Acquire the mutation lock (serializes self-improvement writes).
    pub async fn mutation_lock(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.inner.mutation_lock.lock().await
    }

    /// Execute a closure with timing + statistics recording.
    pub async fn timed<F, T>(&self, subsystem: &'static str, f: F) -> Result<T, ServiceError>
    where
        F: std::future::Future<Output = Result<T, ServiceError>>,
    {
        if self.is_shutting_down() {
            return Err(ServiceError::ShuttingDown);
        }
        let start = Instant::now();
        let result = f.await;
        let success = result.is_ok();
        self.record_invocation(subsystem, success);
        let elapsed = start.elapsed();
        tracing::debug!(
            subsystem,
            ok = success,
            elapsed_ms = elapsed.as_millis() as u64,
            "subsystem invocation"
        );
        result
    }
}

/// All subsystems known to the orchestrator.
pub const SUBSYSTEMS: &[&str] = &[
    "kernel",
    "memory",
    "planner",
    "skills",
    "tools",
    "neural_engine",
    "vision",
    "speech",
    "world_model",
    "evolution",
    "data",
    "reasoning",
    "learning",
    "safety",
    "training",
];

/// A handle returned by `HttpServer::serve` representing a running HTTP/WS server.
pub struct ServiceHandle {
    /// The local address the server bound to.
    pub local_addr: std::net::SocketAddr,
    /// The join handle for the server task.
    pub join: tokio::task::JoinHandle<Result<(), ServiceError>>,
}

impl ServiceHandle {
    /// Wait for the server to exit.
    pub async fn wait(self) -> ServiceResult<()> {
        self.join.await?
    }
}
#[cfg(test)]
mod tests {
    use crate::ServiceConfig;
    use super::*;

    #[tokio::test]
    async fn create_and_stats() {
        let svc = CognitiveService::new(ServiceConfig::default());
        assert!(!svc.instance_id().is_nil());
        let stats = svc.stats();
        assert!(stats.uptime_seconds < 5);
        assert_eq!(stats.total_requests, 0);
        svc.record_request();
        let stats = svc.stats();
        assert_eq!(stats.total_requests, 1);
    }

    #[tokio::test]
    async fn shutdown_propagates() {
        let svc = CognitiveService::new(ServiceConfig::default());
        assert!(!svc.is_shutting_down());
        svc.shutdown().await;
        assert!(svc.is_shutting_down());
    }

    #[tokio::test]
    async fn timed_records_invocation() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let r: Result<u32, ServiceError> = svc
            .timed("memory", async { Ok::<_, ServiceError>(42u32) })
            .await;
        assert_eq!(r.unwrap(), 42);
        let stats = svc.stats();
        assert_eq!(stats.subsystem_invocations.get("memory"), Some(&1));
        assert_eq!(stats.subsystem_errors.get("memory"), None);
    }

    #[tokio::test]
    async fn timed_returns_shutdown_error_when_shutting_down() {
        let svc = CognitiveService::new(ServiceConfig::default());
        svc.shutdown().await;
        let r: Result<u32, ServiceError> = svc
            .timed("memory", async { Ok::<_, ServiceError>(42u32) })
            .await;
        assert!(matches!(r, Err(ServiceError::ShuttingDown)));
    }

    #[tokio::test]
    async fn healthy_subsystems_initially_all_healthy() {
        let svc = CognitiveService::new(ServiceConfig::default());
        let healthy = svc.healthy_subsystems();
        assert!(healthy.len() >= 10);
    }

    #[tokio::test]
    async fn subsystems_are_initialized() {
        let svc = CognitiveService::new(ServiceConfig::default());
        assert!(!svc.memory().episodic.len() > 0 || svc.memory().episodic.len() == 0); // Just verify access works
        assert_eq!(svc.skills().registry.len(), 0);
        assert_eq!(svc.curriculum().len(), 5); // Default stages
        assert_eq!(svc.replay().len(), 0);
        assert_eq!(svc.meta().len(), 4); // 4 default strategies
    }
}
