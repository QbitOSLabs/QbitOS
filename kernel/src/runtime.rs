//! Kernel runtime.
//!
//! Owns the Tokio runtime, drives the cognitive loop, and exposes a
//! [`RuntimeHandle`] for graceful shutdown.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::cognitive_kernel::{CognitiveKernel, CognitiveStageHandler};
use crate::config::KernelConfig;
use crate::event_system::EventBus;
use crate::resource_manager::ResourceManager;
use crate::scheduler::Scheduler;

/// A handle returned by [`KernelRuntime::start`] that can be used to shut the
/// runtime down.
#[derive(Clone)]
pub struct RuntimeHandle {
    shutdown_tx: watch::Sender<bool>,
    join_handle: Arc<std::sync::Mutex<Option<JoinHandle<()>>>>,
}

impl RuntimeHandle {
    /// Request a graceful shutdown. The runtime will finish the current
    /// cognitive cycle and then exit.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Wait for the runtime to fully exit.
    ///
    /// # Errors
    /// Returns the underlying [`JoinError`] if the runtime task panicked.
    pub async fn join(&self) -> Result<(), tokio::task::JoinError> {
        // Scope the lock guard so it's dropped before the .await below —
        // holding a std::sync::Mutex across an await point can deadlock.
        // Use lock().ok() to avoid panicking on poison (a panicked runtime
        // task leaves the mutex poisoned; we still want to join it).
        let handle = { self.join_handle.lock().ok().and_then(|mut g| g.take()) };
        if let Some(h) = handle {
            h.await?;
        }
        Ok(())
    }
}

/// The kernel runtime. Owns the cognitive kernel and drives the loop.
pub struct KernelRuntime {
    config: KernelConfig,
    bus: EventBus,
    scheduler: Scheduler,
    resources: ResourceManager,
    handler: Arc<dyn CognitiveStageHandler>,
}

impl KernelRuntime {
    /// Create a new runtime with the given configuration and stage handler.
    #[must_use]
    pub fn new(config: KernelConfig, handler: Arc<dyn CognitiveStageHandler>) -> Self {
        let bus = EventBus::new(config.event_bus_capacity);
        let scheduler = Scheduler::new(config.max_concurrent_tasks);
        let resources = ResourceManager::new();
        resources.register_agent(uuid::Uuid::now_v7(), config.agent_memory_budget_bytes);
        Self { config, bus, scheduler, resources, handler }
    }

    /// Get a reference to the event bus.
    #[must_use]
    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    /// Get a reference to the scheduler.
    #[must_use]
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Get a reference to the resource manager.
    #[must_use]
    pub fn resources(&self) -> &ResourceManager {
        &self.resources
    }

    /// Start the runtime. Returns a [`RuntimeHandle`] for shutdown.
    ///
    /// The runtime spawns a background task that ticks the cognitive loop at
    /// `config.tick_rate` and runs cycles whenever the kernel receives an
    /// observation via the `observations` channel.
    pub fn start(
        self,
        observations: flume::Receiver<serde_json::Value>,
    ) -> RuntimeHandle {
        let kernel = Arc::new(CognitiveKernel::new(
            self.bus.clone(),
            self.scheduler.clone(),
            self.resources.clone(),
            self.handler,
        ));

        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
        let tick_rate = self.config.tick_rate;
        let poll_resources = self.resources.clone();

        let join_handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(tick_rate);
            loop {
                tokio::select! {
                    biased;
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!(target: "qbitos.kernel", "shutdown requested, draining");
                            break;
                        }
                    }
                    _ = ticker.tick() => {
                        poll_resources.poll(Duration::from_secs(1));
                    }
                    obs = observations.recv_async() => {
                        match obs {
                            Ok(value) => {
                                let kernel = Arc::clone(&kernel);
                                tokio::spawn(async move {
                                    let _cycle = kernel.run_cycle(value).await;
                                });
                            }
                            Err(flume::RecvError::Disconnected) => {
                                tracing::info!(target: "qbitos.kernel", "observation channel closed, exiting");
                                break;
                            }
                        }
                    }
                }
            }
            poll_resources.poll(Duration::ZERO);
            let _ = self.scheduler.shutdown(Duration::from_secs(5)).await;
        });

        RuntimeHandle {
            shutdown_tx,
            join_handle: Arc::new(std::sync::Mutex::new(Some(join_handle))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_runtime() -> KernelRuntime {
        let config = KernelConfig::default();
        let handler = Arc::new(StubHandler);
        KernelRuntime::new(config, handler)
    }

    struct StubHandler;
    impl CognitiveStageHandler for StubHandler {
        fn observe(&self, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"status": "ok"})
        }
        fn understand(&self, _observation: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"type": "test"})
        }
        fn create_goal(&self, _understanding: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
            Some(serde_json::json!({"goal": "test"}))
        }
        fn plan(&self, _goal: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
            Some(serde_json::json!({"plan": "test"}))
        }
        fn execute(&self, _plan: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"executed": true})
        }
        fn verify(&self, _goal: &serde_json::Value, _result: &serde_json::Value, _state: &mut CognitiveState) -> bool {
            true
        }
        fn learn(&self, _cycle: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"learned": true})
        }
        fn improve(&self, _cycle: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
            serde_json::json!({"improved": true})
        }
    }

    #[test]
    fn runtime_construction() {
        let runtime = make_runtime();
        assert_eq!(runtime.bus().len(), 0);
    }

    #[test]
    fn runtime_exposes_scheduler() {
        let runtime = make_runtime();
        assert_eq!(runtime.scheduler().max_concurrent(), 8); // KernelConfig::default
    }

    #[test]
    fn runtime_exposes_resources() {
        let runtime = make_runtime();
        let _ = runtime.resources();
    }

    #[tokio::test]
    async fn runtime_start_returns_handle() {
        let runtime = make_runtime();
        let (tx, rx) = flume::bounded::<serde_json::Value>(4);
        let handle = runtime.start(rx);
        // Drop the sender to trigger the runtime to exit.
        drop(tx);
        // The handle should be joinable.
        let _ = handle.join().await;
    }
}
