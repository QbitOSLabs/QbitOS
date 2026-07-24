//! Configuration for the cognitive kernel.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Top-level kernel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Human-readable instance name (used in logs and event metadata).
    pub instance_name: String,
    /// Persistent state directory (memory snapshots, skill cache, telemetry).
    pub data_dir: PathBuf,
    /// Tick rate of the cognitive loop.
    pub tick_rate: Duration,
    /// Maximum concurrent cognitive tasks the scheduler will admit.
    pub max_concurrent_tasks: usize,
    /// Maximum number of events the bus will buffer before back-pressuring.
    pub event_bus_capacity: usize,
    /// Hard cap on resident memory per cognitive agent (bytes).
    pub agent_memory_budget_bytes: u64,
    /// Whether to enable the reflection engine (default: true).
    pub enable_reflection: bool,
    /// Whether to enable the evolution engine (default: false; requires sandbox).
    pub enable_evolution: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            instance_name: "qbitos-0".into(),
            data_dir: PathBuf::from("/var/lib/qbitos"),
            tick_rate: Duration::from_millis(50),
            max_concurrent_tasks: 8,
            event_bus_capacity: 4_096,
            agent_memory_budget_bytes: 512 * 1024 * 1024,
            enable_reflection: true,
            enable_evolution: false,
        }
    }
}

impl KernelConfig {
    /// Load configuration from a TOML file, falling back to [`Default`] when
    /// the file is missing or unreadable.
    ///
    /// # Errors
    /// Returns an error only if the file exists but is malformed TOML.
    pub fn load_or_default(path: &std::path::Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&raw)?;
        Ok(cfg)
    }

    /// Persist the configuration to a TOML file.
    ///
    /// # Errors
    /// Propagates IO / serialization errors.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let raw = toml::to_string_pretty(self)?;
        std::fs::write(path, raw)?;
        Ok(())
    }
}
