//! Composite tool module: bundles the registry, VM, executor, and
//! built-in tools into a single handle.

use std::sync::Arc;
use std::time::Duration;

use qbitos_memory::MemoryFabric;

use crate::executor::ToolExecutor;
use crate::permissions::{Permission, PermissionStore};
use crate::sandbox::Sandbox;
use crate::tool_vm::{ToolRegistry, ToolVm};
use crate::tools::filesystem::FilesystemTool;
use crate::tools::http::HttpTool;
use crate::tools::terminal::TerminalTool;

/// The composite tool module.
pub struct ToolModule {
    /// The tool registry.
    pub registry: Arc<ToolRegistry>,
    /// The permission store.
    pub permissions: Arc<PermissionStore>,
    /// The tool VM.
    pub vm: Arc<ToolVm>,
    /// The tool executor (with optional memory recording).
    pub executor: Arc<ToolExecutor>,
    /// The sandbox configuration.
    pub sandbox: Sandbox,
}

impl ToolModule {
    /// Build a new tool module with the default built-in tools and no memory
    /// recording.
    #[must_use]
    pub fn new() -> Self {
        let registry = Arc::new(ToolRegistry::new());
        let permissions = Arc::new(PermissionStore::new());
        let vm = Arc::new(ToolVm::new(Arc::clone(&registry), Arc::clone(&permissions)));
        let executor = Arc::new(ToolExecutor::new(Arc::clone(&vm)));
        Self { registry, permissions, vm, executor, sandbox: Sandbox::default() }
    }

    /// Register the default built-in tools (filesystem, terminal, http, calculator, summarizer, file_analyzer, etc.).
    pub fn with_builtin_tools(self) -> Self {
        let root = std::env::var("QBITOS_DATA_DIR").unwrap_or_else(|_| "/var/lib/qbitos".into());
        self.registry.register(Arc::new(FilesystemTool::new(root)));
        self.registry.register(Arc::new(TerminalTool::new(Duration::from_secs(30))));
        self.registry.register(Arc::new(HttpTool::default()));
        // 2026-era advanced tools
        self.registry.register(Arc::new(crate::tools::advanced::CalculatorTool));
        self.registry.register(Arc::new(crate::tools::advanced::SummarizerTool));
        self.registry.register(Arc::new(crate::tools::advanced::FileAnalyzerTool));
        self.registry.register(Arc::new(crate::tools::advanced::SentimentAnalyzerTool));
        self.registry.register(Arc::new(crate::tools::advanced::EntityExtractorTool));
        self.registry.register(Arc::new(crate::tools::advanced::TimeTool));
        // Extended tools (hash, base64, url_parser, json_query, statistics, web_scraper, diff, qr_generator, levenshtein)
        self.registry.register(Arc::new(crate::tools::extended::HashTool));
        self.registry.register(Arc::new(crate::tools::extended::Base64Tool));
        self.registry.register(Arc::new(crate::tools::extended::UrlParserTool));
        self.registry.register(Arc::new(crate::tools::extended::JsonQueryTool));
        self.registry.register(Arc::new(crate::tools::extended::StatisticsTool));
        self.registry.register(Arc::new(crate::tools::extended::WebScraperTool));
        self.registry.register(Arc::new(crate::tools::extended::DiffTool));
        self.registry.register(Arc::new(crate::tools::extended::QrGeneratorTool));
        self.registry.register(Arc::new(crate::tools::extended::LevenshteinTool));
        // 2026-era advanced tools
        self.registry.register(Arc::new(crate::tools::code_analyzer::CodeAnalyzerTool));
        self.registry.register(Arc::new(crate::tools::web_search::WebSearchTool::new()));
        self
    }

    /// Attach a memory fabric so tool outcomes are recorded.
    pub fn with_memory(self, memory: Arc<MemoryFabric>) -> Self {
        let executor = Arc::new(ToolExecutor::new(Arc::clone(&self.vm)).with_memory(memory));
        Self { executor, ..self }
    }

    /// Grant a permission to an agent.
    pub fn grant(&self, agent: uuid::Uuid, perm: Permission) {
        self.permissions.grant(agent, perm);
    }
}

impl Default for ToolModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_module_starts_empty() {
        let module = ToolModule::new();
        assert!(module.vm.registry().list().is_empty());
    }

    #[test]
    fn with_builtin_tools_registers_all() {
        let module = ToolModule::new().with_builtin_tools();
        let tools = module.vm.registry().list();
        assert!(!tools.is_empty(), "expected built-in tools to be registered");
        // Filesystem, terminal, http should always be present.
        assert!(tools.iter().any(|t| t.contains("filesystem")));
        assert!(tools.iter().any(|t| t.contains("terminal")));
        assert!(tools.iter().any(|t| t.contains("http")));
    }

    #[test]
    fn default_is_new() {
        let module = ToolModule::default();
        assert!(module.vm.registry().list().is_empty());
    }

    #[test]
    fn with_memory_returns_new_instance() {
        let memory = std::sync::Arc::new(qbitos_memory::MemoryFabric::with_deterministic_embeddings(64));
        let module = ToolModule::new().with_builtin_tools().with_memory(memory);
        // The module should still have its tools after attaching memory.
        assert!(!module.vm.registry().list().is_empty());
    }

    #[test]
    fn grant_permission_does_not_panic() {
        let module = ToolModule::new();
        let agent = uuid::Uuid::new_v4();
        let perm = crate::permissions::Permission::new("read", "/tmp/*");
        module.grant(agent, perm);
        // No assertion needed — just verify it doesn't panic.
    }
}
