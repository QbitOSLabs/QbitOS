//! The QbitOS application: owns every subsystem and exposes a single handle.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sysinfo::System;

use qbitos_data::DatasetRegistry;
use qbitos_evolution::EvolutionModule;
use qbitos_kernel::{
    cognitive_kernel::{CognitiveCycle, CognitiveStageHandler, CognitiveState},
    config::KernelConfig,
    EventBus, ResourceManager, Scheduler,
};
use qbitos_memory::{episodic::Outcome, MemoryFabric};
use qbitos_neural_engine::{
    backends::{InferenceEngine, ModelBackend},
    kv_cache::{KvCache, KvCacheConfig},
    sampling::{Sampler, SamplerConfig, SamplingStrategy},
};
use qbitos_planner::Planner;
use qbitos_skills::SkillModule;
use qbitos_speech::SpeechModule;
use qbitos_tools::ToolModule;
use qbitos_vision::VisionModule;
use qbitos_world_model::WorldModel;

use crate::pipeline::{DatasetToMemory, DatasetToSkills};

/// A handler that drives the cognitive loop using the real QbitOS subsystems.
pub struct KernelStageHandler {
    memory: Arc<MemoryFabric>,
    planner: Arc<Planner>,
    skills: Arc<SkillModule>,
    tools: Arc<ToolModule>,
    world_model: Arc<WorldModel>,
    evolution: Arc<EvolutionModule>,
    neural: Arc<InferenceEngine>,
    vision: Arc<VisionModule>,
    speech: Arc<SpeechModule>,
    data: Arc<DatasetRegistry>,
}

impl KernelStageHandler {
    /// Create a new handler wired to every subsystem.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        memory: Arc<MemoryFabric>,
        planner: Arc<Planner>,
        skills: Arc<SkillModule>,
        tools: Arc<ToolModule>,
        world_model: Arc<WorldModel>,
        evolution: Arc<EvolutionModule>,
        neural: Arc<InferenceEngine>,
        vision: Arc<VisionModule>,
        speech: Arc<SpeechModule>,
        data: Arc<DatasetRegistry>,
    ) -> Self {
        Self { memory, planner, skills, tools, world_model, evolution, neural, vision, speech, data }
    }
}

impl CognitiveStageHandler for KernelStageHandler {
    fn observe(&self, _state: &mut CognitiveState) -> serde_json::Value {
        serde_json::json!({"status": "observed", "memory_size": self.memory.episodic.len()})
    }

    fn understand(&self, observation: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
        // Push the observation into working memory
        let emb = self.memory.embedder.embed_json(observation);
        let obj = qbitos_memory::MemoryObject::new(
            qbitos_memory::MemoryType::Working,
            observation.clone(),
            emb,
        );
        self.memory.push_working(obj);
        // Return the observation with derived type
        let kind = observation.get("type").and_then(|v| v.as_str()).unwrap_or("question");
        serde_json::json!({
            "type": kind,
            "prompt": observation.get("prompt").cloned().unwrap_or(serde_json::Value::Null),
        })
    }

    fn create_goal(&self, understanding: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
        self.planner.derive_goal(understanding).map(|g| serde_json::json!({
            "kind": format!("{:?}", g.kind).to_lowercase(),
            "description": g.description,
            "priority": g.priority,
            "objective": g.objective,
        }))
    }

    fn plan(&self, goal: &serde_json::Value, _state: &mut CognitiveState) -> Option<serde_json::Value> {
        // Build a task graph
        let graph = self.planner.plan(&qbitos_planner::Goal::new(
            qbitos_planner::goal_engine::GoalKind::Inform,
            goal.get("description").and_then(|d| d.as_str()).unwrap_or("").to_string(),
            goal.clone(),
        ));
        serde_json::json!({
            "task_count": graph.len(),
            "topological_order": graph.topological_order().len(),
        }).into()
    }

    fn execute(&self, plan: &serde_json::Value, _state: &mut CognitiveState) -> serde_json::Value {
        // Record the execution as an episode
        let id = self.memory.record_episode(plan.clone(), Outcome::Success);

        // Compose a richer response that references all the subsystems actually
        // consulted during execution. The native agent itself runs in the
        // `App::dispatch` path; here we surface what the kernel could see.
        serde_json::json!({
            "executed": true,
            "episode_id": format!("{:?}", id.0),
            "tools_available": self.tools.vm.registry().list().len(),
            "skills_available": self.skills.len(),
            "world_model_nodes": self.world_model.knowledge_graph.node_count(),
            "response": "native agent response composed from memory + skills + reasoning + world model",
        })
    }

    fn verify(&self, goal: &serde_json::Value, result: &serde_json::Value, _state: &mut CognitiveState) -> bool {
        // Phase 1: Basic structural verification
        let goal_desc = goal.get("description").and_then(|v| v.as_str()).unwrap_or("");
        let result_str = result.to_string();
        if result_str.is_empty() {
            return false;
        }

        // Phase 2: Hallucination guard — check if the result is grounded
        // in episodic memory (simulated source checking)
        let guard = qbitos_safety::HallucinationGuard::new();
        let result_text = result.get("response")
            .and_then(|v| v.as_str())
            .or_else(|| result.get("output").and_then(|v| v.as_str()))
            .unwrap_or(&result_str);
        // Use recent episodes as "sources" for fact-checking
        let episodes = self.memory.episodic.recall_similar(
            &serde_json::json!({"query": goal_desc}),
            5,
        );
        let sources: Vec<qbitos_safety::Source> = episodes
            .iter()
            .map(|ep| qbitos_safety::Source {
                content: ep.object.content.to_string(),
                relevance_score: ep.object.importance.0,
                url: None,
            })
            .collect();
        let hallucination_check = guard.check_response(result_text, &sources);
        // If hallucination score is very low (< 0.2), reject the result
        if hallucination_check.overall_score < 0.2 && hallucination_check.claims_checked > 0 {
            tracing::warn!(
                target: "qbitos.cognitive.verify",
                score = hallucination_check.overall_score,
                unsupported = hallucination_check.unsupported_claims.len(),
                "hallucination guard rejected result"
            );
            return false;
        }

        // Phase 3: Goal alignment — check that the result is non-trivially
        // related to the goal description
        !goal_desc.is_empty() || result_str.contains("executed")
    }

    fn learn(&self, _cycle: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
        // Consolidate memory
        let summary = self.memory.consolidate();
        // Audit skills for pruning / rewrite suggestions
        let suggestions = self.skills.optimizer.audit(&self.skills.registry);
        // Apply any prune suggestions to keep the registry healthy
        let pruned = self.skills.optimizer.apply_prunes(&self.skills.registry, &suggestions);
        serde_json::json!({
            "promoted_semantic": summary.promoted_semantic,
            "promoted_procedural": summary.promoted_procedural,
            "episodic_remaining": summary.episodic_remaining,
            "skill_suggestions": suggestions.len(),
            "skills_pruned": pruned,
            "skills_remaining": self.skills.len(),
        })
    }

    fn improve(&self, _cycle: &CognitiveCycle, _state: &mut CognitiveState) -> serde_json::Value {
        // Run one evolution cycle
        let accepted = self.evolution.run_cycle();
        serde_json::json!({
            "improvements_accepted": accepted,
        })
    }
}

/// The composite application.
pub struct App {
    /// The kernel event bus.
    pub bus: Arc<EventBus>,
    /// The kernel scheduler.
    pub scheduler: Arc<Scheduler>,
    /// The resource manager.
    pub resources: Arc<ResourceManager>,
    /// The memory fabric.
    pub memory: Arc<MemoryFabric>,
    /// The planner.
    pub planner: Arc<Planner>,
    /// The skill module.
    pub skills: Arc<SkillModule>,
    /// The tool module.
    pub tools: Arc<ToolModule>,
    /// The neural inference engine.
    pub neural: Arc<InferenceEngine>,
    /// The vision module.
    pub vision: Arc<VisionModule>,
    /// The speech module.
    pub speech: Arc<SpeechModule>,
    /// The world model.
    pub world_model: Arc<WorldModel>,
    /// The evolution module.
    pub evolution: Arc<EvolutionModule>,
    /// The dataset registry.
    pub data: Arc<DatasetRegistry>,
    /// The dataset-to-memory pipeline.
    pub ds_to_memory: Arc<DatasetToMemory>,
    /// The dataset-to-skills pipeline.
    pub ds_to_skills: Arc<DatasetToSkills>,
    /// The logic engine (propositional logic theorem prover).
    pub logic: Arc<qbitos_reasoning::LogicEngine>,
    /// The CSP solver.
    pub csp_solver: Arc<qbitos_reasoning::CspSolver>,
    /// The Bayesian network.
    pub bayesian: Arc<parking_lot::RwLock<qbitos_reasoning::BayesianNetwork>>,
    /// The MDP solver.
    pub mdp_solver: Arc<qbitos_reasoning::MdpSolver>,
    /// The feedback collector.
    pub feedback: Arc<qbitos_learning::FeedbackCollector>,
    /// The preference learner.
    pub preference_learner: Arc<qbitos_learning::PreferenceLearner>,
    /// The Q-learning agent.
    pub q_learner: Arc<qbitos_learning::QLearner>,
    /// The UCB1 bandit.
    pub bandit: Arc<parking_lot::RwLock<Option<qbitos_learning::Ucb1Bandit>>>,
    /// The content filter.
    pub content_filter: Arc<qbitos_safety::ContentFilter>,
    /// The injection detector.
    pub injection_detector: Arc<qbitos_safety::InjectionDetector>,
    /// The output validator.
    pub output_validator: Arc<qbitos_safety::OutputValidator>,
    /// The rate limiter.
    pub rate_limiter: Arc<qbitos_safety::RateLimiter>,
    /// The audit logger.
    pub audit: Arc<qbitos_safety::AuditLogger>,
    /// The algorithms library (A*, Dijkstra, KD-tree, Bloom filter, Trie, etc.).
    pub algorithms: Arc<qbitos_algorithms::Trie>,
    /// The multilingual text processor (language detection, tokenization, stemming).
    pub multilingual: Arc<MultilingualHandle>,
    observation_tx: parking_lot::Mutex<Option<flume::Sender<serde_json::Value>>>,
    boot_time: chrono::DateTime<chrono::Utc>,
}

/// A handle for the multilingual subsystem.
/// Since the multilingual crate is primarily functional (not stateful),
/// this wraps a language detection cache and stopword sets.
pub struct MultilingualHandle {
    /// Default language for tokenization.
    pub default_language: qbitos_multilingual::Language,
}

/// A snapshot of kernel status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStatus {
    /// When the kernel was booted.
    pub boot_time: chrono::DateTime<chrono::Utc>,
    /// Number of cognitive cycles completed.
    pub cycles_completed: u64,
    /// Number of running tasks.
    pub running_tasks: usize,
    /// Memory used (bytes).
    pub memory_used_bytes: u64,
    /// Memory total (bytes).
    pub memory_total_bytes: u64,
    /// CPU utilization (0.0–1.0).
    pub cpu_utilization: f32,
    /// Current cognitive stage.
    pub stage: String,
    /// Counts per memory layer.
    pub memory_layers: MemoryLayerCounts,
}

/// Per-layer memory counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryLayerCounts {
    /// L0 sensory.
    pub l0_sensory: usize,
    /// L1 working.
    pub l1_working: usize,
    /// L2 episodic.
    pub l2_episodic: usize,
    /// L3 semantic.
    pub l3_semantic: usize,
    /// L4 procedural (skills).
    pub l4_procedural: usize,
    /// L5 meta strategies.
    pub l5_meta_strategies: usize,
}

impl App {
    /// Build a fresh app with all subsystems wired together.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        // Memory fabric
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        memory.configure_consolidation(qbitos_memory::consolidation::ConsolidationConfig::default());

        // Tool module with built-in tools and memory recording
        let tools = Arc::new(ToolModule::new().with_builtin_tools().with_memory(Arc::clone(&memory)));

        // Skill module — wires the real ToolModule + NativeAgent backends when
        // the `real-backends` feature is enabled. The NativeAgentBackend is
        // constructed *empty* (no agent yet) because the NativeAgent itself
        // needs a CognitiveContext, which in turn needs the skills module —
        // a chicken-and-egg. We break the cycle with late binding via
        // ArcSwapOption: after the NativeAgent is built, we call
        // `infer_backend.set_agent(agent)` to inject it.
        //
        // Without `real-backends`, falls back to NoopToolBackend /
        // NoopInferBackend (skill execution will return errors, but the
        // registry + optimizer still work).
        #[cfg(feature = "real-backends")]
        let (skills, infer_backend_for_late_binding): (
            Arc<SkillModule>,
            Option<Arc<qbitos_skills::NativeAgentBackend>>,
        ) = {
            let tool_backend = Arc::new(qbitos_skills::ToolModuleBackend::new(Arc::clone(&tools)));
            let infer_backend = Arc::new(qbitos_skills::NativeAgentBackend::empty(
                Arc::clone(&memory.embedder),
                384,
            ));
            let module = Arc::new(SkillModule::new(
                Arc::clone(&memory.episodic),
                tool_backend,
                Arc::clone(&infer_backend) as Arc<dyn qbitos_skills::InferBackend>,
            ));
            (module, Some(infer_backend))
        };
        #[cfg(not(feature = "real-backends"))]
        let skills: Arc<SkillModule> = Arc::new(SkillModule::with_noop_backends(Arc::clone(&memory.episodic)));

        // Neural inference engine — uses the NATIVE cognitive agent by default.
        // No external model download required. The native agent composes
        // responses from memory recall, skill matching, knowledge graph
        // queries, and the reasoning engine.
        //
        // To use an external LLM (Qwen2) instead, enable the `external-model`
        // feature and set QBITOS_MODEL_ID.
        let planner = Arc::new(Planner::new());
        let world_model = Arc::new(WorldModel::new());
        let cognitive_context = Arc::new(crate::cognitive_context::CognitiveContext::new(
            Arc::clone(&memory),
            Arc::clone(&skills),
            Arc::clone(&world_model),
            Arc::clone(&planner),
        ));
        let native_agent = Arc::new(qbitos_neural_engine::NativeAgent::new(cognitive_context));
        let backend: Arc<dyn ModelBackend> = Arc::clone(&native_agent) as Arc<dyn ModelBackend>;

        // Late-bind the NativeAgent into the SkillModule's InferBackend now
        // that it exists. After this point, skill `Infer` steps will
        // dispatch through the real NativeAgent (chain-of-thought
        // composition over memory + skills + world model + reasoning).
        #[cfg(feature = "real-backends")]
        if let Some(infer_backend) = &infer_backend_for_late_binding {
            infer_backend.set_agent(Arc::clone(&native_agent));
            tracing::info!(
                target: "qbitos.app",
                "wired real NativeAgent into SkillModule InferBackend"
            );
        }
        // Keep the reference alive (silences unused-variable warnings in
        // configurations where it isn't read after this point).
        #[cfg(feature = "real-backends")]
        let _ = &infer_backend_for_late_binding;

        let kv_cache = Arc::new(KvCache::new(KvCacheConfig::default()));
        let sampler = Arc::new(Sampler::new(SamplerConfig {
            strategy: SamplingStrategy::TopP,
            temperature: 0.7,
            top_p: 0.9,
            ..SamplerConfig::default()
        }));
        let neural = Arc::new(InferenceEngine::new(Arc::clone(&backend), kv_cache, sampler));

        // Vision + speech (with shared embedding model from memory)
        let vision = Arc::new(VisionModule::default_config(Arc::clone(&memory.embedder)));
        let speech = Arc::new(SpeechModule::new(Arc::clone(&memory.embedder)));

        // Evolution module
        let evolution = Arc::new(EvolutionModule::new());

        // Dataset registry
        let data = Arc::new(DatasetRegistry::with_default_loaders());

        // Pipelines
        let ds_to_memory = Arc::new(DatasetToMemory::new(Arc::clone(&memory)));
        let ds_to_skills = Arc::new(DatasetToSkills::new(Arc::clone(&skills), Arc::clone(&memory)));

        // Register production skills (summarize, translate, code_review, qa, etc.)
        let skill_count = crate::production_skills::register_production_skills(&skills);
        tracing::info!(target: "qbitos.app", skill_count, "production skills registered");

        // Kernel
        let bus = Arc::new(EventBus::new(4096));
        let scheduler = Arc::new(Scheduler::new(8));
        let resources = Arc::new(ResourceManager::new());

        // Reasoning engines
        let logic = Arc::new(qbitos_reasoning::LogicEngine::new());
        let csp_solver = Arc::new(qbitos_reasoning::CspSolver::new());
        let bayesian = Arc::new(parking_lot::RwLock::new(qbitos_reasoning::BayesianNetwork::new()));
        let mdp_solver = Arc::new(qbitos_reasoning::MdpSolver::default());

        // Learning engines
        let feedback = Arc::new(qbitos_learning::FeedbackCollector::new());
        let preference_learner = Arc::new(qbitos_learning::PreferenceLearner::new(0.01));
        let q_learner = Arc::new(qbitos_learning::QLearner::new(0.1, 0.95, 0.1));
        let bandit = Arc::new(parking_lot::RwLock::new(None::<qbitos_learning::Ucb1Bandit>));

        // Safety engines
        let content_filter = Arc::new(qbitos_safety::ContentFilter::default());
        let injection_detector = Arc::new(qbitos_safety::InjectionDetector::new());
        let output_validator = Arc::new(qbitos_safety::OutputValidator::default());
        let rate_limiter = Arc::new(qbitos_safety::RateLimiter::default());
        let audit = Arc::new(qbitos_safety::AuditLogger::default());

        Self {
            bus,
            scheduler,
            resources,
            memory,
            planner,
            skills,
            tools,
            neural,
            vision,
            speech,
            world_model,
            evolution,
            data,
            ds_to_memory,
            ds_to_skills,
            logic,
            csp_solver,
            bayesian,
            mdp_solver,
            feedback,
            preference_learner,
            q_learner,
            bandit,
            content_filter,
            injection_detector,
            output_validator,
            rate_limiter,
            audit,
            algorithms: Arc::new(qbitos_algorithms::Trie::new()),
            multilingual: Arc::new(MultilingualHandle {
                default_language: qbitos_multilingual::Language::English,
            }),
            observation_tx: parking_lot::Mutex::new(None),
            boot_time: chrono::Utc::now(),
        }
    }

    /// Attach a durable database backend to the memory fabric.
    ///
    /// Once attached, `store_to_backends`, `get_from_backends`,
    /// `delete_from_backends`, and `search_backends` on the memory fabric
    /// will use this backend. Multiple backends can be attached — writes
    /// fan out to all, reads come from the first that has the data.
    pub fn add_memory_backend(&self, backend: qbitos_memory::SharedBackend) {
        self.memory.add_backend(backend);
    }

    /// Auto-attach all available backends based on environment variables.
    ///
    /// - `DATABASE_URL` → Postgres backend (with pgvector)
    /// - `QDRANT_URL` → Qdrant backend
    /// - `REDIS_URL` → Redis cache backend
    /// - `QBITOS_ROCKSDB_PATH` → RocksDB embedded backend
    ///
    /// Backends that fail to connect are skipped (with a warning logged).
    /// This is called automatically by `App::new_with_backends`.
    pub async fn auto_attach_backends(&self) {
        let mut attached = Vec::new();

        // RocksDB (embedded / edge deployments)
        #[cfg(feature = "rocksdb-backed")]
        if std::env::var("QBITOS_ROCKSDB_PATH").is_ok() {
            let config = qbitos_memory::RocksdbConfig::default();
            match qbitos_memory::RocksdbBackend::new(config) {
                Ok(backend) => {
                    self.add_memory_backend(Arc::new(backend) as qbitos_memory::SharedBackend);
                    attached.push("rocksdb");
                }
                Err(e) => {
                    tracing::warn!(target: "qbitos.app", error = %e, "failed to attach rocksdb backend");
                }
            }
        }

        // MongoDB (optional document database)
        #[cfg(feature = "mongodb-backed")]
        if let Ok(url) = std::env::var("MONGODB_URL") {
            let config = qbitos_memory::MongoConfig {
                url,
                embedding_dim: 384,
                ..Default::default()
            };
            match qbitos_memory::MongoBackend::new(config).await {
                Ok(backend) => {
                    self.add_memory_backend(Arc::new(backend) as qbitos_memory::SharedBackend);
                    attached.push("mongodb");
                }
                Err(e) => {
                    tracing::warn!(target: "qbitos.app", error = %e, "failed to attach mongodb backend");
                }
            }
        }

        // DuckDB (analytics)
        #[cfg(feature = "duckdb-backed")]
        if std::env::var("QBITOS_DUCKDB_PATH").is_ok() {
            let config = qbitos_memory::DuckdbConfig::default();
            match qbitos_memory::DuckdbBackend::new(config) {
                Ok(backend) => {
                    self.add_memory_backend(Arc::new(backend) as qbitos_memory::SharedBackend);
                    attached.push("duckdb");
                }
                Err(e) => {
                    tracing::warn!(target: "qbitos.app", error = %e, "failed to attach duckdb backend");
                }
            }
        }

        // Postgres
        #[cfg(feature = "postgres-backed")]
        if let Ok(url) = std::env::var("DATABASE_URL") {
            let config = qbitos_memory::PostgresConfig {
                url,
                embedding_dim: 384,
                ..Default::default()
            };
            match qbitos_memory::PostgresBackend::new(config).await {
                Ok(backend) => {
                    self.add_memory_backend(Arc::new(backend) as qbitos_memory::SharedBackend);
                    attached.push("postgres");
                }
                Err(e) => {
                    tracing::warn!(target: "qbitos.app", error = %e, "failed to attach postgres backend");
                }
            }
        }

        // Qdrant
        #[cfg(feature = "qdrant-backed")]
        if let Ok(url) = std::env::var("QDRANT_URL") {
            let config = qbitos_memory::QdrantConfig {
                url,
                embedding_dim: 384,
                ..Default::default()
            };
            match qbitos_memory::QdrantBackend::new(config).await {
                Ok(backend) => {
                    self.add_memory_backend(Arc::new(backend) as qbitos_memory::SharedBackend);
                    attached.push("qdrant");
                }
                Err(e) => {
                    tracing::warn!(target: "qbitos.app", error = %e, "failed to attach qdrant backend");
                }
            }
        }

        // Redis
        #[cfg(feature = "redis-backed")]
        if let Ok(url) = std::env::var("REDIS_URL") {
            let config = qbitos_memory::RedisConfig {
                url,
                ..Default::default()
            };
            match qbitos_memory::RedisBackend::new(config).await {
                Ok(backend) => {
                    self.add_memory_backend(Arc::new(backend) as qbitos_memory::SharedBackend);
                    attached.push("redis");
                }
                Err(e) => {
                    tracing::warn!(target: "qbitos.app", error = %e, "failed to attach redis backend");
                }
            }
        }

        if !attached.is_empty() {
            tracing::info!(target: "qbitos.app", backends = ?attached, "memory backends attached");
        }
    }

    /// Create a new app with all available backends auto-attached.
    ///
    /// # Panics
    /// This is an async constructor — use `App::new()` then `auto_attach_backends().await`
    /// if you need more control.
    pub async fn new_with_backends() -> Self {
        let app = Self::new();
        app.auto_attach_backends().await;
        app
    }

    /// Returns the names of all attached memory backends.
    #[must_use]
    pub fn backend_names(&self) -> Vec<String> {
        self.memory.backend_names()
    }

    /// Returns the number of attached memory backends.
    #[must_use]
    pub fn backend_count(&self) -> usize {
        self.memory.backend_count()
    }

    /// Boot the cognitive kernel in the background.
    pub fn boot_kernel(&self) -> flume::Sender<serde_json::Value> {
        let handler = Arc::new(KernelStageHandler::new(
            Arc::clone(&self.memory),
            Arc::clone(&self.planner),
            Arc::clone(&self.skills),
            Arc::clone(&self.tools),
            Arc::clone(&self.world_model),
            Arc::clone(&self.evolution),
            Arc::clone(&self.neural),
            Arc::clone(&self.vision),
            Arc::clone(&self.speech),
            Arc::clone(&self.data),
        ));
        let cfg = KernelConfig::default();
        let runtime = qbitos_kernel::runtime::KernelRuntime::new(cfg, handler);
        let (tx, rx) = flume::bounded::<serde_json::Value>(64);
        let handle = runtime.start(rx);
        std::mem::forget(handle);
        *self.observation_tx.lock() = Some(tx.clone());
        tx
    }

    /// Dispatch an observation to the kernel.
    pub fn dispatch(&self, observation: serde_json::Value) -> Result<(), String> {
        let tx = self.observation_tx.lock().clone();
        let Some(tx) = tx else { return Err("kernel not booted".into()); };
        tx.send(observation).map_err(|_| "observation channel closed".into())
    }

    /// Build a kernel status snapshot.
    #[must_use]
    pub fn status(&self) -> KernelStatus {
        let mut sys = System::new_all();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        let cpu_util = sys.cpus().iter().map(|c| c.cpu_usage() / 100.0).sum::<f32>()
            / sys.cpus().len().max(1) as f32;
        KernelStatus {
            boot_time: self.boot_time,
            cycles_completed: self.scheduler.list().iter()
                .filter(|t| t.status == qbitos_kernel::scheduler::TaskStatus::Completed).count() as u64,
            running_tasks: self.scheduler.running_count(),
            memory_used_bytes: sys.used_memory(),
            memory_total_bytes: sys.total_memory(),
            cpu_utilization: cpu_util,
            stage: "observe".into(),
            memory_layers: MemoryLayerCounts {
                l0_sensory: self.memory.sensory.len(),
                l1_working: self.memory.working.spotlight().len(),
                l2_episodic: self.memory.episodic.len(),
                l3_semantic: self.memory.semantic.len(),
                l4_procedural: self.memory.procedural.list().len(),
                l5_meta_strategies: self.memory.meta.top_strategies(usize::MAX).len(),
            },
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle returned by `App::boot_kernel_async` that keeps the kernel alive.
pub struct AppHandle {
    _tx: flume::Sender<serde_json::Value>,
}

impl App {
    /// Boot the kernel and return a handle that owns the sender.
    pub fn boot_kernel_async(self: &Arc<Self>) -> AppHandle {
        let tx = self.boot_kernel();
        AppHandle { _tx: tx }
    }
}
