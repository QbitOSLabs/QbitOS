//! Subcommands for the QbitOS CLI.

use std::path::PathBuf;
use std::sync::Arc;

use clap::Subcommand;
use colored::Colorize;

use qbitos_data::LoadOptions;

use crate::app::App;

/// All CLI subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Boot the cognitive kernel and drop into an interactive REPL.
    Repl,
    /// Show kernel status (memory layers, cycles, CPU/RAM).
    Status,
    /// Dataset operations.
    #[command(subcommand)]
    Dataset(DatasetCmd),
    /// Memory operations.
    #[command(subcommand)]
    Memory(MemoryCmd),
    /// Skill operations.
    #[command(subcommand)]
    Skill(SkillCmd),
    /// Tool operations.
    #[command(subcommand)]
    Tool(ToolCmd),
    /// World model operations.
    #[command(subcommand)]
    World(WorldCmd),
    /// Evolution engine operations.
    #[command(subcommand)]
    Evolution(EvolutionCmd),
    /// State persistence operations.
    #[command(subcommand)]
    State(StateCmd),
    /// Memory backend operations.
    #[command(subcommand)]
    Backend(BackendCmd),
    /// Reasoning engine operations.
    #[command(subcommand)]
    Reason(ReasonCmd),
    /// Safety operations.
    #[command(subcommand)]
    Safety(SafetyCmd),
    /// Learning operations.
    #[command(subcommand)]
    Learn(LearnCmd),
    /// Run a one-shot cognitive cycle on a prompt.
    Ask {
        /// The prompt to send to the kernel.
        prompt: String,
    },
    /// Language detection and multilingual text processing.
    #[command(subcommand)]
    Language(LanguageCmd),
    /// Algorithm operations (pathfinding, search, data structures).
    #[command(subcommand)]
    Algo(AlgoCmd),
    /// Start the HTTP/REST + WebSocket service layer.
    Serve {
        /// Bind address (e.g. "127.0.0.1:8080"). Defaults to env `QBITOS_SERVE_ADDR` or `127.0.0.1:8080`.
        #[arg(long, default_value = "127.0.0.1:8080", env = "QBITOS_SERVE_ADDR")]
        addr: String,
        /// Instance name. Defaults to env `QBITOS_INSTANCE_NAME` or `qbitos-cli`.
        #[arg(long, default_value = "qbitos-cli", env = "QBITOS_INSTANCE_NAME")]
        name: String,
        /// Disable safety enforcement.
        #[arg(long)]
        no_safety: bool,
        /// Enable self-improvement (writes to skill registry).
        #[arg(long)]
        allow_self_improve: bool,
        /// Enable Bearer-token authentication. Pass the token to register as "admin".
        /// When set, all non-public routes require `Authorization: Bearer <token>`.
        #[arg(long)]
        auth_token: Option<String>,
        /// Attach the cognitive kernel so `/api/v1/cognitive/cycle` dispatches
        /// through the real 8-stage cognitive loop.
        #[arg(long)]
        real_kernel: bool,
        /// Attach a persistent event store backed by the memory fabric's
        /// semantic layer (events survive across restarts). Honors env `QBITOS_PERSISTENT_EVENTS=true`.
        #[arg(long, env = "QBITOS_PERSISTENT_EVENTS")]
        persistent_events: bool,
        /// Use a restricted CORS policy (production-safe).
        #[arg(long)]
        restricted_cors: bool,
    },
}

/// Reasoning subcommands.
#[derive(Subcommand)]
pub enum ReasonCmd {
    /// Prove a proposition using resolution theorem proving.
    Prove {
        /// Proposition in string format: "a implies b" or "a and b".
        expression: String,
    },
    /// Solve a constraint satisfaction problem.
    Solve {
        /// Number of variables.
        n: usize,
    },
}

/// Safety subcommands.
#[derive(Subcommand)]
pub enum SafetyCmd {
    /// Check text for harmful content.
    Check {
        /// Text to check.
        text: String,
    },
    /// Detect prompt injection in input.
    Detect {
        /// Input to check for injection.
        text: String,
    },
    /// Validate an output string.
    Validate {
        /// Output to validate.
        text: String,
    },
    /// Show recent audit events.
    Audit,
}

/// Learning subcommands.
#[derive(Subcommand)]
pub enum LearnCmd {
    /// Add positive feedback for a target.
    Positive {
        /// The target (e.g., response id).
        target: String,
    },
    /// Add negative feedback for a target.
    Negative {
        /// The target.
        target: String,
    },
    /// Show feedback statistics.
    Stats {
        /// The target to show stats for.
        target: String,
    },
    /// Show all collected feedback.
    List,
}

/// Language subcommands.
#[derive(Subcommand)]
pub enum LanguageCmd {
    /// Detect the language of a text.
    Detect {
        /// The text to analyze.
        text: String,
    },
    /// Tokenize text into words.
    Tokenize {
        /// The text to tokenize.
        text: String,
    },
    /// Split text into sentences.
    Sentences {
        /// The text to split.
        text: String,
    },
    /// Stem an English word.
    Stem {
        /// The word to stem.
        word: String,
    },
    /// Transliterate Cyrillic to Latin.
    CyrillicToLatin {
        /// The Cyrillic text.
        text: String,
    },
    /// Transliterate Latin to Cyrillic.
    LatinToCyrillic {
        /// The Latin text.
        text: String,
    },
}

/// Algorithm subcommands.
#[derive(Subcommand)]
pub enum AlgoCmd {
    /// Compute the Levenshtein edit distance between two strings.
    Levenshtein {
        /// First string.
        a: String,
        /// Second string.
        b: String,
    },
    /// Create a Bloom filter and test membership.
    Bloom {
        /// Items to insert (comma-separated).
        items: String,
        /// Item to test.
        test: String,
    },
    /// Find the top-K largest numbers.
    TopK {
        /// Comma-separated numbers.
        numbers: String,
        /// K value.
        k: usize,
    },
    /// Compute statistics on a list of numbers.
    Stats {
        /// Comma-separated numbers.
        numbers: String,
    },
}

/// Memory backend subcommands.
#[derive(Subcommand)]
pub enum BackendCmd {
    /// List all attached memory backends.
    List,
    /// Show the number of attached backends.
    Count,
    /// Attach all available backends based on environment variables.
    Attach,
}

/// State persistence subcommands.
#[derive(Subcommand)]
pub enum StateCmd {
    /// Show the path to the state file.
    Path,
    /// Save current state to disk.
    Save,
    /// Load state from disk (replaces current in-memory state).
    Load,
    /// Show whether a state file exists.
    Exists,
    /// Show environment variable configuration (which QBITOS_* vars are set).
    Env,
}

/// Dataset subcommands.
#[derive(Subcommand)]
pub enum DatasetCmd {
    /// List all registered dataset loaders.
    ListLoaders,
    /// Inspect a dataset (format, schema, sample count).
    Inspect { path: PathBuf },
    /// Load a dataset and print records as JSONL.
    Load {
        path: PathBuf,
        #[arg(long, default_value = "10")]
        max: usize,
        #[arg(long, default_value = "0")]
        offset: usize,
    },
    /// Ingest a dataset into the memory fabric.
    Ingest {
        path: PathBuf,
        #[arg(long)]
        max: Option<usize>,
    },
    /// Mine skills from a dataset (agent trajectories / instructions).
    Mine {
        path: PathBuf,
        #[arg(long)]
        max: Option<usize>,
    },
}

/// Memory subcommands.
#[derive(Subcommand)]
pub enum MemoryCmd {
    /// Show counts for each memory layer.
    Stats,
    /// Recall the k most similar episodes to a query.
    Recall {
        query: String,
        #[arg(long, default_value = "5")]
        k: usize,
    },
    /// Run a consolidation cycle.
    Consolidate,
}

/// Skill subcommands.
#[derive(Subcommand)]
pub enum SkillCmd {
    /// List all registered skills.
    List,
    /// Show details of a specific skill.
    Show { name: String },
}

/// Tool subcommands.
#[derive(Subcommand)]
pub enum ToolCmd {
    /// List all registered tools.
    List,
}

/// World-model subcommands.
#[derive(Subcommand)]
pub enum WorldCmd {
    /// Show stats of the knowledge graph.
    Stats,
    /// Add a triple (subject, predicate, object).
    Add {
        subject: String,
        predicate: String,
        object: String,
    },
}

/// Evolution subcommands.
#[derive(Subcommand)]
pub enum EvolutionCmd {
    /// Run one evolution cycle.
    Run,
    /// List all benchmark results.
    Results,
}

/// Dispatch a subcommand against an app.
///
/// # Errors
/// Returns an error if the command fails.
pub async fn dispatch(app: &Arc<App>, command: Command) -> anyhow::Result<()> {
    match command {
        Command::Repl => crate::commands::repl::run(app).await,
        Command::Status => {
            let status = app.status();
            println!("{}", serde_json::to_string_pretty(&status)?);
            Ok(())
        }
        Command::Ask { prompt } => {
            // Boot the kernel inline so the one-shot ask works
            app.boot_kernel();
            crate::commands::repl::ask_once(app, &prompt).await
        }
        Command::Dataset(cmd) => handle_dataset(app, cmd).await,
        Command::Memory(cmd) => handle_memory(app, cmd).await,
        Command::Skill(cmd) => handle_skill(app, cmd).await,
        Command::Tool(cmd) => handle_tool(app, cmd).await,
        Command::World(cmd) => handle_world(app, cmd).await,
        Command::Evolution(cmd) => handle_evolution(app, cmd).await,
        Command::State(cmd) => handle_state(app, cmd).await,
        Command::Backend(cmd) => handle_backend(app, cmd).await,
        Command::Reason(cmd) => handle_reason(app, cmd).await,
        Command::Safety(cmd) => handle_safety(app, cmd).await,
        Command::Learn(cmd) => handle_learn(app, cmd).await,
        Command::Language(cmd) => handle_language(cmd).await,
        Command::Algo(cmd) => handle_algo(cmd).await,
        Command::Serve { addr, name, no_safety, allow_self_improve, auth_token, real_kernel, persistent_events, restricted_cors } => {
            handle_serve(
                app,
                addr,
                name,
                !no_safety,
                allow_self_improve,
                auth_token,
                real_kernel,
                persistent_events,
                restricted_cors,
            )
            .await
        }
    }
}

async fn handle_reason(app: &Arc<App>, cmd: ReasonCmd) -> anyhow::Result<()> {
    match cmd {
        ReasonCmd::Prove { expression } => {
            // Simple parser: "a implies b" → Implies(Atom("a"), Atom("b"))
            let parts: Vec<&str> = expression.split(" implies ").collect();
            let prop = if parts.len() == 2 {
                qbitos_reasoning::Proposition::Implies(
                    Box::new(qbitos_reasoning::Proposition::atom(parts[0].trim())),
                    Box::new(qbitos_reasoning::Proposition::atom(parts[1].trim())),
                )
            } else {
                qbitos_reasoning::Proposition::atom(expression.trim())
            };
            let result = app.logic.prove(&prop);
            println!("Provable: {}", result.proven);
            println!("Steps: {}", result.steps);
            println!("Clauses: {}", result.clauses.len());
        }
        ReasonCmd::Solve { n } => {
            println!("CSP solver with {n} variables — use the API for programmatic access");
        }
    }
    Ok(())
}

async fn handle_safety(app: &Arc<App>, cmd: SafetyCmd) -> anyhow::Result<()> {
    match cmd {
        SafetyCmd::Check { text } => {
            let result = app.content_filter.filter(&text);
            println!("Passed: {}", result.passed);
            println!("Risk score: {:.2}", result.risk_score);
            if !result.flagged_categories.is_empty() {
                println!("Flagged: {:?}", result.flagged_categories);
            }
            if let Some(sanitized) = &result.sanitized {
                println!("Sanitized: {sanitized}");
            }
        }
        SafetyCmd::Detect { text } => {
            let result = app.injection_detector.detect(&text);
            println!("Injection detected: {}", result.detected);
            println!("Risk score: {:.2}", result.risk_score);
            if !result.injection_types.is_empty() {
                println!("Types: {:?}", result.injection_types);
            }
            if result.detected {
                app.audit.warn("injection_detected", &format!("detected in: {text}"));
            }
        }
        SafetyCmd::Validate { text } => {
            let result = app.output_validator.validate(&text);
            println!("Valid: {}", result.valid);
            if !result.failed_rules.is_empty() {
                println!("Failed rules: {:?}", result.failed_rules);
                for err in &result.errors {
                    println!("  - {err}");
                }
            }
            println!("Passed rules: {:?}", result.passed_rules);
        }
        SafetyCmd::Audit => {
            let events = app.audit.events();
            if events.is_empty() {
                println!("(no audit events)");
            } else {
                println!("{} audit events:", events.len());
                for event in events.iter().take(20) {
                    println!("  [{:?}] {} — {}", event.level, event.category, event.message);
                }
            }
        }
    }
    Ok(())
}

async fn handle_learn(app: &Arc<App>, cmd: LearnCmd) -> anyhow::Result<()> {
    match cmd {
        LearnCmd::Positive { target } => {
            app.feedback.add(qbitos_learning::Feedback::positive(&target));
            app.audit.info("feedback", &format!("positive feedback for: {target}"));
            println!("Added positive feedback for: {target}");
        }
        LearnCmd::Negative { target } => {
            app.feedback.add(qbitos_learning::Feedback::negative(&target));
            app.audit.info("feedback", &format!("negative feedback for: {target}"));
            println!("Added negative feedback for: {target}");
        }
        LearnCmd::Stats { target } => {
            let stats = app.feedback.stats_for(&target);
            println!("Feedback for: {target}");
            println!("  Count: {}", stats.count);
            println!("  Mean score: {:.2}", stats.mean_score);
            println!("  Positive: {}", stats.positive);
            println!("  Negative: {}", stats.negative);
            if let Some(rating) = stats.avg_rating {
                println!("  Avg rating: {rating:.1}");
            }
        }
        LearnCmd::List => {
            let all = app.feedback.all();
            if all.is_empty() {
                println!("(no feedback collected)");
            } else {
                println!("{} feedback entries:", all.len());
                for f in all.iter().take(20) {
                    println!("  {} — {:?} — {}", f.timestamp.format("%H:%M:%S"), f.feedback_type, f.target);
                }
            }
        }
    }
    Ok(())
}

async fn handle_backend(app: &Arc<App>, cmd: BackendCmd) -> anyhow::Result<()> {
    match cmd {
        BackendCmd::List => {
            let names = app.backend_names();
            if names.is_empty() {
                println!("(no backends attached — run `qbitos backend attach` or set env vars)");
            } else {
                println!("{} ({} attached):", "Memory Backends".bold(), names.len());
                for name in &names {
                    println!("  - {name}");
                }
            }
        }
        BackendCmd::Count => {
            println!("{}", app.backend_count());
        }
        BackendCmd::Attach => {
            println!("{} attaching available backends...", "›".cyan());
            app.auto_attach_backends().await;
            let names = app.backend_names();
            if names.is_empty() {
                println!("  no backends could be attached");
                println!("  set one of: DATABASE_URL, QDRANT_URL, REDIS_URL, MONGODB_URL, QBITOS_ROCKSDB_PATH, QBITOS_DUCKDB_PATH");
            } else {
                println!("  {} backend(s) attached: {}", names.len(), names.join(", "));
            }
        }
    }
    Ok(())
}

async fn handle_state(app: &Arc<App>, cmd: StateCmd) -> anyhow::Result<()> {
    let persistence = crate::persistence::Persistence::new();
    match cmd {
        StateCmd::Path => {
            println!("{}", persistence.path().display());
        }
        StateCmd::Save => {
            persistence.save(app)?;
            println!("State saved to {}", persistence.path().display());
        }
        StateCmd::Load => {
            if !persistence.exists() {
                println!("(no state file at {})", persistence.path().display());
            } else {
                persistence.load(app)?;
                println!("State loaded from {}", persistence.path().display());
            }
        }
        StateCmd::Exists => {
            println!("{}", persistence.exists());
        }
        StateCmd::Env => {
            println!("{}\n", "QbitOS Environment Configuration".bold());
            let vars = [
                "QBITOS_DATA_DIR",
                "QBITOS_TOKEN",
                "RUST_LOG",
                "QBITOS_LOG_FORMAT",
                "DATABASE_URL",
                "QDRANT_URL",
                "REDIS_URL",
                "MONGODB_URL",
                "QBITOS_ROCKSDB_PATH",
                "QBITOS_DUCKDB_PATH",
                "QBITOS_MODEL_DIR",
                "HF_HOME",
                "QBITOS_SERVE_ADDR",
                "QBITOS_INSTANCE_NAME",
                "QBITOS_MAX_CONCURRENT_REQUESTS",
                "QBITOS_REQUEST_TIMEOUT_MS",
                "QBITOS_EVENT_BUS_CAPACITY",
                "QBITOS_REPLAY_CAPACITY",
                "QBITOS_EMBEDDING_DIM",
                "QBITOS_HNSW_EF_SEARCH",
                "QBITOS_PG_MAX_CONNECTIONS",
                "QBITOS_PG_CONNECT_TIMEOUT",
                "QBITOS_ALLOW_SELF_IMPROVE",
                "QBITOS_SAFETY_ENFORCED",
                "QBITOS_AUDIT_LOG",
                "QBITOS_PERSISTENT_EVENTS",
            ];
            for var in &vars {
                match std::env::var(var) {
                    Ok(val) => {
                        if val.is_empty() {
                            println!("  {:<35} (empty)", var);
                        } else if var.contains("TOKEN") || var.contains("PASSWORD") {
                            println!("  {:<35} {} (hidden)", var, "✓ set".green());
                        } else {
                            println!("  {:<35} {} = {}", var, "✓".green(), val);
                        }
                    }
                    Err(_) => {
                        println!("  {:<35} {}", var, "· unset".dimmed());
                    }
                }
            }
        }
    }
    Ok(())
}

async fn handle_dataset(app: &Arc<App>, cmd: DatasetCmd) -> anyhow::Result<()> {
    match cmd {
        DatasetCmd::ListLoaders => {
            println!("{:<15} {:<10} {}", "ID".bold(), "Extensions".bold(), "Name".bold());
            println!("{}", "-".repeat(80));
            for loader in app.data.list() {
                println!("{:<15} {:<10} {}", loader.id(), loader.extensions().join(","), loader.name());
            }
        }
        DatasetCmd::Inspect { path } => {
            let loader = app.data.select(&path)?;
            println!("{} {}", "Format:".bold(), loader.id());
            println!("{} {}", "Name:".bold(), loader.name());
            let schema = loader.detect_schema(&path).await?;
            if !schema.is_empty() {
                println!("{}", "Schema:".bold());
                for field in &schema.fields {
                    println!("  - {}: {:?}{}", field.name, field.field_type, if field.nullable { " (nullable)" } else { "" });
                }
            }
        }
        DatasetCmd::Load { path, max, offset } => {
            let loader = app.data.select(&path)?;
            let opts = LoadOptions::new().with_max_records(max).with_offset(offset);
            let mut stream = loader.load_with_options(&path, &opts).await?;
            while let Some(record) = stream.next_record().await? {
                println!("{}", serde_json::to_string(&record)?);
            }
        }
        DatasetCmd::Ingest { path, max } => {
            println!("{} ingesting {} → memory fabric", "›".cyan(), path.display());
            let result = app.ds_to_memory.ingest_path(&app.data, &path, max).await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        DatasetCmd::Mine { path, max } => {
            println!("{} mining skills from {}", "›".cyan(), path.display());
            let result = app.ds_to_skills.mine_from_path(&app.data, &path, max).await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
    }
    Ok(())
}

async fn handle_memory(app: &Arc<App>, cmd: MemoryCmd) -> anyhow::Result<()> {
    match cmd {
        MemoryCmd::Stats => {
            let status = app.status();
            println!("{}:", "Memory Layers".bold());
            println!("  L0 Sensory:     {}", status.memory_layers.l0_sensory);
            println!("  L1 Working:     {}", status.memory_layers.l1_working);
            println!("  L2 Episodic:    {}", status.memory_layers.l2_episodic);
            println!("  L3 Semantic:    {}", status.memory_layers.l3_semantic);
            println!("  L4 Procedural:  {}", status.memory_layers.l4_procedural);
            println!("  L5 Meta:        {}", status.memory_layers.l5_meta_strategies);
        }
        MemoryCmd::Recall { query, k } => {
            let episodes = app.memory.episodic.recall_similar(
                &serde_json::json!({"query": query}),
                k,
            );
            if episodes.is_empty() {
                println!("(no episodes found)");
            } else {
                for (i, ep) in episodes.iter().enumerate() {
                    println!("{} [{}] {}", "#".cyan(), i, ep.occurred_at);
                    println!("    {}", serde_json::to_string(&ep.object.content)?);
                }
            }
        }
        MemoryCmd::Consolidate => {
            let summary = app.memory.consolidate();
            println!("{}:", "Consolidation Summary".bold());
            println!("  Promoted to semantic:   {}", summary.promoted_semantic);
            println!("  Promoted to procedural: {}", summary.promoted_procedural);
            println!("  Episodic remaining:     {}", summary.episodic_remaining);
        }
    }
    Ok(())
}

async fn handle_skill(app: &Arc<App>, cmd: SkillCmd) -> anyhow::Result<()> {
    match cmd {
        SkillCmd::List => {
            let skills = app.skills.registry.list();
            if skills.is_empty() {
                println!("(no skills registered)");
            } else {
                println!("{:<30} {:<10} {:<12} {:<10}", "Name".bold(), "Conf.".bold(), "Invocations".bold(), "Success".bold());
                println!("{}", "-".repeat(70));
                for s in skills {
                    println!("{:<30} {:<10.2} {:<12} {:<10.2}",
                        s.name, s.confidence(), s.execution_count, s.success_rate());
                }
            }
        }
        SkillCmd::Show { name } => {
            let skill = app.skills.registry.get_by_name(&name)
                .ok_or_else(|| anyhow::anyhow!("skill `{name}` not found"))?;
            println!("{}: {}", "Name".bold(), skill.name);
            println!("{}: {}", "Purpose".bold(), skill.description);
            println!("{}: {:.2}", "Confidence".bold(), skill.confidence());
            println!("{}: {} (success rate: {:.2})", "Invocations".bold(), skill.execution_count, skill.success_rate());
            println!("{}: {}", "Tags".bold(), skill.tags.join(", "));
            println!("{}:", "Steps".bold());
            for (i, step) in skill.steps.iter().enumerate() {
                println!("  {}. {:?} — {}", i, step.kind, step.name);
            }
            println!("{}:", "Success Criteria".bold());
            for c in &skill.success_criteria {
                println!("  - {c}");
            }
        }
    }
    Ok(())
}

async fn handle_tool(app: &Arc<App>, cmd: ToolCmd) -> anyhow::Result<()> {
    match cmd {
        ToolCmd::List => {
            let tools = app.tools.vm.registry().list();
            if tools.is_empty() {
                println!("(no tools registered)");
            } else {
                println!("{}:", "Registered Tools".bold());
                for t in tools {
                    println!("  - {t}");
                }
            }
        }
    }
    Ok(())
}

async fn handle_world(app: &Arc<App>, cmd: WorldCmd) -> anyhow::Result<()> {
    match cmd {
        WorldCmd::Stats => {
            println!("{}: {} nodes, {} edges",
                "Knowledge Graph".bold(),
                app.world_model.knowledge_graph.node_count(),
                app.world_model.knowledge_graph.edge_count());
            println!("{}: {} entities",
                "Entity Engine".bold(),
                app.world_model.entity_engine.len());
        }
        WorldCmd::Add { subject, predicate, object } => {
            app.world_model.knowledge_graph.add(&subject, &predicate, &object);
            println!("Added: {subject} —{predicate}→ {object}");
        }
    }
    Ok(())
}

async fn handle_evolution(app: &Arc<App>, cmd: EvolutionCmd) -> anyhow::Result<()> {
    match cmd {
        EvolutionCmd::Run => {
            let accepted = app.evolution.run_cycle();
            println!("Evolution cycle complete: {accepted} improvements accepted");
        }
        EvolutionCmd::Results => {
            let results = app.evolution.evaluator.results();
            if results.is_empty() {
                println!("(no benchmark results yet)");
            } else {
                println!("{}:", "Benchmark Results".bold());
                for r in results {
                    println!("  [{}] benchmark {} — passed: {}",
                        r.run_at, r.benchmark_id, r.passed);
                }
            }
        }
    }
    Ok(())
}

pub mod repl;

/// Handle the `serve` subcommand: start the HTTP/REST + WebSocket service.
#[allow(clippy::too_many_arguments)]
async fn handle_serve(
    app: &Arc<App>,
    addr: String,
    name: String,
    safety_enforced: bool,
    allow_self_improve: bool,
    auth_token: Option<String>,
    real_kernel: bool,
    persistent_events: bool,
    restricted_cors: bool,
) -> anyhow::Result<()> {
    use qbitos_services::{
        AuthConfig, AuthService, CognitiveService, EventStoreConfig, HttpServer,
        PersistentEventStore, ServiceConfig,
    };
    use std::sync::Arc;

    let socket_addr: std::net::SocketAddr = addr
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid address {addr}: {e}"))?;

    let config = ServiceConfig {
        instance_name: name.clone(),
        safety_enforced,
        self_improvement_enabled: allow_self_improve,
        ..ServiceConfig::default()
    };

    let svc = CognitiveService::new(config);
    let stats = svc.stats();
    println!("{}", "QbitOS Service".cyan().bold());
    println!("  instance_id   : {}", svc.instance_id());
    println!("  instance_name : {name}");
    println!("  bind_addr     : {addr}");
    println!("  api_version   : {}", qbitos_services::API_VERSION);
    println!("  safety        : {safety_enforced}");
    println!("  self_improve  : {allow_self_improve}");
    println!("  auth          : {}", if auth_token.is_some() { "enabled (Bearer)" } else { "disabled" });
    println!("  real_kernel   : {real_kernel}");
    println!("  persist_events: {persistent_events}");
    println!("  restricted_cors: {restricted_cors}");
    println!("  subsystems    : {} healthy", svc.healthy_subsystems().len());
    let _ = stats;

    // Optionally attach the real cognitive kernel.
    if real_kernel {
        // Build a CognitiveKernel directly with the App's stage handler.
        // We construct a fresh kernel using the App's subsystems.
        let handler = std::sync::Arc::new(
            crate::app::KernelStageHandler::new(
                std::sync::Arc::clone(&app.memory),
                std::sync::Arc::clone(&app.planner),
                std::sync::Arc::clone(&app.skills),
                std::sync::Arc::clone(&app.tools),
                std::sync::Arc::clone(&app.world_model),
                std::sync::Arc::clone(&app.evolution),
                std::sync::Arc::clone(&app.neural),
                std::sync::Arc::clone(&app.vision),
                std::sync::Arc::clone(&app.speech),
                std::sync::Arc::clone(&app.data),
            ),
        ) as std::sync::Arc<dyn qbitos_kernel::cognitive_kernel::CognitiveStageHandler>;
        let bus = qbitos_kernel::event_system::EventBus::new(1024);
        let scheduler = qbitos_kernel::scheduler::Scheduler::new(16);
        let resources = qbitos_kernel::resource_manager::ResourceManager::new();
        let kernel = std::sync::Arc::new(qbitos_kernel::cognitive_kernel::CognitiveKernel::new(
            bus,
            scheduler,
            resources,
            handler,
        ));
        svc.attach_kernel(kernel);
        println!("  {} attached real cognitive kernel", "→".cyan().bold());
    }

    // Optionally attach the persistent event store.
    if persistent_events {
        let memory = svc.memory().clone();
        let store = Arc::new(PersistentEventStore::new(
            memory,
            EventStoreConfig::default(),
        ));
        // Replay any persisted events from previous sessions.
        let replayed = store.replay_into_buffer().await;
        svc.attach_event_store(store);
        println!(
            "  {} attached persistent event store (replayed {replayed} events)",
            "→".cyan().bold()
        );
    }

    // Build the HTTP server with optional auth.
    let mut server = HttpServer::new(svc.clone());
    if let Some(token) = auth_token {
        let mut auth_config = AuthConfig {
            enabled: true,
            ..AuthConfig::default()
        };
        // If real_kernel is enabled, allow public read-only access to GETs.
        if !real_kernel {
            auth_config.allow_public_read = false;
        }
        let auth = AuthService::new(auth_config);
        auth.register_token("admin", token);
        server = server.with_auth(auth);
        println!("  {} registered auth principal 'admin'", "→".cyan().bold());
    }
    if restricted_cors {
        server = server.with_restricted_cors();
        println!("  {} using restricted CORS policy", "→".cyan().bold());
    }

    let handle = server.serve(socket_addr).await?;
    println!(
        "{} listening on http://{}",
        "→".cyan().bold(),
        handle.local_addr
    );
    println!(
        "{} OpenAPI spec: http://{}/api/v1/openapi.json",
        "→".cyan().bold(),
        handle.local_addr
    );
    println!(
        "{} WebSocket:    ws://{}/api/v1/events/ws",
        "→".cyan().bold(),
        handle.local_addr
    );
    println!("{}", "Press Ctrl+C to stop.".dimmed());

    handle.wait().await?;
    Ok(())
}


/// Handle the `language` subcommand.
async fn handle_language(cmd: LanguageCmd) -> anyhow::Result<()> {
    use qbitos_multilingual as ml;
    match cmd {
        LanguageCmd::Detect { text } => {
            let scores = ml::detect_language_with_scores(&text);
            println!("Language detection results:");
            for s in &scores {
                println!("  {:10} ({}): {:.4}", s.language.name(), s.language.code(), s.score);
            }
            if let Some(top) = scores.first() {
                println!("\nDetected: {} ({})", top.language.name(), top.language.code());
            }
        }
        LanguageCmd::Tokenize { text } => {
            let tokens = ml::word_tokenize(&text);
            println!("Tokens ({}):", tokens.len());
            for t in &tokens {
                println!("  [{}, {}) {}", t.start, t.end, t.text);
            }
        }
        LanguageCmd::Sentences { text } => {
            let sentences = ml::sentence_tokenize(&text);
            println!("Sentences ({}):", sentences.len());
            for (i, s) in sentences.iter().enumerate() {
                println!("  {}. {}", i + 1, s);
            }
        }
        LanguageCmd::Stem { word } => {
            let stemmed = ml::stem_english(&word);
            println!("{} → {}", word, stemmed);
        }
        LanguageCmd::CyrillicToLatin { text } => {
            let latin = ml::cyrillic_to_latin(&text);
            println!("{} → {}", text, latin);
        }
        LanguageCmd::LatinToCyrillic { text } => {
            let cyrillic = ml::latin_to_cyrillic(&text);
            println!("{} → {}", text, cyrillic);
        }
    }
    Ok(())
}

/// Handle the `algo` subcommand.
async fn handle_algo(cmd: AlgoCmd) -> anyhow::Result<()> {
    use qbitos_algorithms as alg;
    match cmd {
        AlgoCmd::Levenshtein { a, b } => {
            let dist = alg::levenshtein_distance(&a, &b);
            let max_len = a.len().max(b.len());
            let similarity = if max_len == 0 { 1.0 } else { 1.0 - dist as f64 / max_len as f64 };
            println!("Levenshtein distance: {}", dist);
            println!("Similarity: {:.4}", similarity);
            println!("  a: {}", a);
            println!("  b: {}", b);
        }
        AlgoCmd::Bloom { items, test } => {
            let items: Vec<&str> = items.split(',').map(|s| s.trim()).collect();
            let mut bf = alg::BloomFilter::new(items.len().max(10), 0.01);
            for item in &items {
                bf.insert(item);
            }
            println!("Bloom filter: {} items, {} bits, {} hashes, {:.4}% estimated FP rate",
                bf.len(), bf.memory_bytes() * 8, 0, bf.estimated_fp_rate() * 100.0);
            println!("Testing '{}': {}", test, if bf.contains(&test) { "possibly present" } else { "definitely absent" });
        }
        AlgoCmd::TopK { numbers, k } => {
            let nums: Vec<i64> = numbers.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let mut tracker = alg::TopK::new(k);
            tracker.extend(nums.clone());
            let top = tracker.into_sorted_vec();
            println!("Top-{} of {:?}:", k, nums);
            for (i, v) in top.iter().enumerate() {
                println!("  {}. {}", i + 1, v);
            }
        }
        AlgoCmd::Stats { numbers } => {
            let nums: Vec<f64> = numbers.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if nums.is_empty() {
                println!("No valid numbers provided.");
                return Ok(());
            }
            let n = nums.len();
            let sum: f64 = nums.iter().sum();
            let mean = sum / n as f64;
            let variance: f64 = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let std_dev = variance.sqrt();
            let mut sorted = nums.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            };
            println!("Statistics for {:?}:", nums);
            println!("  count:    {}", n);
            println!("  sum:      {:.4}", sum);
            println!("  mean:     {:.4}", mean);
            println!("  median:   {:.4}", median);
            println!("  std_dev:  {:.4}", std_dev);
            println!("  min:      {:.4}", sorted.first().unwrap());
            println!("  max:      {:.4}", sorted.last().unwrap());
            println!("  range:    {:.4}", sorted.last().unwrap() - sorted.first().unwrap());
        }
    }
    Ok(())
}
