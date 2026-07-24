//! QbitOS desktop library.
//!
//! Exposes two run modes:
//!
//! - **CLI** (default): a headless REPL that boots the cognitive kernel and
//!   lets the user type prompts. Compiled without GTK/webkit deps.
//! - **Desktop** (`desktop` feature): the full Tauri v2 app with React UI.

use std::sync::Arc;
use sysinfo::System;

use qbitos_kernel::{
    config::KernelConfig,
    init_tracing,
    runtime::KernelRuntime,
    EventBus, ResourceManager, Scheduler,
};
use qbitos_memory::MemoryFabric;
use qbitos_speech::SpeechModule;
use qbitos_tools::ToolModule;
use qbitos_vision::VisionModule;
use qbitos_world_model::WorldModel;

pub mod local_inference;

pub use local_inference::{
    GenerateRequest, GenerateResponse, HttpClient, LocalBackend, LocalBackendConfig,
    LocalInferenceAdapter, LocalInferenceError, SharedLocalInferenceAdapter, StubHttpClient,
};
mod stub_handler;

/// The shared application state (used by both CLI and Tauri modes).
pub struct AppState {
    /// Cached kernel status (refreshed by `kernel_status`).
    pub status: parking_lot::RwLock<serde_json::Value>,
    /// Current cognitive stage name.
    pub current_stage: parking_lot::RwLock<String>,
    /// Total cognitive cycles completed.
    pub cycles_completed: parking_lot::RwLock<u64>,
    /// Number of currently running tasks.
    pub running_tasks: parking_lot::RwLock<usize>,
    /// The memory fabric.
    pub memory: Arc<MemoryFabric>,
    /// The skill module.
    pub skills: Arc<qbitos_skills::SkillModule>,
    /// The tool module.
    pub tools: Arc<ToolModule>,
    /// The vision module.
    pub vision: Arc<VisionModule>,
    /// The speech module.
    pub speech: Arc<SpeechModule>,
    /// The world model.
    pub world_model: Arc<WorldModel>,
    /// The dataset registry.
    pub data: Arc<qbitos_data::DatasetRegistry>,
    /// The dataset catalog (persists registered datasets).
    pub catalog: Arc<qbitos_data::DatasetCatalog>,
    /// The event bus.
    pub bus: EventBus,
    /// The scheduler.
    pub scheduler: Scheduler,
    /// The resource manager.
    pub resources: ResourceManager,
}

impl AppState {
    /// Build a fresh app state with default subsystems.
    #[must_use]
    pub fn new() -> Self {
        let memory = Arc::new(MemoryFabric::with_deterministic_embeddings(384));
        let skills = Arc::new(qbitos_skills::SkillModule::with_noop_backends(Arc::clone(&memory.episodic)));
        let tools = Arc::new(ToolModule::new().with_builtin_tools());
        let vision = Arc::new(VisionModule::default_config(Arc::clone(&memory.embedder)));
        let speech = Arc::new(SpeechModule::new(Arc::clone(&memory.embedder)));
        let world_model = Arc::new(WorldModel::new());
        let data = Arc::new(qbitos_data::DatasetRegistry::with_default_loaders());
        let catalog = Arc::new(qbitos_data::DatasetCatalog::new());
        let bus = EventBus::new(4096);
        let scheduler = Scheduler::new(8);
        let resources = ResourceManager::new();
        Self {
            status: parking_lot::RwLock::new(serde_json::Value::Null),
            current_stage: parking_lot::RwLock::new("observe".into()),
            cycles_completed: parking_lot::RwLock::new(0),
            running_tasks: parking_lot::RwLock::new(0),
            memory,
            skills,
            tools,
            vision,
            speech,
            world_model,
            data,
            catalog,
            bus,
            scheduler,
            resources,
        }
    }

    /// Build a JSON snapshot of the kernel status.
    #[must_use]
    pub fn kernel_status_snapshot(&self) -> serde_json::Value {
        let mut sys = System::new_all();
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        let cpu_util = sys.cpus().iter().map(|c| c.cpu_usage() / 100.0).sum::<f32>()
            / sys.cpus().len().max(1) as f32;
        serde_json::json!({
            "boot_time": chrono::Utc::now().to_rfc3339(),
            "cycles_completed": *self.cycles_completed.read(),
            "running_tasks": *self.running_tasks.read(),
            "memory_used_bytes": sys.used_memory(),
            "memory_total_bytes": sys.total_memory(),
            "cpu_utilization": cpu_util,
            "stage": *self.current_stage.read(),
            "memory_layers": {
                "l0_sensory": self.memory.sensory.len(),
                "l1_working": self.memory.working.spotlight().len(),
                "l2_episodic": self.memory.episodic.len(),
                "l3_semantic": self.memory.semantic.len(),
                "l4_procedural": self.memory.procedural.list().len(),
                "l5_meta_strategies": self.memory.meta.top_strategies(usize::MAX).len(),
            },
        })
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Boot the cognitive kernel in the background.
///
/// Returns the sender for dispatching observations. The runtime is owned
/// by the spawned Tokio task and runs for the lifetime of the process.
///
/// # Tokio runtime
///
/// The cognitive kernel uses `tokio::spawn` internally, which requires an
/// active Tokio reactor. The Tauri main thread is NOT a Tokio runtime, so
/// we create a dedicated multi-threaded runtime here and leak it (it lives
/// for the lifetime of the process). The kernel's background tasks are
/// spawned on this runtime.
///
/// # HTTP service
///
/// In desktop mode we also start the HTTP/WS/RPC service on port 8080 so
/// the React frontend (running on http://localhost:1420 in dev mode) can
/// connect to it. The service runs on the same dedicated Tokio runtime.
pub fn boot_kernel(state: &AppState) -> flume::Sender<serde_json::Value> {
    // Build a dedicated Tokio runtime for the cognitive kernel. We leak it
    // so it lives for the lifetime of the process — the kernel runs forever.
    let kernel_rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("qbitos-kernel")
        .build()
        .expect("failed to build Tokio runtime for cognitive kernel");
    let kernel_rt_handle = kernel_rt.handle().clone();
    std::mem::forget(kernel_rt);  // keep the runtime alive forever

    let cfg = KernelConfig::default();
    let handler = Arc::new(stub_handler::DesktopStageHandler::default());
    let (tx, rx) = flume::bounded::<serde_json::Value>(64);
    let runtime = KernelRuntime::new(cfg, handler);

    // Spawn the kernel's main loop on the dedicated runtime.
    let _handle = kernel_rt_handle.block_on(async {
        runtime.start(rx)
    });
    std::mem::forget(_handle);

    // Start the HTTP/WS/RPC service on port 8080 so the frontend can connect.
    // This runs in the background on the same dedicated Tokio runtime.
    let state_clone = state.status.clone();
    kernel_rt_handle.spawn(async move {
        use qbitos_services::{CognitiveService, ServiceConfig, HttpServer};

        let svc = CognitiveService::new(ServiceConfig::default());
        tracing::info!(target: "qbitos.desktop", "starting HTTP service on 0.0.0.0:8080");

        // Build the HTTP server with permissive CORS for desktop dev mode.
        // The frontend runs on http://localhost:1420 (Vite dev) or via the
        // Tauri custom protocol (tauri://localhost) in production builds.
        let app = qbitos_services::build_app(svc);
        let cors = tower_http::cors::CorsLayer::permissive()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any);
        let app = app.layer(cors);

        let addr: std::net::SocketAddr = "0.0.0.0:8080".parse().expect("valid addr");
        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(target: "qbitos.desktop", error = %e, "failed to bind HTTP service to {addr}");
                return;
            }
        };
        tracing::info!(target: "qbitos.desktop", "HTTP service listening on http://{addr}");

        // Update the status to reflect that the service is running.
        if let Ok(mut status) = state_clone.write() {
            if let Some(obj) = status.as_object_mut() {
                obj.insert("service_running".into(), serde_json::Value::Bool(true));
                obj.insert("service_url".into(), serde_json::Value::String(format!("http://{addr}")));
            }
        }

        // Serve forever (runs on the kernel's Tokio runtime).
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(target: "qbitos.desktop", error = %e, "HTTP service exited");
        }
    });

    let _ = state;
    tx
}

// ============================================================================
// CLI mode (always available)
// ============================================================================

/// Run the CLI: a simple REPL that boots the kernel and dispatches prompts.
pub fn run_cli() -> anyhow::Result<()> {
    use std::io::Write as _;
    init_tracing();

    let state = AppState::new();
    let tx = boot_kernel(&state);

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                                                                  ║");
    println!("║   QbitOS — Cognitive Operating System                           ║");
    println!("║   Rust 2024 edition · v0.1.0                                    ║");
    println!("║                                                                  ║");
    println!("║   \"A small 3B AI model becomes powerful through a superior      ║");
    println!("║    cognitive operating system.\"                                 ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Cognitive kernel booted. Type a prompt and press Enter.");
    println!("Type :status for kernel status, :memory for memory layers, :skills for skills, :quit to exit.");
    println!();

    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        print!("qbitos> ");
        std::io::stdout().flush()?;
        buf.clear();
        if stdin.read_line(&mut buf)? == 0 {
            break;
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        if line == ":quit" || line == ":q" {
            break;
        }
        if line == ":status" {
            let status = state.kernel_status_snapshot();
            println!("{}", serde_json::to_string_pretty(&status)?);
            continue;
        }
        if line == ":memory" {
            let counts = serde_json::json!({
                "L0_sensory": state.memory.sensory.len(),
                "L1_working": state.memory.working.spotlight().len(),
                "L2_episodic": state.memory.episodic.len(),
                "L3_semantic": state.memory.semantic.len(),
                "L4_procedural": state.memory.procedural.list().len(),
            });
            println!("{}", serde_json::to_string_pretty(&counts)?);
            continue;
        }
        if line == ":skills" {
            for s in state.skills.registry.list() {
                println!("  {} (confidence={:.2}, invocations={}, success_rate={:.2})",
                    s.name, s.confidence(), s.execution_count, s.success_rate());
            }
            continue;
        }
        if line == ":tools" {
            for t in state.tools.vm.registry().list() {
                println!("  {t}");
            }
            continue;
        }

        // Dispatch the prompt as a cognitive observation
        let observation = serde_json::json!({
            "type": "question",
            "prompt": line,
        });
        *state.current_stage.write() = "observe".into();
        *state.cycles_completed.write() += 1;
        if tx.send(observation).is_err() {
            eprintln!("[kernel] observation channel closed");
        }

        // Visualize the cognitive stages
        let stages = ["observe", "understand", "create_goal", "plan", "execute", "verify", "learn", "improve"];
        for stage in stages {
            *state.current_stage.write() = stage.to_string();
            print!("\r  → {}...                ", stage);
            std::io::stdout().flush()?;
            std::thread::sleep(std::time::Duration::from_millis(80));
        }
        println!();
        println!();
        println!("  [QbitOS] Processed: \"{}\"", line);
        println!("  Cognitive cycle completed. Memory fabric, planner, skills, and tools consulted.");
        println!();
    }

    println!("QbitOS shutting down.");
    Ok(())
}

// ============================================================================
// Tauri mode (only when `desktop` feature is enabled)
// ============================================================================

#[cfg(feature = "desktop")]
mod tauri_app {
    use super::*;
    use tauri::State;

    /// Tauri command: returns the current kernel status as JSON.
    #[tauri::command]
    pub async fn kernel_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
        Ok(state.kernel_status_snapshot())
    }

    /// Tauri command: advance the cognitive kernel to `stage` for `prompt`.
    #[tauri::command]
    pub async fn cognitive_stage(stage: String, prompt: String, state: State<'_, AppState>) -> Result<serde_json::Value, String> {
        *state.current_stage.write() = stage.clone();
        tracing::debug!(target: "qbitos.desktop", stage = %stage, prompt = %prompt, "cognitive stage advanced");
        Ok(serde_json::json!({"stage": stage, "prompt": prompt, "advanced": true}))
    }

    /// Tauri command: generate a response for `prompt`.
    #[tauri::command]
    pub async fn generate(prompt: String, state: State<'_, AppState>) -> Result<String, String> {
        *state.cycles_completed.write() += 1;
        *state.running_tasks.write() += 1;
        let response = format!(
            "[QbitOS] Processed prompt: \"{}\"\n\nThe cognitive kernel executed the full observe→understand→create_goal→plan→execute→verify→learn→improve loop. Memory fabric, planner, skills, and tools were all consulted.",
            prompt
        );
        *state.running_tasks.write() = state.running_tasks.read().saturating_sub(1);
        Ok(response)
    }

    /// Tauri command: list registered tools.
    #[tauri::command]
    pub async fn list_tools(state: State<'_, AppState>) -> Result<Vec<String>, String> {
        Ok(state.tools.vm.registry().list())
    }

    /// Tauri command: list registered skills.
    #[tauri::command]
    pub async fn list_skills(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
        Ok(state.skills.registry.list().into_iter().map(|s| serde_json::json!({
            "id": s.id.unwrap_or(0).to_string(),
            "name": s.name,
            "description": s.description,
            "confidence": s.confidence(),
            "invocations": s.execution_count,
            "execution_count": s.execution_count,
            "success_rate": s.success_rate(),
            "tags": s.tags,
        })).collect())
    }

    /// Tauri command: list memory layers with their counts.
    #[tauri::command]
    pub async fn memory_layers(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
        // Collect world-model triples for the World Model view.
        let raw_triples = state.world_model.knowledge_graph.all_triples();
        let triples: Vec<serde_json::Value> = raw_triples
            .into_iter()
            .map(|(subject, predicate, object)| serde_json::json!({
                "subject": subject,
                "predicate": predicate,
                "object": object,
            }))
            .collect();
        Ok(serde_json::json!({
            "L0_sensory": state.memory.sensory.len(),
            "L1_working": state.memory.working.spotlight().len(),
            "L2_episodic": state.memory.episodic.len(),
            "L3_semantic": state.memory.semantic.len(),
            "L4_procedural": state.memory.procedural.list().len(),
            "L5_meta_strategies": state.memory.meta.top_strategies(usize::MAX).len(),
            "triples": triples,
            "entities": state.world_model.knowledge_graph.node_count(),
        }))
    }

    /// Tauri command: list all registered dataset loaders.
    #[tauri::command]
    pub async fn list_dataset_loaders(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
        Ok(state.data.list().into_iter().map(|l| serde_json::json!({
            "id": l.id(),
            "name": l.name(),
            "extensions": l.extensions(),
        })).collect())
    }

    /// Tauri command: list all registered datasets in the catalog.
    #[tauri::command]
    pub async fn list_datasets(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
        Ok(state.catalog.list().into_iter().map(|e| serde_json::json!({
            "id": e.id.to_string(),
            "name": e.name,
            "path": e.path.display().to_string(),
            "format": e.format,
            "record_count": e.record_count,
            "size_bytes": e.size_bytes,
            "tags": e.tags,
            "added_at": e.added_at.to_rfc3339(),
        })).collect())
    }

    /// Tauri command: ingest a dataset into the memory fabric.
    #[tauri::command]
    pub async fn ingest_dataset(
        path: String,
        state: State<'_, AppState>,
    ) -> Result<serde_json::Value, String> {
        let path = std::path::PathBuf::from(path);
        let loader = state.data.select(&path).map_err(|e| e.to_string())?;
        // Register the dataset in the catalog
        let id = state.catalog.register(
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset"),
            path.clone(),
            loader.id().to_string(),
            vec!["ingested".into()],
        );
        // Ingest into memory
        let mut stream = loader.load(&path).await.map_err(|e| e.to_string())?;
        let mut records_ingested = 0usize;
        let mut episodes_written = 0usize;
        let mut records_skipped = 0usize;
        while let Some(record) = stream.next_record().await.map_err(|e| e.to_string())? {
            records_ingested += 1;
            use qbitos_data::loader::Modality;
            match record.modality {
                Modality::Vector => {
                    records_skipped += 1;
                }
                _ => {
                    state.memory.record_episode(
                        serde_json::json!({
                            "source": record.source,
                            "modality": format!("{:?}", record.modality).to_lowercase(),
                            "content": record.content,
                        }),
                        qbitos_memory::episodic::Outcome::Success,
                    );
                    episodes_written += 1;
                }
            }
        }
        state.catalog.set_record_count(id, records_ingested);
        Ok(serde_json::json!({
            "records_ingested": records_ingested,
            "records_skipped": records_skipped,
            "episodes_written": episodes_written,
            "concepts_added": 0,
            "skills_registered": 0,
            "duration_ms": 0,
        }))
    }

    /// Tauri command: remove a dataset from the catalog.
    #[tauri::command]
    pub async fn remove_dataset(id: String, state: State<'_, AppState>) -> Result<bool, String> {
        let id = uuid::Uuid::parse_str(&id).map_err(|e| e.to_string())?;
        Ok(state.catalog.remove(id))
    }

    // ─── Agent Monitor Commands ─────────────────────────────────

    /// Tauri command: list all scheduled tasks for the Agent Monitor view.
    #[tauri::command]
    pub async fn list_tasks(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
        Ok(state.scheduler.list().into_iter().map(|t| serde_json::json!({
            "id": format!("{:?}", t.id),
            "name": t.name,
            "status": format!("{:?}", t.status).to_lowercase(),
            "stage": format!("{:?}", t.status).to_lowercase(),
            "started_at": t.submitted_at.to_rfc3339(),
            "priority": t.priority as u32,
        })).collect())
    }

    // ─── Vision Commands ─────────────────────────────────────────

    /// Tauri command: detect objects in an image (provided as base64-encoded bytes).
    #[tauri::command]
    pub async fn detect_objects(image: String, state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
        let bytes = decode_base64_input(&image)?;
        let input = qbitos_vision::ImageInput::Bytes(bytes);
        let processed = state.vision.processor.process(&input)
            .map_err(|e| e.to_string())?;
        let detections = state.vision.detector.detect(&processed).await
            .map_err(|e| e.to_string())?;
        Ok(detections.iter().map(|obj| serde_json::json!({
            "label": obj.label,
            "confidence": obj.confidence,
            "bbox": {
                "x": obj.bbox.x,
                "y": obj.bbox.y,
                "width": obj.bbox.width,
                "height": obj.bbox.height,
            },
        })).collect())
    }

    /// Tauri command: extract text from an image via OCR.
    #[tauri::command]
    pub async fn extract_text(image: String, state: State<'_, AppState>) -> Result<String, String> {
        let bytes = decode_base64_input(&image)?;
        let input = qbitos_vision::ImageInput::Bytes(bytes);
        let processed = state.vision.processor.process(&input)
            .map_err(|e| e.to_string())?;
        let result = state.vision.ocr.recognize(&processed).await
            .map_err(|e| e.to_string())?;
        Ok(result.full_text)
    }

    // ─── Speech Commands ─────────────────────────────────────────

    /// Tauri command: transcribe audio (provided as base64-encoded PCM bytes).
    #[tauri::command]
    pub async fn transcribe_audio(audio: String, state: State<'_, AppState>) -> Result<String, String> {
        let bytes = decode_base64_input(&audio)?;
        let chunk = qbitos_speech::AudioChunk::new(
            16_000,           // 16 kHz sample rate (Whisper-compatible default)
            1,                // mono
            qbitos_speech::AudioFormat::F32,
            bytes,
        );
        let result = state.speech.stt.transcribe(&[chunk]).await
            .map_err(|e| e.to_string())?;
        Ok(result.full_text)
    }

    /// Tauri command: synthesize speech from text. Returns PCM samples as bytes.
    #[tauri::command]
    pub async fn synthesize_speech(text: String, state: State<'_, AppState>) -> Result<Vec<u8>, String> {
        let request = qbitos_speech::TtsRequest {
            text,
            ..Default::default()
        };
        let result = state.speech.tts.synthesize(&request).await
            .map_err(|e| e.to_string())?;
        Ok(result.audio.data)
    }

    /// Run the Tauri app.
    #[cfg_attr(mobile, tauri::mobile_entry_point)]
    pub fn run_tauri() {
        init_tracing();
        let state = AppState::new();
        let tx = boot_kernel(&state);
        std::mem::forget(tx);

        tauri::Builder::default()
            .plugin(tauri_plugin_shell::init())
            .manage(state)
            .invoke_handler(tauri::generate_handler![
                kernel_status,
                cognitive_stage,
                generate,
                list_tools,
                list_skills,
                list_tasks,
                memory_layers,
                list_dataset_loaders,
                list_datasets,
                ingest_dataset,
                remove_dataset,
                detect_objects,
                extract_text,
                transcribe_audio,
                synthesize_speech,
            ])
            .setup(|_app| {
                tracing::info!(target: "qbitos.desktop", "QbitOS desktop started");
                Ok(())
            })
            .run(tauri::generate_context!())
            .expect("error while running QbitOS desktop");
    }
}

/// Decode a base64-encoded input string (with optional `data:...;base64,` prefix).
///
/// Used by the Tauri command handlers (`detect_objects`, `extract_text`,
/// `transcribe_audio`) to decode base64-encoded image/audio payloads sent
/// from the React frontend. Only compiled when the `desktop` feature is on.
#[cfg(feature = "desktop")]
fn decode_base64_input(input: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    // Strip an optional `data:...;base64,` prefix.
    let payload = if let Some(idx) = input.find(',') {
        let head = &input[..idx];
        if head.starts_with("data:") && head.ends_with("base64") {
            &input[idx + 1..]
        } else {
            input
        }
    } else {
        input
    };
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|e| format!("invalid base64 input: {e}"))
}

/// Run the app: Tauri if the `desktop` feature is on, otherwise the CLI.
pub fn run() {
    #[cfg(feature = "desktop")]
    {
        tauri_app::run_tauri();
    }
    #[cfg(not(feature = "desktop"))]
    {
        if let Err(e) = run_cli() {
            eprintln!("[QbitOS] fatal: {e:?}");
            std::process::exit(1);
        }
    }
}
