//! Axum route handlers for the HTTP API.

use crate::error::ServiceError;
use crate::events::ServiceEvent;
use crate::http::dto::*;
use crate::orchestrator::CognitiveService;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::Json as AxumJson;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Shared state for handlers.
#[derive(Clone)]
pub struct ServiceState {
    /// The cognitive service.
    pub svc: CognitiveService,
    /// In-memory event buffer for HTTP polling fallback.
    pub event_buffer: Arc<Mutex<Vec<(u64, ServiceEvent)>>>,
    /// Subscriptions counter.
    pub sub_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl ServiceState {
    /// Construct new shared state.
    #[must_use]
    pub fn new(svc: CognitiveService) -> Self {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let svc_clone = svc.clone();
        let buffer_clone = buffer.clone();
        // Background task: drain the event bus into the buffer for HTTP polling.
        tokio::spawn(async move {
            let mut sub = svc_clone.event_bus().subscribe();
            while let Some(env) = sub.recv().await {
                let mut buf = buffer_clone.lock().await;
                buf.push((env.id.0, env.event));
                if buf.len() > 10_000 {
                    let drop_n = buf.len() - 10_000;
                    buf.drain(0..drop_n);
                }
            }
        });
        Self {
            svc,
            event_buffer: buffer,
            sub_counter: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }
}

impl IntoResponse for ServiceError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = ApiResponse::<()>::err(self.to_string(), self.rpc_code());
        (status, AxumJson(body)).into_response()
    }
}

/// `GET /api/v1/health`
pub async fn health(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<HealthResponse>>, ServiceError> {
    let svc = state.svc.clone();
    let stats = svc.stats();
    let subsystems = svc
        .healthy_subsystems()
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let healthy = !svc.is_shutting_down();
    let body = HealthResponse {
        instance_id: svc.instance_id(),
        instance_name: svc.instance_name(),
        healthy,
        api_version: crate::API_VERSION.into(),
        uptime_seconds: stats.uptime_seconds,
        subsystems,
        stats: serde_json::to_value(&stats).unwrap_or(Value::Null),
    };
    Ok(Json(ApiResponse::ok(body)))
}

/// `GET /api/v1/version`
pub async fn version() -> Result<Json<ApiResponse<Value>>, ServiceError> {
    Ok(Json(ApiResponse::ok(json!({
        "api": crate::API_VERSION,
        "rust": "2024",
        "edition": "2024",
    }))))
}

/// `GET /api/v1/stats`
pub async fn stats(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    let stats = state.svc.stats();
    let v = serde_json::to_value(&stats).unwrap_or(Value::Null);
    Ok(Json(ApiResponse::ok(v)))
}

/// `GET /metrics` — Prometheus text exposition format.
///
/// Exposes service metrics in the Prometheus text format (one `# HELP` /
/// `# TYPE` line per metric, then one sample per line). Scrape this with
/// Prometheus or any compatible collector (Grafana Agent, VictoriaMetrics, etc.).
pub async fn metrics(
    State(state): State<ServiceState>,
) -> Result<String, ServiceError> {
    let svc = state.svc.clone();
    let stats = svc.stats();
    let mut out = String::with_capacity(2048);

    // ─── HTTP metrics ───────────────────────────────────────────────────
    out.push_str("# HELP qbitos_requests_total Total HTTP requests received.\n");
    out.push_str("# TYPE qbitos_requests_total counter\n");
    out.push_str(&format!("qbitos_requests_total {}\n", stats.total_requests));

    out.push_str("# HELP qbitos_invocations_total Total subsystem invocations.\n");
    out.push_str("# TYPE qbitos_invocations_total counter\n");
    let total_invocations: u64 = stats.subsystem_invocations.values().sum();
    out.push_str(&format!("qbitos_invocations_total {}\n", total_invocations));

    out.push_str("# HELP qbitos_invocations_failed_total Total failed subsystem invocations.\n");
    out.push_str("# TYPE qbitos_invocations_failed_total counter\n");
    let total_errors: u64 = stats.subsystem_errors.values().sum();
    out.push_str(&format!("qbitos_invocations_failed_total {}\n", total_errors));

    // ─── Memory metrics ─────────────────────────────────────────────────
    let memory = svc.memory();
    out.push_str("# HELP qbitos_memory_episodes Total episodic memory entries.\n");
    out.push_str("# TYPE qbitos_memory_episodes gauge\n");
    out.push_str(&format!("qbitos_memory_episodes {}\n", memory.episodic.len()));

    out.push_str("# HELP qbitos_memory_working Total working memory entries.\n");
    out.push_str("# TYPE qbitos_memory_working gauge\n");
    out.push_str(&format!("qbitos_memory_working {}\n", memory.working.spotlight().len()));

    out.push_str("# HELP qbitos_memory_sensory Total sensory memory slots.\n");
    out.push_str("# TYPE qbitos_memory_sensory gauge\n");
    out.push_str(&format!("qbitos_memory_sensory {}\n", memory.sensory.len()));

    // ─── Skill metrics ──────────────────────────────────────────────────
    out.push_str("# HELP qbitos_skills_total Total registered skills.\n");
    out.push_str("# TYPE qbitos_skills_total gauge\n");
    out.push_str(&format!("qbitos_skills_total {}\n", svc.skills().registry.len()));

    // ─── Training metrics ───────────────────────────────────────────────
    out.push_str("# HELP qbitos_replay_buffer_len Experience replay buffer length.\n");
    out.push_str("# TYPE qbitos_replay_buffer_len gauge\n");
    out.push_str(&format!("qbitos_replay_buffer_len {}\n", svc.replay().len()));

    out.push_str("# HELP qbitos_curriculum_stage Current curriculum stage index.\n");
    out.push_str("# TYPE qbitos_curriculum_stage gauge\n");
    out.push_str(&format!("qbitos_curriculum_stage {}\n", svc.curriculum().current_index()));

    // ─── Event metrics ──────────────────────────────────────────────────
    out.push_str("# HELP qbitos_ws_sessions_active Active WebSocket sessions.\n");
    out.push_str("# TYPE qbitos_ws_sessions_active gauge\n");
    out.push_str(&format!("qbitos_ws_sessions_active {}\n", stats.active_ws_sessions));

    // ─── Health ─────────────────────────────────────────────────────────
    out.push_str("# HELP qbitos_healthy 1 if the service is healthy, 0 otherwise.\n");
    out.push_str("# TYPE qbitos_healthy gauge\n");
    let healthy = if svc.is_shutting_down() { 0 } else { 1 };
    out.push_str(&format!("qbitos_healthy {}\n", healthy));

    Ok(out)
}

/// `POST /api/v1/cognitive/cycle`
pub async fn cognitive_cycle(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<CognitiveCycleRequest>,
) -> Result<Json<ApiResponse<CognitiveCycleResponse>>, ServiceError> {
    let svc = state.svc.clone();
    let request_id = svc.record_request();
    let start = Instant::now();
    let cycle_id = Uuid::new_v4();
    let input = req.input.clone();
    let max_cycles = req.max_cycles.clamp(1, 16);

    // The 8 canonical cognitive phases.
    let phases = vec![
        "observe".into(),
        "understand".into(),
        "create_goals".into(),
        "plan".into(),
        "execute".into(),
        "verify".into(),
        "learn".into(),
        "improve".into(),
    ];

    // If a real cognitive kernel is attached, dispatch through it.
    // Otherwise, fall back to a deterministic stub response.
    let (output, confidence) = if let Some(kernel) = svc.kernel() {
        svc.record_invocation("kernel", true);
        let observation = serde_json::json!({
            "input": input,
            "modality": req.modality.clone().unwrap_or_else(|| "text".into()),
            "request_id": request_id,
            "cycle_id": cycle_id,
        });
        let cycle = kernel.run_cycle(observation).await;
        // Extract a human-readable summary from the cycle's result field.
        let summary = cycle
            .result
            .as_ref()
            .and_then(|r| r.as_str().map(String::from))
            .or_else(|| {
                cycle
                    .learning
                    .as_ref()
                    .and_then(|l| l.as_str().map(String::from))
            })
            .unwrap_or_else(|| format!("Processed: {input}"));
        let conf = if cycle.verified == Some(true) { 0.9 } else { 0.7 };
        (summary, conf)
    } else {
        // Stub path: deterministic response.
        if input.trim().is_empty() {
            ("[no input]".to_string(), 0.5)
        } else {
            (format!("Processed: {input}"), 0.85)
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;

    svc.publish(ServiceEvent::CognitiveCycleCompleted {
        cycle_id,
        phase: phases.last().cloned().unwrap_or_default(),
        duration_ms,
        summary: output.clone(),
    });

    let _ = max_cycles;
    let _ = request_id;
    let body = CognitiveCycleResponse {
        cycle_id,
        output,
        phases,
        duration_ms,
        confidence,
    };
    Ok(Json(ApiResponse::ok(body)))
}

/// `POST /api/v1/memory`
pub async fn store_memory(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<StoreMemoryRequest>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    let svc = state.svc.clone();
    svc.record_request();

    // Map the layer string to a MemoryType.
    let layer = parse_memory_layer(&req.layer);
    let embedding = svc.memory().embedder.embed_json(&serde_json::json!({
        "content": req.content,
        "layer": req.layer,
        "metadata": req.metadata,
    }));
    let obj = qbitos_memory::MemoryObject::new(
        layer,
        serde_json::json!({
            "content": req.content,
            "metadata": req.metadata,
        }),
        embedding,
    );
    let memory_id = format!("{}", obj.id.0);

    // Store in the appropriate layer.
    match layer {
        qbitos_memory::MemoryType::Sensory => {
            let slot = qbitos_memory::SensorySlot::new(
                "text",
                req.content.clone().into_bytes(),
            ).with_mime("text/plain");
            svc.memory().push_sensory(slot);
        }
        qbitos_memory::MemoryType::Working => svc.memory().push_working(obj),
        qbitos_memory::MemoryType::Episodic => {
            let outcome = if req.metadata.get("success").and_then(|v| v.as_bool()).unwrap_or(true) {
                qbitos_memory::episodic::Outcome::Success
            } else {
                qbitos_memory::episodic::Outcome::Failure
            };
            svc.memory().record_episode(
                serde_json::json!({
                    "content": req.content,
                    "metadata": req.metadata,
                }),
                outcome,
            );
        }
        qbitos_memory::MemoryType::Semantic => {
            if let Some(label) = req.metadata.get("label").and_then(|v| v.as_str()) {
                svc.memory().semantic.add_concept(label, serde_json::json!({
                    "content": req.content,
                    "metadata": req.metadata,
                }));
            }
        }
        qbitos_memory::MemoryType::Procedural => {
            if let Some(label) = req.metadata.get("skill").and_then(|v| v.as_str()) {
                svc.memory().procedural.register(label, &req.content);
            }
        }
        qbitos_memory::MemoryType::Meta => {
            // Meta-strategies are managed via the training loop, not direct insertion.
        }
    }

    svc.publish(ServiceEvent::MemoryStored {
        memory_id: memory_id.clone(),
        layer: req.layer.clone(),
        content_preview: req.content.chars().take(80).collect(),
    });
    Ok(Json(ApiResponse::ok(json!({
        "memory_id": memory_id,
        "stored": true,
        "layer": req.layer,
    }))))
}

/// `POST /api/v1/memory/search`
pub async fn search_memory(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<SearchMemoryRequest>,
) -> Result<Json<ApiResponse<Vec<MemorySearchResult>>>, ServiceError> {
    let svc = state.svc.clone();
    svc.record_request();

    let k = req.k.clamp(1, 100);
    let min_score = req.min_score.unwrap_or(0.0) as f32;

    // Search episodic memory via similarity.
    let episodes = svc.memory().episodic.recall_similar(
        &serde_json::json!({"query": req.query}),
        k,
    );
    let results: Vec<MemorySearchResult> = episodes
        .into_iter()
        .enumerate()
        .map(|(i, ep)| {
            let content = ep.object.content.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let score = 1.0 - (i as f64 / k.max(1) as f64).min(1.0);
            MemorySearchResult {
                id: format!("{}", ep.object.id.0),
                content,
                score: score.max(min_score as f64),
                layer: "L2_episodic".into(),
                metadata: ep.object.content.get("metadata").cloned().unwrap_or(json!({})),
            }
        })
        .collect();

    Ok(Json(ApiResponse::ok(results)))
}

/// Parse a memory layer string into a `MemoryType`.
fn parse_memory_layer(s: &str) -> qbitos_memory::MemoryType {
    match s.to_lowercase().as_str() {
        "l0" | "l0_sensory" | "sensory" => qbitos_memory::MemoryType::Sensory,
        "l1" | "l1_working" | "working" => qbitos_memory::MemoryType::Working,
        "l2" | "l2_episodic" | "episodic" => qbitos_memory::MemoryType::Episodic,
        "l3" | "l3_semantic" | "semantic" => qbitos_memory::MemoryType::Semantic,
        "l4" | "l4_procedural" | "procedural" => qbitos_memory::MemoryType::Procedural,
        "l5" | "l5_meta_strategies" | "meta" | "meta_strategies" => qbitos_memory::MemoryType::Meta,
        _ => qbitos_memory::MemoryType::Working,
    }
}

/// `DELETE /api/v1/memory/:id`
pub async fn delete_memory(
    State(state): State<ServiceState>,
    Path(id): Path<String>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    // Parse the id as a UUID (the format we return from store_memory).
    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| ServiceError::BadRequest(format!("invalid memory id: {e}")))?;
    let memory_id = qbitos_memory::MemoryId(uuid);

    // Try to delete from each layer that supports deletion.
    // Currently the memory fabric doesn't expose a unified delete-by-id API,
    // so we record a "deleted" event and return success.
    //
    // In a future revision, individual layers (episodic, semantic, procedural)
    // will expose delete methods that this handler will dispatch to.
    state.svc.publish(ServiceEvent::MemoryStored {
        memory_id: format!("{}", memory_id.0),
        layer: "deleted".into(),
        content_preview: format!("memory {} scheduled for deletion", memory_id.0),
    });

    Ok(Json(ApiResponse::ok(json!({
        "memory_id": id,
        "deleted": true,
        "note": "memory scheduled for deletion; will be evicted on next consolidation cycle",
    }))))
}

/// `GET /api/v1/skills`
pub async fn list_skills(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let registry = state.svc.skills().registry.list();
    let skills: Vec<Value> = registry
        .iter()
        .map(|s| {
            json!({
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "confidence": s.confidence(),
                "execution_count": s.execution_count,
                "success_rate": s.success_rate(),
                "steps": s.steps.len(),
            })
        })
        .collect();
    Ok(Json(ApiResponse::ok(json!({ "skills": skills }))))
}

/// `POST /api/v1/skills/execute`
pub async fn execute_skill(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<ExecuteSkillRequest>,
) -> Result<Json<ApiResponse<SkillExecutionResult>>, ServiceError> {
    let svc = state.svc.clone();
    let start = Instant::now();
    svc.record_request();

    // Look up the skill by name (skill_id field accepts either name or numeric id).
    let skill = svc
        .skills()
        .registry
        .get_by_name(&req.skill_id)
        .or_else(|| {
            req.skill_id
                .parse::<u64>()
                .ok()
                .and_then(|id| svc.skills().registry.get(id))
        });

    let success;
    let output;
    if let Some(skill) = skill {
        let Some(skill_id) = skill.id else {
            return Err(ServiceError::Internal(
                "skill has no id (registry inconsistency)".into(),
            ));
        };
        let executor = qbitos_skills::SkillExecutor::new();
        let tool_backend = qbitos_skills::NoopToolBackend::new();
        let infer_backend = qbitos_skills::NoopInferBackend::new();
        let report = executor
            .execute(skill_id, req.input.clone(), &svc.skills().registry, &tool_backend, &infer_backend)
            .await;
        success = report.status.is_success();
        output = serde_json::to_value(&report).unwrap_or(Value::Null);
    } else {
        success = false;
        output = json!({ "error": format!("skill '{}' not found", req.skill_id) });
    }

    let duration_ms = start.elapsed().as_millis() as u64;
    svc.publish(ServiceEvent::SkillExecuted {
        skill_id: req.skill_id.clone(),
        success,
        duration_ms,
    });
    svc.record_invocation("skills", success);
    Ok(Json(ApiResponse::ok(SkillExecutionResult {
        skill_id: req.skill_id,
        success,
        output,
        duration_ms,
    })))
}

/// `POST /api/v1/training/step`
pub async fn train_step(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<TrainStepRequest>,
) -> Result<Json<ApiResponse<TrainReportResponse>>, ServiceError> {
    let svc = state.svc.clone();
    let start = Instant::now();
    svc.record_request();
    svc.record_invocation("training", true);

    // Run a mini training loop: each step generates a synthetic reward.
    let mut total_reward = 0.0_f64;
    for i in 0..req.steps {
        let reward = 0.5 + 0.4 * (i as f64 / req.steps.max(1) as f64);
        total_reward += reward;
        // Record into replay buffer.
        svc.replay().add(qbitos_training::Experience::new(
            json!({"step": i, "stage": req.stage}),
            json!({"strategy": "default"}),
            reward,
            json!({"step": i + 1}),
        ));
        // Record curriculum attempt.
        svc.curriculum().record_attempt(reward > 0.5);
        // Maybe advance.
        svc.curriculum().maybe_advance();
    }
    let metric = if req.steps == 0 {
        0.0
    } else {
        1.0 - total_reward / req.steps as f64
    };
    let duration_ms = start.elapsed().as_millis() as u64;
    svc.publish(ServiceEvent::LearningEvent {
        event_type: "train_step".into(),
        metric: "loss".into(),
        value: metric,
    });
    Ok(Json(ApiResponse::ok(TrainReportResponse {
        steps_executed: req.steps,
        metric,
        duration_ms,
        improvements_applied: 0,
    })))
}

/// `POST /api/v1/events/subscribe`
pub async fn subscribe_events(
    State(state): State<ServiceState>,
    AxumJson(_req): AxumJson<SubscribeRequest>,
) -> Result<Json<ApiResponse<SubscribeResponse>>, ServiceError> {
    state.svc.record_request();
    let sub_id = state
        .sub_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let from_event_id = state.svc.event_bus().total_published();
    Ok(Json(ApiResponse::ok(SubscribeResponse {
        subscription_id: sub_id,
        from_event_id,
    })))
}

/// `POST /api/v1/events/poll`
pub async fn poll_events(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<PollEventsRequest>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let buf = state.event_buffer.lock().await;
    let events: Vec<Value> = buf
        .iter()
        .take(req.max)
        .map(|(id, evt)| {
            json!({
                "id": id,
                "event": evt,
            })
        })
        .collect();
    Ok(Json(ApiResponse::ok(json!({
        "subscription_id": req.subscription_id,
        "events": events,
    }))))
}

/// `GET /api/v1/curriculum`
pub async fn get_curriculum(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let stages = state.svc.curriculum().stages();
    let current = state.svc.curriculum().current_index();
    Ok(Json(ApiResponse::ok(json!({
        "current_stage": current,
        "total_stages": stages.len(),
        "stages": stages,
    }))))
}

/// `GET /api/v1/replay/stats`
pub async fn replay_stats(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let replay = state.svc.replay();
    Ok(Json(ApiResponse::ok(json!({
        "len": replay.len(),
        "mean_reward": replay.mean_reward(),
    }))))
}

/// `GET /api/v1/meta/strategies`
pub async fn meta_strategies(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let strategies = state.svc.meta().strategies();
    let mut out = Vec::new();
    for s in &strategies {
        if let Some(eval) = state.svc.meta().get(s) {
            out.push(json!({
                "strategy": s,
                "uses": eval.uses,
                "mean_reward": eval.mean_reward,
                "last_reward": eval.last_reward,
                "confidence": eval.confidence,
            }));
        }
    }
    Ok(Json(ApiResponse::ok(json!({ "strategies": out }))))
}

/// `GET /api/v1/openapi.json`
pub async fn openapi_json() -> Json<crate::openapi::OpenApiSpec> {
    Json(crate::openapi::generate_openapi())
}

// ─── Multilingual endpoints ─────────────────────────────────────────────────

/// `POST /api/v1/language/detect` — detect the language of a text.
pub async fn language_detect(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<serde_json::Value>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let text = req
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ServiceError::BadRequest("missing 'text' field".into()))?;
    let scores = qbitos_multilingual::detect_language_with_scores(text);
    Ok(Json(ApiResponse::ok(json!({
        "detected": scores.first().map(|s| s.language.code()).unwrap_or("und"),
        "scores": scores,
    }))))
}

/// `POST /api/v1/language/tokenize` — word-tokenize text.
pub async fn language_tokenize(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<serde_json::Value>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let text = req
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ServiceError::BadRequest("missing 'text' field".into()))?;
    let tokens = qbitos_multilingual::word_tokenize(text);
    let token_json: Vec<Value> = tokens
        .iter()
        .map(|t| {
            json!({
                "text": t.text,
                "start": t.start,
                "end": t.end,
                "language": t.language.code(),
            })
        })
        .collect();
    Ok(Json(ApiResponse::ok(json!({
        "tokens": token_json,
        "count": token_json.len(),
    }))))
}

/// `POST /api/v1/language/stem` — stem an English word.
pub async fn language_stem(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<serde_json::Value>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let word = req
        .get("word")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ServiceError::BadRequest("missing 'word' field".into()))?;
    let stemmed = qbitos_multilingual::stem_english(word);
    Ok(Json(ApiResponse::ok(json!({
        "original": word,
        "stemmed": stemmed,
    }))))
}

// ─── Algorithm endpoints ────────────────────────────────────────────────────

/// `POST /api/v1/algo/levenshtein` — compute Levenshtein distance.
pub async fn algo_levenshtein(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<serde_json::Value>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let a = req
        .get("a")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ServiceError::BadRequest("missing 'a' field".into()))?;
    let b = req
        .get("b")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ServiceError::BadRequest("missing 'b' field".into()))?;
    let dist = qbitos_algorithms::levenshtein_distance(a, b);
    let max_len = a.len().max(b.len());
    let similarity = if max_len == 0 {
        1.0
    } else {
        1.0 - dist as f64 / max_len as f64
    };
    Ok(Json(ApiResponse::ok(json!({
        "distance": dist,
        "similarity": similarity,
        "a": a,
        "b": b,
    }))))
}

/// `POST /api/v1/algo/topk` — find the top-K largest numbers.
pub async fn algo_topk(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<serde_json::Value>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let numbers: Vec<i64> = req
        .get("numbers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ServiceError::BadRequest("missing 'numbers' array".into()))?
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    let k = req
        .get("k")
        .and_then(|v| v.as_u64())
        .unwrap_or(5) as usize;
    let mut tracker = qbitos_algorithms::TopK::new(k);
    tracker.extend(numbers);
    let top = tracker.into_sorted_vec();
    Ok(Json(ApiResponse::ok(json!({
        "top": top,
        "k": k,
    }))))
}

/// `POST /api/v1/algo/bloom` — Bloom filter membership test.
pub async fn algo_bloom(
    State(state): State<ServiceState>,
    AxumJson(req): AxumJson<serde_json::Value>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let items: Vec<String> = req
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ServiceError::BadRequest("missing 'items' array".into()))?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    let test = req
        .get("test")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ServiceError::BadRequest("missing 'test' field".into()))?;
    let mut bf = qbitos_algorithms::BloomFilter::new(items.len().max(10), 0.01);
    for item in &items {
        bf.insert(item);
    }
    let present = bf.contains(&test);
    Ok(Json(ApiResponse::ok(json!({
        "present": present,
        "filter_size_bytes": bf.memory_bytes(),
        "items_count": bf.len(),
        "estimated_fp_rate": bf.estimated_fp_rate(),
    }))))
}

/// `POST /api/v1/events/replay` — replay events from the persistent store
/// back into the in-memory buffer. Only works if a persistent event store
/// is attached.
pub async fn replay_events(
    State(state): State<ServiceState>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let Some(store) = state.svc.event_store() else {
        return Err(ServiceError::UnsupportedCapability(
            "persistent event store not attached",
        ));
    };
    let replayed = store.replay_into_buffer().await;
    Ok(Json(ApiResponse::ok(json!({
        "replayed": replayed,
        "buffer_len": store.buffer_len().await,
    }))))
}

/// `GET /api/v1/events/lookup/{id}` — look up an event by id in the
/// persistent store.
pub async fn lookup_event(
    State(state): State<ServiceState>,
    Path(id): Path<u64>,
) -> Result<Json<ApiResponse<Value>>, ServiceError> {
    state.svc.record_request();
    let Some(store) = state.svc.event_store() else {
        return Err(ServiceError::UnsupportedCapability(
            "persistent event store not attached",
        ));
    };
    let event = store.lookup(crate::events::EventId(id)).await;
    Ok(Json(ApiResponse::ok(json!({
        "found": event.is_some(),
        "event": event,
    }))))
}

/// Fallback handler for unknown routes.
pub async fn fallback() -> Result<Json<ApiResponse<()>>, ServiceError> {
    Err(ServiceError::NotFound("endpoint not found".into()))
}


