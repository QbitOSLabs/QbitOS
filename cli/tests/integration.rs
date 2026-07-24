//! End-to-end integration tests for the QbitOS CLI.
//!
//! These tests verify that the full cognitive loop runs end-to-end:
//! dataset → memory fabric → planner → skills → tools → evolution.

use std::sync::Arc;

use qbitos_cli::App;

#[tokio::test]
async fn app_boots_all_subsystems() {
    let app = Arc::new(App::new());
    // Verify every subsystem is present
    assert!(!app.data.list().is_empty(), "dataset registry should have loaders");
    assert!(!app.tools.vm.registry().list().is_empty(), "tool registry should have tools");
    // Boot the kernel
    let tx = app.boot_kernel();
    let observation = serde_json::json!({"type": "question", "prompt": "hello"});
    tx.send(observation).unwrap();
    // Give the kernel a moment to process
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
}

#[tokio::test]
async fn dataset_ingest_writes_to_memory() {
    use qbitos_data::DatasetRegistry;
    let app = Arc::new(App::new());
    let registry = DatasetRegistry::with_default_loaders();

    // Write a small JSONL dataset to a temp file (with .jsonl extension)
    let tmp = tempfile::Builder::new().suffix(".jsonl").tempfile().unwrap();
    std::fs::write(tmp.path(), "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n").unwrap();

    let result = app.ds_to_memory.ingest_path(&registry, tmp.path(), None).await.unwrap();
    assert_eq!(result.records_ingested, 3);
    assert_eq!(result.episodes_written, 3);
    assert_eq!(app.memory.episodic.len(), 3);
}

#[tokio::test]
async fn dataset_mine_extracts_skills() {
    use qbitos_data::DatasetRegistry;
    let app = Arc::new(App::new());
    let registry = DatasetRegistry::with_default_loaders();

    // Write a trajectory dataset (filename must contain "trajectory")
    let dir = std::env::temp_dir().join("qbitos-cli-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trajectory.jsonl");
    std::fs::write(&path,
        r#"{"goal":"Find weather","steps":[{"action":"observe"},{"action":"tool"},{"action":"verify"}],"outcome":"success"}
{"goal":"Debug error","steps":[{"action":"infer"},{"action":"tool"},{"action":"verify"}],"outcome":"success"}
"#,
    ).unwrap();

    let result = app.ds_to_skills.mine_from_path(&registry, &path, None).await.unwrap();
    assert_eq!(result.records_ingested, 2);
    assert_eq!(result.skills_registered, 2);
    assert!(app.skills.registry.get_by_name("find_weather").is_some());
    assert!(app.skills.registry.get_by_name("debug_error").is_some());
}

#[tokio::test]
async fn cognitive_loop_runs_full_cycle() {
    let app = Arc::new(App::new());
    app.boot_kernel();
    // Dispatch a prompt
    app.dispatch(serde_json::json!({
        "type": "question",
        "prompt": "What is 2+2?",
    })).unwrap();
    // Give the kernel time to run the cycle
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    // The kernel should have written at least one episode
    assert!(app.memory.episodic.len() >= 1, "cognitive loop should produce episodes");
}

#[tokio::test]
async fn memory_consolidation_works() {
    let app = Arc::new(App::new());
    // Record some episodes
    for i in 0..5 {
        app.memory.record_episode(
            serde_json::json!({"event": "test", "i": i}),
            qbitos_memory::episodic::Outcome::Success,
        );
    }
    assert_eq!(app.memory.episodic.len(), 5);
    // Run consolidation
    let summary = app.memory.consolidate();
    assert!(summary.episodic_remaining <= 5);
}

#[tokio::test]
async fn evolution_cycle_runs() {
    let app = Arc::new(App::new());
    // Run an evolution cycle (should not panic even with no benchmarks)
    let accepted = app.evolution.run_cycle();
    // With no benchmarks, no improvements should be accepted
    assert_eq!(accepted, 0);
}

#[tokio::test]
async fn world_model_accepts_triples() {
    let app = Arc::new(App::new());
    app.world_model.knowledge_graph.add("dog", "is_a", "mammal");
    app.world_model.knowledge_graph.add("cat", "is_a", "mammal");
    assert_eq!(app.world_model.knowledge_graph.node_count(), 3); // dog, cat, mammal
    assert_eq!(app.world_model.knowledge_graph.edge_count(), 2);
}

#[tokio::test]
async fn tool_registry_has_builtin_tools() {
    let app = Arc::new(App::new());
    let tools = app.tools.vm.registry().list();
    assert!(tools.contains(&"filesystem".to_string()));
    assert!(tools.contains(&"terminal".to_string()));
    assert!(tools.contains(&"http".to_string()));
}

#[tokio::test]
async fn skill_registry_supports_lifecycle() {
    use qbitos_skills::registry::{Skill, SkillId};
    let app = Arc::new(App::new());
    let id = app.skills.registry.register(Skill::new("test_skill", "a test"));
    app.skills.registry.record_execution(id, true);
    app.skills.registry.record_execution(id, true);
    app.skills.registry.record_execution(id, false);
    let skill = app.skills.registry.get(id).unwrap();
    assert_eq!(skill.execution_count, 3);
    assert!((skill.success_rate() - (2.0 / 3.0)).abs() < 1e-6);
}

#[tokio::test]
async fn status_returns_all_memory_layers() {
    let app = Arc::new(App::new());
    let status = app.status();
    // All layers should be present in the status
    assert_eq!(status.memory_layers.l0_sensory, 0);
    assert_eq!(status.memory_layers.l1_working, 0);
    assert_eq!(status.memory_layers.l2_episodic, 0);
    assert_eq!(status.memory_layers.l3_semantic, 0);
    assert_eq!(status.memory_layers.l4_procedural, 0);
    assert_eq!(status.memory_layers.l5_meta_strategies, 0);
}

/// Verifies that when the `real-backends` feature is enabled, the App wires
/// the real ToolModuleBackend + NativeAgentBackend into the SkillModule.
/// When the feature is disabled, falls back to noop backends.
#[cfg(feature = "real-backends")]
#[tokio::test]
async fn real_backends_wired_into_skill_module() {
    use qbitos_skills::ToolBackend;

    let app = Arc::new(App::new());

    // The SkillModule's tool backend should be a ToolModuleBackend that
    // proxies to the App's ToolModule. Listing tools should return the
    // built-in tools (filesystem, terminal, http, calculator, etc.).
    let tool_list = app.skills.tool.list_tools();
    assert!(!tool_list.is_empty(), "real ToolModuleBackend should expose built-in tools");
    assert!(
        tool_list.iter().any(|t| t.contains("calculator") || t.contains("calc")),
        "expected a calculator tool in: {tool_list:?}"
    );

    // The InferBackend should be a NativeAgentBackend with late binding.
    // After App::new() completes, the NativeAgent should be bound and
    // infer() should produce a non-stub response (i.e. one that doesn't
    // contain the "[native-agent-stub]" marker).
    let result = app.skills.infer.infer("hello", &serde_json::json!({})).await;
    // The NativeAgent's compose_response may produce any text, but it
    // should NOT contain the stub marker (which only appears when no
    // agent is bound).
    assert!(
        !result.text.contains("[native-agent-stub]"),
        "expected real NativeAgent response, got stub: {}",
        result.text
    );
    assert!(result.confidence >= 0.0, "confidence should be non-negative");
}

/// Verifies that a skill with an `Infer` step dispatches through the
/// NativeAgent (not the stub) when real-backends is enabled.
#[cfg(feature = "real-backends")]
#[tokio::test]
async fn real_backends_skill_execution_uses_native_agent() {
    use qbitos_skills::registry::{Skill, SkillStep, StepKind};

    let app = Arc::new(App::new());

    // Register a simple skill: Infer → Return.
    let mut skill = Skill::new("test_infer", "tests NativeAgent inference");
    skill.add_step(SkillStep::new(
        "infer",
        StepKind::Infer,
        serde_json::json!({"prompt": "What is 2+2?"}),
    ));
    skill.add_step(SkillStep::new(
        "return",
        StepKind::Return,
        serde_json::json!({}),
    ));
    let id = app.skills.registry.register(skill);

    // Execute the skill. With real-backends, the Infer step dispatches
    // through NativeAgent::compose_response, which produces a real answer.
    let report = app.skills.execute(id, serde_json::json!({})).await;
    assert!(
        report.status.is_success(),
        "skill execution should succeed, got status: {:?}",
        report.status
    );
    assert_eq!(report.steps.len(), 2, "Infer + Return = 2 steps");
}

/// Verifies that a skill with a `Tool` step dispatches through the real
/// ToolModule (not the noop backend) when real-backends is enabled.
#[cfg(feature = "real-backends")]
#[tokio::test]
async fn real_backends_skill_execution_uses_tool_module() {
    use qbitos_skills::registry::{Skill, SkillStep, StepKind};

    let app = Arc::new(App::new());

    // Register a skill that calls the calculator tool.
    let mut skill = Skill::new("test_calc", "tests calculator tool");
    skill.add_step(SkillStep::new(
        "call_calc",
        StepKind::Tool,
        serde_json::json!({"tool": "calculator", "args": {"expression": "2+2"}}),
    ));
    skill.add_step(SkillStep::new(
        "return",
        StepKind::Return,
        serde_json::json!({}),
    ));
    let id = app.skills.registry.register(skill);

    let report = app.skills.execute(id, serde_json::json!({})).await;
    // The calculator tool may or may not succeed depending on its API,
    // but the call should at least produce a structured result (not a
    // noop-backend error like "noop tool backend").
    let step_output = report
        .steps
        .first()
        .map(|s| serde_json::to_string(&s.output).unwrap_or_default())
        .unwrap_or_default();
    assert!(
        !step_output.contains("noop tool backend"),
        "expected real tool dispatch, got noop: {step_output}"
    );
}
