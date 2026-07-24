//! The interactive REPL.

use std::sync::Arc;

use colored::Colorize;

use crate::app::App;

/// Run a one-shot cognitive cycle on a single prompt.
///
/// # Errors
/// Returns an error if dispatch fails.
pub async fn ask_once(app: &Arc<App>, prompt: &str) -> anyhow::Result<()> {
    let observation = serde_json::json!({
        "type": "question",
        "prompt": prompt,
    });
    app.dispatch(observation).map_err(|e| anyhow::anyhow!(e))?;

    // Visualize the cognitive stages
    let stages = ["observe", "understand", "create_goal", "plan", "execute", "verify", "learn", "improve"];
    for stage in stages {
        println!("  {} {}", "→".cyan(), stage);
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    }
    println!();

    // Generate a real response via the native cognitive agent
    let request = qbitos_neural_engine::InferenceRequest {
        prompt: prompt.to_string(),
        max_tokens: 256,
        ..Default::default()
    };
    match app.neural.generate(&request).await {
        Ok(response) => {
            println!("{} {}", "✓".green(), response.text);
            println!();
            println!("  {} {}ms · {} sources · finish: {}",
                "›".cyan(),
                response.duration_ms,
                0, // sources count not available from InferenceResponse
                response.finish_reason,
            );
        }
        Err(e) => {
            println!("{} [QbitOS] Processed: \"{}\" (native agent: {e})", "✓".green(), prompt);
        }
    }
    Ok(())
}

/// Run the interactive REPL.
///
/// # Errors
/// Returns an error if stdin fails.
pub async fn run(app: &Arc<App>) -> anyhow::Result<()> {
    // Boot the cognitive kernel
    app.boot_kernel();

    // Auto-load persisted state on startup
    let persistence = crate::persistence::Persistence::new();
    if persistence.exists() {
        match persistence.load(app) {
            Ok(()) => println!("{} state loaded from {}", "›".cyan(), persistence.path().display()),
            Err(e) => eprintln!("{} failed to load state: {e}", "warning:".yellow()),
        }
    }

    println!();
    println!("{}", "╔══════════════════════════════════════════════════════════════════╗".magenta());
    println!("{}", "║                                                                  ║".magenta());
    println!("{}", "║   QbitOS — Cognitive Operating System                           ║".magenta());
    println!("{}", "║   Rust 2024 edition · v0.1.0                                    ║".magenta());
    println!("{}", "║                                                                  ║".magenta());
    println!("{}", "║   \"A small 3B AI model becomes powerful through a superior      ║".magenta());
    println!("{}", "║    cognitive operating system.\"                                 ║".magenta());
    println!("{}", "║                                                                  ║".magenta());
    println!("{}", "╚══════════════════════════════════════════════════════════════════╝".magenta());
    println!();
    println!("Cognitive kernel booted. Type a prompt and press Enter.");
    println!("Commands: :status, :memory, :skills, :tools, :world, :evolution, :safety <text>, :inject <text>, :feedback <pos|neg|stats> <target>, :audit, :save, :load, :quit");
    println!();

    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let readline = rl.readline("qbitos> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() { continue; }
                let _ = rl.add_history_entry(line);
                if line == ":quit" || line == ":q" { break; }
                if line == ":status" {
                    let status = app.status();
                    println!("{}", serde_json::to_string_pretty(&status)?);
                    continue;
                }
                if line == ":memory" {
                    let s = app.status();
                    println!("L0={} L1={} L2={} L3={} L4={} L5={}",
                        s.memory_layers.l0_sensory,
                        s.memory_layers.l1_working,
                        s.memory_layers.l2_episodic,
                        s.memory_layers.l3_semantic,
                        s.memory_layers.l4_procedural,
                        s.memory_layers.l5_meta_strategies);
                    continue;
                }
                if line == ":skills" {
                    for s in app.skills.registry.list() {
                        println!("  {} (conf={:.2}, invocations={}, success_rate={:.2})",
                            s.name, s.confidence(), s.execution_count, s.success_rate());
                    }
                    continue;
                }
                if line == ":tools" {
                    for t in app.tools.vm.registry().list() {
                        println!("  {t}");
                    }
                    continue;
                }
                if line == ":world" {
                    let kg = &app.world_model.knowledge_graph;
                    println!("  Knowledge graph: {} nodes, {} edges", kg.node_count(), kg.edge_count());
                    if kg.edge_count() > 0 {
                        println!("  Triples:");
                        for (s, p, o) in kg.all_triples() {
                            println!("    {s} —{p}→ {o}");
                        }
                    }
                    continue;
                }
                if line.starts_with(":world add ") {
                    let parts: Vec<&str> = line.strip_prefix(":world add ").unwrap().splitn(3, ' ').collect();
                    if parts.len() == 3 {
                        app.world_model.knowledge_graph.add(parts[0], parts[1], parts[2]);
                        println!("  Added: {} —{}→ {}", parts[0], parts[1], parts[2]);
                    } else {
                        println!("  Usage: :world add <subject> <predicate> <object>");
                    }
                    continue;
                }
                if line == ":evolution" {
                    let accepted = app.evolution.run_cycle();
                    println!("  Evolution cycle: {accepted} improvements accepted");
                    continue;
                }
                if line.starts_with(":safety ") {
                    let text = line.strip_prefix(":safety ").unwrap().trim();
                    let result = app.content_filter.filter(text);
                    if result.passed {
                        println!("  ✓ Safe (risk: {:.2})", result.risk_score);
                    } else {
                        println!("  ✗ Blocked (risk: {:.2}) — {:?}", result.risk_score, result.flagged_categories);
                    }
                    continue;
                }
                if line.starts_with(":inject ") {
                    let text = line.strip_prefix(":inject ").unwrap().trim();
                    let result = app.injection_detector.detect(text);
                    if result.detected {
                        println!("  ✗ Injection detected (risk: {:.2}) — {:?}", result.risk_score, result.injection_types);
                        app.audit.warn("injection_detected", text);
                    } else {
                        println!("  ✓ No injection (risk: {:.2})", result.risk_score);
                    }
                    continue;
                }
                if line.starts_with(":feedback ") {
                    let parts: Vec<&str> = line.strip_prefix(":feedback ").unwrap().splitn(2, ' ').collect();
                    if parts.len() == 2 {
                        match parts[0] {
                            "pos" | "positive" => {
                                app.feedback.add(qbitos_learning::Feedback::positive(parts[1]));
                                println!("  ✓ Positive feedback for: {}", parts[1]);
                            }
                            "neg" | "negative" => {
                                app.feedback.add(qbitos_learning::Feedback::negative(parts[1]));
                                println!("  ✓ Negative feedback for: {}", parts[1]);
                            }
                            "stats" => {
                                let stats = app.feedback.stats_for(parts[1]);
                                println!("  Feedback for: {} — count={}, score={:.2}, pos={}, neg={}",
                                    parts[1], stats.count, stats.mean_score, stats.positive, stats.negative);
                            }
                            _ => println!("  Usage: :feedback <pos|neg|stats> <target>"),
                        }
                    } else {
                        println!("  Usage: :feedback <pos|neg|stats> <target>");
                    }
                    continue;
                }
                if line == ":audit" {
                    let events = app.audit.events();
                    if events.is_empty() {
                        println!("  (no audit events)");
                    } else {
                        println!("  {} audit events:", events.len());
                        for event in events.iter().take(10) {
                            println!("    [{:?}] {} — {}", event.level, event.category, event.message);
                        }
                    }
                    continue;
                }
                if line == ":save" {
                    match persistence.save(app) {
                        Ok(()) => println!("  State saved to {}", persistence.path().display()),
                        Err(e) => eprintln!("{} failed to save: {e}", "error:".red()),
                    }
                    continue;
                }
                if line == ":load" {
                    match persistence.load(app) {
                        Ok(()) => println!("  State loaded from {}", persistence.path().display()),
                        Err(e) => eprintln!("{} failed to load: {e}", "error:".red()),
                    }
                    continue;
                }
                if line.starts_with(":ingest ") {
                    let path = line.strip_prefix(":ingest ").unwrap().trim();
                    match app.ds_to_memory.ingest_path(&app.data, std::path::Path::new(path), None).await {
                        Ok(result) => {
                            println!("  {} {} records → {} episodes, {} concepts",
                                "›".cyan(),
                                result.records_ingested,
                                result.episodes_written,
                                result.concepts_added,
                            );
                        }
                        Err(e) => eprintln!("{} ingest failed: {e}", "error:".red()),
                    }
                    continue;
                }
                // Otherwise: dispatch as a cognitive observation
                if let Err(e) = ask_once(app, line).await {
                    eprintln!("{} {e}", "error:".red());
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) | Err(rustyline::error::ReadlineError::Eof) => {
                println!("Exiting...");
                // Auto-save on exit
                if let Err(e) = persistence.save(app) {
                    eprintln!("{} failed to auto-save: {e}", "warning:".yellow());
                } else {
                    println!("  State auto-saved to {}", persistence.path().display());
                }
                break;
            }
            Err(e) => {
                eprintln!("{} {e}", "error:".red());
                break;
            }
        }
    }
    Ok(())
}
