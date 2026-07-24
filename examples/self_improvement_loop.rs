//! Example: a full self-improvement loop using the training crate.
//!
//! Runs entirely in-process — no server required.
//!
//! ```sh
//! cargo run --example self_improvement_loop
//! ```

use qbitos_training::{
    Curriculum, ExperienceReplay, MetaLearner, PerformanceMetrics,
    SelfImprovementCycle, TrainingConfig, TrainingLoop,
};
use std::sync::Arc;

fn main() {
    println!("── QbitOS Self-Improvement Loop ──");
    println!();

    // 1. Configure the training loop.
    let config = TrainingConfig {
        max_steps: 200,
        batch_size: 16,
        use_curriculum: true,
        use_experience_replay: true,
        use_self_play: false,
        use_self_improvement: true,
        use_meta_learning: true,
        replay_capacity: 5_000,
        log_every: 50,
    };
    let loop_ = TrainingLoop::new(config);

    // 2. Run the loop with a synthetic reward function that improves over time.
    println!("Running 200 training steps...");
    let report = loop_.run(|step, stage| {
        // Simulate a learning curve: starts at 0.3, asymptotes at 0.95.
        let progress = step as f64 / 200.0;
        let base = 0.3 + 0.65 * (1.0 - (-progress * 3.0).exp());
        // Add small noise to simulate stochasticity.
        let noise = ((step as f64).sin() * 0.05).abs();
        let reward = (base + noise).min(1.0);
        let _ = stage;
        reward
    });

    // 3. Report results.
    println!();
    println!("── Training Report ──");
    println!("  steps_executed:    {}", report.steps_executed);
    println!("  mean_reward:       {:.4}", report.mean_reward);
    println!("  final_loss:        {:.4}", report.final_loss);
    println!("  total_duration:    {}ms", report.total_duration_ms);
    println!("  curriculum_advanced: {}", report.curriculum_advanced);
    println!("  improvements_applied: {}", report.improvements_applied);
    println!();

    // 4. Per-strategy rewards.
    println!("── Meta-Learning Strategy Rewards ──");
    for (s, r) in &report.strategy_rewards {
        println!("  {s}: {r:.4}");
    }
    println!();

    // 5. Demonstrate individual subsystems.
    println!("── Individual Subsystems ──");

    // Curriculum.
    let curriculum = Curriculum::with_default_stages();
    println!("Curriculum: {} stages", curriculum.len());
    for stage in curriculum.stages() {
        println!("  - {} ({})", stage.name, stage.difficulty);
    }
    println!();

    // Experience replay.
    let replay = ExperienceReplay::new(1000);
    for i in 0..50 {
        replay.add(qbitos_training::Experience::new(
            serde_json::json!({"step": i}),
            serde_json::json!({"action": "noop"}),
            0.5 + 0.01 * i as f64,
            serde_json::json!({"step": i + 1}),
        ));
    }
    println!("ExperienceReplay: {} experiences, mean_reward={:.4}",
        replay.len(), replay.mean_reward());
    let batch = replay.sample(5);
    println!("  sampled {} experiences", batch.len());
    println!();

    // Self-improvement.
    let cycle = SelfImprovementCycle::new();
    let mut metrics = PerformanceMetrics::default();
    metrics.skill_success_rates.insert("summarize".into(), 0.3);
    metrics.skill_success_rates.insert("translate".into(), 0.85);
    metrics.skill_latencies_ms.insert("translate".into(), 8_000.0);
    metrics.memory_recall_precision = 0.5;
    metrics.reasoning_accuracy = 0.55;
    let weaknesses = cycle.detect_weaknesses(&metrics);
    println!("SelfImprovement: detected {} weaknesses", weaknesses.len());
    for w in &weaknesses {
        println!("  - [{:.2}] {}: {}", w.severity, w.target, w.description);
    }
    println!();

    // Meta-learner.
    let meta = Arc::new(MetaLearner::new());
    for _ in 0..10 {
        meta.record("curriculum", 0.7);
        meta.record("experience_replay", 0.5);
        meta.record("self_play", 0.6);
    }
    let best = meta.best_strategy();
    println!("MetaLearner: best strategy = {:?}", best);
    println!();

    println!("✓ Self-improvement loop complete.");
}
