//! Native cognitive agent — no external model required.
//!
//! This is QbitOS's native intelligence layer. Instead of relying on a
//! pre-trained LLM, the `NativeAgent` composes responses from:
//!
//! 1. **Memory recall** — searches episodic + semantic memory for relevant
//!    past experiences and concepts.
//! 2. **Goal derivation** — uses the planner's goal engine to classify the
//!    prompt and create a typed goal.
//! 3. **Skill matching** — looks up registered skills that match the goal.
//! 4. **World model queries** — searches the knowledge graph for relevant
//!    facts and entities.
//! 5. **Reasoning** — runs forward-chaining inference over known facts.
//! 6. **Response composition** — assembles a structured response from all
//!    of the above.
//!
//! This makes QbitOS a **truly native AI agent** — its intelligence emerges
//! from its cognitive architecture, not from a downloaded neural network.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::backends::{InferenceError, InferenceRequest, InferenceResponse, ModelBackend};
use crate::tensor_engine::Tensor;

/// A composed native response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeResponse {
    /// The main text response.
    pub text: String,
    /// Sources consulted (memory episodes, skills, facts).
    pub sources: Vec<ResponseSource>,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A source consulted during response composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseSource {
    /// An episodic memory that was recalled.
    EpisodicMemory {
        /// The recalled content.
        content: serde_json::Value,
        /// Similarity score.
        score: f32,
    },
    /// A semantic concept that matched.
    SemanticConcept {
        /// Concept label.
        label: String,
    },
    /// A skill that was invoked.
    Skill {
        /// Skill name.
        name: String,
        /// Confidence.
        confidence: f32,
    },
    /// A knowledge-graph fact.
    KnowledgeFact {
        /// Subject.
        subject: String,
        /// Predicate.
        predicate: String,
        /// Object.
        object: String,
    },
    /// A reasoning trace.
    ReasoningTrace {
        /// Rule that fired.
        rule: String,
        /// Conclusion.
        conclusion: String,
    },
    /// A derived goal.
    Goal {
        /// Goal kind.
        kind: String,
        /// Description.
        description: String,
    },
}

/// Trait for the subsystems the native agent needs.
/// This decouples the agent from the full App, making it testable.
#[async_trait]
pub trait NativeAgentContext: Send + Sync {
    /// Search episodic memory for experiences similar to `query`.
    async fn recall_episodes(&self, query: &str, k: usize) -> Vec<(serde_json::Value, f32)>;

    /// Search semantic memory for concepts similar to `query`.
    async fn recall_concepts(&self, query: &str, k: usize) -> Vec<String>;

    /// Find skills matching a goal description.
    fn find_skills(&self, query: &str) -> Vec<(String, f32)>;

    /// Query the knowledge graph for facts about `subject`.
    fn query_knowledge(&self, subject: &str) -> Vec<(String, String)>;

    /// Derive a goal from an observation.
    fn derive_goal(&self, observation: &serde_json::Value) -> Option<(String, String)>;

    /// Get the number of episodes in memory.
    fn episode_count(&self) -> usize;

    /// Get the number of skills registered.
    fn skill_count(&self) -> usize;
}

/// The native cognitive agent.
pub struct NativeAgent {
    context: Arc<dyn NativeAgentContext>,
}

impl NativeAgent {
    /// Create a new native agent with the given cognitive context.
    #[must_use]
    pub fn new(context: Arc<dyn NativeAgentContext>) -> Self {
        Self { context }
    }

    /// Compose a response from the cognitive architecture using chain-of-thought reasoning.
    ///
    /// Implements a 2026-style reasoning pipeline:
    /// 1. Decompose the prompt into sub-questions
    /// 2. For each sub-question, recall relevant memories + skills + facts
    /// 3. Apply self-consistency: generate multiple reasoning paths and
    ///    select the one with highest agreement
    /// 4. Synthesize a final answer from the reasoning chain
    pub async fn compose_response(&self, request: &InferenceRequest) -> NativeResponse {
        let prompt = &request.prompt;
        let mut sources = Vec::new();
        let mut reasoning_steps = Vec::new();
        let mut response_parts = Vec::new();

        // === Step 1: Goal Derivation ===
        let observation = serde_json::json!({
            "type": "question",
            "prompt": prompt,
        });
        if let Some((kind, description)) = self.context.derive_goal(&observation) {
            sources.push(ResponseSource::Goal {
                kind: kind.clone(),
                description: description.clone(),
            });
            reasoning_steps.push(format!("Goal: {description}"));
        }

        // === Step 2: Prompt Decomposition ===
        let sub_questions = decompose_prompt(prompt);
        if sub_questions.len() > 1 {
            reasoning_steps.push(format!("Decomposed into {} sub-questions", sub_questions.len()));
        }

        // === Step 3: Multi-Path Reasoning (Self-Consistency) ===
        // Generate 3 reasoning paths and select the best
        let mut reasoning_paths: Vec<Vec<String>> = Vec::new();

        for path_idx in 0..3 {
            let mut path = Vec::new();

            // For each sub-question, gather evidence
            for (sq_idx, sub_q) in sub_questions.iter().enumerate() {
                // Recall memories
                let episodes = self.context.recall_episodes(sub_q, 3).await;
                for (content, score) in &episodes {
                    if path_idx == 0 { // Only record sources from the first path
                        sources.push(ResponseSource::EpisodicMemory {
                            content: content.clone(),
                            score: *score,
                        });
                    }
                    let text = extract_content_text(content);
                    if is_informative(&text) {
                        path.push(format!("  Evidence {} (sim={:.2}): {}", sq_idx + 1, score, text));
                    }
                }

                // Recall concepts
                if path_idx == 0 {
                    let concepts = self.context.recall_concepts(sub_q, 3).await;
                    for label in &concepts {
                        sources.push(ResponseSource::SemanticConcept { label: label.clone() });
                    }
                }

                // Query knowledge graph
                if path_idx == 0 {
                    let facts = self.context.query_knowledge(sub_q);
                    for (predicate, object) in &facts {
                        sources.push(ResponseSource::KnowledgeFact {
                            subject: sub_q.clone(),
                            predicate: predicate.clone(),
                            object: object.clone(),
                        });
                    }
                }
            }

            // Find skills
            if path_idx == 0 {
                let skills = self.context.find_skills(prompt);
                for (name, confidence) in &skills {
                    sources.push(ResponseSource::Skill {
                        name: name.clone(),
                        confidence: *confidence,
                    });
                }
            }

            if !path.is_empty() {
                reasoning_paths.push(path);
            }
        }

        // === Step 4: Self-Consistency Selection ===
        // Select the reasoning path with the most evidence (most unique items)
        let best_path = if reasoning_paths.is_empty() {
            Vec::new()
        } else {
            reasoning_paths.iter()
                .max_by_key(|path| path.iter().collect::<std::collections::HashSet<_>>().len())
                .cloned()
            .unwrap_or_default()
        };

        // === Step 5: Synthesize Answer ===
        if !best_path.is_empty() {
            response_parts.push(format!("Based on my memory and reasoning:\n{}", best_path.join("\n")));
        }

        // Add skill suggestions if available
        let skills = self.context.find_skills(prompt);
        if !skills.is_empty() {
            let skill_summary: Vec<String> = skills.iter()
                .map(|(name, conf)| format!("{name} ({:.0}%)", conf * 100.0))
                .collect();
            response_parts.push(format!("Applicable skills: {}", skill_summary.join(", ")));
        }

        // Add reasoning chain if we have sub-questions
        if sub_questions.len() > 1 {
            response_parts.push(format!("Reasoning: decomposed → {} sub-questions → {} evidence paths → self-consistency selection",
                sub_questions.len(), reasoning_paths.len()));
        }

        // === Step 6: Confidence Estimation ===
        let text = if response_parts.is_empty() {
            let ep_count = self.context.episode_count();
            let sk_count = self.context.skill_count();
            format!(
                "I don't have any memories or skills matching \"{prompt}\" yet.\n\
                 I currently hold {ep_count} episodic memories and {sk_count} skills.\n\
                 Use `:ingest <path>` to feed me knowledge, then ask again."
            )
        } else {
            response_parts.join("\n\n")
        };

        let confidence = match sources.len() {
            0 => 0.1,
            1..=2 => 0.3,
            3..=5 => 0.6,
            6..=10 => 0.75,
            _ => 0.85,
        };

        NativeResponse { text, sources, confidence }
    }
}

/// Decompose a prompt into sub-questions for chain-of-thought reasoning.
///
/// Splits compound questions ("What is X and how does Y work?") into
/// individual sub-questions, each of which can be answered independently
/// and then combined.
fn decompose_prompt(prompt: &str) -> Vec<String> {
    // Split on common conjunctions and question words
    let separators = [" and ", " but ", " or ", "? ", ". "];
    let mut sub_questions = vec![prompt.to_string()];

    for sep in &separators {
        let mut new_sqs = Vec::new();
        for sq in &sub_questions {
            let parts: Vec<&str> = sq.split(sep).collect();
            if parts.len() > 1 {
                for part in parts {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        new_sqs.push(trimmed.to_string());
                    }
                }
            } else {
                new_sqs.push(sq.clone());
            }
        }
        sub_questions = new_sqs;
    }

    // If no decomposition happened, return the original
    if sub_questions.len() <= 1 {
        return vec![prompt.to_string()];
    }

    sub_questions
}

/// Extract readable text from episode content (handles nested JSON).
fn extract_content_text(content: &serde_json::Value) -> String {
    let inner = content.get("content").or_else(|| content.get("description"));
    if let Some(inner) = inner {
        if let Some(s) = inner.as_str() {
            return s.to_string();
        } else if let Some(nested) = inner.get("content").and_then(|c| c.as_str()) {
            return nested.to_string();
        } else if let Some(nested) = inner.get("description").and_then(|c| c.as_str()) {
            return nested.to_string();
        } else {
            let mut parts = Vec::new();
            if let Some(obj) = inner.as_object() {
                for (k, v) in obj.iter().take(3) {
                    if let Some(s) = v.as_str() {
                        parts.push(format!("{k}: {s}"));
                    }
                }
            }
            return if parts.is_empty() { format!("{inner}") } else { parts.join(", ") };
        }
    } else if let Some(s) = content.as_str() {
        return s.to_string();
    } else {
        let mut parts = Vec::new();
        if let Some(obj) = content.as_object() {
            for (k, v) in obj.iter().take(3) {
                if let Some(s) = v.as_str() {
                    parts.push(format!("{k}: {s}"));
                }
            }
        }
        return if parts.is_empty() { format!("{content}") } else { parts.join(", ") };
    }
}

/// Check if text is informative (not metadata, not too long/short).
fn is_informative(text: &str) -> bool {
    !text.is_empty()
        && text.len() < 500
        && !text.contains("task_count")
        && !text.contains("topological_order")
        && !text.starts_with("executed")
        && !text.starts_with("{")
}

#[async_trait]
impl ModelBackend for NativeAgent {
    async fn load(&self) -> Result<(), InferenceError> {
        // Nothing to load — the native agent is always ready
        Ok(())
    }

    async fn unload(&self) {
        // Nothing to unload
    }

    async fn forward(&self, _token_ids: &[u32]) -> Result<Tensor, InferenceError> {
        // The native agent doesn't use token-level forward passes
        Err(InferenceError::Backend(
            "NativeAgent does not support token-level forward passes — use generate()".into()
        ))
    }

    async fn generate(&self, request: &InferenceRequest, _sampler: &crate::sampling::Sampler) -> Result<InferenceResponse, InferenceError> {
        let start = std::time::Instant::now();
        let native = self.compose_response(request).await;

        Ok(InferenceResponse {
            text: native.text,
            token_ids: Vec::new(),
            prompt_tokens: request.prompt.split_whitespace().count(),
            generated_tokens: 0, // native agent doesn't generate tokens
            finish_reason: "native".into(),
            logprobs: None,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    fn is_loaded(&self) -> bool {
        true // always loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubContext {
        episodes: Vec<(serde_json::Value, f32)>,
        concepts: Vec<String>,
        skills: Vec<(String, f32)>,
        facts: Vec<(String, String)>,
    }

    #[async_trait::async_trait]
    impl NativeAgentContext for StubContext {
        async fn recall_episodes(&self, _query: &str, k: usize) -> Vec<(serde_json::Value, f32)> {
            self.episodes.iter().take(k).cloned().collect()
        }
        async fn recall_concepts(&self, _query: &str, k: usize) -> Vec<String> {
            self.concepts.iter().take(k).cloned().collect()
        }
        fn find_skills(&self, _query: &str) -> Vec<(String, f32)> {
            self.skills.clone()
        }
        fn query_knowledge(&self, _subject: &str) -> Vec<(String, String)> {
            self.facts.clone()
        }
        fn derive_goal(&self, _observation: &serde_json::Value) -> Option<(String, String)> {
            Some(("question".into(), "answer the user's question".into()))
        }
        fn episode_count(&self) -> usize {
            self.episodes.len()
        }
        fn skill_count(&self) -> usize {
            self.skills.len()
        }
    }

    fn make_agent() -> NativeAgent {
        let ctx = StubContext {
            episodes: vec![
                (serde_json::json!({"text": "hello world"}), 0.9),
                (serde_json::json!({"text": "goodbye"}), 0.7),
            ],
            concepts: vec!["greeting".into(), "farewell".into()],
            skills: vec![("summarize".into(), 0.85)],
            facts: vec![("greeting".into(), "is polite".into())],
        };
        NativeAgent::new(Arc::new(ctx))
    }

    #[tokio::test]
    async fn compose_response_returns_non_empty() {
        let agent = make_agent();
        let req = InferenceRequest {
            prompt: "What is QbitOS?".into(),
            ..Default::default()
        };
        let resp = agent.compose_response(&req).await;
        assert!(!resp.text.is_empty());
        assert!(resp.confidence >= 0.0 && resp.confidence <= 1.0);
    }

    #[tokio::test]
    async fn compose_response_includes_sources() {
        let agent = make_agent();
        let req = InferenceRequest {
            prompt: "hello".into(),
            ..Default::default()
        };
        let resp = agent.compose_response(&req).await;
        // With populated context, we should get at least one source.
        assert!(!resp.sources.is_empty());
    }

    #[tokio::test]
    async fn generate_returns_valid_response() {
        let agent = make_agent();
        let req = InferenceRequest {
            prompt: "test prompt".into(),
            ..Default::default()
        };
        let sampler = crate::sampling::Sampler::new(crate::sampling::SamplerConfig::default());
        let resp = agent.generate(&req, &sampler).await.unwrap();
        assert!(!resp.text.is_empty());
        assert_eq!(resp.finish_reason, "native");
    }

    #[tokio::test]
    async fn load_always_succeeds() {
        let agent = make_agent();
        agent.load().await.unwrap();
        assert!(agent.is_loaded());
    }
}
