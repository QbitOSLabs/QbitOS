//! Real implementation of `NativeAgentContext` that wires the native agent
//! to QbitOS's actual memory fabric, skill registry, world model, and planner.

use std::sync::Arc;
use async_trait::async_trait;
use qbitos_neural_engine::NativeAgentContext;
use qbitos_memory::MemoryFabric;
use qbitos_planner::Planner;
use qbitos_skills::SkillModule;
use qbitos_world_model::WorldModel;

/// The real native agent context — connects the agent to all QbitOS subsystems.
pub struct CognitiveContext {
    memory: Arc<MemoryFabric>,
    skills: Arc<SkillModule>,
    world_model: Arc<WorldModel>,
    planner: Arc<Planner>,
}

impl CognitiveContext {
    /// Create a new context wired to all subsystems.
    #[must_use]
    pub fn new(
        memory: Arc<MemoryFabric>,
        skills: Arc<SkillModule>,
        world_model: Arc<WorldModel>,
        planner: Arc<Planner>,
    ) -> Self {
        Self { memory, skills, world_model, planner }
    }
}

#[async_trait]
impl NativeAgentContext for CognitiveContext {
    async fn recall_episodes(&self, query: &str, k: usize) -> Vec<(serde_json::Value, f32)> {
        let episodes = self.memory.episodic.recall_similar(
            &serde_json::json!({"query": query}),
            k,
        );
        episodes.into_iter().map(|ep| {
            let score = ep.object.importance.0;
            (ep.object.content, score)
        }).collect()
    }

    async fn recall_concepts(&self, query: &str, k: usize) -> Vec<String> {
        let concepts = self.memory.semantic.find_similar(query, k);
        concepts.into_iter().map(|c| c.label).collect()
    }

    fn find_skills(&self, _query: &str) -> Vec<(String, f32)> {
        // List all skills and rank by confidence
        let mut skills: Vec<(String, f32)> = self.skills.registry.list()
            .into_iter()
            .map(|s| {
                let conf = s.confidence();
                (s.name, conf)
            })
            .collect();
        // Sort by confidence descending
        skills.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        skills.truncate(3);
        skills
    }

    fn query_knowledge(&self, subject: &str) -> Vec<(String, String)> {
        // Try to look up the subject in the knowledge graph
        if let Some(node_id) = self.world_model.knowledge_graph.lookup(subject) {
            let triples = self.world_model.knowledge_graph.outgoing(node_id);
            triples.into_iter()
                .map(|t| (t.predicate, self.world_model.knowledge_graph.label(t.object).unwrap_or_default()))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn derive_goal(&self, observation: &serde_json::Value) -> Option<(String, String)> {
        self.planner.derive_goal(observation).map(|g| {
            (format!("{:?}", g.kind).to_lowercase(), g.description)
        })
    }

    fn episode_count(&self) -> usize {
        self.memory.episodic.len()
    }

    fn skill_count(&self) -> usize {
        self.skills.registry.len()
    }
}
