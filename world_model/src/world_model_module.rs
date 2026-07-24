//! Composite world model.

use std::sync::Arc;

use crate::causal_engine::{CausalGraph, CausalReasoner};
use crate::entity_engine::EntityEngine;
use crate::knowledge_graph::KnowledgeGraph;
use crate::simulation_engine::Simulation;

/// The composite world model.
pub struct WorldModel {
    /// The knowledge graph.
    pub knowledge_graph: Arc<KnowledgeGraph>,
    /// The entity engine.
    pub entity_engine: Arc<EntityEngine>,
    /// The causal graph.
    pub causal_graph: Arc<CausalGraph>,
    /// The causal reasoner.
    pub causal_reasoner: Arc<CausalReasoner>,
    /// The simulation engine.
    pub simulation: Arc<Simulation>,
}

impl Default for WorldModel {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldModel {
    /// Build a new empty world model.
    #[must_use]
    pub fn new() -> Self {
        let knowledge_graph = Arc::new(KnowledgeGraph::new());
        let entity_engine = Arc::new(EntityEngine::new());
        let causal_graph = Arc::new(CausalGraph::new());
        let causal_reasoner = Arc::new(CausalReasoner::new(Arc::clone(&causal_graph)));
        let simulation = Arc::new(Simulation::new(Arc::clone(&causal_graph), Arc::clone(&causal_reasoner)));
        Self {
            knowledge_graph,
            entity_engine,
            causal_graph,
            causal_reasoner,
            simulation,
        }
    }
}
