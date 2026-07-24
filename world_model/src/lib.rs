//! QbitOS World Model
//!
//! An internal representation of reality. Combines:
//!
//! - A knowledge graph (`knowledge_graph`)
//! - An entity engine (`entity_engine`)
//! - A causal reasoning engine (`causal_engine`)
//! - A simulation engine (`simulation_engine`)

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod causal_engine;
pub mod jepa;
pub mod entity_engine;
pub mod knowledge_graph;
pub mod simulation_engine;

pub use causal_engine::{CausalGraph, CausalLink, CausalReasoner, CausalRule};
pub use jepa::{Jepa, JepaConfig, JepaError};
pub use entity_engine::{Entity, EntityEngine, EntityId, EntityKind, EntityRelation};
pub use knowledge_graph::{KnowledgeGraph, NodeId, Triple};
pub use simulation_engine::{Simulation, SimulationResult, SimulationStep};

/// Re-export of the composite world model.
pub use world_model_module::WorldModel;

mod world_model_module;
