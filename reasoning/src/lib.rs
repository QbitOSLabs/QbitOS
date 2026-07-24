//! QbitOS Reasoning — Advanced reasoning engines.
//!
//! Implements:
//!
//! 1. **Propositional logic engine** — resolution-based theorem proving
//! 2. **Constraint satisfaction problem (CSP) solver** — backtracking with
//!    arc consistency (AC-3) and forward checking
//! 3. **Bayesian network** — probabilistic inference with variable elimination
//! 4. **Markov decision process (MDP) solver** — value iteration and policy
//!    iteration for sequential decision making
//! 5. **Game tree search** — minimax with alpha-beta pruning

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod bayesian;
pub mod chain_verification;
pub mod graph_of_thoughts;
pub mod csp;
pub mod game_tree;
pub mod logic;
pub mod mdp;

pub use bayesian::{BayesianNetwork, ConditionalProbability, RandomVariable};
pub use csp::{CspSolver, Constraint, CspProblem, Domain, Variable};
pub use game_tree::{GameTree, MinimaxResult, minimax};
pub use graph_of_thoughts::{
    GotConfig, GotError, GotSolution, GraphOfThoughts, HeuristicScorer, Thought, ThoughtGenerator,
    ThoughtId, ThoughtScorer, TransformKind,
};
pub use logic::{LogicEngine, Proposition, ProofResult};
pub use mdp::{MdpSolver, MdpState, MdpAction, Policy};
pub use chain_verification::ChainVerifier;
