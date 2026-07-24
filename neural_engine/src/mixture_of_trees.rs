//! Mixture-of-Trees (MoT) Reasoning — sparse expert activation with tree-based reasoning.
//!
//! Based on: "Learning to Select and Weigh Reasoning Paths for Efficient LLM
//! Reasoning" (AAAI 2026). MoT integrates sparse expert activation with
//! structured tree-based reasoning, allowing the model to:
//!
//! 1. Generate multiple reasoning trees (experts) in parallel.
//! 2. Use a learned router to select which trees to fully expand.
//! 3. Weight the contributions of different trees via attention.
//! 4. Achieve better accuracy than single-tree ToT at lower compute cost.
//!
//! Key insight from 2026 research: not all reasoning branches are worth
//! expanding. MoT uses a lightweight router to decide which branches to
//! pursue, similar to how MoE models route tokens to specific experts.

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A node in the Mixture-of-Trees reasoning structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotNode {
    /// The reasoning text at this node.
    pub thought: String,
    /// Value score in [0, 1] assigned by the value function.
    pub value: f32,
    /// Depth in the tree.
    pub depth: usize,
    /// Which expert (tree) this node belongs to.
    pub expert_id: usize,
    /// Children nodes.
    pub children: Vec<MotNode>,
    /// Whether this node was expanded or pruned.
    pub status: NodeStatus,
    /// Estimated compute cost of expanding this node.
    pub compute_cost: f32,
}

/// Status of a node in the MoT search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    /// Node has been generated but not evaluated.
    Pending,
    /// Node has been evaluated but not expanded.
    Evaluated,
    /// Node has been expanded into children.
    Expanded,
    /// Node was pruned by the router (not worth expanding).
    Pruned,
    /// Node is a leaf (terminal answer).
    Leaf,
}

/// Configuration for Mixture-of-Trees reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotConfig {
    /// Number of expert trees to generate in parallel.
    pub num_experts: usize,
    /// Maximum depth of each tree.
    pub max_depth: usize,
    /// Maximum number of children per node (branching factor).
    pub branching_factor: usize,
    /// Top-K nodes to expand per round (sparse activation).
    pub top_k_expand: usize,
    /// Minimum value score to expand a node.
    pub min_expand_value: f32,
    /// Router threshold: nodes below this value are pruned.
    pub prune_threshold: f32,
    /// Whether to use weighted aggregation across experts.
    pub weighted_aggregation: bool,
    /// Temperature for expert selection (softmax temperature).
    pub router_temperature: f32,
    /// Maximum total compute budget (number of node expansions).
    pub compute_budget: usize,
}

impl Default for MotConfig {
    fn default() -> Self {
        Self {
            num_experts: 4,
            max_depth: 5,
            branching_factor: 3,
            top_k_expand: 2,
            min_expand_value: 0.3,
            prune_threshold: 0.15,
            weighted_aggregation: true,
            router_temperature: 0.7,
            compute_budget: 50,
        }
    }
}

/// Result of Mixture-of-Trees reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotResult {
    /// The best answer found.
    pub answer: String,
    /// Confidence in the answer [0, 1].
    pub confidence: f32,
    /// Which expert produced the best answer.
    pub best_expert: usize,
    /// Total nodes expanded.
    pub nodes_expanded: usize,
    /// Total nodes pruned.
    pub nodes_pruned: usize,
    /// Expert weights (how much each expert contributed).
    pub expert_weights: Vec<f32>,
    /// All candidate answers from different experts.
    pub candidate_answers: Vec<(usize, String, f32)>,
    /// Total compute used.
    pub compute_used: usize,
}

/// Trait for the reasoning engine used by MoT.
pub trait MotReasoner: Send + Sync {
    /// Generate initial thoughts for multiple experts.
    fn generate_expert_thoughts(&self, problem: &str, num_experts: usize) -> Vec<String>;

    /// Expand a node: generate children thoughts.
    fn expand_node(&self, problem: &str, node: &MotNode, branching_factor: usize) -> Vec<String>;

    /// Evaluate a thought: assign a value score [0, 1].
    fn evaluate(&self, problem: &str, thought: &str) -> f32;

    /// Synthesize a final answer from the best reasoning paths.
    fn synthesize(&self, problem: &str, paths: &[&MotNode]) -> (String, f32);

    /// Route: decide which experts to activate (returns weights).
    fn route(&self, problem: &str, expert_roots: &[MotNode], temperature: f32) -> Vec<f32>;
}

/// Run Mixture-of-Trees reasoning.
///
/// # Algorithm
///
/// 1. Generate `num_experts` initial reasoning trees in parallel.
/// 2. Use a router to assign weights to each expert (sparse activation).
/// 3. Expand the top-K nodes across all weighted experts per round.
/// 4. Prune nodes below the threshold.
/// 5. Continue until max depth, compute budget, or a high-confidence answer is found.
/// 6. Aggregate answers from all experts using the router weights.
#[must_use]
pub fn run_mixture_of_trees(
    problem: &str,
    reasoner: &dyn MotReasoner,
    config: &MotConfig,
) -> MotResult {
    let mut compute_used = 0usize;
    let mut nodes_expanded = 0usize;
    let mut nodes_pruned = 0usize;

    // Step 1: Generate initial thoughts for each expert.
    let initial_thoughts = reasoner.generate_expert_thoughts(problem, config.num_experts);

    // Step 2: Create expert root nodes.
    let mut experts: Vec<MotNode> = initial_thoughts
        .into_iter()
        .enumerate()
        .map(|(i, thought)| {
            let value = reasoner.evaluate(problem, &thought);
            MotNode {
                thought,
                value,
                depth: 0,
                expert_id: i,
                children: Vec::new(),
                status: NodeStatus::Evaluated,
                compute_cost: 1.0,
            }
        })
        .collect();

    // Step 3: Route — compute expert weights.
    let expert_weights = reasoner.route(problem, &experts, config.router_temperature);

    // Step 4: Iterative expansion with sparse activation.
    let mut all_leaves: Vec<(usize, &MotNode)> = Vec::new();

    for depth in 1..=config.max_depth {
        if compute_used >= config.compute_budget {
            break;
        }

        // Collect all pending nodes across experts, weighted by expert weight.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new(); // (weighted_score, expert_idx, node_idx)
        for (ei, expert) in experts.iter().enumerate() {
            let weight = expert_weights[ei];
            collect_expandable(expert, depth - 1, weight, &mut candidates, ei);
        }

        // Sort by weighted score descending.
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Select top-K to expand (sparse activation).
        let to_expand: Vec<_> = candidates
            .into_iter()
            .filter(|(score, _, _)| *score >= config.min_expand_value)
            .take(config.top_k_expand)
            .collect();

        if to_expand.is_empty() {
            break;
        }

        // Expand selected nodes.
        for (score, ei, _ni) in &to_expand {
            if compute_used >= config.compute_budget {
                break;
            }
            let expert = &mut experts[*ei];

            // Find the best node at the previous depth to expand.
            if let Some(node) = find_best_at_depth(expert, depth - 1) {
                let children_thoughts = reasoner.expand_node(problem, node, config.branching_factor);
                node.status = NodeStatus::Expanded;
                node.children = children_thoughts
                    .into_iter()
                    .map(|thought| {
                        let value = reasoner.evaluate(problem, &thought);
                        compute_used += 1;
                        nodes_expanded += 1;
                        MotNode {
                            thought,
                            value,
                            depth,
                            expert_id: *ei,
                            children: Vec::new(),
                            status: if value < config.prune_threshold {
                                nodes_pruned += 1;
                                NodeStatus::Pruned
                            } else if depth >= config.max_depth {
                                NodeStatus::Leaf
                            } else {
                                NodeStatus::Evaluated
                            },
                            compute_cost: 1.0,
                        }
                    })
                    .collect();
            }
        }
    }

    // Step 5: Collect leaf nodes from all experts.
    for expert in &experts {
        collect_leaves(expert, &mut all_leaves);
    }

    // Step 6: Synthesize answer from the best paths.
    let mut best_paths: Vec<&MotNode> = all_leaves
        .iter()
        .map(|(_, n)| *n)
        .collect();
    best_paths.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(Ordering::Equal));
    best_paths.truncate(5);

    let (answer, confidence) = if best_paths.is_empty() {
        // Fallback: use the root nodes.
        let roots: Vec<&MotNode> = experts.iter().collect();
        reasoner.synthesize(problem, &roots)
    } else {
        reasoner.synthesize(problem, &best_paths)
    };

    // Find which expert contributed the best answer.
    let best_expert = best_paths
        .first()
        .map(|n| n.expert_id)
        .unwrap_or(0);

    // Collect candidate answers from each expert's best leaf.
    let candidate_answers: Vec<(usize, String, f32)> = experts
        .iter()
        .enumerate()
        .map(|(i, expert)| {
            let best = find_best_leaf(expert);
            (
                i,
                best.map(|n| n.thought.clone()).unwrap_or_default(),
                best.map(|n| n.value).unwrap_or(0.0),
            )
        })
        .collect();

    MotResult {
        answer,
        confidence,
        best_expert,
        nodes_expanded,
        nodes_pruned,
        expert_weights,
        candidate_answers,
        compute_used,
    }
}

fn collect_expandable(
    node: &MotNode,
    target_depth: usize,
    expert_weight: f32,
    out: &mut Vec<(f32, usize, usize)>,
    expert_idx: usize,
) {
    if node.depth == target_depth && node.status == NodeStatus::Evaluated {
        let weighted_score = node.value * expert_weight;
        out.push((weighted_score, expert_idx, 0));
    }
    for child in &node.children {
        collect_expandable(child, target_depth, expert_weight, out, expert_idx);
    }
}

fn find_best_at_depth<'a>(node: &'a mut MotNode, depth: usize) -> Option<&'a mut MotNode> {
    if node.depth == depth && node.status == NodeStatus::Evaluated {
        return Some(node);
    }
    let mut best: Option<&'a mut MotNode> = None;
    let mut best_val = -1.0_f32;
    for child in &mut node.children {
        if let Some(found) = find_best_at_depth(child, depth) {
            if found.value > best_val {
                best_val = found.value;
                best = Some(found);
            }
        }
    }
    best
}

fn collect_leaves<'a>(node: &'a MotNode, out: &mut Vec<(usize, &'a MotNode)>) {
    if node.status == NodeStatus::Leaf || node.children.is_empty() {
        if node.status != NodeStatus::Pruned {
            out.push((node.expert_id, node));
        }
        return;
    }
    for child in &node.children {
        collect_leaves(child, out);
    }
}

fn find_best_leaf<'a>(node: &'a MotNode) -> Option<&'a MotNode> {
    if node.children.is_empty() {
        return if node.status == NodeStatus::Pruned { None } else { Some(node) };
    }
    let mut best: Option<&'a MotNode> = None;
    let mut best_val = -1.0_f32;
    for child in &node.children {
        if let Some(found) = find_best_leaf(child) {
            if found.value > best_val {
                best_val = found.value;
                best = Some(found);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub reasoner that returns fixed values.
    struct StubReasoner;

    impl MotReasoner for StubReasoner {
        fn generate_expert_thoughts(&self, _problem: &str, num_experts: usize) -> Vec<String> {
            (0..num_experts).map(|i| format!("Expert {i} initial thought")).collect()
        }

        fn expand_node(&self, _problem: &str, node: &MotNode, bf: usize) -> Vec<String> {
            (0..bf)
                .map(|i| format!("Child {i} of '{}'", node.thought))
                .collect()
        }

        fn evaluate(&self, _problem: &str, thought: &str) -> f32 {
            // Deterministic value based on thought length.
            0.3 + (thought.len() as f32 % 0.7)
        }

        fn synthesize(&self, _problem: &str, paths: &[&MotNode]) -> (String, f32) {
            if paths.is_empty() {
                return ("no answer".into(), 0.0);
            }
            let best = paths
                .iter()
                .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal))
                .unwrap();
            (best.thought.clone(), best.value)
        }

        fn route(&self, _problem: &str, experts: &[MotNode], _temp: f32) -> Vec<f32> {
            let total: f32 = experts.iter().map(|e| e.value).sum();
            if total == 0.0 {
                return vec![1.0 / experts.len() as f32; experts.len()];
            }
            experts.iter().map(|e| e.value / total).collect()
        }
    }

    #[test]
    fn mot_produces_answer() {
        let result = run_mixture_of_trees("What is 2+2?", &StubReasoner, &MotConfig::default());
        assert!(!result.answer.is_empty());
        assert!(result.confidence > 0.0);
        assert_eq!(result.expert_weights.len(), 4);
        assert!(!result.candidate_answers.is_empty());
    }

    #[test]
    fn mot_respects_compute_budget() {
        let config = MotConfig {
            compute_budget: 5,
            ..MotConfig::default()
        };
        let result = run_mixture_of_trees("test", &StubReasoner, &config);
        assert!(result.compute_used <= 5);
    }

    #[test]
    fn mot_prunes_low_value_nodes() {
        let config = MotConfig {
            prune_threshold: 0.9, // Very high — most nodes should be pruned
            ..MotConfig::default()
        };
        let result = run_mixture_of_trees("test", &StubReasoner, &config);
        assert!(result.nodes_pruned > 0);
    }

    #[test]
    fn mot_config_defaults() {
        let c = MotConfig::default();
        assert_eq!(c.num_experts, 4);
        assert_eq!(c.max_depth, 5);
        assert_eq!(c.branching_factor, 3);
        assert_eq!(c.top_k_expand, 2);
    }
}
