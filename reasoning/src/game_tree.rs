//! Game tree search with minimax and alpha-beta pruning.

use serde::{Deserialize, Serialize};

/// Result of a game tree search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimaxResult {
    /// Best move (action index).
    pub best_move: usize,
    /// Value of the best move.
    pub value: f64,
    /// Nodes explored.
    pub nodes_explored: usize,
}

/// Trait for game state evaluation.
pub trait GameState: Send + Sync {
    /// Get available moves.
    fn moves(&self) -> Vec<String>;
    /// Apply a move and return the new state.
    fn apply(&self, move_idx: usize) -> Self;
    /// Evaluate the state from the perspective of the maximizing player.
    /// Returns +1 for win, -1 for loss, 0 for draw.
    fn evaluate(&self) -> f64;
    /// Whether the game is over.
    fn is_terminal(&self) -> bool;
    /// Whose turn it is: `true` = maximizer, `false` = minimizer.
    fn is_maximizer_turn(&self) -> bool;
}

/// Minimax with alpha-beta pruning.
///
/// `max_depth`: maximum search depth.
/// Returns the best move index and its value.
#[must_use]
pub fn minimax<S: GameState + Clone>(
    state: &S,
    max_depth: usize,
) -> MinimaxResult {
    let mut nodes = 0usize;
    let moves = state.moves();
    if moves.is_empty() {
        return MinimaxResult { best_move: 0, value: state.evaluate(), nodes_explored: 1 };
    }

    let mut best_move = 0;
    let mut best_value = if state.is_maximizer_turn() { f64::NEG_INFINITY } else { f64::INFINITY };
    let mut alpha = f64::NEG_INFINITY;
    let mut beta = f64::INFINITY;

    for (i, _) in moves.iter().enumerate() {
        nodes += 1;
        let child = state.apply(i);
        let value = minimax_value(&child, max_depth - 1, alpha, beta, &mut nodes);

        if state.is_maximizer_turn() {
            if value > best_value {
                best_value = value;
                best_move = i;
            }
            alpha = alpha.max(best_value);
        } else {
            if value < best_value {
                best_value = value;
                best_move = i;
            }
            beta = beta.min(best_value);
        }

        if beta <= alpha { break; } // Prune
    }

    MinimaxResult { best_move, value: best_value, nodes_explored: nodes }
}

fn minimax_value<S: GameState + Clone>(
    state: &S,
    depth: usize,
    mut alpha: f64,
    mut beta: f64,
    nodes: &mut usize,
) -> f64 {
    *nodes += 1;

    if depth == 0 || state.is_terminal() {
        return state.evaluate();
    }

    let moves = state.moves();
    if moves.is_empty() { return state.evaluate(); }

    if state.is_maximizer_turn() {
        let mut value = f64::NEG_INFINITY;
        for (i, _) in moves.iter().enumerate() {
            let child = state.apply(i);
            value = value.max(minimax_value(&child, depth - 1, alpha, beta, nodes));
            alpha = alpha.max(value);
            if beta <= alpha { break; }
        }
        value
    } else {
        let mut value = f64::INFINITY;
        for (i, _) in moves.iter().enumerate() {
            let child = state.apply(i);
            value = value.min(minimax_value(&child, depth - 1, alpha, beta, nodes));
            beta = beta.min(value);
            if beta <= alpha { break; }
        }
        value
    }
}

/// A simple game tree for testing.
pub struct GameTree {
    /// The tree as a flat array: each node has a value and children indices.
    pub nodes: Vec<GameTreeNode>,
}

/// A node in the game tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameTreeNode {
    /// Node value (for leaf nodes).
    pub value: f64,
    /// Children indices.
    pub children: Vec<usize>,
    /// Whether this is a maximizer node.
    pub is_maximizer: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_picks_best_move() {
        // Simple tree: maximizer has 3 children with values 3, 5, 2
        let tree = GameTree {
            nodes: vec![
                GameTreeNode { value: 0.0, children: vec![1, 2, 3], is_maximizer: true },
                GameTreeNode { value: 3.0, children: vec![], is_maximizer: false },
                GameTreeNode { value: 5.0, children: vec![], is_maximizer: false },
                GameTreeNode { value: 2.0, children: vec![], is_maximizer: false },
            ],
        };

        // Simple game state wrapper for the tree
        #[derive(Clone)]
        struct TreeState {
            tree: Vec<GameTreeNode>,
            current: usize,
        }

        impl GameState for TreeState {
            fn moves(&self) -> Vec<String> {
                self.tree[self.current].children.iter().map(|i| i.to_string()).collect()
            }
            fn apply(&self, move_idx: usize) -> Self {
                let child = self.tree[self.current].children[move_idx];
                Self { tree: self.tree.clone(), current: child }
            }
            fn evaluate(&self) -> f64 { self.tree[self.current].value }
            fn is_terminal(&self) -> bool { self.tree[self.current].children.is_empty() }
            fn is_maximizer_turn(&self) -> bool { self.tree[self.current].is_maximizer }
        }

        let state = TreeState { tree: tree.nodes, current: 0 };
        let result = minimax(&state, 3);
        assert_eq!(result.best_move, 1); // Move index 1 → value 5
        assert!((result.value - 5.0).abs() < 1e-10);
    }
}
