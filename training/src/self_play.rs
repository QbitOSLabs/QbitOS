//! Self-play engine: the agent plays against itself to generate training data.

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The outcome of a self-play game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameOutcome {
    /// Player 1 (the agent) won.
    Win,
    /// Player 1 lost (player 2 won).
    Loss,
    /// The game was a draw.
    Draw,
    /// The game was not completed (timeout or error).
    Incomplete,
}

impl GameOutcome {
    /// Convert to a numeric reward from the perspective of player 1.
    #[must_use]
    pub fn reward(self) -> f64 {
        match self {
            Self::Win => 1.0,
            Self::Loss => -1.0,
            Self::Draw => 0.0,
            Self::Incomplete => 0.0,
        }
    }
}

/// A single self-play game record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPlayResult {
    /// Game id.
    pub game_id: uuid::Uuid,
    /// Final outcome.
    pub outcome: GameOutcome,
    /// Number of moves played.
    pub moves: usize,
    /// Total game duration in milliseconds.
    pub duration_ms: u64,
    /// Move history (state, action pairs).
    pub history: Vec<(serde_json::Value, serde_json::Value)>,
}

/// The self-play engine: simulates games where the agent plays both sides.
pub struct SelfPlayEngine {
    /// Maximum moves per game.
    pub max_moves: usize,
    /// Game timeout in milliseconds.
    pub timeout_ms: u64,
    /// History of completed games.
    pub history: Arc<RwLock<Vec<SelfPlayResult>>>,
    /// Win/loss/draw statistics.
    pub stats: Arc<RwLock<AHashMap<String, u64>>>,
}

impl Default for SelfPlayEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfPlayEngine {
    /// Construct a new self-play engine.
    #[must_use]
    pub fn new() -> Self {
        let mut stats = AHashMap::new();
        stats.insert("wins".into(), 0);
        stats.insert("losses".into(), 0);
        stats.insert("draws".into(), 0);
        stats.insert("incomplete".into(), 0);
        stats.insert("total_games".into(), 0);
        Self {
            max_moves: 100,
            timeout_ms: 10_000,
            history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(stats)),
        }
    }

    /// Play a single game. The `policy` closure is called for each move and
    /// should return an action given the current state. The `apply` closure
    /// applies an action to a state, returning the next state. The `is_terminal`
    /// closure determines whether a state ends the game, and `evaluate` returns
    /// the outcome from a terminal state.
    pub fn play<F, A, T, E>(&self, policy: F, apply: A, is_terminal: T, evaluate: E) -> SelfPlayResult
    where
        F: Fn(&serde_json::Value, usize) -> serde_json::Value,
        A: Fn(&serde_json::Value, &serde_json::Value) -> serde_json::Value,
        T: Fn(&serde_json::Value) -> bool,
        E: Fn(&serde_json::Value) -> GameOutcome,
    {
        let start = std::time::Instant::now();
        let game_id = uuid::Uuid::new_v4();
        let mut state = serde_json::json!({"initial": true, "turn": 0});
        let mut history = Vec::new();
        let mut moves = 0;
        let mut outcome = GameOutcome::Incomplete;

        for mv_idx in 0..self.max_moves {
            if is_terminal(&state) {
                outcome = evaluate(&state);
                break;
            }
            if start.elapsed().as_millis() as u64 > self.timeout_ms {
                break;
            }
            let action = policy(&state, mv_idx);
            let next = apply(&state, &action);
            history.push((state.clone(), action));
            state = next;
            moves += 1;
        }

        if matches!(outcome, GameOutcome::Incomplete) && is_terminal(&state) {
            outcome = evaluate(&state);
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let result = SelfPlayResult {
            game_id,
            outcome,
            moves,
            duration_ms,
            history,
        };

        // Update stats.
        let mut stats = self.stats.write();
        match outcome {
            GameOutcome::Win => *stats.entry("wins".into()).or_insert(0) += 1,
            GameOutcome::Loss => *stats.entry("losses".into()).or_insert(0) += 1,
            GameOutcome::Draw => *stats.entry("draws".into()).or_insert(0) += 1,
            GameOutcome::Incomplete => *stats.entry("incomplete".into()).or_insert(0) += 1,
        }
        *stats.entry("total_games".into()).or_insert(0) += 1;
        drop(stats);

        self.history.write().push(result.clone());
        result
    }

    /// Get statistics.
    #[must_use]
    pub fn stats(&self) -> AHashMap<String, u64> {
        self.stats.read().clone()
    }

    /// Number of games played.
    #[must_use]
    pub fn games_played(&self) -> usize {
        self.history.read().len()
    }

    /// Win rate from player 1's perspective.
    #[must_use]
    pub fn win_rate(&self) -> f64 {
        let stats = self.stats.read();
        let total = *stats.get("total_games").unwrap_or(&0);
        if total == 0 {
            return 0.0;
        }
        let wins = *stats.get("wins").unwrap_or(&0);
        wins as f64 / total as f64
    }

    /// Clear all history.
    pub fn clear(&self) {
        self.history.write().clear();
        let mut stats = self.stats.write();
        stats.insert("wins".into(), 0);
        stats.insert("losses".into(), 0);
        stats.insert("draws".into(), 0);
        stats.insert("incomplete".into(), 0);
        stats.insert("total_games".into(), 0);
    }
}

impl std::fmt::Debug for SelfPlayEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfPlayEngine")
            .field("games_played", &self.games_played())
            .field("win_rate", &self.win_rate())
            .field("max_moves", &self.max_moves)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_simple_game_with_terminal() {
        let engine = SelfPlayEngine::new();
        // A trivial game: state increments a counter, terminal when >= 5.
        let result = engine.play(
            |state, _i| {
                let n = state.get("n").and_then(|v| v.as_u64()).unwrap_or(0);
                serde_json::json!({"increment": n + 1})
            },
            |state, action| {
                let n = state.get("n").and_then(|v| v.as_u64()).unwrap_or(0);
                let inc = action.get("increment").and_then(|v| v.as_u64()).unwrap_or(1);
                serde_json::json!({"n": n + inc})
            },
            |state| state.get("n").and_then(|v| v.as_u64()).unwrap_or(0) >= 5,
            |state| {
                let n = state.get("n").and_then(|v| v.as_u64()).unwrap_or(0);
                if n >= 5 {
                    GameOutcome::Win
                } else {
                    GameOutcome::Loss
                }
            },
        );
        assert_eq!(result.outcome, GameOutcome::Win);
        assert!(result.moves <= 5);
    }

    #[test]
    fn stats_track_outcomes() {
        let engine = SelfPlayEngine::new();
        for _ in 0..3 {
            engine.play(
                |_state, _i| serde_json::json!({"done": true}),
                |state, _action| {
                    let mut s = state.clone();
                    s["done"] = serde_json::json!(true);
                    s
                },
                |state| state.get("done").and_then(|v| v.as_bool()).unwrap_or(false),
                |_| GameOutcome::Win,
            );
        }
        let stats = engine.stats();
        assert_eq!(*stats.get("wins").unwrap(), 3);
        assert_eq!(*stats.get("total_games").unwrap(), 3);
        assert!((engine.win_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn outcome_reward_mapping() {
        assert_eq!(GameOutcome::Win.reward(), 1.0);
        assert_eq!(GameOutcome::Loss.reward(), -1.0);
        assert_eq!(GameOutcome::Draw.reward(), 0.0);
    }

    #[test]
    fn max_moves_caps_runaway_games() {
        let mut engine = SelfPlayEngine::new();
        engine.max_moves = 5;
        let result = engine.play(
            |state, _i| serde_json::json!({"x": state.get("x").unwrap_or(&serde_json::json!(0))}),
            |state, _action| {
                let x = state.get("x").and_then(|v| v.as_u64()).unwrap_or(0);
                serde_json::json!({"x": x + 1})
            },
            |_| false, // Never terminal
            |_| GameOutcome::Win,
        );
        assert_eq!(result.outcome, GameOutcome::Incomplete);
        assert_eq!(result.moves, 5);
    }
}
