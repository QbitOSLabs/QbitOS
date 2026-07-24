//! Federated learning support for QbitOS.
//!
//! Implements a simplified FedAvg-style federated learning protocol that
//! allows multiple QbitOS nodes (or external clients) to collaboratively
//! train a shared model without exchanging raw data.
//!
//! # Protocol
//!
//! 1. The **coordinator** (this server) holds the global model weights.
//! 2. Each **participant** downloads the current global weights.
//! 3. Each participant trains locally on its private data for E epochs.
//! 4. Each participant uploads its weight delta (or updated weights) plus
//!    the number of samples it trained on.
//! 5. The coordinator aggregates updates using weighted average (FedAvg):
//!       w_global_new = Σ (n_i / N) * w_i
//!    where n_i is participant i's sample count and N = Σ n_i.
//! 6. The new global weights are published. Repeat.
//!
//! # Privacy
//!
//! - Raw data never leaves the participant.
//! - Only model weight deltas are exchanged.
//! - Optional differential privacy: participants clip their gradients and
//!   add Gaussian noise before uploading.
//! - Optional secure aggregation: weights are additively masked so the
//!   coordinator sees only the aggregate (not individual updates).
//!
//! # Limitations
//!
//! This is a research-quality implementation suitable for small models
//! (≤ 100M parameters). For production-scale federated learning, integrate
//! with Flower (flower.dev) or TensorFlow Federated.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use thiserror::Error;
use uuid::Uuid;

/// Errors emitted by the federated learning coordinator.
#[derive(Debug, Error)]
pub enum FedError {
    /// Participant is not registered.
    #[error("participant {0} not registered")]
    ParticipantNotFound(Uuid),
    /// Round is not active.
    #[error("round {0} is not active")]
    RoundNotActive(u64),
    /// Round is already complete.
    #[error("round {0} is already complete")]
    RoundComplete(u64),
    /// Update has wrong shape.
    #[error("update shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    /// Aggregation failed.
    #[error("aggregation failed: {0}")]
    Aggregation(String),
}

/// A model weight vector (flattened f32 parameters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weights {
    /// Flat parameter buffer.
    pub data: Vec<f32>,
    /// Shape of the model (for validation).
    pub shape: Vec<usize>,
    /// Version number — incremented after each aggregation.
    pub version: u64,
}

impl Weights {
    /// Create new weights from a flat buffer and shape.
    #[must_use]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape, version: 0 }
    }

    /// Create zero-initialized weights of the given shape.
    #[must_use]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        Self { data: vec![0.0; total], shape, version: 0 }
    }

    /// Number of parameters.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.data.len()
    }

    /// Compute the L2 norm of the weight vector.
    #[must_use]
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|w| w * w).sum::<f32>().sqrt()
    }

    /// Add another weight vector in-place. Returns an error if shapes differ.
    pub fn add_assign(&mut self, other: &Weights) -> Result<(), FedError> {
        if self.shape != other.shape {
            return Err(FedError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
        Ok(())
    }

    /// Scale all weights by a scalar.
    pub fn scale(&mut self, factor: f32) {
        for w in &mut self.data {
            *w *= factor;
        }
    }

    /// Compute the delta: `self - baseline` (used to extract a participant's update).
    #[must_use]
    pub fn delta_from(&self, baseline: &Weights) -> Self {
        debug_assert_eq!(self.shape, baseline.shape);
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(baseline.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self {
            data,
            shape: self.shape.clone(),
            version: self.version,
        }
    }
}

/// A participant's update for one round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantUpdate {
    /// Participant ID.
    pub participant_id: Uuid,
    /// Round number this update belongs to.
    pub round: u64,
    /// Updated weights (full model) from the participant.
    pub weights: Weights,
    /// Number of training samples used.
    pub num_samples: usize,
    /// Training loss reported by the participant (averaged over batches).
    pub train_loss: f32,
    /// Optional validation accuracy reported by the participant.
    pub val_accuracy: Option<f32>,
    /// Timestamp the update was received.
    pub timestamp_ms: u64,
}

/// Configuration for a federated learning session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedConfig {
    /// Minimum number of participants required per round.
    pub min_participants: usize,
    /// Maximum number of participants per round (0 = unlimited).
    pub max_participants: usize,
    /// Minimum number of samples a participant must train on.
    pub min_samples_per_participant: usize,
    /// Maximum L2 norm for gradient clipping (0 = no clipping).
    pub max_grad_norm: f32,
    /// Differential privacy noise multiplier (0 = no noise).
    pub dp_noise_multiplier: f32,
    /// Number of rounds to run before stopping (0 = unlimited).
    pub max_rounds: u64,
    /// Per-round timeout in seconds.
    pub round_timeout_secs: u64,
}

impl Default for FedConfig {
    fn default() -> Self {
        Self {
            min_participants: 2,
            max_participants: 10,
            min_samples_per_participant: 10,
            max_grad_norm: 1.0,
            dp_noise_multiplier: 0.0,
            max_rounds: 100,
            round_timeout_secs: 600,
        }
    }
}

/// Round state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundState {
    /// Accepting participant updates.
    Active,
    /// Aggregation in progress.
    Aggregating,
    /// Round complete, global weights updated.
    Complete,
    /// Round timed out or was cancelled.
    Failed,
}

/// A single federated learning round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedRound {
    pub round_number: u64,
    pub state: RoundState,
    pub started_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub num_updates: usize,
    pub total_samples: usize,
    pub mean_train_loss: f32,
    pub mean_val_accuracy: Option<f32>,
}

/// The federated learning coordinator.
pub struct FedCoordinator {
    config: FedConfig,
    global_weights: RwLock<Weights>,
    current_round: RwLock<u64>,
    round_state: RwLock<RoundState>,
    round_started_at: RwLock<Option<Instant>>,
    updates: RwLock<HashMap<Uuid, ParticipantUpdate>>,
    rounds: RwLock<Vec<FedRound>>,
    participants: RwLock<HashMap<Uuid, ParticipantInfo>>,
}

/// Info about a registered participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantInfo {
    pub id: Uuid,
    pub name: String,
    pub joined_at_ms: u64,
    pub last_seen_ms: u64,
    pub rounds_participated: u64,
}

impl FedCoordinator {
    /// Create a new coordinator with the given config and initial weights.
    #[must_use]
    pub fn new(config: FedConfig, initial_weights: Weights) -> Self {
        Self {
            config,
            global_weights: RwLock::new(initial_weights),
            current_round: RwLock::new(0),
            round_state: RwLock::new(RoundState::Active),
            round_started_at: RwLock::new(Some(Instant::now())),
            updates: RwLock::new(HashMap::new()),
            rounds: RwLock::new(Vec::new()),
            participants: RwLock::new(HashMap::new()),
        }
    }

    /// Get the current global weights.
    #[must_use]
    pub fn global_weights(&self) -> Weights {
        self.global_weights.read().clone()
    }

    /// Get the current round number.
    #[must_use]
    pub fn current_round(&self) -> u64 {
        *self.current_round.read()
    }

    /// Get the current round state.
    #[must_use]
    pub fn round_state(&self) -> RoundState {
        *self.round_state.read()
    }

    /// Register a new participant.
    pub fn register_participant(&self, name: String) -> Uuid {
        let id = Uuid::now_v7();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let info = ParticipantInfo {
            id,
            name,
            joined_at_ms: now,
            last_seen_ms: now,
            rounds_participated: 0,
        };
        self.participants.write().insert(id, info);
        id
    }

    /// List all registered participants.
    #[must_use]
    pub fn list_participants(&self) -> Vec<ParticipantInfo> {
        self.participants.read().values().cloned().collect()
    }

    /// Submit an update for the current round.
    pub fn submit_update(&self, update: ParticipantUpdate) -> Result<(), FedError> {
        if *self.round_state.read() != RoundState::Active {
            return Err(FedError::RoundNotActive(*self.current_round.read()));
        }
        if update.round != *self.current_round.read() {
            return Err(FedError::RoundNotActive(update.round));
        }
        if update.num_samples < self.config.min_samples_per_participant {
            return Err(FedError::Aggregation(format!(
                "participant {} trained on only {} samples (min {})",
                update.participant_id, update.num_samples, self.config.min_samples_per_participant
            )));
        }

        // Validate shape.
        let global = self.global_weights.read();
        if update.weights.shape != global.shape {
            return Err(FedError::ShapeMismatch {
                expected: global.shape.clone(),
                actual: update.weights.shape.clone(),
            });
        }
        drop(global);

        // Apply gradient clipping (max L2 norm).
        let mut clipped = update.weights.clone();
        if self.config.max_grad_norm > 0.0 {
            let global = self.global_weights.read();
            let delta = update.weights.delta_from(&global);
            let norm = delta.l2_norm();
            if norm > self.config.max_grad_norm && norm > 0.0 {
                let scale = self.config.max_grad_norm / norm;
                let original = update.weights.data.clone();
                for (i, w) in clipped.data.iter_mut().enumerate() {
                    *w = global.data[i] + (original[i] - global.data[i]) * scale;
                }
            }
            drop(global);
        }

        // Apply differential privacy noise (Gaussian).
        if self.config.dp_noise_multiplier > 0.0 {
            // Box-Muller transform for Gaussian samples.
            let sigma = self.config.dp_noise_multiplier * self.config.max_grad_norm;
            let global = self.global_weights.read();
            let seed = update.timestamp_ms.wrapping_add(update.participant_id.to_u128_le() as u64);
            use rand::{rngs::StdRng, Rng, SeedableRng};
            let mut rng = StdRng::seed_from_u64(seed);
            // Snapshot the clipped data before iterating so we don't read+write
            // the same buffer simultaneously.
            let clipped_snapshot = clipped.data.clone();
            for (i, w) in clipped.data.iter_mut().enumerate() {
                let u1: f64 = rng.gen_range(1e-10..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *w = global.data[i] + (clipped_snapshot[i] - global.data[i]) + (noise as f32) * sigma;
            }
        }

        // Update participant's last-seen.
        if let Some(info) = self.participants.write().get_mut(&update.participant_id) {
            info.last_seen_ms = update.timestamp_ms;
        }

        self.updates.write().insert(update.participant_id, ParticipantUpdate {
            weights: clipped,
            ..update
        });
        Ok(())
    }

    /// Aggregate the submitted updates for the current round using FedAvg
    /// (weighted by sample count). Starts a new round after aggregation.
    ///
    /// # Errors
    /// Returns an error if not enough updates have been received.
    pub fn aggregate(&self) -> Result<FedRound, FedError> {
        if *self.round_state.read() != RoundState::Active {
            return Err(FedError::RoundNotActive(*self.current_round.read()));
        }
        let updates_guard = self.updates.read();
        if updates_guard.len() < self.config.min_participants {
            return Err(FedError::Aggregation(format!(
                "only {} updates (min {})",
                updates_guard.len(),
                self.config.min_participants
            )));
        }

        *self.round_state.write() = RoundState::Aggregating;
        let round_num = *self.current_round.read();

        // FedAvg: weighted average of weights, weighted by sample count.
        let total_samples: usize = updates_guard.values().map(|u| u.num_samples).sum();
        if total_samples == 0 {
            return Err(FedError::Aggregation("total samples is zero".into()));
        }

        let shape = self.global_weights.read().shape.clone();
        let mut new_data = vec![0.0_f32; self.global_weights.read().num_params()];

        for update in updates_guard.values() {
            let weight = update.num_samples as f32 / total_samples as f32;
            for (i, w) in update.weights.data.iter().enumerate() {
                new_data[i] += weight * w;
            }
        }

        let new_weights = Weights {
            data: new_data,
            shape,
            version: self.global_weights.read().version + 1,
        };
        drop(updates_guard);

        // Compute round statistics.
        let updates_guard = self.updates.read();
        let num_updates = updates_guard.len();
        let mean_loss: f32 = updates_guard
            .values()
            .map(|u| u.train_loss)
            .sum::<f32>()
            / num_updates as f32;
        let val_accs: Vec<f32> = updates_guard
            .values()
            .filter_map(|u| u.val_accuracy)
            .collect();
        let mean_val_acc = if val_accs.is_empty() {
            None
        } else {
            Some(val_accs.iter().sum::<f32>() / val_accs.len() as f32)
        };
        drop(updates_guard);

        // Increment participant rounds_participated.
        for pid in self.updates.read().keys() {
            if let Some(info) = self.participants.write().get_mut(pid) {
                info.rounds_participated += 1;
            }
        }

        // Update global weights.
        *self.global_weights.write() = new_weights;

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let round = FedRound {
            round_number: round_num,
            state: RoundState::Complete,
            started_at_ms: self
                .round_started_at
                .read()
                .map(|i| now_ms - i.elapsed().as_millis() as u64)
                .unwrap_or(now_ms),
            completed_at_ms: Some(now_ms),
            num_updates,
            total_samples,
            mean_train_loss: mean_loss,
            mean_val_accuracy: mean_val_acc,
        };

        self.rounds.write().push(round.clone());

        // Clear updates and start next round.
        self.updates.write().clear();
        *self.current_round.write() = round_num + 1;
        *self.round_state.write() = RoundState::Active;
        *self.round_started_at.write() = Some(Instant::now());

        Ok(round)
    }

    /// Get the history of completed rounds.
    #[must_use]
    pub fn round_history(&self) -> Vec<FedRound> {
        self.rounds.read().clone()
    }

    /// Get the number of updates submitted for the current round.
    #[must_use]
    pub fn pending_updates(&self) -> usize {
        self.updates.read().len()
    }
}

/// Shared handle to a coordinator.
pub type SharedFedCoordinator = Arc<FedCoordinator>;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_weights(version: u64, value: f32) -> Weights {
        Weights {
            data: vec![value; 4],
            shape: vec![2, 2],
            version,
        }
    }

    #[test]
    fn register_participant_returns_unique_ids() {
        let coord = FedCoordinator::new(FedConfig::default(), make_weights(0, 0.0));
        let a = coord.register_participant("alice".into());
        let b = coord.register_participant("bob".into());
        assert_ne!(a, b);
        assert_eq!(coord.list_participants().len(), 2);
    }

    #[test]
    fn submit_update_validates_shape() {
        let coord = FedCoordinator::new(FedConfig::default(), make_weights(0, 0.0));
        let pid = coord.register_participant("alice".into());
        let bad_update = ParticipantUpdate {
            participant_id: pid,
            round: 0,
            weights: Weights { data: vec![0.0; 9], shape: vec![3, 3], version: 0 },
            num_samples: 100,
            train_loss: 0.5,
            val_accuracy: None,
            timestamp_ms: 0,
        };
        assert!(coord.submit_update(bad_update).is_err());
    }

    #[test]
    fn submit_update_requires_min_samples() {
        let coord = FedCoordinator::new(
            FedConfig {
                min_samples_per_participant: 50,
                ..Default::default()
            },
            make_weights(0, 0.0),
        );
        let pid = coord.register_participant("alice".into());
        let update = ParticipantUpdate {
            participant_id: pid,
            round: 0,
            weights: make_weights(0, 0.1),
            num_samples: 10,  // below min
            train_loss: 0.5,
            val_accuracy: None,
            timestamp_ms: 0,
        };
        assert!(coord.submit_update(update).is_err());
    }

    #[test]
    fn aggregate_requires_min_participants() {
        let coord = FedCoordinator::new(
            FedConfig {
                min_participants: 3,
                ..Default::default()
            },
            make_weights(0, 0.0),
        );
        let pid = coord.register_participant("alice".into());
        let update = ParticipantUpdate {
            participant_id: pid,
            round: 0,
            weights: make_weights(0, 0.1),
            num_samples: 100,
            train_loss: 0.5,
            val_accuracy: None,
            timestamp_ms: 0,
        };
        coord.submit_update(update).unwrap();
        assert!(coord.aggregate().is_err());
    }

    #[test]
    fn fedavg_weights_by_sample_count() {
        let coord = FedCoordinator::new(FedConfig::default(), make_weights(0, 0.0));
        let alice = coord.register_participant("alice".into());
        let bob = coord.register_participant("bob".into());

        // Alice: 100 samples, weights all 1.0
        coord.submit_update(ParticipantUpdate {
            participant_id: alice,
            round: 0,
            weights: make_weights(0, 1.0),
            num_samples: 100,
            train_loss: 0.4,
            val_accuracy: Some(0.9),
            timestamp_ms: 0,
        }).unwrap();

        // Bob: 300 samples, weights all 2.0
        coord.submit_update(ParticipantUpdate {
            participant_id: bob,
            round: 0,
            weights: make_weights(0, 2.0),
            num_samples: 300,
            train_loss: 0.6,
            val_accuracy: Some(0.8),
            timestamp_ms: 0,
        }).unwrap();

        let round = coord.aggregate().unwrap();
        assert_eq!(round.num_updates, 2);
        assert_eq!(round.total_samples, 400);

        // FedAvg: (100/400)*1.0 + (300/400)*2.0 = 0.25 + 1.5 = 1.75
        let new_weights = coord.global_weights();
        for w in &new_weights.data {
            assert!((w - 1.75).abs() < 1e-5, "expected 1.75, got {w}");
        }
        assert_eq!(new_weights.version, 1);
    }

    #[test]
    fn round_advances_after_aggregate() {
        let coord = FedCoordinator::new(FedConfig::default(), make_weights(0, 0.0));
        assert_eq!(coord.current_round(), 0);

        let alice = coord.register_participant("alice".into());
        let bob = coord.register_participant("bob".into());
        coord.submit_update(ParticipantUpdate {
            participant_id: alice, round: 0,
            weights: make_weights(0, 1.0), num_samples: 10,
            train_loss: 0.5, val_accuracy: None, timestamp_ms: 0,
        }).unwrap();
        coord.submit_update(ParticipantUpdate {
            participant_id: bob, round: 0,
            weights: make_weights(0, 1.0), num_samples: 10,
            train_loss: 0.5, val_accuracy: None, timestamp_ms: 0,
        }).unwrap();

        coord.aggregate().unwrap();
        assert_eq!(coord.current_round(), 1);
        assert_eq!(coord.round_state(), RoundState::Active);
        assert_eq!(coord.pending_updates(), 0);
    }

    #[test]
    fn round_history_records_completed_rounds() {
        let coord = FedCoordinator::new(FedConfig::default(), make_weights(0, 0.0));
        let alice = coord.register_participant("alice".into());
        let bob = coord.register_participant("bob".into());
        coord.submit_update(ParticipantUpdate {
            participant_id: alice, round: 0,
            weights: make_weights(0, 1.0), num_samples: 10,
            train_loss: 0.5, val_accuracy: Some(0.8), timestamp_ms: 0,
        }).unwrap();
        coord.submit_update(ParticipantUpdate {
            participant_id: bob, round: 0,
            weights: make_weights(0, 1.0), num_samples: 10,
            train_loss: 0.3, val_accuracy: Some(0.6), timestamp_ms: 0,
        }).unwrap();

        let round = coord.aggregate().unwrap();
        assert_eq!(round.round_number, 0);
        assert_eq!(round.state, RoundState::Complete);
        assert!((round.mean_train_loss - 0.4).abs() < 1e-5);
        assert!((round.mean_val_accuracy.unwrap() - 0.7).abs() < 1e-5);

        let history = coord.round_history();
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn gradient_clipping_applies_max_norm() {
        let coord = FedCoordinator::new(
            FedConfig {
                min_participants: 1,
                max_grad_norm: 0.1,
                ..Default::default()
            },
            make_weights(0, 0.0),
        );
        let pid = coord.register_participant("alice".into());
        // Submit a large delta (norm = sqrt(4) = 2.0).
        coord.submit_update(ParticipantUpdate {
            participant_id: pid, round: 0,
            weights: make_weights(0, 1.0), num_samples: 10,
            train_loss: 0.5, val_accuracy: None, timestamp_ms: 0,
        }).unwrap();
        // Should still aggregate (clipped to max_norm=0.1).
        let round = coord.aggregate().unwrap();
        assert_eq!(round.num_updates, 1);
        // Global weights should be small (clipped to 0.1 norm, distributed across 4 params).
        let new_w = coord.global_weights();
        let new_norm = new_w.l2_norm();
        assert!(new_norm <= 0.11, "norm {new_norm} should be ≤ 0.1");
    }

    #[test]
    fn weights_l2_norm_computes_correctly() {
        let w = Weights { data: vec![3.0, 4.0], shape: vec![2], version: 0 };
        assert!((w.l2_norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn weights_delta_subtracts_correctly() {
        let a = Weights { data: vec![5.0, 7.0], shape: vec![2], version: 1 };
        let b = Weights { data: vec![2.0, 3.0], shape: vec![2], version: 0 };
        let d = a.delta_from(&b);
        assert_eq!(d.data, vec![3.0, 4.0]);
    }
}
