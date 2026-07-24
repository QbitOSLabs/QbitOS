//! Preference learning from pairwise comparisons.
//!
//! Implements a Bradley-Terry model for learning preferences
//! from pairwise comparisons (chosen vs rejected).

use ahash::AHashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// A preference pair: `chosen` is preferred over `rejected`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    /// The prompt/question.
    pub prompt: String,
    /// The preferred response.
    pub chosen: String,
    /// The rejected response.
    pub rejected: String,
}

/// A preference model using Bradley-Terry scores.
///
/// Each response has a learned "quality score" θ. The probability that
/// response A is preferred over response B is:
/// `P(A > B) = sigmoid(θ_A - θ_B)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceModel {
    /// Quality scores for each response (keyed by response hash).
    scores: AHashMap<String, f64>,
    /// Learning rate.
    learning_rate: f64,
}

impl Default for PreferenceModel {
    fn default() -> Self {
        Self { scores: AHashMap::new(), learning_rate: 0.01 }
    }
}

impl PreferenceModel {
    /// Create a new model.
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self { scores: AHashMap::new(), learning_rate }
    }

    /// Get the quality score for a response (defaults to 0.0).
    #[must_use]
    pub fn score(&self, response: &str) -> f64 {
        self.scores.get(response).copied().unwrap_or(0.0)
    }

    /// Predict P(chosen > rejected) = sigmoid(θ_chosen - θ_rejected).
    #[must_use]
    pub fn predict(&self, chosen: &str, rejected: &str) -> f64 {
        let diff = self.score(chosen) - self.score(rejected);
        1.0 / (1.0 + (-diff).exp())
    }

    /// Update the model from a preference pair using gradient ascent.
    pub fn update(&mut self, pair: &PreferencePair) {
        let _theta_chosen = self.score(&pair.chosen);
        let _theta_rejected = self.score(&pair.rejected);
        let prob = self.predict(&pair.chosen, &pair.rejected);

        // Gradient: d/dθ_chosen [log P] = 1 - prob
        // Gradient: d/dθ_rejected [log P] = -(1 - prob) = prob - 1
        let grad = 1.0 - prob;
        *self.scores.entry(pair.chosen.clone()).or_insert(0.0) += self.learning_rate * grad;
        *self.scores.entry(pair.rejected.clone()).or_insert(0.0) -= self.learning_rate * grad;
    }

    /// Train on a batch of preference pairs.
    pub fn train(&mut self, pairs: &[PreferencePair], epochs: usize) {
        for _ in 0..epochs {
            for pair in pairs {
                self.update(pair);
            }
        }
    }

    /// Rank responses by quality score (best first).
    #[must_use]
    pub fn rank(&self, responses: &[String]) -> Vec<(String, f64)> {
        let mut ranked: Vec<(String, f64)> = responses.iter()
            .map(|r| (r.clone(), self.score(r)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Number of learned responses.
    #[must_use]
    pub fn len(&self) -> usize { self.scores.len() }

    /// Returns `true` if the model is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.scores.is_empty() }
}

/// A preference learner that collects pairs and trains the model.
pub struct PreferenceLearner {
    model: RwLock<PreferenceModel>,
    pairs: RwLock<Vec<PreferencePair>>,
}

impl Default for PreferenceLearner {
    fn default() -> Self {
        Self { model: RwLock::new(PreferenceModel::default()), pairs: RwLock::new(Vec::new()) }
    }
}

impl PreferenceLearner {
    /// Create a new learner.
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            model: RwLock::new(PreferenceModel::new(learning_rate)),
            pairs: RwLock::new(Vec::new()),
        }
    }

    /// Add a preference pair and update the model incrementally.
    pub fn add_preference(&self, pair: PreferencePair) {
        self.model.write().update(&pair);
        self.pairs.write().push(pair);
    }

    /// Train the model on all collected pairs for additional epochs.
    pub fn train_batch(&self, epochs: usize) {
        let pairs = self.pairs.read().clone();
        self.model.write().train(&pairs, epochs);
    }

    /// Get the current preference model.
    #[must_use]
    pub fn model(&self) -> PreferenceModel {
        self.model.read().clone()
    }

    /// Get all collected pairs.
    #[must_use]
    pub fn pairs(&self) -> Vec<PreferencePair> {
        self.pairs.read().clone()
    }

    /// Rank responses by learned preference.
    #[must_use]
    pub fn rank(&self, responses: &[String]) -> Vec<(String, f64)> {
        self.model.read().rank(responses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_model_learns() {
        let mut model = PreferenceModel::new(0.1);
        let pair = PreferencePair {
            prompt: "What is 2+2?".into(),
            chosen: "4".into(),
            rejected: "5".into(),
        };
        // Train for many epochs
        model.train(&[pair], 100);
        // The chosen response should have a higher score
        assert!(model.score("4") > model.score("5"));
        // Prediction should be > 0.5 (favoring chosen)
        assert!(model.predict("4", "5") > 0.5);
    }

    #[test]
    fn preference_learner_incremental() {
        let learner = PreferenceLearner::new(0.1);
        learner.add_preference(PreferencePair {
            prompt: "q1".into(), chosen: "good".into(), rejected: "bad".into(),
        });
        learner.add_preference(PreferencePair {
            prompt: "q2".into(), chosen: "good".into(), rejected: "bad".into(),
        });
        learner.add_preference(PreferencePair {
            prompt: "q3".into(), chosen: "good".into(), rejected: "bad".into(),
        });
        let ranked = learner.rank(&["good".into(), "bad".into()]);
        assert_eq!(ranked[0].0, "good");
        assert!(ranked[0].1 > ranked[1].1);
    }

    #[test]
    fn bradley_terry_predict() {
        let mut model = PreferenceModel::new(0.01);
        model.scores.insert("a".into(), 2.0);
        model.scores.insert("b".into(), 0.0);
        // P(a > b) = sigmoid(2 - 0) = sigmoid(2) ≈ 0.88
        let prob = model.predict("a", "b");
        assert!(prob > 0.8 && prob < 0.95, "expected ~0.88, got {prob}");
    }
}
