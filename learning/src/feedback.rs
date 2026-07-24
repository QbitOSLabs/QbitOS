//! Feedback collection and aggregation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Type of feedback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackType {
    /// Thumbs up / positive.
    Positive,
    /// Thumbs down / negative.
    Negative,
    /// Star rating (1-5).
    Rating(u8),
    /// Text comment.
    Comment(String),
    /// Correction (expected vs actual).
    Correction {
        /// Expected output.
        expected: String,
        /// Actual output received.
        actual: String },
}

/// A single feedback entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feedback {
    /// Unique id.
    pub id: uuid::Uuid,
    /// What the feedback is about (e.g., response id).
    pub target: String,
    /// The feedback type.
    pub feedback_type: FeedbackType,
    /// When the feedback was given.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional user/session id.
    pub user: Option<String>,
}

impl Feedback {
    /// Create new positive feedback.
    #[must_use]
    pub fn positive(target: impl Into<String>) -> Self {
        Self { id: uuid::Uuid::now_v7(), target: target.into(), feedback_type: FeedbackType::Positive,
            timestamp: chrono::Utc::now(), user: None }
    }

    /// Create new negative feedback.
    #[must_use]
    pub fn negative(target: impl Into<String>) -> Self {
        Self { id: uuid::Uuid::now_v7(), target: target.into(), feedback_type: FeedbackType::Negative,
            timestamp: chrono::Utc::now(), user: None }
    }

    /// Create a rating.
    #[must_use]
    pub fn rating(target: impl Into<String>, stars: u8) -> Self {
        Self { id: uuid::Uuid::now_v7(), target: target.into(), feedback_type: FeedbackType::Rating(stars),
            timestamp: chrono::Utc::now(), user: None }
    }

    /// Create a correction.
    #[must_use]
    pub fn correction(target: impl Into<String>, expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self { id: uuid::Uuid::now_v7(), target: target.into(),
            feedback_type: FeedbackType::Correction { expected: expected.into(), actual: actual.into() },
            timestamp: chrono::Utc::now(), user: None }
    }

    /// Returns the numeric score of this feedback (-1, 0, +1, or rating/5).
    #[must_use]
    pub fn score(&self) -> f64 {
        match &self.feedback_type {
            FeedbackType::Positive => 1.0,
            FeedbackType::Negative => -1.0,
            FeedbackType::Rating(stars) => (*stars as f64 - 3.0) / 2.0, // 1→-1, 3→0, 5→+1
            FeedbackType::Comment(_) => 0.0,
            FeedbackType::Correction { .. } => -0.5,
        }
    }
}

/// Aggregated feedback statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStats {
    /// Total feedback count.
    pub count: usize,
    /// Mean score.
    pub mean_score: f64,
    /// Positive count.
    pub positive: usize,
    /// Negative count.
    pub negative: usize,
    /// Average rating (if applicable).
    pub avg_rating: Option<f64>,
}

/// Feedback collector — accumulates and aggregates user feedback.
pub struct FeedbackCollector {
    feedback: Arc<RwLock<Vec<Feedback>>>,
}

impl Default for FeedbackCollector {
    fn default() -> Self { Self::new() }
}

impl FeedbackCollector {
    /// Create a new collector.
    #[must_use]
    pub fn new() -> Self {
        Self { feedback: Arc::new(RwLock::new(Vec::new())) }
    }

    /// Add feedback.
    pub fn add(&self, feedback: Feedback) {
        self.feedback.write().push(feedback);
    }

    /// Get all feedback for a target.
    #[must_use]
    pub fn get_for_target(&self, target: &str) -> Vec<Feedback> {
        self.feedback.read().iter().filter(|f| f.target == target).cloned().collect()
    }

    /// Get statistics for a target.
    #[must_use]
    pub fn stats_for(&self, target: &str) -> FeedbackStats {
        let feedback = self.get_for_target(target);
        let count = feedback.len();
        if count == 0 {
            return FeedbackStats { count: 0, mean_score: 0.0, positive: 0, negative: 0, avg_rating: None };
        }

        let scores: Vec<f64> = feedback.iter().map(|f| f.score()).collect();
        let mean_score = scores.iter().sum::<f64>() / count as f64;
        let positive = feedback.iter().filter(|f| f.score() > 0.0).count();
        let negative = feedback.iter().filter(|f| f.score() < 0.0).count();

        let ratings: Vec<u8> = feedback.iter()
            .filter_map(|f| match &f.feedback_type {
                FeedbackType::Rating(stars) => Some(*stars),
                _ => None,
            })
            .collect();
        let avg_rating = if ratings.is_empty() { None } else {
            Some(ratings.iter().map(|&r| r as f64).sum::<f64>() / ratings.len() as f64)
        };

        FeedbackStats { count, mean_score, positive, negative, avg_rating }
    }

    /// Get all feedback.
    #[must_use]
    pub fn all(&self) -> Vec<Feedback> {
        self.feedback.read().clone()
    }

    /// Get all corrections.
    #[must_use]
    pub fn corrections(&self) -> Vec<Feedback> {
        self.feedback.read().iter()
            .filter(|f| matches!(f.feedback_type, FeedbackType::Correction { .. }))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_scores() {
        assert_eq!(Feedback::positive("x").score(), 1.0);
        assert_eq!(Feedback::negative("x").score(), -1.0);
        assert_eq!(Feedback::rating("x", 5).score(), 1.0);
        assert_eq!(Feedback::rating("x", 1).score(), -1.0);
        assert_eq!(Feedback::rating("x", 3).score(), 0.0);
    }

    #[test]
    fn collector_aggregates() {
        let collector = FeedbackCollector::new();
        collector.add(Feedback::positive("response1"));
        collector.add(Feedback::positive("response1"));
        collector.add(Feedback::negative("response1"));
        collector.add(Feedback::rating("response1", 5));
        collector.add(Feedback::positive("response2"));

        let stats = collector.stats_for("response1");
        assert_eq!(stats.count, 4);
        assert_eq!(stats.positive, 3);
        assert_eq!(stats.negative, 1);
        assert!(stats.mean_score > 0.0);
        assert_eq!(stats.avg_rating, Some(5.0));
    }

    #[test]
    fn corrections_extracted() {
        let collector = FeedbackCollector::new();
        collector.add(Feedback::correction("r1", "correct answer", "wrong answer"));
        collector.add(Feedback::positive("r2"));
        let corrections = collector.corrections();
        assert_eq!(corrections.len(), 1);
    }
}
