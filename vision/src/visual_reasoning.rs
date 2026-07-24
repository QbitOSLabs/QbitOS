//! Visual reasoning.
//!
//! Combines visual perception with the cognitive kernel's reasoning engine
//! to answer questions about images (visual question answering), generate
//! captions, and detect anomalies.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::image_processor::ProcessedImage;
use crate::object_detection::DetectedObject;
use crate::ocr::OcrResult;

/// A visual reasoning result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualReasoningResult {
    /// Natural-language answer.
    pub answer: String,
    /// Detected objects.
    pub objects: Vec<DetectedObject>,
    /// OCR result.
    pub ocr: Option<OcrResult>,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Errors emitted by the visual reasoner.
#[derive(Debug, Error)]
pub enum VisualReasoningError {
    /// The reasoner is not loaded.
    #[error("visual reasoner not loaded")]
    NotLoaded,
    /// Reasoning failed.
    #[error("visual reasoning failed: {0}")]
    ReasoningFailed(String),
}

/// The visual reasoner.
pub struct VisualReasoner {
    loaded: std::sync::atomic::AtomicBool,
}

impl Default for VisualReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualReasoner {
    /// Create a new visual reasoner.
    #[must_use]
    pub fn new() -> Self {
        Self { loaded: std::sync::atomic::AtomicBool::new(false) }
    }

    /// Load the reasoner.
    pub async fn load(&self) -> Result<(), VisualReasoningError> {
        self.loaded.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Answer a question about an image.
    ///
    /// # Errors
    /// Returns [`VisualReasoningError`] if reasoning fails.
    pub async fn answer_question(
        &self,
        _image: &ProcessedImage,
        question: &str,
        objects: &[DetectedObject],
        ocr: Option<&OcrResult>,
    ) -> Result<VisualReasoningResult, VisualReasoningError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Err(VisualReasoningError::NotLoaded);
        }
        // Stub: produce a templated answer based on detected objects and OCR.
        let obj_summary = if objects.is_empty() {
            "no objects".to_string()
        } else {
            objects.iter()
                .map(|o| format!("{} ({:.0}%)", o.label, o.confidence * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let ocr_summary = ocr.map(|r| r.full_text.clone()).unwrap_or_default();
        let answer = format!(
            "Question: {question}\nDetected: {obj_summary}.\nText: {ocr_summary}"
        );
        Ok(VisualReasoningResult {
            answer,
            objects: objects.to_vec(),
            ocr: ocr.cloned(),
            confidence: 0.5,
            duration_ms: 0,
        })
    }

    /// Generate a caption for an image.
    ///
    /// # Errors
    /// Returns [`VisualReasoningError`] if captioning fails.
    pub async fn caption(&self, _image: &ProcessedImage, objects: &[DetectedObject]) -> Result<String, VisualReasoningError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Err(VisualReasoningError::NotLoaded);
        }
        if objects.is_empty() {
            Ok("An image with no detected objects.".into())
        } else {
            Ok(format!(
                "An image containing {}.",
                objects.iter().map(|o| o.label.clone()).collect::<Vec<_>>().join(", ")
            ))
        }
    }
}
