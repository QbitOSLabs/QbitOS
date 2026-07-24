//! Composite vision module: bundles the image processor, vision encoder,
//! OCR, object detector, visual memory, visual reasoner, and cross-modal
//! aligner into a single handle.

use std::sync::Arc;

use qbitos_memory::embedding::SharedEmbeddingModel;

use crate::cross_modal_aligner::CrossModalAligner;
use crate::image_processor::ImageProcessor;
use crate::object_detection::ObjectDetector;
use crate::ocr::OcrEngine;
use crate::visual_memory::VisualMemory;
use crate::visual_reasoning::VisualReasoner;
use crate::vision_encoder::{VisionEncoder, VisionEncoderConfig};

/// The composite vision module.
pub struct VisionModule {
    /// The image processor.
    pub processor: Arc<ImageProcessor>,
    /// The vision encoder.
    pub encoder: Arc<VisionEncoder>,
    /// The OCR engine.
    pub ocr: Arc<OcrEngine>,
    /// The object detector.
    pub detector: Arc<ObjectDetector>,
    /// The visual memory.
    pub memory: Arc<VisualMemory>,
    /// The visual reasoner.
    pub reasoner: Arc<VisualReasoner>,
    /// The cross-modal aligner (vision ↔ text).
    pub aligner: Arc<CrossModalAligner>,
}

impl VisionModule {
    /// Build a new vision module with the given embedding model and encoder config.
    #[must_use]
    pub fn new(embedder: SharedEmbeddingModel, encoder_config: VisionEncoderConfig) -> Self {
        let latent_dim = encoder_config.embed_dim.min(256);
        Self {
            processor: Arc::new(ImageProcessor::new()),
            encoder: Arc::new(VisionEncoder::new(encoder_config)),
            ocr: Arc::new(OcrEngine::new()),
            detector: Arc::new(ObjectDetector::new()),
            memory: Arc::new(VisualMemory::new(embedder)),
            reasoner: Arc::new(VisualReasoner::new()),
            aligner: Arc::new(CrossModalAligner::new(latent_dim, latent_dim, latent_dim)),
        }
    }

    /// Build a new vision module with default configuration.
    #[must_use]
    pub fn default_config(embedder: SharedEmbeddingModel) -> Self {
        Self::new(embedder, VisionEncoderConfig::default())
    }

    /// Load all models.
    ///
    /// # Errors
    /// Returns an error if any model fails to load.
    pub async fn load_all(&self) -> anyhow::Result<()> {
        self.encoder.load().await?;
        self.ocr.load().await?;
        self.detector.load().await?;
        self.reasoner.load().await?;
        Ok(())
    }
}
