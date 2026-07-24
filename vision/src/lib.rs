//! QbitOS Vision Intelligence
//!
//! Native visual perception pipeline:
//!
//! ```text
//! Image / Camera Input
//!   → Vision Encoder
//!   → Visual Understanding
//!   → Visual Memory
//!   → Reasoning
//!   → Action
//! ```
//!
//! Supports Vision Transformers (ViT), CLIP-style encoders, SigLIP, and
//! multimodal LLM architectures. Implemented on top of `image-rs` for I/O
//! and `candle-core` (via `qbitos-neural-engine`) for tensor ops.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod advanced_processing;
pub mod cross_modal_aligner;
pub mod window_attention;
pub mod selective_ssm;
pub mod image_processor;
pub mod ocr;
pub mod object_detection;
pub mod vision_encoder;
pub mod visual_memory;
pub mod visual_reasoning;

pub use advanced_processing::{
    canny_edges, gaussian_blur, histogram, histogram_equalize, kmeans_quantize, otsu_threshold,
    sobel_edges, threshold, GrayImage, RgbImage,
};
pub use image_processor::{ImageInput, ImageProcessor, ProcessedImage};
pub use ocr::{OcrEngine, OcrResult, TextRegion};
pub use object_detection::{BoundingBox, DetectedObject, ObjectDetector};
pub use vision_encoder::{VisionEncoder, VisionEncoderConfig, VisionEncoderKind};
pub use visual_memory::{VisualMemory, VisualMemoryEntry};
pub use visual_reasoning::{VisualReasoner, VisualReasoningResult};

/// Re-export of the composite vision module.
pub use vision_module::VisionModule;
pub use cross_modal_aligner::CrossModalAligner;

mod vision_module;
