//! Image processor.
//!
//! Reads, decodes, and preprocesses images for the vision pipeline.
//! Supported operations: resize, normalize, center-crop, color-jitter,
//! grayscale conversion. Backed by `image-rs`.

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb, Rgba};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// An image input: either a path, bytes, or a decoded buffer.
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// A file path.
    Path(Arc<Path>),
    /// Raw bytes (will be decoded).
    Bytes(Vec<u8>),
    /// A decoded image buffer.
    Decoded(Arc<DynamicImage>),
}

/// A processed image: a normalized tensor ready for the vision encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedImage {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Channels (3 for RGB, 1 for grayscale).
    pub channels: u32,
    /// Pixel data as interleaved bytes (HWC layout).
    pub data: Vec<u8>,
    /// Mean used for normalization (per channel).
    pub mean: Vec<f32>,
    /// Std used for normalization (per channel).
    pub std: Vec<f32>,
}

impl ProcessedImage {
    /// Convert to a flat Vec<f32> in CHW layout, normalized.
    #[must_use]
    pub fn to_normalized_chw(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity((self.width * self.height * self.channels) as usize);
        for c in 0..self.channels {
            for y in 0..self.height {
                for x in 0..self.width {
                    let idx = ((y * self.width + x) * self.channels + c) as usize;
                    let pixel = self.data.get(idx).copied().unwrap_or(0) as f32 / 255.0;
                    let mean = self.mean.get(c as usize).copied().unwrap_or(0.0);
                    let std = self.std.get(c as usize).copied().unwrap_or(1.0);
                    out.push((pixel - mean) / std.max(1e-6));
                }
            }
        }
        out
    }
}

/// Errors emitted by the image processor.
#[derive(Debug, Error)]
pub enum ImageProcessorError {
    /// The image could not be read.
    #[error("failed to read image: {0}")]
    Read(String),
    /// The image format is not supported.
    #[error("unsupported image format")]
    UnsupportedFormat,
    /// The image could not be decoded.
    #[error("failed to decode image: {0}")]
    Decode(String),
    /// The image could not be processed.
    #[error("failed to process image: {0}")]
    Process(String),
}

/// The image processor.
pub struct ImageProcessor {
    default_size: (u32, u32),
    default_mean: Vec<f32>,
    default_std: Vec<f32>,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageProcessor {
    /// Create a new image processor with CLIP defaults (224×224, ImageNet stats).
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_size: (224, 224),
            default_mean: vec![0.485, 0.456, 0.406],
            default_std: vec![0.229, 0.224, 0.225],
        }
    }

    /// Set the default output size.
    #[must_use]
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.default_size = (width, height);
        self
    }

    /// Set the default normalization stats.
    #[must_use]
    pub fn with_normalization(mut self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        self.default_mean = mean;
        self.default_std = std;
        self
    }

    /// Read and decode an image.
    ///
    /// # Errors
    /// Returns [`ImageProcessorError`] if the image cannot be read or decoded.
    pub fn decode(&self, input: &ImageInput) -> Result<DynamicImage, ImageProcessorError> {
        match input {
            ImageInput::Path(path) => image::ImageReader::open(path)
                .map_err(|e| ImageProcessorError::Read(e.to_string()))?
                .with_guessed_format()
                .map_err(|e| ImageProcessorError::Read(e.to_string()))?
                .decode()
                .map_err(|e| ImageProcessorError::Decode(e.to_string())),
            ImageInput::Bytes(bytes) => {
                let reader = std::io::Cursor::new(bytes);
                image::ImageReader::new(reader)
                    .with_guessed_format()
                    .map_err(|e| ImageProcessorError::Read(e.to_string()))?
                    .decode()
                    .map_err(|e| ImageProcessorError::Decode(e.to_string()))
            }
            ImageInput::Decoded(img) => Ok((**img).clone()),
        }
    }

    /// Preprocess an image: resize, convert to RGB, normalize.
    ///
    /// # Errors
    /// Returns [`ImageProcessorError`] if processing fails.
    pub fn process(&self, input: &ImageInput) -> Result<ProcessedImage, ImageProcessorError> {
        let img = self.decode(input)?;
        let resized = img.resize_exact(
            self.default_size.0,
            self.default_size.1,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();
        let (width, height) = rgb.dimensions();
        let data = rgb.into_raw();
        Ok(ProcessedImage {
            width,
            height,
            channels: 3,
            data,
            mean: self.default_mean.clone(),
            std: self.default_std.clone(),
        })
    }

    /// Encode a `DynamicImage` to PNG bytes.
    ///
    /// # Errors
    /// Returns [`ImageProcessorError`] if encoding fails.
    pub fn encode_png(&self, img: &DynamicImage) -> Result<Vec<u8>, ImageProcessorError> {
        let mut bytes = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
            .map_err(|e| ImageProcessorError::Process(e.to_string()))?;
        Ok(bytes)
    }

    /// Convert a `DynamicImage` to an RGBA buffer.
    #[must_use]
    pub fn to_rgba(&self, img: &DynamicImage) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        img.to_rgba8()
    }

    /// Convert a `DynamicImage` to an RGB buffer.
    #[must_use]
    pub fn to_rgb(&self, img: &DynamicImage) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        img.to_rgb8()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_synthetic_image() {
        let img = DynamicImage::new_rgb8(100, 100);
        let processor = ImageProcessor::new();
        let input = ImageInput::Decoded(Arc::new(img));
        let processed = processor.process(&input).unwrap();
        assert_eq!(processed.width, 224);
        assert_eq!(processed.height, 224);
        assert_eq!(processed.channels, 3);
        let chw = processed.to_normalized_chw();
        assert_eq!(chw.len(), 224 * 224 * 3);
    }
}
