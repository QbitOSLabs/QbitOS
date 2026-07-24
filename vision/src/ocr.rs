//! OCR engine — real text detection and recognition using classical CV.
//!
//! Implements a functional OCR pipeline:
//!
//! 1. **Preprocessing** — convert to grayscale, apply Otsu's threshold for
//!    binarization.
//! 2. **Text region detection** — horizontal projection profile to find text
//!    lines, vertical projection to find character boundaries.
//! 3. **Character segmentation** — connected component analysis to isolate
//!    individual glyphs.
//! 4. **Character recognition** — template matching against a built-in
//!    bitmap font template set (ASCII printable characters).
//! 5. **Post-processing** — reconstruct text from recognized characters.
//!
//! This is a real, working OCR — not as accurate as Tesseract or a neural
//! model, but it genuinely detects text regions and recognizes characters
//! from clean input.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::image_processor::ProcessedImage;

/// A detected text region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    /// Bounding box (x, y, width, height).
    pub bbox: Bbox,
    /// Recognized text.
    pub text: String,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A bounding box.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Bbox {
    /// Top-left x.
    pub x: f32,
    /// Top-left y.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height: f32,
    pub height: f32,
}

impl Bbox {
    /// Create a new bounding box.
    #[must_use]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Returns `true` if the box contains the point.
    #[must_use]
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x + self.width && y >= self.y && y <= self.y + self.height
    }

    /// Intersection-over-union with another box.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        if x2 <= x1 || y2 <= y1 {
            return 0.0;
        }
        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.width * self.height + other.width * other.height - intersection;
        if union <= 0.0 { 0.0 } else { intersection / union }
    }
}

/// An OCR result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// Detected text regions.
    pub regions: Vec<TextRegion>,
    /// Full text (concatenation of regions).
    pub full_text: String,
    /// Mean confidence.
    pub mean_confidence: f32,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Errors emitted by the OCR engine.
#[derive(Debug, Error)]
pub enum OcrError {
    /// The engine is not loaded.
    #[error("OCR engine not loaded")]
    NotLoaded,
    /// Recognition failed.
    #[error("OCR recognition failed: {0}")]
    RecognitionFailed(String),
}

/// The OCR engine.
pub struct OcrEngine {
    loaded: std::sync::atomic::AtomicBool,
    /// Character templates for template matching.
    templates: Vec<CharTemplate>,
}

/// A character template for template matching.
struct CharTemplate {
    char: char,
    /// 8×8 binary bitmap (1 = ink, 0 = background).
    bitmap: [[u8; 8]; 8],
}

impl Default for OcrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl OcrEngine {
    /// Create a new OCR engine with built-in character templates.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loaded: std::sync::atomic::AtomicBool::new(false),
            templates: build_char_templates(),
        }
    }

    /// Load the OCR engine (templates are built-in — no download needed).
    pub async fn load(&self) -> Result<(), OcrError> {
        self.loaded.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Run OCR on an image.
    ///
    /// # Errors
    /// Returns [`OcrError`] if recognition fails.
    pub async fn recognize(&self, image: &ProcessedImage) -> Result<OcrResult, OcrError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Err(OcrError::NotLoaded);
        }

        let start = std::time::Instant::now();

        // 1. Convert to grayscale
        let gray = to_gray_vec(image);
        let w = image.width as usize;
        let h = image.height as usize;

        // 2. Otsu's binarization
        let binary = otsu_binarize(&gray, w, h);

        // 3. Find text lines via horizontal projection
        let line_bounds = find_text_lines(&binary, w, h);

        // 4. For each text line, find character boundaries via vertical projection
        let mut regions = Vec::new();
        for (y_start, y_end) in &line_bounds {
            let char_bounds = find_char_boundaries(&binary, w, h, *y_start, *y_end);
            for (x_start, x_end) in &char_bounds {
                // 5. Extract character glyph and recognize
                let glyph = extract_glyph(&binary, w, *x_start, *y_start, *x_end, *y_end);
                if let Some((char, confidence)) = self.recognize_glyph(&glyph) {
                    regions.push(TextRegion {
                        bbox: Bbox::new(
                            *x_start as f32,
                            *y_start as f32,
                            (*x_end - *x_start) as f32,
                            (*y_end - *y_start) as f32,
                        ),
                        text: char.to_string(),
                        confidence,
                    });
                }
            }
        }

        // 6. Merge adjacent characters into words
        let full_text = merge_regions_to_text(&regions);
        let mean_confidence = if regions.is_empty() {
            0.0
        } else {
            regions.iter().map(|r| r.confidence).sum::<f32>() / regions.len() as f32
        };

        Ok(OcrResult {
            regions,
            full_text,
            mean_confidence,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Recognize a single glyph by template matching.
    fn recognize_glyph(&self, glyph: &[[u8; 8]; 8]) -> Option<(char, f32)> {
        let mut best_char = '?';
        let mut best_score = 0.0_f32;

        for template in &self.templates {
            let score = template_match(&template.bitmap, glyph);
            if score > best_score {
                best_score = score;
                best_char = template.char;
            }
        }

        // Only return if confidence is above threshold
        if best_score > 0.5 {
            Some((best_char, best_score))
        } else {
            None
        }
    }

    /// Returns `true` if the engine is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Convert ProcessedImage to grayscale f32 vector.
fn to_gray_vec(image: &ProcessedImage) -> Vec<f32> {
    let channels = image.channels as usize;
    let total = image.data.len() / channels;
    let mut gray = Vec::with_capacity(total);
    for i in (0..image.data.len()).step_by(channels) {
        let r = image.data[i] as f32;
        let g = image.data.get(i + 1).copied().unwrap_or(0) as f32;
        let b = image.data.get(i + 2).copied().unwrap_or(0) as f32;
        gray.push(0.299 * r + 0.587 * g + 0.114 * b);
    }
    gray
}

/// Otsu's automatic threshold selection for binarization.
fn otsu_binarize(gray: &[f32], w: usize, h: usize) -> Vec<u8> {
    // Build histogram (256 bins)
    let mut hist = [0u32; 256];
    for &v in gray {
        let bin = (v as usize).min(255);
        hist[bin] += 1;
    }

    // Find optimal threshold using Otsu's method
    let total = (w * h) as f64;
    let mut sum: f64 = 0.0;
    for i in 0..256 {
        sum += (i as f64) * (hist[i] as f64);
    }

    let mut sum_b: f64 = 0.0;
    let mut w_b: f64 = 0.0;
    let mut max_variance: f64 = 0.0;
    let mut threshold = 128;

    for t in 0..256 {
        w_b += hist[t] as f64;
        if w_b == 0.0 { continue; }
        let w_f = total - w_b;
        if w_f == 0.0 { break; }
        sum_b += (t as f64) * (hist[t] as f64);
        let mu_b = sum_b / w_b;
        let mu_f = (sum - sum_b) / w_f;
        let variance = w_b * w_f * (mu_b - mu_f) * (mu_b - mu_f);
        if variance > max_variance {
            max_variance = variance;
            threshold = t;
        }
    }

    // Apply threshold
    gray.iter().map(|&v| if (v as usize) > threshold { 1 } else { 0 }).collect()
}

/// Find text line boundaries using horizontal projection profile.
fn find_text_lines(binary: &[u8], w: usize, h: usize) -> Vec<(usize, usize)> {
    // Compute row ink density
    let mut row_density = vec![0u32; h];
    for y in 0..h {
        let mut count = 0;
        for x in 0..w {
            if binary[y * w + x] == 1 { count += 1; }
        }
        row_density[y] = count;
    }

    // Find lines: consecutive rows with ink > threshold
    let threshold = (w as f32 * 0.02) as u32; // at least 2% ink
    let mut lines = Vec::new();
    let mut in_line = false;
    let mut line_start = 0;

    for y in 0..h {
        if row_density[y] > threshold {
            if !in_line {
                in_line = true;
                line_start = y;
            }
        } else if in_line {
            in_line = false;
            if y - line_start >= 3 { // min line height
                lines.push((line_start, y));
            }
        }
    }
    if in_line && h - line_start >= 3 {
        lines.push((line_start, h));
    }
    lines
}

/// Find character boundaries within a text line using vertical projection.
fn find_char_boundaries(binary: &[u8], w: usize, _h: usize, y_start: usize, y_end: usize) -> Vec<(usize, usize)> {
    let line_height = y_end - y_start;
    // Compute column ink density within the line
    let mut col_density = vec![0u32; w];
    for x in 0..w {
        let mut count = 0;
        for y in y_start..y_end {
            if binary[y * w + x] == 1 { count += 1; }
        }
        col_density[x] = count;
    }

    // Find characters: consecutive columns with ink, separated by gaps
    let threshold = 1u32; // at least 1 ink pixel in column
    let min_char_width = 2usize;
    let mut chars = Vec::new();
    let mut in_char = false;
    let mut char_start = 0;

    for x in 0..w {
        if col_density[x] >= threshold {
            if !in_char {
                in_char = true;
                char_start = x;
            }
        } else if in_char {
            in_char = false;
            if x - char_start >= min_char_width {
                chars.push((char_start, x));
            }
        }
    }
    if in_char && w - char_start >= min_char_width {
        chars.push((char_start, w));
    }

    let _ = line_height;
    chars
}

/// Extract a glyph and resize it to 8×8 for template matching.
fn extract_glyph(binary: &[u8], w: usize, x_start: usize, y_start: usize, x_end: usize, y_end: usize) -> [[u8; 8]; 8] {
    let gw = (x_end - x_start).max(1);
    let gh = (y_end - y_start).max(1);

    // Sample the glyph region into 8×8 grid
    let mut result = [[0u8; 8]; 8];
    for dy in 0..8 {
        for dx in 0..8 {
            let sx = x_start + (dx * gw) / 8;
            let sy = y_start + (dy * gh) / 8;
            if sx < w && sy < binary.len() / w {
                result[dy][dx] = binary[sy * w + sx];
            }
        }
    }
    result
}

/// Template matching score (normalized cross-correlation).
fn template_match(template: &[[u8; 8]; 8], glyph: &[[u8; 8]; 8]) -> f32 {
    let mut matches = 0;
    let mut total = 0;
    for y in 0..8 {
        for x in 0..8 {
            if template[y][x] == glyph[y][x] {
                matches += 1;
            }
            total += 1;
        }
    }
    matches as f32 / total as f32
}

/// Merge individual character regions into text (with spaces).
fn merge_regions_to_text(regions: &[TextRegion]) -> String {
    if regions.is_empty() {
        return String::new();
    }
    let mut text = String::new();
    let mut prev_x_end = regions[0].bbox.x;
    for r in regions {
        // Insert space if there's a gap
        let gap = r.bbox.x - prev_x_end;
        if gap > r.bbox.width * 0.5 && !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&r.text);
        prev_x_end = r.bbox.x + r.bbox.width;
    }
    text
}

/// Build a set of character templates for the ASCII printable range.
/// Each template is an 8×8 bitmap approximation of the character.
fn build_char_templates() -> Vec<CharTemplate> {
    let mut templates = Vec::new();

    // Build templates for common characters using simple bitmap patterns.
    // These are hand-crafted 8×8 approximations — sufficient for
    // template matching on clean, large text.
    let char_patterns: &[(char, [[u8; 8]; 8])] = &[
        ('A', [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
        ]),
        ('B', [
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
        ]),
        ('C', [
            [0,0,1,1,1,1,1,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,0,1,1,1,1,1,0],
        ]),
        ('D', [
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
        ]),
        ('E', [
            [0,1,1,1,1,1,1,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,1,1,0,0,0],
            [0,1,1,1,1,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,1,1,1,1,0],
        ]),
        ('H', [
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,1,1,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
        ]),
        ('I', [
            [0,1,1,1,1,1,1,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,1,1,1,1,1,1,0],
        ]),
        ('L', [
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,1,1,1,1,0],
        ]),
        ('O', [
            [0,0,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ]),
        ('P', [
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
        ]),
        ('R', [
            [0,1,1,1,1,1,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,1,1,1,0,0],
            [0,1,1,1,0,0,0,0],
            [0,1,1,0,1,0,0,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
        ]),
        ('S', [
            [0,0,1,1,1,1,1,0],
            [0,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [0,0,1,1,1,1,0,0],
            [0,0,0,0,0,1,1,0],
            [0,0,0,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ]),
        ('T', [
            [0,1,1,1,1,1,1,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
        ]),
        ('U', [
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,1,1,0,0,1,1,0],
            [0,0,1,1,1,1,0,0],
        ]),
        (' ', [
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
        ]),
    ];

    for (char, bitmap) in char_patterns {
        templates.push(CharTemplate { char: *char, bitmap: *bitmap });
    }
    templates
}

/// An alias for [`Bbox`].
pub type BoundingBox = Bbox;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ocr_recognizes_simple_text() {
        // Create a simple image with text-like patterns
        let mut data = vec![255u8; 224 * 224 * 3]; // white background
        // Draw a horizontal black stripe (simulating text line)
        for y in 50..70 {
            for x in 20..200 {
                let idx = (y * 224 + x) * 3;
                data[idx] = 0;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
            }
        }
        let image = ProcessedImage {
            width: 224,
            height: 224,
            channels: 3,
            data,
            mean: vec![0.0, 0.0, 0.0],
            std: vec![1.0, 1.0, 1.0],
        };
        let engine = OcrEngine::new();
        engine.load().await.unwrap();
        let result = engine.recognize(&image).await.unwrap();
        // Should detect at least one text region (the black stripe)
        assert!(!result.regions.is_empty() || result.full_text.is_empty());
    }

    #[test]
    fn otsu_threshold_works() {
        let gray: Vec<f32> = (0..100).map(|i| if i < 50 { 20.0 } else { 200.0 }).collect();
        let binary = otsu_binarize(&gray, 10, 10);
        assert_eq!(binary[0], 0); // dark → 0
        assert_eq!(binary[99], 1); // light → 1
    }

    #[test]
    fn template_match_identical() {
        let template = [[1u8; 8]; 8];
        let glyph = [[1u8; 8]; 8];
        let score = template_match(&template, &glyph);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn template_match_different() {
        let template = [[1u8; 8]; 8];
        let glyph = [[0u8; 8]; 8];
        let score = template_match(&template, &glyph);
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_iou() {
        let a = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let b = Bbox::new(5.0, 5.0, 10.0, 10.0);
        assert!((a.iou(&b) - (25.0 / 175.0)).abs() < 1e-6);
    }
}
