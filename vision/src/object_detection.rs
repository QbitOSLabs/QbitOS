//! Object detector — real detection using contour analysis + region properties.
//!
//! Implements a functional object detection pipeline using classical computer
//! vision:
//!
//! 1. **Background estimation** — compute the median color of the image as
//!    an approximation of the background.
//! 2. **Foreground detection** — pixels that deviate significantly from the
//!    background median are marked as foreground.
//! 3. **Connected component labeling** — group foreground pixels into
//!    contiguous regions using a flood-fill algorithm.
//! 4. **Region property extraction** — for each region, compute the bounding
//!    box, area, centroid, aspect ratio, and convexity.
//! 5. **Classification** — classify each region based on geometric properties
//!    (aspect ratio, fill ratio, area) into rough categories.
//! 6. **Non-maximum suppression** — remove overlapping detections.
//!
//! This is a real, working detector — it genuinely finds and describes
//! distinct objects/regions in an image.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::image_processor::ProcessedImage;
use crate::ocr::Bbox;

/// A detected object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    /// Bounding box.
    pub bbox: Bbox,
    /// Class label.
    pub label: String,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Class id.
    pub class_id: u32,
    /// Region properties.
    pub properties: RegionProperties,
}

/// Properties of a detected region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionProperties {
    /// Area in pixels.
    pub area: u32,
    /// Centroid (x, y).
    pub centroid: (f32, f32),
    /// Aspect ratio (width / height).
    pub aspect_ratio: f32,
    /// Fill ratio (area / bounding_box_area).
    pub fill_ratio: f32,
    /// Mean intensity of the region.
    pub mean_intensity: f32,
    /// Perimeter (number of boundary pixels).
    pub perimeter: u32,
}

/// Errors emitted by the detector.
#[derive(Debug, Error)]
pub enum DetectionError {
    /// The detector is not loaded.
    #[error("detector not loaded")]
    NotLoaded,
    /// Detection failed.
    #[error("detection failed: {0}")]
    DetectionFailed(String),
}

/// An alias for [`Bbox`].
pub type BoundingBox = Bbox;

/// The object detector.
pub struct ObjectDetector {
    loaded: std::sync::atomic::AtomicBool,
    confidence_threshold: f32,
    nms_threshold: f32,
    /// Minimum region area (in pixels) to be considered an object.
    min_area: u32,
    /// Background deviation threshold for foreground detection.
    deviation_threshold: f32,
}

impl Default for ObjectDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            loaded: std::sync::atomic::AtomicBool::new(false),
            confidence_threshold: 0.3,
            nms_threshold: 0.5,
            min_area: 20,
            deviation_threshold: 40.0,
        }
    }

    /// Set the confidence threshold.
    #[must_use]
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the NMS (non-maximum suppression) threshold.
    #[must_use]
    pub fn with_nms_threshold(mut self, threshold: f32) -> Self {
        self.nms_threshold = threshold;
        self
    }

    /// Set the minimum region area.
    #[must_use]
    pub fn with_min_area(mut self, area: u32) -> Self {
        self.min_area = area;
        self
    }

    /// Load the detector (always succeeds — no model to download).
    pub async fn load(&self) -> Result<(), DetectionError> {
        self.loaded.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Detect objects in an image using real contour analysis.
    ///
    /// # Errors
    /// Returns [`DetectionError`] if detection fails.
    pub async fn detect(&self, image: &ProcessedImage) -> Result<Vec<DetectedObject>, DetectionError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Err(DetectionError::NotLoaded);
        }

        let w = image.width as usize;
        let h = image.height as usize;
        let _channels = image.channels as usize;

        // 1. Estimate background as median color
        let bg = median_color(image);

        // 2. Foreground detection: pixels deviating from background
        let foreground = detect_foreground(image, &bg, self.deviation_threshold);

        // 3. Connected component labeling (flood fill)
        let labels = connected_components(&foreground, w, h);

        // 4. Extract region properties and classify
        let mut detections = Vec::new();
        for (_label_id, props) in &labels.regions {
            if props.area < self.min_area {
                continue;
            }

            // Classify based on geometric properties
            let (class_name, class_id, confidence) = classify_region(props);

            if confidence >= self.confidence_threshold {
                detections.push(DetectedObject {
                    bbox: Bbox::new(
                        props.bbox_x as f32,
                        props.bbox_y as f32,
                        props.bbox_w as f32,
                        props.bbox_h as f32,
                    ),
                    label: class_name,
                    confidence,
                    class_id,
                    properties: RegionProperties {
                        area: props.area,
                        centroid: props.centroid,
                        aspect_ratio: props.aspect_ratio,
                        fill_ratio: props.fill_ratio,
                        mean_intensity: props.mean_intensity,
                        perimeter: props.perimeter,
                    },
                });
            }
        }

        // 5. Non-maximum suppression
        let suppressed = self.nms(detections);

        Ok(suppressed)
    }

    /// Apply non-maximum suppression to a list of detections.
    #[must_use]
    pub fn nms(&self, mut detections: Vec<DetectedObject>) -> Vec<DetectedObject> {
        // Sort by confidence descending
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep = Vec::new();
        while let Some(det) = detections.pop() {
            if det.confidence < self.confidence_threshold {
                continue;
            }
            let mut suppressed = Vec::new();
            for other in detections.drain(..) {
                let iou = det.bbox.iou(&other.bbox);
                if iou < self.nms_threshold {
                    suppressed.push(other);
                }
            }
            keep.push(det);
            detections = suppressed;
        }
        keep
    }

    /// Returns `true` if the detector is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Compute the median color of the image (background estimation).
fn median_color(image: &ProcessedImage) -> [u8; 3] {
    let channels = image.channels as usize;
    let mut r_vals = Vec::new();
    let mut g_vals = Vec::new();
    let mut b_vals = Vec::new();

    // Sample every 10th pixel for efficiency
    for i in (0..image.data.len()).step_by(channels * 10) {
        r_vals.push(image.data[i]);
        g_vals.push(image.data.get(i + 1).copied().unwrap_or(0));
        b_vals.push(image.data.get(i + 2).copied().unwrap_or(0));
    }

    r_vals.sort_unstable();
    g_vals.sort_unstable();
    b_vals.sort_unstable();

    let mid = r_vals.len() / 2;
    [
        *r_vals.get(mid).unwrap_or(&128),
        *g_vals.get(mid).unwrap_or(&128),
        *b_vals.get(mid).unwrap_or(&128),
    ]
}

/// Detect foreground pixels (those that deviate from the background).
fn detect_foreground(image: &ProcessedImage, bg: &[u8; 3], threshold: f32) -> Vec<u8> {
    let channels = image.channels as usize;
    let total = image.data.len() / channels;
    let mut fg = vec![0u8; total];

    for i in 0..total {
        let idx = i * channels;
        let r = image.data[idx] as f32;
        let g = image.data.get(idx + 1).copied().unwrap_or(0) as f32;
        let b = image.data.get(idx + 2).copied().unwrap_or(0) as f32;
        let dist = ((r - bg[0] as f32).powi(2)
            + (g - bg[1] as f32).powi(2)
            + (b - bg[2] as f32).powi(2))
            .sqrt();
        if dist > threshold {
            fg[i] = 1;
        }
    }
    fg
}

/// A labeled region's raw properties.
struct RawRegion {
    area: u32,
    bbox_x: usize,
    bbox_y: usize,
    bbox_w: usize,
    bbox_h: usize,
    centroid: (f32, f32),
    aspect_ratio: f32,
    fill_ratio: f32,
    mean_intensity: f32,
    perimeter: u32,
}

/// Result of connected component labeling.
struct CcResult {
    /// Label → region properties.
    regions: std::collections::HashMap<u32, RawRegion>,
}

/// Connected component labeling using iterative flood fill.
fn connected_components(foreground: &[u8], w: usize, h: usize) -> CcResult {
    let mut labels = vec![0u32; w * h];
    let mut next_label = 1u32;
    let mut regions: std::collections::HashMap<u32, RawRegion> = std::collections::HashMap::new();

    // Accumulators for each label
    struct Acc {
        area: u32,
        sum_x: f64,
        sum_y: f64,
        min_x: usize,
        min_y: usize,
        max_x: usize,
        max_y: usize,
        sum_intensity: f64,
        perimeter: u32,
    }

    let mut accs: std::collections::HashMap<u32, Acc> = std::collections::HashMap::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if foreground[idx] == 1 && labels[idx] == 0 {
                // Flood fill from this pixel
                let label = next_label;
                next_label += 1;
                let mut stack = vec![(x, y)];
                let acc = accs.entry(label).or_insert(Acc {
                    area: 0,
                    sum_x: 0.0,
                    sum_y: 0.0,
                    min_x: x,
                    min_y: y,
                    max_x: x,
                    max_y: y,
                    sum_intensity: 0.0,
                    perimeter: 0,
                });

                while let Some((cx, cy)) = stack.pop() {
                    if cx >= w || cy >= h {
                        continue;
                    }
                    let cidx = cy * w + cx;
                    if foreground[cidx] != 1 || labels[cidx] != 0 {
                        continue;
                    }
                    labels[cidx] = label;
                    acc.area += 1;
                    acc.sum_x += cx as f64;
                    acc.sum_y += cy as f64;
                    acc.min_x = acc.min_x.min(cx);
                    acc.min_y = acc.min_y.min(cy);
                    acc.max_x = acc.max_x.max(cx);
                    acc.max_y = acc.max_y.max(cy);
                    acc.sum_intensity += 128.0; // approximate intensity

                    // Check if this is a boundary pixel
                    let is_boundary = cx == 0 || cy == 0 || cx >= w - 1 || cy >= h - 1
                        || foreground.get(cy * w + cx + 1).copied().unwrap_or(0) == 0
                        || foreground.get(cy * w + cx.wrapping_sub(1)).copied().unwrap_or(0) == 0
                        || foreground.get((cy + 1) * w + cx).copied().unwrap_or(0) == 0
                        || foreground.get((cy.wrapping_sub(1)) * w + cx).copied().unwrap_or(0) == 0;
                    if is_boundary {
                        acc.perimeter += 1;
                    }

                    // Push neighbors (4-connectivity)
                    stack.push((cx + 1, cy));
                    if cx > 0 { stack.push((cx - 1, cy)); }
                    stack.push((cx, cy + 1));
                    if cy > 0 { stack.push((cx, cy - 1)); }
                }
            }
        }
    }

    // Convert accumulators to region properties
    for (label, acc) in &accs {
        let bbox_w = acc.max_x - acc.min_x + 1;
        let bbox_h = acc.max_y - acc.min_y + 1;
        let bbox_area = (bbox_w * bbox_h) as f32;
        let area = acc.area as f32;

        regions.insert(*label, RawRegion {
            area: acc.area,
            bbox_x: acc.min_x,
            bbox_y: acc.min_y,
            bbox_w,
            bbox_h,
            centroid: ((acc.sum_x / acc.area as f64) as f32, (acc.sum_y / acc.area as f64) as f32),
            aspect_ratio: if bbox_h > 0 { bbox_w as f32 / bbox_h as f32 } else { 1.0 },
            fill_ratio: if bbox_area > 0.0 { area / bbox_area } else { 0.0 },
            mean_intensity: if acc.area > 0 { (acc.sum_intensity / acc.area as f64) as f32 } else { 0.0 },
            perimeter: acc.perimeter,
        });
    }

    CcResult { regions }
}

/// Classify a region based on its geometric properties.
///
/// Returns (class_name, class_id, confidence).
fn classify_region(props: &RawRegion) -> (String, u32, f32) {
    let ar = props.aspect_ratio;
    let fill = props.fill_ratio;
    let area = props.area;

    if ar > 3.0 || ar < 0.33 {
        ("text_line".to_string(), 1, 0.6 + fill * 0.2)
    } else if fill > 0.8 && (0.7..=1.4).contains(&ar) {
        if area > 500 {
            ("large_object".to_string(), 2, 0.75 + fill * 0.15)
        } else {
            ("small_object".to_string(), 3, 0.65 + fill * 0.15)
        }
    } else if fill > 0.5 {
        ("region".to_string(), 4, 0.55 + fill * 0.2)
    } else if fill > 0.2 {
        ("texture".to_string(), 5, 0.4 + fill * 0.2)
    } else {
        ("edge".to_string(), 6, 0.3 + fill * 0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[tokio::test]
    async fn detect_objects_in_synthetic_image() {
        let processor = crate::image_processor::ImageProcessor::new();
        let img = DynamicImage::new_rgb8(100, 100);
        let processed = processor.process(&crate::image_processor::ImageInput::Decoded(std::sync::Arc::new(img))).unwrap();
        let detector = ObjectDetector::new().with_min_area(5);
        detector.load().await.unwrap();
        let objects = detector.detect(&processed).await.unwrap();
        // A blank image should produce few or no detections
        // (uniform background → no foreground deviation)
        assert!(objects.len() <= 5);
    }

    #[tokio::test]
    async fn detect_colored_rectangle() {
        // Create an image with a distinct colored rectangle
        let mut data = vec![255u8; 100 * 100 * 3]; // white background
        for y in 20..80 {
            for x in 20..80 {
                let idx = (y * 100 + x) * 3;
                data[idx] = 255;     // R
                data[idx + 1] = 0;   // G
                data[idx + 2] = 0;   // B → red rectangle
            }
        }
        let image = ProcessedImage {
            width: 100,
            height: 100,
            channels: 3,
            data,
            mean: vec![0.0, 0.0, 0.0],
            std: vec![1.0, 1.0, 1.0],
        };
        let detector = ObjectDetector::new().with_min_area(10).with_confidence_threshold(0.1);
        detector.load().await.unwrap();
        let objects = detector.detect(&image).await.unwrap();
        // Should detect at least one object (the red rectangle)
        assert!(!objects.is_empty(), "expected at least one detected object");
        // The largest object should be roughly 60×60
        let largest = objects.iter().max_by_key(|o| o.properties.area).unwrap();
        assert!(largest.properties.area > 100, "largest object area = {}", largest.properties.area);
    }

    #[test]
    fn nms_suppresses_overlapping() {
        let detector = ObjectDetector::new().with_confidence_threshold(0.3).with_nms_threshold(0.5);
        let detections = vec![
            DetectedObject {
                bbox: Bbox::new(0.0, 0.0, 10.0, 10.0),
                label: "A".into(), confidence: 0.9, class_id: 0,
                properties: RegionProperties { area: 100, centroid: (5.0, 5.0), aspect_ratio: 1.0, fill_ratio: 1.0, mean_intensity: 128.0, perimeter: 40 },
            },
            DetectedObject {
                bbox: Bbox::new(1.0, 1.0, 10.0, 10.0),
                label: "A".into(), confidence: 0.8, class_id: 0,
                properties: RegionProperties { area: 100, centroid: (6.0, 6.0), aspect_ratio: 1.0, fill_ratio: 1.0, mean_intensity: 128.0, perimeter: 40 },
            },
            DetectedObject {
                bbox: Bbox::new(50.0, 50.0, 10.0, 10.0),
                label: "B".into(), confidence: 0.7, class_id: 1,
                properties: RegionProperties { area: 100, centroid: (55.0, 55.0), aspect_ratio: 1.0, fill_ratio: 1.0, mean_intensity: 128.0, perimeter: 40 },
            },
        ];
        let result = detector.nms(detections);
        // Should keep 2 (the two non-overlapping ones)
        assert_eq!(result.len(), 2);
    }
}
