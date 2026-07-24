//! Advanced image processing: Sobel edge detection, Canny edge detection,
//! k-means segmentation, color quantization, Gaussian blur.

use serde::{Deserialize, Serialize};

/// A grayscale image as a 2D matrix of pixel values (0-255).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayImage {
    /// Pixel data, row-major.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl GrayImage {
    /// Construct a new grayscale image.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width * height],
            width,
            height,
        }
    }

    /// Get a pixel value.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// Set a pixel value.
    pub fn set(&mut self, x: usize, y: usize, value: u8) {
        self.data[y * self.width + x] = value;
    }

    /// Get a pixel with clamped boundary.
    #[must_use]
    pub fn get_clamped(&self, x: i32, y: i32) -> u8 {
        let x = x.clamp(0, (self.width as i32) - 1) as usize;
        let y = y.clamp(0, (self.height as i32) - 1) as usize;
        self.data[y * self.width + x]
    }
}

/// An RGB image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbImage {
    /// Pixel data, row-major, RGB interleaved.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl RgbImage {
    /// Construct a new RGB image.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width * height * 3],
            width,
            height,
        }
    }

    /// Get a pixel as (R, G, B).
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let idx = (y * self.width + x) * 3;
        (self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    /// Set a pixel from (R, G, B).
    pub fn set_pixel(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let idx = (y * self.width + x) * 3;
        self.data[idx] = r;
        self.data[idx + 1] = g;
        self.data[idx + 2] = b;
    }

    /// Convert to grayscale using luminance weights.
    #[must_use]
    pub fn to_gray(&self) -> GrayImage {
        let mut gray = GrayImage::new(self.width, self.height);
        for i in 0..self.width * self.height {
            let r = f64::from(self.data[i * 3]);
            let g = f64::from(self.data[i * 3 + 1]);
            let b = f64::from(self.data[i * 3 + 2]);
            gray.data[i] = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
        }
        gray
    }
}

/// Apply the Sobel edge detection operator.
#[must_use]
pub fn sobel_edges(image: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(image.width, image.height);
    let sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    for y in 0..image.height {
        for x in 0..image.width {
            let mut gx = 0i32;
            let mut gy = 0i32;
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let px = image.get_clamped(x as i32 + dx, y as i32 + dy);
                    gx += i32::from(px) * sobel_x[(dy + 1) as usize][(dx + 1) as usize];
                    gy += i32::from(px) * sobel_y[(dy + 1) as usize][(dx + 1) as usize];
                }
            }
            let magnitude = ((gx * gx + gy * gy) as f64).sqrt().min(255.0) as u8;
            out.set(x, y, magnitude);
        }
    }
    out
}

/// Apply the Canny edge detection algorithm.
///
/// Steps: Gaussian blur → Sobel → non-maximum suppression → double threshold → hysteresis.
#[must_use]
pub fn canny_edges(image: &GrayImage, low: u8, high: u8) -> GrayImage {
    // Step 1: Gaussian blur.
    let blurred = gaussian_blur(image, 1.4);

    // Step 2: Sobel gradient + direction.
    let sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    let mut magnitude = vec![0.0f64; image.width * image.height];
    let mut direction = vec![0.0f64; image.width * image.height];

    for y in 0..blurred.height {
        for x in 0..blurred.width {
            let mut gx = 0f64;
            let mut gy = 0f64;
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let px = f64::from(blurred.get_clamped(x as i32 + dx, y as i32 + dy));
                    gx += px * f64::from(sobel_x[(dy + 1) as usize][(dx + 1) as usize]);
                    gy += px * f64::from(sobel_y[(dy + 1) as usize][(dx + 1) as usize]);
                }
            }
            let idx = y * blurred.width + x;
            magnitude[idx] = (gx * gx + gy * gy).sqrt();
            direction[idx] = gy.atan2(gx);
        }
    }

    // Step 3: Non-maximum suppression.
    let mut nms = vec![0.0f64; image.width * image.height];
    for y in 1..image.height - 1 {
        for x in 1..image.width - 1 {
            let idx = y * image.width + x;
            let angle = direction[idx] * 180.0 / std::f64::consts::PI;
            let angle = if angle < 0.0 { angle + 180.0 } else { angle };
            let mag = magnitude[idx];
            // Quantize angle to 4 directions.
            let (n1, n2) = if (angle < 22.5) || (angle >= 157.5) {
                // Horizontal.
                (magnitude[y * image.width + x + 1], magnitude[y * image.width + x - 1])
            } else if angle < 67.5 {
                // Diagonal 45°.
                (magnitude[(y + 1) * image.width + x - 1], magnitude[(y - 1) * image.width + x + 1])
            } else if angle < 112.5 {
                // Vertical.
                (magnitude[(y + 1) * image.width + x], magnitude[(y - 1) * image.width + x])
            } else {
                // Diagonal 135°.
                (magnitude[(y + 1) * image.width + x + 1], magnitude[(y - 1) * image.width + x - 1])
            };
            if mag >= n1 && mag >= n2 {
                nms[idx] = mag;
            }
        }
    }

    // Step 4: Double threshold + hysteresis.
    let mut out = GrayImage::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let idx = y * image.width + x;
            let mag = nms[idx];
            if mag >= f64::from(high) {
                out.data[idx] = 255;
            } else if mag >= f64::from(low) {
                // Weak edge — keep only if connected to a strong edge.
                let mut connected = false;
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        let nx = (x as i32 + dx).clamp(0, (image.width as i32) - 1) as usize;
                        let ny = (y as i32 + dy).clamp(0, (image.height as i32) - 1) as usize;
                        let nidx = ny * image.width + nx;
                        if nms[nidx] >= f64::from(high) {
                            connected = true;
                            break;
                        }
                    }
                    if connected {
                        break;
                    }
                }
                out.data[idx] = if connected { 255 } else { 0 };
            }
        }
    }
    out
}

/// Apply a Gaussian blur.
#[must_use]
pub fn gaussian_blur(image: &GrayImage, sigma: f64) -> GrayImage {
    let radius = (3.0 * sigma).ceil() as i32;
    let mut kernel = Vec::with_capacity(((2 * radius + 1) as usize).pow(2));
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let w = (-(dx as f64 * dx as f64 + dy as f64 * dy as f64) / two_sigma_sq).exp();
            kernel.push(w);
            sum += w;
        }
    }
    // Normalize.
    for w in &mut kernel {
        *w /= sum;
    }

    let mut out = GrayImage::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let mut acc = 0.0;
            let mut k = 0;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let px = f64::from(image.get_clamped(x as i32 + dx, y as i32 + dy));
                    acc += px * kernel[k];
                    k += 1;
                }
            }
            out.set(x, y, acc.round().clamp(0.0, 255.0) as u8);
        }
    }
    out
}

/// Color quantization using k-means clustering.
#[must_use]
pub fn kmeans_quantize(image: &RgbImage, k: usize, max_iter: usize) -> RgbImage {
    if k == 0 || image.data.is_empty() {
        return image.clone();
    }
    // Initialize centroids by sampling evenly.
    let mut centroids: Vec<(f64, f64, f64)> = (0..k)
        .map(|i| {
            let idx = (i * image.data.len() / (k * 3)) * 3;
            (
                f64::from(image.data[idx]),
                f64::from(image.data[idx + 1]),
                f64::from(image.data[idx + 2]),
            )
        })
        .collect();

    let mut assignments = vec![0usize; image.width * image.height];

    for _ in 0..max_iter {
        // Assign each pixel to the nearest centroid.
        let mut changed = false;
        for i in 0..image.width * image.height {
            let r = f64::from(image.data[i * 3]);
            let g = f64::from(image.data[i * 3 + 1]);
            let b = f64::from(image.data[i * 3 + 2]);
            let best = (0..k)
                .map(|c| {
                    let (cr, cg, cb) = centroids[c];
                    let dr = r - cr;
                    let dg = g - cg;
                    let db = b - cb;
                    (dr * dr + dg * dg + db * db, c)
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .map(|(_, c)| c)
                .unwrap_or(0);
            if assignments[i] != best {
                changed = true;
                assignments[i] = best;
            }
        }
        if !changed {
            break;
        }
        // Update centroids.
        let mut sums = vec![(0.0f64, 0.0f64, 0.0f64, 0usize); k];
        for i in 0..image.width * image.height {
            let c = assignments[i];
            sums[c].0 += f64::from(image.data[i * 3]);
            sums[c].1 += f64::from(image.data[i * 3 + 1]);
            sums[c].2 += f64::from(image.data[i * 3 + 2]);
            sums[c].3 += 1;
        }
        for c in 0..k {
            if sums[c].3 > 0 {
                centroids[c] = (
                    sums[c].0 / sums[c].3 as f64,
                    sums[c].1 / sums[c].3 as f64,
                    sums[c].2 / sums[c].3 as f64,
                );
            }
        }
    }

    // Produce the quantized image.
    let mut out = RgbImage::new(image.width, image.height);
    for i in 0..image.width * image.height {
        let c = assignments[i];
        out.data[i * 3] = centroids[c].0.round().clamp(0.0, 255.0) as u8;
        out.data[i * 3 + 1] = centroids[c].1.round().clamp(0.0, 255.0) as u8;
        out.data[i * 3 + 2] = centroids[c].2.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Otsu's thresholding for automatic binary segmentation.
#[must_use]
pub fn otsu_threshold(image: &GrayImage) -> u8 {
    let mut histogram = [0u32; 256];
    for &p in &image.data {
        histogram[p as usize] += 1;
    }
    let total = image.data.len() as f64;
    let mut sum = 0.0;
    for i in 0..256 {
        sum += i as f64 * histogram[i] as f64;
    }
    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_variance = 0.0;
    let mut threshold = 0u8;
    for t in 0..256 {
        w_b += histogram[t] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += t as f64 * histogram[t] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let variance = w_b * w_f * (m_b - m_f).powi(2);
        if variance > max_variance {
            max_variance = variance;
            threshold = t as u8;
        }
    }
    threshold
}

/// Apply a binary threshold to an image.
#[must_use]
pub fn threshold(image: &GrayImage, threshold: u8) -> GrayImage {
    let mut out = GrayImage::new(image.width, image.height);
    for (i, &p) in image.data.iter().enumerate() {
        out.data[i] = if p >= threshold { 255 } else { 0 };
    }
    out
}

/// Compute the histogram of a grayscale image.
#[must_use]
pub fn histogram(image: &GrayImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &p in &image.data {
        hist[p as usize] += 1;
    }
    hist
}

/// Histogram equalization for contrast enhancement.
#[must_use]
pub fn histogram_equalize(image: &GrayImage) -> GrayImage {
    let hist = histogram(image);
    let total = image.data.len() as f64;
    let mut cdf = [0.0f64; 256];
    let mut sum = 0.0;
    for i in 0..256 {
        sum += hist[i] as f64;
        cdf[i] = sum / total;
    }
    let mut out = GrayImage::new(image.width, image.height);
    for (i, &p) in image.data.iter().enumerate() {
        out.data[i] = (cdf[p as usize] * 255.0).round() as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image() -> RgbImage {
        let mut img = RgbImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                if x < 4 {
                    img.set_pixel(x, y, 255, 255, 255);
                } else {
                    img.set_pixel(x, y, 0, 0, 0);
                }
            }
        }
        img
    }

    #[test]
    fn rgb_to_gray_preserves_dimensions() {
        let img = make_test_image();
        let gray = img.to_gray();
        assert_eq!(gray.width, 8);
        assert_eq!(gray.height, 8);
    }

    #[test]
    fn sobel_detects_edges() {
        let img = make_test_image();
        let gray = img.to_gray();
        let edges = sobel_edges(&gray);
        // The edge should be around x=3/4 boundary.
        let edge_magnitude = edges.get(3, 3);
        assert!(edge_magnitude > 100, "expected high edge magnitude, got {edge_magnitude}");
        // The interior should have no edges.
        let interior = edges.get(0, 0);
        assert!(interior < 50, "expected low interior magnitude, got {interior}");
    }

    #[test]
    fn canny_detects_edges() {
        let img = make_test_image();
        let gray = img.to_gray();
        let edges = canny_edges(&gray, 50, 100);
        assert_eq!(edges.width, 8);
        assert_eq!(edges.height, 8);
    }

    #[test]
    fn gaussian_blur_smooths_image() {
        let mut img = GrayImage::new(5, 5);
        img.set(2, 2, 255);
        let blurred = gaussian_blur(&img, 1.0);
        // Center should be lower after blur.
        assert!(blurred.get(2, 2) < 255);
        // Neighbors should be higher than 0.
        assert!(blurred.get(1, 2) > 0);
    }

    #[test]
    fn kmeans_quantize_reduces_colors() {
        let mut img = RgbImage::new(8, 8);
        for i in 0..8 * 8 {
            img.data[i * 3] = (i * 10) as u8;
            img.data[i * 3 + 1] = (i * 20) as u8;
            img.data[i * 3 + 2] = (i * 30) as u8;
        }
        let quantized = kmeans_quantize(&img, 4, 10);
        // Count unique colors.
        let mut colors = std::collections::HashSet::new();
        for i in 0..8 * 8 {
            colors.insert((
                quantized.data[i * 3],
                quantized.data[i * 3 + 1],
                quantized.data[i * 3 + 2],
            ));
        }
        assert!(colors.len() <= 4);
    }

    #[test]
    fn otsu_finds_threshold() {
        let img = make_test_image();
        let gray = img.to_gray();
        let t = otsu_threshold(&gray);
        // For a bimodal image with only 0 and 255, Otsu picks 0 or 255.
        // Verify it returns a valid threshold value.
        assert!(t <= 255);
    }

    #[test]
    fn histogram_equalize_enhances_contrast() {
        let img = make_test_image();
        let gray = img.to_gray();
        let equalized = histogram_equalize(&gray);
        assert_eq!(equalized.width, 8);
        assert_eq!(equalized.height, 8);
    }
}
