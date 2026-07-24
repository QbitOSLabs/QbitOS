//! Quantization.
//!
//! Supports multiple quantization formats:
//!
//! - INT8, INT4, NF4, FP8 (runtime quantization)
//! - GPTQ, AWQ (loaded from pre-quantized checkpoints)
//! - GGUF (llama.cpp-compatible)

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The quantization dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationDtype {
    /// FP32 (no quantization).
    F32,
    /// FP16.
    F16,
    /// BF16.
    BF16,
    /// INT8.
    Int8,
    /// INT4.
    Int4,
    /// NF4 (NormalFloat 4-bit).
    Nf4,
    /// FP8.
    Fp8,
    /// GPTQ.
    Gptq,
    /// AWQ.
    Awq,
    /// GGUF (llama.cpp format).
    Gguf,
}

impl QuantizationDtype {
    /// Bits per weight.
    #[must_use]
    pub fn bits(&self) -> usize {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::Int8 | Self::Fp8 => 8,
            Self::Int4 | Self::Nf4 | Self::Gptq | Self::Awq | Self::Gguf => 4,
        }
    }

    /// Whether this is a quantized format.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

/// Errors emitted by the quantizer.
#[derive(Debug, Error)]
pub enum QuantizationError {
    /// The shape is not supported for quantization.
    #[error("unsupported shape: {0:?}")]
    UnsupportedShape(Vec<usize>),
    /// The dtype is not supported for quantization.
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(QuantizationDtype),
    /// An underlying tensor error.
    #[error("tensor error: {0}")]
    Tensor(String),
}

/// A quantizer: converts FP32/FP16 weights to lower-precision formats.
pub struct Quantizer {
    dtype: QuantizationDtype,
    group_size: usize,
}

impl Quantizer {
    /// Create a new quantizer.
    #[must_use]
    pub fn new(dtype: QuantizationDtype, group_size: usize) -> Self {
        Self { dtype, group_size }
    }

    /// Quantize a flat slice of FP32 weights. Returns the quantized bytes
    /// and the per-group scales (if applicable).
    ///
    /// # Errors
    /// Returns [`QuantizationError`] if the dtype is not supported.
    pub fn quantize(&self, weights: &[f32]) -> Result<QuantizedTensor, QuantizationError> {
        match self.dtype {
            QuantizationDtype::F32 => Ok(QuantizedTensor {
                dtype: self.dtype,
                shape: vec![weights.len()],
                data: weights.iter().flat_map(|w| w.to_le_bytes()).collect(),
                scales: Vec::new(),
                zeros: Vec::new(),
            }),
            QuantizationDtype::F16 => Ok(QuantizedTensor {
                dtype: self.dtype,
                shape: vec![weights.len()],
                data: weights.iter().flat_map(|w| half::f16::from_f32(*w).to_le_bytes()).collect(),
                scales: Vec::new(),
                zeros: Vec::new(),
            }),
            QuantizationDtype::Int8 => self.quantize_int8(weights),
            QuantizationDtype::Int4 => self.quantize_int4(weights),
            QuantizationDtype::Nf4 => self.quantize_nf4(weights),
            QuantizationDtype::Fp8 => self.quantize_fp8(weights),
            QuantizationDtype::BF16 => Ok(QuantizedTensor {
                dtype: self.dtype,
                shape: vec![weights.len()],
                data: weights.iter().flat_map(|w| half::bf16::from_f32(*w).to_le_bytes()).collect(),
                scales: Vec::new(),
                zeros: Vec::new(),
            }),
            QuantizationDtype::Gptq | QuantizationDtype::Awq | QuantizationDtype::Gguf => {
                // Pre-quantized checkpoints are loaded directly, not quantized at runtime
                Err(QuantizationError::UnsupportedDtype(self.dtype))
            }
        }
    }

    /// Dequantize back to FP32.
    ///
    /// # Errors
    /// Returns [`QuantizationError`] if the dtype is not supported.
    pub fn dequantize(&self, qt: &QuantizedTensor) -> Result<Vec<f32>, QuantizationError> {
        match qt.dtype {
            QuantizationDtype::F32 => {
                let mut out = Vec::with_capacity(qt.data.len() / 4);
                for chunk in qt.data.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(out)
            }
            QuantizationDtype::F16 => {
                let mut out = Vec::with_capacity(qt.data.len() / 2);
                for chunk in qt.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(half::f16::from_bits(bits).to_f32());
                }
                Ok(out)
            }
            QuantizationDtype::BF16 => {
                let mut out = Vec::with_capacity(qt.data.len() / 2);
                for chunk in qt.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(half::bf16::from_bits(bits).to_f32());
                }
                Ok(out)
            }
            QuantizationDtype::Int8 => self.dequantize_int8(qt),
            QuantizationDtype::Int4 => self.dequantize_int4(qt),
            QuantizationDtype::Nf4 => self.dequantize_nf4(qt),
            QuantizationDtype::Fp8 => self.dequantize_fp8(qt),
            QuantizationDtype::Gptq | QuantizationDtype::Awq | QuantizationDtype::Gguf => {
                Err(QuantizationError::UnsupportedDtype(qt.dtype))
            }
        }
    }

    fn quantize_int8(&self, weights: &[f32]) -> Result<QuantizedTensor, QuantizationError> {
        let group_size = self.group_size.max(1);
        let mut data = Vec::with_capacity(weights.len());
        let mut scales = Vec::new();
        let mut zeros = Vec::new();
        for chunk in weights.chunks(group_size) {
            let (min, max) = chunk.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
            let scale = (max - min) / 255.0;
            let zero = min;
            scales.push(scale);
            zeros.push(zero);
            for &w in chunk {
                let q = if scale > 1e-12 { ((w - zero) / scale).round().clamp(0.0, 255.0) as u8 } else { 128 };
                data.push(q);
            }
        }
        Ok(QuantizedTensor {
            dtype: QuantizationDtype::Int8,
            shape: vec![weights.len()],
            data,
            scales: scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
            zeros: zeros.iter().flat_map(|z| z.to_le_bytes()).collect(),
        })
    }

    fn dequantize_int8(&self, qt: &QuantizedTensor) -> Result<Vec<f32>, QuantizationError> {
        let group_size = self.group_size.max(1);
        let scales: Vec<f32> = qt.scales.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let zeros: Vec<f32> = qt.zeros.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::with_capacity(qt.data.len());
        for (i, &q) in qt.data.iter().enumerate() {
            let group_idx = i / group_size;
            let scale = scales.get(group_idx).copied().unwrap_or(1.0);
            let zero = zeros.get(group_idx).copied().unwrap_or(0.0);
            out.push(zero + scale * q as f32);
        }
        Ok(out)
    }

    fn quantize_int4(&self, weights: &[f32]) -> Result<QuantizedTensor, QuantizationError> {
        let group_size = self.group_size.max(1);
        let mut data = Vec::with_capacity((weights.len() + 1) / 2);
        let mut scales = Vec::new();
        let mut zeros = Vec::new();
        for chunk in weights.chunks(group_size) {
            let (min, max) = chunk.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
            let scale = (max - min) / 15.0;
            let zero = min;
            scales.push(scale);
            zeros.push(zero);
            let quants: Vec<u8> = chunk.iter()
                .map(|&w| {
                    if scale > 1e-12 { ((w - zero) / scale).round().clamp(0.0, 15.0) as u8 } else { 8 }
                })
                .collect();
            // Pack two 4-bit values into one byte
            for pair in quants.chunks(2) {
                let lo = pair[0] & 0x0F;
                let hi = pair.get(1).copied().unwrap_or(0) & 0x0F;
                data.push((hi << 4) | lo);
            }
        }
        Ok(QuantizedTensor {
            dtype: QuantizationDtype::Int4,
            shape: vec![weights.len()],
            data,
            scales: scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
            zeros: zeros.iter().flat_map(|z| z.to_le_bytes()).collect(),
        })
    }

    fn dequantize_int4(&self, qt: &QuantizedTensor) -> Result<Vec<f32>, QuantizationError> {
        let group_size = self.group_size.max(1);
        let scales: Vec<f32> = qt.scales.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let zeros: Vec<f32> = qt.zeros.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::with_capacity(qt.shape.first().copied().unwrap_or(0));
        for (i, &byte) in qt.data.iter().enumerate() {
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            for q in [lo, hi] {
                let token_idx = i * 2;
                let group_idx = token_idx / group_size;
                let scale = scales.get(group_idx).copied().unwrap_or(1.0);
                let zero = zeros.get(group_idx).copied().unwrap_or(0.0);
                out.push(zero + scale * q as f32);
            }
        }
        // Truncate to actual length
        let target_len = qt.shape.first().copied().unwrap_or(0);
        out.truncate(target_len);
        Ok(out)
    }

    fn quantize_nf4(&self, weights: &[f32]) -> Result<QuantizedTensor, QuantizationError> {
        // NF4 lookup table from the QLoRA paper
        const NF4_LUT: [f32; 16] = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625534, 0.16093020141124725, 0.24611228108406067, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        let group_size = self.group_size.max(1);
        let mut data = Vec::with_capacity((weights.len() + 1) / 2);
        let mut scales = Vec::new();
        for chunk in weights.chunks(group_size) {
            let abs_max = chunk.iter().map(|v| v.abs()).fold(0.0_f32, f32::max).max(1e-12);
            scales.push(abs_max);
            let quants: Vec<u8> = chunk.iter()
                .map(|&w| {
                    let normalized = w / abs_max;
                    // Find the nearest NF4 value
                    let mut best_idx = 0u8;
                    let mut best_dist = f32::INFINITY;
                    for (i, &nf4_val) in NF4_LUT.iter().enumerate() {
                        let dist = (nf4_val - normalized).abs();
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = i as u8;
                        }
                    }
                    best_idx
                })
                .collect();
            for pair in quants.chunks(2) {
                let lo = pair[0] & 0x0F;
                let hi = pair.get(1).copied().unwrap_or(0) & 0x0F;
                data.push((hi << 4) | lo);
            }
        }
        Ok(QuantizedTensor {
            dtype: QuantizationDtype::Nf4,
            shape: vec![weights.len()],
            data,
            scales: scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
            zeros: Vec::new(),
        })
    }

    fn dequantize_nf4(&self, qt: &QuantizedTensor) -> Result<Vec<f32>, QuantizationError> {
        const NF4_LUT: [f32; 16] = [
            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
            0.07958029955625634, 0.16093020141124725, 0.24611228108406067, 0.33791524171829224,
            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
        ];
        let group_size = self.group_size.max(1);
        let scales: Vec<f32> = qt.scales.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::with_capacity(qt.shape.first().copied().unwrap_or(0));
        for (i, &byte) in qt.data.iter().enumerate() {
            let lo = (byte & 0x0F) as usize;
            let hi = ((byte >> 4) & 0x0F) as usize;
            for q_idx in [lo, hi] {
                let token_idx = i * 2;
                let group_idx = token_idx / group_size;
                let scale = scales.get(group_idx).copied().unwrap_or(1.0);
                out.push(NF4_LUT[q_idx] * scale);
            }
        }
        out.truncate(qt.shape.first().copied().unwrap_or(0));
        Ok(out)
    }

    fn quantize_fp8(&self, weights: &[f32]) -> Result<QuantizedTensor, QuantizationError> {
        // Simplified FP8 (E4M3) quantization
        let mut data = Vec::with_capacity(weights.len());
        let abs_max = weights.iter().map(|v| v.abs()).fold(0.0_f32, f32::max).max(1e-12);
        let scale = abs_max / 240.0; // E4M3 max = 240
        for &w in weights {
            let scaled = (w / scale).clamp(-240.0, 240.0);
            // Simple quantization: round to nearest int and pack as i8
            data.push(scaled.round() as i8 as u8);
        }
        Ok(QuantizedTensor {
            dtype: QuantizationDtype::Fp8,
            shape: vec![weights.len()],
            data,
            scales: scale.to_le_bytes().to_vec(),
            zeros: Vec::new(),
        })
    }

    fn dequantize_fp8(&self, qt: &QuantizedTensor) -> Result<Vec<f32>, QuantizationError> {
        let scale = f32::from_le_bytes([
            qt.scales.get(0).copied().unwrap_or(0),
            qt.scales.get(1).copied().unwrap_or(0),
            qt.scales.get(2).copied().unwrap_or(0),
            qt.scales.get(3).copied().unwrap_or(0),
        ]);
        Ok(qt.data.iter().map(|&b| (b as i8 as f32) * scale).collect())
    }
}

/// A quantized tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// The dtype.
    pub dtype: QuantizationDtype,
    /// The shape.
    pub shape: Vec<usize>,
    /// The quantized data.
    pub data: Vec<u8>,
    /// Per-group scales (FP32, flattened).
    pub scales: Vec<u8>,
    /// Per-group zeros (FP32, flattened).
    pub zeros: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_round_trip() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q = Quantizer::new(QuantizationDtype::Int8, 16);
        let qt = q.quantize(&weights).unwrap();
        let dequant = q.dequantize(&qt).unwrap();
        for (i, (a, b)) in weights.iter().zip(dequant.iter()).enumerate() {
            assert!((a - b).abs() < 0.05, "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn nf4_round_trip() {
        let weights: Vec<f32> = (0..32).map(|i| ((i as f32) / 32.0 - 0.5) * 2.0).collect();
        let q = Quantizer::new(QuantizationDtype::Nf4, 8);
        let qt = q.quantize(&weights).unwrap();
        let dequant = q.dequantize(&qt).unwrap();
        // NF4 has 16 levels over [-1, 1]; tolerance ~0.13
        for (i, (a, b)) in weights.iter().zip(dequant.iter()).enumerate() {
            assert!((a - b).abs() < 0.2, "mismatch at {i}: {a} vs {b}");
        }
    }
}
