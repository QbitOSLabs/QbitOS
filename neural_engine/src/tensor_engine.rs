//! Tensor engine.
//!
//! A thin wrapper around `candle_core::Tensor` that adds:
//!
//! - A serializable shape & dtype
//! - Convenience constructors for common shapes
//! - Conversion to/from `ndarray::ArrayD<f32>` for interop with the rest of QbitOS

use candle_core::{DType, Device as CandleDevice};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// A serializable tensor type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// The shape.
    pub shape: Vec<usize>,
    /// The dtype.
    pub dtype: TensorDtype,
    /// The backing storage (CPU only — for GPU tensors, use `CandleTensor`).
    pub data: Vec<u8>,
}

impl Tensor {
    /// Create a new tensor from raw bytes.
    #[must_use]
    pub fn new(shape: Vec<usize>, dtype: TensorDtype, data: Vec<u8>) -> Self {
        Self { shape, dtype, data }
    }

    /// Create a new FP32 tensor from a flat `Vec<f32>` and shape.
    #[must_use]
    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self { shape, dtype: TensorDtype::F32, data: bytes }
    }

    /// Convert to an `ndarray::ArrayD<f32>` (only for FP32 tensors).
    ///
    /// # Errors
    /// Returns [`TensorError::DtypeMismatch`] if the dtype is not F32, or
    /// [`TensorError::ShapeMismatch`] if the data length doesn't match the shape.
    pub fn to_array_f32(&self) -> Result<ndarray::ArrayD<f32>, TensorError> {
        if self.dtype != TensorDtype::F32 {
            return Err(TensorError::DtypeMismatch {
                expected: TensorDtype::F32,
                got: self.dtype,
            });
        }
        let expected_len: usize = self.shape.iter().product();
        let actual_len = self.data.len() / 4;
        if expected_len != actual_len {
            return Err(TensorError::ShapeMismatch {
                expected: expected_len,
                got: actual_len,
            });
        }
        let data: Vec<f32> = self.data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&self.shape), data)
            .map_err(|e| TensorError::Ndarray(e.to_string()))
    }

    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert to a `candle_core::Tensor` on the given device.
    ///
    /// # Errors
    /// Returns an error if the conversion fails.
    pub fn to_candle(&self, device: &CandleDevice) -> Result<candle_core::Tensor, TensorError> {
        let candle_dtype = match self.dtype {
            TensorDtype::F32 => DType::F32,
            TensorDtype::F16 => DType::F16,
            TensorDtype::BF16 => DType::BF16,
            TensorDtype::U8 => DType::U8,
            TensorDtype::U32 => DType::U32,
            TensorDtype::I64 => DType::I64,
        };
        candle_core::Tensor::from_raw_buffer(&self.data, candle_dtype, &self.shape, device)
            .map_err(|e| TensorError::Candle(e.to_string()))
    }

    /// Convert from a `candle_core::Tensor` (moves to CPU if necessary).
    ///
    /// # Errors
    /// Returns an error if the conversion fails.
    pub fn from_candle(t: &candle_core::Tensor) -> Result<Self, TensorError> {
        let cpu = CandleDevice::Cpu;
        let t = if !matches!(t.device(), CandleDevice::Cpu) {
            t.to_device(&cpu).map_err(|e| TensorError::Candle(e.to_string()))?
        } else {
            t.clone()
        };
        let dtype = match t.dtype() {
            DType::F32 => TensorDtype::F32,
            DType::F16 => TensorDtype::F16,
            DType::BF16 => TensorDtype::BF16,
            DType::U8 => TensorDtype::U8,
            DType::U32 => TensorDtype::U32,
            DType::I64 => TensorDtype::I64,
            // Other dtypes (I16, I32, F64, F8E4M3, F6E2M3) are not supported by the
            // serializable Tensor envelope yet — fall back to F32 via cast.
            _ => {
                let cast = t.to_dtype(DType::F32).map_err(|e| TensorError::Candle(e.to_string()))?;
                return Self::from_candle(&cast);
            }
        };
        let shape = t.dims().to_vec();
        // Read raw bytes by computing nbytes and using flatten + cast
        let nbytes: usize = shape.iter().product::<usize>() * dtype.size();
        let data: Vec<u8> = match dtype {
            TensorDtype::F32 => {
                let flat = t.flatten_all().map_err(|e| TensorError::Candle(e.to_string()))?;
                let v: Vec<f32> = flat.to_vec1().map_err(|e| TensorError::Candle(e.to_string()))?;
                v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            TensorDtype::F16 | TensorDtype::BF16 => {
                // candle doesn't expose u16 via WithDType, so cast to u32 then narrow
                let cast = t.to_dtype(DType::U32).map_err(|e| TensorError::Candle(e.to_string()))?;
                let flat = cast.flatten_all().map_err(|e| TensorError::Candle(e.to_string()))?;
                let v: Vec<u32> = flat.to_vec1().map_err(|e| TensorError::Candle(e.to_string()))?;
                v.into_iter().map(|x| x as u16).flat_map(|x| x.to_le_bytes()).collect()
            }
            TensorDtype::U8 => {
                let flat = t.flatten_all().map_err(|e| TensorError::Candle(e.to_string()))?;
                flat.to_vec1::<u8>().map_err(|e| TensorError::Candle(e.to_string()))?
            }
            TensorDtype::U32 => {
                let flat = t.flatten_all().map_err(|e| TensorError::Candle(e.to_string()))?;
                let v: Vec<u32> = flat.to_vec1().map_err(|e| TensorError::Candle(e.to_string()))?;
                v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
            TensorDtype::I64 => {
                let flat = t.flatten_all().map_err(|e| TensorError::Candle(e.to_string()))?;
                let v: Vec<i64> = flat.to_vec1().map_err(|e| TensorError::Candle(e.to_string()))?;
                v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
            }
        };
        let _ = nbytes;
        Ok(Self { shape, dtype, data })
    }
}

/// The dtype of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDtype {
    /// FP32.
    F32,
    /// FP16.
    F16,
    /// BF16.
    BF16,
    /// U8.
    U8,
    /// U32.
    U32,
    /// I64.
    I64,
}

impl TensorDtype {
    /// Size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::U8 => 1,
            Self::I64 => 8,
        }
    }
}

/// Errors emitted by the tensor engine.
#[derive(Debug, Error)]
pub enum TensorError {
    /// Dtype mismatch.
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DtypeMismatch {
        /// Expected dtype.
        expected: TensorDtype,
        /// Actual dtype.
        got: TensorDtype,
    },
    /// Shape mismatch.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// An error from the underlying candle-core library.
    #[error("candle error: {0}")]
    Candle(String),
    /// An error from ndarray.
    #[error("ndarray error: {0}")]
    Ndarray(String),
}

/// A shared tensor handle.
pub type SharedTensor = Arc<Tensor>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f32_round_trip() {
        let t = Tensor::from_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr = t.to_array_f32().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.iter().sum::<f32>(), 21.0);
    }
}
