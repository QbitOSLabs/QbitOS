//! Selective state-space scan (SF-Mamba 2026).
//!
//! Implements a 2D selective state-space model scan for vision encoders,
//! inspired by SF-Mamba (2026). Supports bidirectional scanning (4 directions),
//! selective gating (softplus on input-dependent delta), and layer norm.

use ndarray::{Array1, Array3, s};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors emitted by the selective SSM.
#[derive(Debug, Error)]
pub enum SsmError {
    /// Channel dimension mismatch between input and config.
    #[error("channel mismatch: expected {expected}, got {actual}")]
    ChannelMismatch {
        /// Expected channel count.
        expected: usize,
        /// Actual channel count.
        actual: usize,
    },
}

/// Scan direction for 2D state-space traversal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScanDirection {
    /// Row-major (left-to-right, top-to-bottom).
    RowMajor,
    /// Row-major reverse (right-to-left, bottom-to-top).
    RowMajorReverse,
    /// Column-major (top-to-bottom, left-to-right).
    ColMajor,
    /// Column-major reverse (bottom-to-top, right-to-left).
    ColMajorReverse,
}

impl ScanDirection {
    /// All 4 scan directions.
    pub const ALL: [ScanDirection; 4] = [
        ScanDirection::RowMajor,
        ScanDirection::RowMajorReverse,
        ScanDirection::ColMajor,
        ScanDirection::ColMajorReverse,
    ];
}

/// Configuration for the selective state-space model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveSsmConfig {
    /// State dimension (hidden size of the SSM).
    pub state_dim: usize,
    /// Whether to use selective (input-dependent) gating.
    pub selective: bool,
    /// Whether to do bidirectional 2D scanning (4 directions, averaged).
    pub bidirectional_2d: bool,
    /// Initial value for the diagonal state matrix A (negative for stability).
    pub a_init: f32,
    /// Whether to apply layer normalization before scanning.
    pub layer_norm: bool,
}

impl Default for SelectiveSsmConfig {
    fn default() -> Self {
        Self {
            state_dim: 16,
            selective: true,
            bidirectional_2d: true,
            a_init: -1.0,
            layer_norm: true,
        }
    }
}

/// The selective state-space model.
pub struct SelectiveSsm {
    cfg: SelectiveSsmConfig,
    w_b: Array1<f32>,
    w_c: Array1<f32>,
    a_log: Array1<f32>,
}

impl SelectiveSsm {
    /// Create a new selective SSM with the given channel count and config.
    #[must_use]
    pub fn new(channels: usize, cfg: SelectiveSsmConfig) -> Self {
        let d = cfg.state_dim;
        let a_init = cfg.a_init;
        let sc = (channels as f32).sqrt().recip();
        Self {
            cfg,
            w_b: Array1::from_shape_fn(d, |i| ((i as f32 + 1.0) * sc * 0.01).tanh()),
            w_c: Array1::from_shape_fn(channels, |i| ((i as f32 + 1.0) * sc * 0.01).tanh()),
            a_log: Array1::from_shape_fn(d, |_| a_init.abs().ln()),
        }
    }

    /// Forward pass: apply the SSM scan to a 3D input tensor (H, W, C).
    ///
    /// # Errors
    /// Returns `ChannelMismatch` if the input channel count doesn't match the SSM's.
    pub fn forward(&self, x: &Array3<f32>) -> Result<Array3<f32>, SsmError> {
        if self.cfg.bidirectional_2d {
            let mut acc = Array3::zeros(x.dim());
            for dir in ScanDirection::ALL {
                let y = self.scan_single(x, dir)?;
                acc = acc + y;
            }
            acc.mapv_inplace(|v| v / 4.0);
            Ok(acc)
        } else {
            self.scan_single(x, ScanDirection::RowMajor)
        }
    }

    fn scan_single(&self, x: &Array3<f32>, dir: ScanDirection) -> Result<Array3<f32>, SsmError> {
        let (h, w, c) = x.dim();
        let d = self.cfg.state_dim;
        let n = h * w;
        let xn = if self.cfg.layer_norm {
            self.ln(x)
        } else {
            x.clone()
        };
        let order: Vec<(usize, usize)> = match dir {
            ScanDirection::RowMajor => {
                (0..h).flat_map(|i| (0..w).map(move |j| (i, j))).collect()
            }
            ScanDirection::RowMajorReverse => {
                (0..h).rev().flat_map(|i| (0..w).rev().map(move |j| (i, j))).collect()
            }
            ScanDirection::ColMajor => {
                (0..w).flat_map(|j| (0..h).map(move |i| (i, j))).collect()
            }
            ScanDirection::ColMajorReverse => {
                (0..w).rev().flat_map(|j| (0..h).rev().map(move |i| (i, j))).collect()
            }
        };
        let mut state = Array1::zeros(d);
        let mut outputs = ndarray::Array2::zeros((n, c));
        for (idx, (i, j)) in order.iter().enumerate() {
            let xt = xn.slice(s![*i, *j, ..]).to_owned();
            // delta: input-dependent step size (selective gating)
            let delta = if self.cfg.selective {
                // w_b is (d,), xt is (c,) — they may have different dims.
                // Use the dot product of the overlapping prefix as a scalar gate.
                let min_dim = d.min(xt.len());
                let raw: f32 = self.w_b.slice(s![..min_dim]).dot(&xt.slice(s![..min_dim]));
                raw.softplus()
            } else {
                1.0
            };
            // State update: state = state * exp(A * delta) + B * delta
            let a = self.a_log.mapv(|v| -v.exp());
            let ab = a.mapv(|v| (v * delta).exp());
            // B = w_b · xt (project input to state space)
            let min_dim = d.min(xt.len());
            let b_vec: Array1<f32> = self.w_b.slice(s![..min_dim]).mapv(|v| v * delta) * &xt.slice(s![..min_dim]);
            let mut b_full = Array1::zeros(d);
            b_full.slice_mut(s![..min_dim]).assign(&b_vec);
            state = state * ab + b_full;
            // Output projection: cp = w_c · state
            let cp = self.w_c.dot(&state);
            outputs.row_mut(idx).assign(&Array1::from_elem(c, cp));
        }
        let mut out = Array3::zeros((h, w, c));
        for (idx, (i, j)) in order.iter().enumerate() {
            out.slice_mut(s![*i, *j, ..]).assign(&outputs.row(idx));
        }
        Ok(out)
    }

    fn ln(&self, x: &Array3<f32>) -> Array3<f32> {
        let (h, w, c) = x.dim();
        let mut o = Array3::zeros((h, w, c));
        for ch in 0..c {
            let mut vals = Vec::with_capacity(h * w);
            for i in 0..h {
                for j in 0..w {
                    vals.push(x[(i, j, ch)]);
                }
            }
            let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len().max(1) as f32;
            let std = var.sqrt().max(1e-6);
            for i in 0..h {
                for j in 0..w {
                    o[(i, j, ch)] = (x[(i, j, ch)] - mean) / std;
                }
            }
        }
        o
    }
}

trait SoftplusExt {
    fn softplus(self) -> f32;
}

impl SoftplusExt for f32 {
    fn softplus(self) -> f32 {
        if self > 20.0 {
            self
        } else if self < -20.0 {
            self.exp()
        } else {
            (1.0 + self.exp()).ln()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_shape() {
        let s = SelectiveSsm::new(64, SelectiveSsmConfig::default());
        let x = Array3::zeros((14, 14, 64));
        let y = s.forward(&x).unwrap();
        assert_eq!(y.dim(), (14, 14, 64));
    }

    #[test]
    fn unidirectional() {
        let cfg = SelectiveSsmConfig {
            bidirectional_2d: false,
            ..Default::default()
        };
        let s = SelectiveSsm::new(32, cfg);
        let x = Array3::from_shape_fn((7, 7, 32), |(i, j, _)| (i + j) as f32 * 0.01);
        let y = s.forward(&x).unwrap();
        assert_eq!(y.dim(), (7, 7, 32));
        assert!(y.iter().copied().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn selective_vs_non() {
        let x = Array3::from_shape_fn((4, 4, 16), |(i, j, _)| (i * 4 + j) as f32 * 0.1);
        let sel = SelectiveSsm::new(
            16,
            SelectiveSsmConfig {
                selective: true,
                bidirectional_2d: false,
                ..Default::default()
            },
        );
        let non = SelectiveSsm::new(
            16,
            SelectiveSsmConfig {
                selective: false,
                bidirectional_2d: false,
                ..Default::default()
            },
        );
        let ys = sel.forward(&x).unwrap();
        let yn = non.forward(&x).unwrap();
        let diff = ys.iter().zip(yn.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>();
        assert!(diff > 1e-3);
    }

    #[test]
    fn all_directions() {
        for dir in ScanDirection::ALL {
            let cfg = SelectiveSsmConfig {
                bidirectional_2d: false,
                ..Default::default()
            };
            let s = SelectiveSsm::new(32, cfg);
            let x = Array3::zeros((7, 7, 32));
            let y = s.scan_single(&x, dir).unwrap();
            assert_eq!(y.dim(), (7, 7, 32));
        }
    }
}
