//! Attention engine.
//!
//! Implements multiple attention variants:
//!
//! - Multi-Head Attention (MHA)
//! - Grouped Query Attention (GQA)
//! - Multi-Query Attention (MQA)
//! - Sliding Window Attention
//! - Ring Attention (distributed)
//! - Flash Attention (via `candle-flash-attn` when available)

use serde::{Deserialize, Serialize};

/// The kind of attention to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    /// Multi-Head Attention: every head has its own K and V.
    Mha,
    /// Grouped Query Attention: K and V are shared across groups of heads.
    Gqa,
    /// Multi-Query Attention: a single K and V is shared across all heads.
    Mqa,
    /// Sliding Window Attention: only attend to the last `window_size` tokens.
    SlidingWindow,
    /// Ring Attention: distributed attention across multiple GPUs.
    Ring,
    /// Flash Attention: memory-efficient exact attention.
    Flash,
}

/// Configuration for an attention layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Kind of attention.
    pub kind: AttentionKind,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key/value heads (for GQA; equals `num_heads` for MHA, 1 for MQA).
    pub num_kv_heads: usize,
    /// Head dimension (typically `hidden_dim / num_heads`).
    pub head_dim: usize,
    /// Sliding window size (only used for `SlidingWindow`).
    pub window_size: Option<usize>,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Scaling factor for the attention scores (typically `1 / sqrt(head_dim)`).
    pub scale: f32,
    /// Dropout probability (0.0 = no dropout).
    pub dropout: f32,
}

impl AttentionConfig {
    /// Create a new MHA config.
    #[must_use]
    pub fn mha(hidden_dim: usize, num_heads: usize, max_seq_len: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            kind: AttentionKind::Mha,
            hidden_dim,
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            window_size: None,
            max_seq_len,
            scale,
            dropout: 0.0,
        }
    }

    /// Create a new GQA config.
    #[must_use]
    pub fn gqa(hidden_dim: usize, num_heads: usize, num_kv_heads: usize, max_seq_len: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            kind: AttentionKind::Gqa,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            window_size: None,
            max_seq_len,
            scale,
            dropout: 0.0,
        }
    }

    /// Create a new MQA config.
    #[must_use]
    pub fn mqa(hidden_dim: usize, num_heads: usize, max_seq_len: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            kind: AttentionKind::Mqa,
            hidden_dim,
            num_heads,
            num_kv_heads: 1,
            head_dim,
            window_size: None,
            max_seq_len,
            scale,
            dropout: 0.0,
        }
    }

    /// Create a new Sliding Window config.
    #[must_use]
    pub fn sliding_window(hidden_dim: usize, num_heads: usize, window_size: usize, max_seq_len: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            kind: AttentionKind::SlidingWindow,
            hidden_dim,
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            window_size: Some(window_size),
            max_seq_len,
            scale,
            dropout: 0.0,
        }
    }
}

/// Trait that an attention layer must implement.
pub trait Attention: Send + Sync {
    /// Compute attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape `[batch, num_heads, seq_len, head_dim]`.
    /// * `key` - Key tensor of shape `[batch, num_kv_heads, kv_len, head_dim]`.
    /// * `value` - Value tensor of shape `[batch, num_kv_heads, kv_len, head_dim]`.
    /// * `mask` - Optional attention mask.
    ///
    /// # Errors
    /// Returns an error if the shapes are incompatible.
    fn forward(
        &self,
        query: &crate::tensor_engine::Tensor,
        key: &crate::tensor_engine::Tensor,
        value: &crate::tensor_engine::Tensor,
        mask: Option<&crate::tensor_engine::Tensor>,
    ) -> Result<crate::tensor_engine::Tensor, AttentionError>;
}

/// Errors emitted by attention layers.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    /// Shape mismatch.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },
    /// An underlying tensor error.
    #[error("tensor error: {0}")]
    Tensor(String),
    /// Unsupported attention kind.
    #[error("unsupported attention kind: {0:?}")]
    Unsupported(AttentionKind),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mha_config() {
        let cfg = AttentionConfig::mha(768, 12, 2048);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_kv_heads, 12);
        assert!((cfg.scale - 0.125).abs() < 1e-3);
    }

    #[test]
    fn gqa_config() {
        let cfg = AttentionConfig::gqa(768, 12, 4, 2048);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.kind, AttentionKind::Gqa);
    }
}
