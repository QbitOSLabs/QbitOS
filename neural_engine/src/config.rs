//! Engine configuration.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the neural engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// The device to run inference on.
    pub device: DeviceConfig,
    /// Quantization settings.
    pub quantization: QuantizationConfig,
    /// KV cache settings.
    pub kv_cache: KvCacheSpec,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Maximum batch size for continuous batching.
    pub max_batch_size: usize,
    /// Directory for downloaded models.
    pub model_dir: PathBuf,
    /// Whether to enable speculative decoding.
    pub enable_speculative_decoding: bool,
    /// Whether to enable flash attention when available.
    pub enable_flash_attention: bool,
    /// Number of threads to use for CPU inference (0 = auto).
    pub num_threads: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let model_dir = std::env::var("QBITOS_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/var/lib/qbitos/models"));
        Self {
            device: DeviceConfig::Auto,
            quantization: QuantizationConfig::default(),
            kv_cache: KvCacheSpec::default(),
            max_seq_len: 4096,
            max_batch_size: 8,
            model_dir,
            enable_speculative_decoding: true,
            enable_flash_attention: true,
            num_threads: 0,
        }
    }
}

/// Device configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceConfig {
    /// Auto-detect: use CUDA if available, else CPU.
    Auto,
    /// CPU only.
    Cpu,
    /// CUDA GPU with the given device index.
    Cuda(usize),
    /// Metal (macOS only).
    Metal,
}

/// Quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// The quantization dtype to use.
    pub dtype: QuantizationDtypeSpec,
    /// Whether to use quantized matmul.
    pub use_quantized_matmul: bool,
    /// Group size for group-wise quantization.
    pub group_size: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            dtype: QuantizationDtypeSpec::F16,
            use_quantized_matmul: false,
            group_size: 64,
        }
    }
}

/// Quantization dtype specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationDtypeSpec {
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
    /// NF4.
    Nf4,
    /// FP8.
    Fp8,
}

/// KV cache specification (lightweight mirror of [`crate::kv_cache::KvCacheConfig`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheSpec {
    /// Maximum number of tokens to cache.
    pub max_tokens: usize,
    /// Whether to enable KV cache compression.
    pub enable_compression: bool,
    /// Compression ratio (1.0 = no compression).
    pub compression_ratio: f32,
}

impl Default for KvCacheSpec {
    fn default() -> Self {
        Self {
            max_tokens: 8192,
            enable_compression: true,
            compression_ratio: 0.5,
        }
    }
}
