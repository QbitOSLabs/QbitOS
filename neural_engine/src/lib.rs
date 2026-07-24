//! QbitOS Neural Engine
//!
//! Native Rust inference engine built on [`candle-core`]. Supports:
//!
//! - 3B–4B parameter models (Llama, Qwen, Gemma, Mistral, Phi)
//! - Attention variants: MHA, GQA, MQA, sliding window, ring, flash
//! - Sequence models: Transformer, SSM (Mamba), hybrid
//! - KV cache management + compression, context compression
//! - Continuous batching, speculative decoding
//! - Quantization: INT8, INT4, FP8, NF4, GPTQ, AWQ, GGUF
//!
//! The engine exposes a single [`InferenceEngine`] trait that the cognitive
//! kernel calls; concrete implementations live in [`backends`].

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod advanced_cognition;
pub mod moe;
pub use moe::{MoeConfig, MoeError, MoeOutput, MixtureOfExperts};
pub mod attention;
pub mod backends;
pub mod config;
pub mod kv_cache;
pub mod model_manager;
pub mod native_agent;
pub mod quantization;
pub mod react;
pub mod sampling;
pub mod tensor_engine;

pub use advanced_cognition::{
    ConstitutionalChecker, ConstitutionalResult, Principle, SelfReflection, ReflectionResult,
    TreeOfThought, ThoughtNode, TotConfig, UncertaintyEstimate, estimate_uncertainty,
};
pub use attention::{Attention, AttentionConfig, AttentionKind};
pub use backends::{CandleBackend, InferenceEngine, InferenceRequest, InferenceResponse, ModelBackend};
pub use config::{DeviceConfig, EngineConfig, QuantizationConfig};
pub use kv_cache::{KvCache, KvCacheConfig, KvCacheEntry};
pub use model_manager::{ModelHandle, ModelId, ModelManager};
pub use native_agent::{NativeAgent, NativeAgentContext, NativeResponse, ResponseSource};
pub use quantization::{QuantizationDtype, Quantizer};
pub use sampling::{Sampler, SamplerConfig, SamplingStrategy};
pub use react::{
    run_react, ReActAction, ReActConfig, ReActReasoner, ReActStep, ReActToolExecutor,
    ReActTrajectory,
};
pub use tensor_engine::{Tensor, TensorDtype};

/// Semantic version of the neural engine.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
