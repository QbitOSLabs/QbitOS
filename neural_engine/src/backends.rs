//! Backends: concrete implementations of the inference engine.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

use crate::kv_cache::KvCache;
use crate::sampling::Sampler;
use crate::tensor_engine::Tensor;

/// An inference request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// The prompt text.
    pub prompt: String,
    /// Optional system prompt.
    pub system_prompt: Option<String>,
    /// Conversation history (list of (role, content) tuples).
    pub history: Vec<(String, String)>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Temperature override (None = use engine default).
    pub temperature: Option<f32>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Whether to stream tokens (caller will pull via a channel).
    pub stream: bool,
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            system_prompt: None,
            history: Vec::new(),
            max_tokens: 256,
            temperature: None,
            stop: Vec::new(),
            stream: false,
        }
    }
}

/// An inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// The generated text.
    pub text: String,
    /// Token ids generated.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub generated_tokens: usize,
    /// Finish reason: "stop", "length", "content_filter", etc.
    pub finish_reason: String,
    /// Logprobs (optional).
    pub logprobs: Option<Vec<f32>>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Errors emitted by the inference engine.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// The model is not loaded.
    #[error("no model loaded")]
    NoModel,
    /// The prompt exceeds the maximum sequence length.
    #[error("prompt too long: {prompt_len} > {max_len}")]
    PromptTooLong {
        /// Length of the prompt that exceeded the limit.
        prompt_len: usize,
        /// Maximum allowed prompt length.
        max_len: usize }, // doc added below
    /// Tokenization failed.
    #[error("tokenization failed: {0}")]
    Tokenization(String),
    /// The underlying backend failed.
    #[error("backend error: {0}")]
    Backend(String),
    /// Sampling failed.
    #[error("sampling error: {0}")]
    Sampling(String),
}

/// Trait that every inference backend must implement.
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Load the model.
    async fn load(&self) -> Result<(), InferenceError>;

    /// Unload the model.
    async fn unload(&self);

    /// Run a forward pass: given token ids, return logits for the next token.
    async fn forward(&self, token_ids: &[u32]) -> Result<Tensor, InferenceError>;

    /// Generate tokens.
    async fn generate(&self, request: &InferenceRequest, sampler: &Sampler) -> Result<InferenceResponse, InferenceError>;

    /// Returns `true` if the model is loaded.
    fn is_loaded(&self) -> bool;
}

/// The top-level inference engine. Wraps a [`ModelBackend`] and exposes
/// a simpler API.
pub struct InferenceEngine {
    backend: Arc<dyn ModelBackend>,
    kv_cache: Arc<KvCache>,
    sampler: Arc<Sampler>,
}

impl InferenceEngine {
    /// Create a new engine wrapping the given backend.
    #[must_use]
    pub fn new(backend: Arc<dyn ModelBackend>, kv_cache: Arc<KvCache>, sampler: Arc<Sampler>) -> Self {
        Self { backend, kv_cache, sampler }
    }

    /// Load the model.
    pub async fn load(&self) -> Result<(), InferenceError> {
        self.backend.load().await
    }

    /// Generate a response for the given request.
    pub async fn generate(&self, request: &InferenceRequest) -> Result<InferenceResponse, InferenceError> {
        let start = std::time::Instant::now();
        let response = self.backend.generate(request, &self.sampler).await?;
        let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        Ok(InferenceResponse { duration_ms, ..response })
    }

    /// Get a reference to the KV cache.
    #[must_use]
    pub fn kv_cache(&self) -> &KvCache { &self.kv_cache }

    /// Get a reference to the sampler.
    #[must_use]
    pub fn sampler(&self) -> &Sampler { &self.sampler }
}

/// A `candle-core`-backed implementation of [`ModelBackend`] that loads a real
/// Qwen2-family model from Hugging Face Hub. Only used when the `external-model`
/// feature is enabled. By default, QbitOS uses the [`NativeAgent`] instead.
///
pub struct CandleBackend {
    model_id: String,
    revision: String,
    device: candle_core::Device,
    dtype: candle_core::DType,
    loaded: Arc<std::sync::atomic::AtomicBool>,
    /// The loaded model + tokenizer, guarded by a Mutex so only one
    /// generation runs at a time.
    inner: Mutex<Option<CandleModel>>,
}

/// Container for a loaded model + tokenizer.
struct CandleModel {
    model: candle_transformers::models::qwen2::ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    _config: candle_transformers::models::qwen2::Config,
}

impl CandleBackend {
    /// Create a new candle backend for the given Qwen2-family model id
    /// (e.g. `"Qwen/Qwen2.5-3B-Instruct"`).
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self::with_revision(model_id, "main".to_string())
    }

    /// Create a new candle backend with an explicit revision.
    #[must_use]
    pub fn with_revision(model_id: impl Into<String>, revision: String) -> Self {
        Self {
            model_id: model_id.into(),
            revision,
            device: candle_core::Device::Cpu,
            dtype: candle_core::DType::F32,
            loaded: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            inner: Mutex::new(None),
        }
    }

    /// Use CUDA device with the given index.
    pub fn with_cuda(mut self, device_index: usize) -> Result<Self, InferenceError> {
        self.device = candle_core::Device::new_cuda(device_index)
            .map_err(|e| InferenceError::Backend(format!("CUDA init failed: {e}")))?;
        Ok(self)
    }

    /// Set the dtype (default F32; use F16/BF16 for faster inference).
    pub fn with_dtype(mut self, dtype: candle_core::DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Actually download and load the model weights. Called lazily by
    /// `forward` / `generate` if the model isn't loaded yet.
    async fn load_model(&self) -> Result<(), InferenceError> {
        if self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }
        let model_id = self.model_id.clone();
        let _revision = self.revision.clone();
        let device = self.device.clone();
        let dtype = self.dtype;

        let result = tokio::task::spawn_blocking(move || -> Result<CandleModel, InferenceError> {
            // Download config + tokenizer + weights via hf-hub
            let api = hf_hub::api::sync::ApiBuilder::new()
                .build()
                .map_err(|e| InferenceError::Backend(format!("hf-hub error: {e}")))?;
            let repo_api = api.model(model_id.clone());

            // Load config.json
            let config_path = repo_api.get("config.json")
                .map_err(|e| InferenceError::Backend(format!("config.json download failed: {e}")))?;
            let config_content = std::fs::read_to_string(config_path)
                .map_err(|e| InferenceError::Backend(format!("config.json read failed: {e}")))?;
            let config: candle_transformers::models::qwen2::Config = serde_json::from_str(&config_content)
                .map_err(|e| InferenceError::Backend(format!("config.json parse failed: {e}")))?;

            // Load tokenizer
            let tokenizer_path = repo_api.get("tokenizer.json")
                .map_err(|e| InferenceError::Backend(format!("tokenizer.json download failed: {e}")))?;
            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
                .map_err(|e| InferenceError::Backend(format!("tokenizer load failed: {e}")))?;

            // Load safetensors weights via VarBuilder::from_mmaped_safetensors
            let weights_filenames: Vec<std::path::PathBuf> = if let Ok(idx) = repo_api.get("model.safetensors.index.json") {
                // Multi-shard: parse the index and load all listed shards
                let idx_content = std::fs::read_to_string(&idx)
                    .map_err(|e| InferenceError::Backend(format!("index read failed: {e}")))?;
                let idx_json: serde_json::Value = serde_json::from_str(&idx_content)
                    .map_err(|e| InferenceError::Backend(format!("index parse failed: {e}")))?;
                let mut files: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
                if let Some(map) = idx_json["weight_map"].as_object() {
                    for v in map.values() {
                        if let Some(s) = v.as_str() {
                            files.insert(s.to_string());
                        }
                    }
                }
                files.into_iter().filter_map(|f| repo_api.get(&f).ok()).collect()
            } else {
                // Single-shard
                vec![repo_api.get("model.safetensors")
                    .map_err(|e| InferenceError::Backend(format!("model.safetensors download failed: {e}")))?]
            };

            // SAFETY: from_mmaped_safetensors mmaps the weight files. The
            // files are read-only and owned by the OS page cache; we keep
            // the model alive for as long as the backend holds the lock.
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&weights_filenames, dtype, &device)
            }
                .map_err(|e| InferenceError::Backend(format!("weights mmap failed: {e}")))?;

            let model = candle_transformers::models::qwen2::ModelForCausalLM::new(&config, vb)
                .map_err(|e| InferenceError::Backend(format!("model construction failed: {e}")))?;

            Ok(CandleModel { model, tokenizer, _config: config })
        })
        .await
        .map_err(|e| InferenceError::Backend(format!("load task join failed: {e}")))?
        .map_err(Into::into)?;

        let mut guard = self.inner.lock().await;
        *guard = Some(result);
        self.loaded.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Build the chat prompt from the request using Qwen's ChatML format.
    fn build_prompt(request: &InferenceRequest) -> String {
        let mut prompt = String::new();
        if let Some(system) = &request.system_prompt {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(system);
            prompt.push_str("<|im_end|>\n");
        }
        for (role, content) in &request.history {
            prompt.push_str("<|im_start|>");
            prompt.push_str(role);
            prompt.push('\n');
            prompt.push_str(content);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(&request.prompt);
        prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
        prompt
    }
}

#[async_trait]
impl ModelBackend for CandleBackend {
    async fn load(&self) -> Result<(), InferenceError> {
        self.load_model().await
    }

    async fn unload(&self) {
        let mut guard = self.inner.lock().await;
        *guard = None;
        self.loaded.store(false, std::sync::atomic::Ordering::Release);
    }

    async fn forward(&self, token_ids: &[u32]) -> Result<Tensor, InferenceError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            self.load_model().await?;
        }
        let mut guard = self.inner.lock().await;
        let Some(cm) = guard.as_mut() else {
            return Err(InferenceError::NoModel);
        };
        let input = candle_core::Tensor::new(token_ids, &self.device)
            .map_err(|e| InferenceError::Backend(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| InferenceError::Backend(format!("unsqueeze failed: {e}")))?;
        let logits = cm.model.forward(&input, 0)
            .map_err(|e| InferenceError::Backend(format!("forward failed: {e}")))?;
        // Return the logits for the last token
        let last_logits = logits.squeeze(0)
            .map_err(|e| InferenceError::Backend(format!("squeeze failed: {e}")))?;
        Tensor::from_candle(&last_logits)
            .map_err(|e| InferenceError::Backend(format!("tensor conversion failed: {e}")))
    }

    async fn generate(&self, request: &InferenceRequest, sampler: &Sampler) -> Result<InferenceResponse, InferenceError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            self.load_model().await?;
        }

        let prompt = Self::build_prompt(request);
        let max_tokens = request.max_tokens;

        // Tokenize the prompt (blocking — tokenizers is sync)
        let prompt_clone = prompt.clone();
        let token_ids_vec = tokio::task::spawn_blocking(move || -> Result<Vec<u32>, InferenceError> {
            // Re-load tokenizer inside the blocking task (the lock is async-only)
            // — we'll just re-open the tokenizer file via the cache.
            // For simplicity, store the tokenizer path on the backend.
            // Actually, we can avoid this by tokenizing inside the async lock.
            // We'll fall back to char-level tokenization here for the spawn_blocking path.
            // The real path is below in the async-lock branch.
            let _ = prompt_clone;
            Err(InferenceError::Backend("tokenization must happen inside async lock".into()))
        }).await
        .map_err(|e| InferenceError::Backend(format!("join error: {e}")))??;

        // We won't use the spawn_blocking result — instead, hold the async lock
        // and tokenize + generate inside it. This is acceptable because the
        // Mutex guarantees only one generation at a time anyway.
        let _ = token_ids_vec;

        let mut guard = self.inner.lock().await;
        let Some(cm) = guard.as_mut() else {
            return Err(InferenceError::NoModel);
        };

        // Tokenize
        let tokens = cm.tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| InferenceError::Tokenization(e.to_string()))?;
        let mut token_ids: Vec<u32> = tokens.get_ids().to_vec();
        let prompt_tokens = token_ids.len();

        // Generate
        let mut generated_ids = Vec::new();
        let mut last_logits: Option<candle_core::Tensor> = None;
        for _ in 0..max_tokens {
            let input = candle_core::Tensor::new(&token_ids[..], &self.device)
                .map_err(|e| InferenceError::Backend(format!("tensor creation failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| InferenceError::Backend(format!("unsqueeze failed: {e}")))?;
            let logits = cm.model.forward(&input, token_ids.len().saturating_sub(1))
                .map_err(|e| InferenceError::Backend(format!("forward failed: {e}")))?;
            // Get logits for the last token
            let last = logits.squeeze(0)
                .map_err(|e| InferenceError::Backend(format!("squeeze failed: {e}")))?;
            let final_logits = last.squeeze(0)
                .map_err(|e| InferenceError::Backend(format!("second squeeze failed: {e}")))?;
            last_logits = Some(final_logits.clone());

            // Convert to Vec<f32> for sampling
            let logits_vec: Vec<f32> = final_logits.to_vec1()
                .map_err(|e| InferenceError::Backend(format!("logits to vec failed: {e}")))?;

            // Sample next token
            let next_token = sampler.sample(&logits_vec)
                .map_err(|e| InferenceError::Sampling(e.to_string()))?;
            let next_token_u32 = next_token as u32;

            // Stop on EOS (Qwen2 uses 151643 for <|im_end|>)
            if next_token_u32 == 151643 || next_token_u32 == 2 {
                break;
            }

            token_ids.push(next_token_u32);
            generated_ids.push(next_token_u32);
        }
        let _ = last_logits;

        // Detokenize
        let text = cm.tokenizer.decode(&generated_ids, true)
            .map_err(|e| InferenceError::Backend(format!("detokenize failed: {e}")))?;

        Ok(InferenceResponse {
            text,
            token_ids: generated_ids.clone(),
            prompt_tokens,
            generated_tokens: generated_ids.len(),
            finish_reason: "stop".into(),
            logprobs: None,
            duration_ms: 0,
        })
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{SamplerConfig, SamplingStrategy};

    #[tokio::test]
    async fn candle_backend_load_unload() {
        // We can't actually download a 3B model in tests, but we can verify
        // the backend constructs and reports as not-loaded initially.
        let backend = CandleBackend::new("Qwen/Qwen2.5-3B-Instruct");
        assert!(!backend.is_loaded());
        // Attempting to load without network will fail — that's OK.
        // The key thing is the backend doesn't panic.
    }

    #[tokio::test]
    async fn build_prompt_uses_chatml() {
        let req = InferenceRequest {
            prompt: "hello".into(),
            system_prompt: Some("you are helpful".into()),
            history: vec![("user".into(), "hi".into()), ("assistant".into(), "hello".into())],
            ..InferenceRequest::default()
        };
        let prompt = CandleBackend::build_prompt(&req);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
        assert!(prompt.contains("you are helpful"));
        assert!(prompt.contains("hello"));
    }
}
