//! Embeddings.
//!
//! Every memory carries an embedding. In production this is produced by a
//! sentence-transformer or a multimodal encoder running on the neural engine.
//! In tests we use [`EmbeddingModelStub`] which produces deterministic
//! hash-based vectors.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A dense embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The vector data.
    pub data: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding from a vector.
    #[must_use]
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// L2 norm.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize in place.
    pub fn normalize(&mut self) {
        let n = self.norm().max(1e-12);
        for x in &mut self.data {
            *x /= n;
        }
    }

    /// Cosine similarity to another embedding.
    #[must_use]
    pub fn cosine(&self, other: &Self) -> f32 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }
        let dot: f32 = self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum();
        let na = self.norm().max(1e-12);
        let nb = other.norm().max(1e-12);
        dot / (na * nb)
    }

    /// Convert to an [`ndarray::Array1`].
    #[must_use]
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.data.clone())
    }
}

/// Trait for models that produce an [`Embedding`] from arbitrary content.
pub trait EmbeddingModel: Send + Sync {
    /// Embed a piece of text.
    fn embed_text(&self, text: &str) -> Embedding;

    /// Embed a JSON value (multimodal: text, image caption, etc.).
    fn embed_json(&self, value: &serde_json::Value) -> Embedding {
        let text = serde_json::to_string(value).unwrap_or_default();
        self.embed_text(&text)
    }

    /// Dimensionality of the embedding vectors produced by this model.
    /// Required so callers can size downstream structures (HNSW graphs,
    /// storage buffers) without first invoking the model.
    fn dim(&self) -> usize;
}

/// A deterministic embedding model that produces hash-based 384-dim vectors from
/// the SHA-256 of the input. Useful for tests and bootstrapping.
#[derive(Debug, Default, Clone)]
pub struct EmbeddingModelStub {
    dim: usize,
}

impl EmbeddingModelStub {
    /// Create a new deterministic embedding model with the given dimensionality.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbeddingModel for EmbeddingModelStub {
    fn embed_text(&self, text: &str) -> Embedding {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();
        // Use the hash as a seed to produce `dim` values.
        let mut data = Vec::with_capacity(self.dim);
        let mut state = [0u8; 32];
        state.copy_from_slice(&hash);
        for i in 0..self.dim {
            let byte = state[i % 32];
            // Map to [-1, 1]
            let v = (byte as f32 / 255.0) * 2.0 - 1.0;
            data.push(v);
            // Rotate the state
            state[i % 32] = state[i % 32].wrapping_add(17);
        }
        let mut e = Embedding::new(data);
        e.normalize();
        e
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// A type-erased embedding model.
pub type SharedEmbeddingModel = Arc<dyn EmbeddingModel + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity() {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0, 0.0]);
        assert!((a.cosine(&b).abs()) < 1e-6);
        let c = Embedding::new(vec![1.0, 0.0, 0.0]);
        assert!((a.cosine(&c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_model_is_deterministic() {
        let m = EmbeddingModelStub::new(384);
        let a = m.embed_text("hello");
        let b = m.embed_text("hello");
        assert_eq!(a.data, b.data);
    }
}
