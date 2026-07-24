//! Audio memory.
//!
//! Stores audio snippets (voice commands, environmental sounds) for later
//! retrieval. Each snippet is indexed by its transcription embedding so
//! "what did the user say about X?" queries can be served.

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use qbitos_memory::embedding::Embedding;

use crate::audio_engine::AudioChunk;

/// An audio memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMemoryEntry {
    /// Unique id.
    pub id: Uuid,
    /// When the audio was captured.
    pub captured_at: DateTime<Utc>,
    /// Source: "microphone", "file", "tts".
    pub source: String,
    /// Transcription (if available).
    pub transcription: Option<String>,
    /// Speaker id (if known).
    pub speaker: Option<String>,
    /// Embedding of the transcription.
    pub embedding: Embedding,
    /// Audio metadata (sample rate, channels, duration).
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Audio memory.
pub struct AudioMemory {
    entries: Arc<RwLock<HashMap<Uuid, AudioMemoryEntry>>>,
    embedder: Arc<dyn qbitos_memory::embedding::EmbeddingModel + Send + Sync>,
}

impl AudioMemory {
    /// Create a new audio memory.
    #[must_use]
    pub fn new(embedder: Arc<dyn qbitos_memory::embedding::EmbeddingModel + Send + Sync>) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            embedder,
        }
    }

    /// Record an audio chunk.
    pub fn record(
        &self,
        source: impl Into<String>,
        transcription: Option<String>,
        speaker: Option<String>,
        chunk: &AudioChunk,
    ) -> Uuid {
        let transcription_text = transcription.clone().unwrap_or_default();
        let embedding = if transcription_text.is_empty() {
            Embedding::new(vec![])
        } else {
            self.embedder.embed_text(&transcription_text)
        };
        let id = Uuid::now_v7();
        let entry = AudioMemoryEntry {
            id,
            captured_at: Utc::now(),
            source: source.into(),
            transcription,
            speaker,
            embedding,
            sample_rate: chunk.sample_rate,
            channels: chunk.channels,
            duration_ms: u64::try_from(chunk.duration().as_millis()).unwrap_or(0),
        };
        self.entries.write().insert(id, entry);
        id
    }

    /// Find entries similar to the query.
    #[must_use]
    pub fn find_similar(&self, query: &str, k: usize) -> Vec<AudioMemoryEntry> {
        let query_emb = self.embedder.embed_text(query);
        let entries = self.entries.read();
        let mut scored: Vec<(f32, AudioMemoryEntry)> = entries.values()
            .filter(|e| !e.embedding.data.is_empty())
            .map(|e| (e.embedding.cosine(&query_emb), e.clone()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, e)| e).collect()
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }
}
