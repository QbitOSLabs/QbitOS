//! Composite speech module.

use std::sync::Arc;

use qbitos_memory::embedding::SharedEmbeddingModel;

use crate::audio_engine::AudioEngine;
use crate::audio_memory::AudioMemory;
use crate::speech_to_text::{SttConfig, SttEngine};
use crate::text_to_speech::{TtsConfig, TtsEngine};
use crate::voice_manager::VoiceManager;

/// The composite speech module.
pub struct SpeechModule {
    /// The audio engine.
    pub audio: Arc<AudioEngine>,
    /// The STT engine.
    pub stt: Arc<SttEngine>,
    /// The TTS engine.
    pub tts: Arc<TtsEngine>,
    /// The audio memory.
    pub memory: Arc<AudioMemory>,
    /// The voice manager.
    pub voice: Arc<VoiceManager>,
}

impl SpeechModule {
    /// Build a new speech module.
    #[must_use]
    pub fn new(embedder: SharedEmbeddingModel) -> Self {
        let audio = Arc::new(AudioEngine::default());
        let stt = Arc::new(SttEngine::new(SttConfig::default()));
        let tts = Arc::new(TtsEngine::new(TtsConfig::default()));
        let memory = Arc::new(AudioMemory::new(embedder));
        let voice = Arc::new(VoiceManager::new(Arc::clone(&audio), Arc::clone(&stt), Arc::clone(&tts)));
        Self { audio, stt, tts, memory, voice }
    }

    /// Load all models.
    ///
    /// # Errors
    /// Returns an error if any model fails to load.
    pub async fn load_all(&self) -> anyhow::Result<()> {
        self.stt.load().await?;
        self.tts.load().await?;
        Ok(())
    }
}
