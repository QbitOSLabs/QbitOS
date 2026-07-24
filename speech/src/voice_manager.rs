//! Voice manager.
//!
//! Manages the lifecycle of a voice session: capture → STT → reasoning →
//! TTS → playback. Provides a simple API for "voice in, voice out"
//! interactions.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::audio_engine::AudioEngine;
use crate::speech_to_text::{SttEngine, SttResult};
use crate::text_to_speech::{TtsEngine, TtsRequest, TtsResult, VoiceProfile};

/// A voice session: a single conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSession {
    /// Unique id.
    pub id: Uuid,
    /// Started at.
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// STT result (filled in after capture).
    pub stt: Option<SttResult>,
    /// TTS result (filled in after synthesis).
    pub tts: Option<TtsResult>,
    /// Current state.
    pub state: VoiceSessionState,
}

/// State of a voice session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceSessionState {
    /// Capturing audio.
    Capturing,
    /// Transcribing.
    Transcribing,
    /// Reasoning (the kernel is processing the transcription).
    Reasoning,
    /// Synthesizing speech.
    Synthesizing,
    /// Playing back.
    Playing,
    /// Done.
    Done,
}

/// The voice manager.
pub struct VoiceManager {
    audio: Arc<AudioEngine>,
    stt: Arc<SttEngine>,
    tts: Arc<TtsEngine>,
    default_voice: VoiceProfile,
    sessions: Arc<Mutex<std::collections::HashMap<Uuid, VoiceSession>>>,
}

impl VoiceManager {
    /// Create a new voice manager.
    #[must_use]
    pub fn new(audio: Arc<AudioEngine>, stt: Arc<SttEngine>, tts: Arc<TtsEngine>) -> Self {
        Self {
            audio,
            stt,
            tts,
            default_voice: VoiceProfile::default(),
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Set the default voice.
    pub fn set_default_voice(&mut self, voice: VoiceProfile) {
        self.default_voice = voice;
    }

    /// Start a new voice session.
    #[must_use]
    pub fn start_session(&self) -> Uuid {
        let id = Uuid::now_v7();
        let session = VoiceSession {
            id,
            started_at: chrono::Utc::now(),
            stt: None,
            tts: None,
            state: VoiceSessionState::Capturing,
        };
        self.sessions.lock().insert(id, session);
        id
    }

    /// Capture audio for `duration` and run STT.
    ///
    /// # Errors
    /// Propagates [`crate::speech_to_text::SttError`].
    pub async fn capture_and_transcribe(&self, session_id: Uuid) -> Result<SttResult, crate::speech_to_text::SttError> {
        // Update session state
        {
            let mut sessions = self.sessions.lock();
            if let Some(s) = sessions.get_mut(&session_id) {
                s.state = VoiceSessionState::Transcribing;
            }
        }
        // Drain capture buffer
        let chunks = self.audio.drain_captures();
        let result = self.stt.transcribe(&chunks).await?;
        {
            let mut sessions = self.sessions.lock();
            if let Some(s) = sessions.get_mut(&session_id) {
                s.stt = Some(result.clone());
                s.state = VoiceSessionState::Reasoning;
            }
        }
        Ok(result)
    }

    /// Synthesize speech for a text response.
    ///
    /// # Errors
    /// Propagates [`crate::text_to_speech::TtsError`].
    pub async fn synthesize_response(&self, session_id: Uuid, text: &str) -> Result<TtsResult, crate::text_to_speech::TtsError> {
        {
            let mut sessions = self.sessions.lock();
            if let Some(s) = sessions.get_mut(&session_id) {
                s.state = VoiceSessionState::Synthesizing;
            }
        }
        let request = TtsRequest {
            text: text.into(),
            voice: self.default_voice.clone(),
            sample_rate: self.audio.sample_rate(),
        };
        let result = self.tts.synthesize(&request).await?;
        self.audio.enqueue_playback(result.audio.clone());
        {
            let mut sessions = self.sessions.lock();
            if let Some(s) = sessions.get_mut(&session_id) {
                s.tts = Some(result.clone());
                s.state = VoiceSessionState::Playing;
            }
        }
        Ok(result)
    }

    /// End a session.
    pub fn end_session(&self, session_id: Uuid) {
        let mut sessions = self.sessions.lock();
        if let Some(s) = sessions.get_mut(&session_id) {
            s.state = VoiceSessionState::Done;
        }
    }

    /// Get a session by id.
    #[must_use]
    pub fn session(&self, session_id: Uuid) -> Option<VoiceSession> {
        self.sessions.lock().get(&session_id).cloned()
    }
}
