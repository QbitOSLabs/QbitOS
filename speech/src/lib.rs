//! QbitOS Speech Intelligence
//!
//! Voice communication pipeline:
//!
//! ```text
//! Audio Input → STT (Whisper) → Text → Reasoning → Text → TTS (VITS) → Audio Output
//! ```
//!
//! Built on `cpal` for audio I/O, `symphonia` for decoding, `rustfft` for
//! spectral analysis, and `candle-core` (via `qbitos-neural-engine`) for
//! neural inference.

#![cfg_attr(not(test), warn(clippy::pedantic))]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate, clippy::missing_errors_doc)]

pub mod advanced_tts;
pub mod neural_codec_tts;
pub mod audio_engine;
pub mod audio_memory;
pub mod speech_to_text;
pub mod text_to_speech;
pub mod voice_manager;

pub use advanced_tts::{
    synthesize_ssml, parse_ssml, Emphasis, Prosody, SpeechSample, SsmlDocument, SsmlElement,
    VoiceProfile as AdvancedVoiceProfile,
};
pub use audio_engine::{AudioChunk, AudioEngine, AudioFormat, AudioInput, AudioOutput};
pub use audio_memory::{AudioMemory, AudioMemoryEntry};
pub use speech_to_text::{SttEngine, SttResult, Transcription};
pub use text_to_speech::{TtsEngine, TtsRequest, TtsResult, VoiceProfile};
pub use voice_manager::{VoiceManager, VoiceSession};

/// Re-export of the composite speech module.
pub use speech_module::SpeechModule;

mod speech_module;
