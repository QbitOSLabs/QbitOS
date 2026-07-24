//! Audio engine.
//!
//! Wraps `cpal` for cross-platform audio I/O. Provides:
//!
//! - Microphone capture
//! - Speaker playback
//! - Audio format conversion (resampling, channel conversion)

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// 16-bit signed integer.
    I16,
    /// 32-bit float.
    F32,
}

impl AudioFormat {
    /// Size in bytes per sample.
    #[must_use]
    pub fn sample_size(&self) -> usize {
        match self {
            Self::I16 => 2,
            Self::F32 => 4,
        }
    }
}

/// An audio chunk: a fixed-size buffer of samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunk {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Sample format.
    pub format: AudioFormat,
    /// Interleaved samples (as bytes).
    pub data: Vec<u8>,
    /// Timestamp (ms from start of stream).
    pub timestamp_ms: u64,
}

impl AudioChunk {
    /// Create a new chunk.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u16, format: AudioFormat, data: Vec<u8>) -> Self {
        Self { sample_rate, channels, format, data, timestamp_ms: 0 }
    }

    /// Duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        let samples = self.data.len() / self.format.sample_size() / self.channels as usize;
        Duration::from_secs_f64(samples as f64 / self.sample_rate as f64)
    }

    /// Number of samples (per channel).
    #[must_use]
    pub fn num_samples(&self) -> usize {
        self.data.len() / self.format.sample_size() / self.channels as usize
    }

    /// Convert to F32 samples (interleaved).
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        match self.format {
            AudioFormat::F32 => self.data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            AudioFormat::I16 => self.data.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect(),
        }
    }
}

/// An audio input source.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Microphone (live capture).
    Microphone,
    /// Audio file (path).
    File(std::path::PathBuf),
    /// In-memory bytes (WAV/MP3/FLAC).
    Bytes(Vec<u8>),
}

/// An audio output sink.
#[derive(Debug, Clone)]
pub enum AudioOutput {
    /// Speaker (live playback).
    Speaker,
    /// Audio file (path).
    File(std::path::PathBuf),
}

/// Errors emitted by the audio engine.
#[derive(Debug, Error)]
pub enum AudioEngineError {
    /// No audio device available.
    #[error("no audio device available")]
    NoDevice,
    /// Audio I/O failed.
    #[error("audio I/O failed: {0}")]
    Io(String),
    /// Unsupported format.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// The audio engine.
pub struct AudioEngine {
    sample_rate: u32,
    channels: u16,
    format: AudioFormat,
    chunk_duration_ms: u64,
    capture_buffer: Arc<Mutex<Vec<AudioChunk>>>,
    playback_buffer: Arc<Mutex<Vec<AudioChunk>>>,
}

impl AudioEngine {
    /// Create a new audio engine with the given format.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u16, format: AudioFormat) -> Self {
        Self {
            sample_rate,
            channels,
            format,
            chunk_duration_ms: 20,
            capture_buffer: Arc::new(Mutex::new(Vec::new())),
            playback_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Set the chunk duration (in milliseconds).
    pub fn set_chunk_duration(&mut self, ms: u64) {
        self.chunk_duration_ms = ms;
    }

    /// Synthesize a sine wave for testing.
    #[must_use]
    pub fn synthesize_sine(&self, freq: f32, duration: Duration) -> AudioChunk {
        let num_samples = (duration.as_secs_f64() * self.sample_rate as f64) as usize;
        let mut data = Vec::with_capacity(num_samples * self.format.sample_size() * self.channels as usize);
        for i in 0..num_samples {
            let t = i as f32 / self.sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
            for _ in 0..self.channels {
                match self.format {
                    AudioFormat::F32 => data.extend_from_slice(&sample.to_le_bytes()),
                    AudioFormat::I16 => {
                        let v = (sample * 32767.0) as i16;
                        data.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        AudioChunk::new(self.sample_rate, self.channels, self.format, data)
    }

    /// Push a captured chunk into the buffer (used by the capture thread).
    pub fn push_capture(&self, chunk: AudioChunk) {
        self.capture_buffer.lock().push(chunk);
    }

    /// Drain captured chunks.
    pub fn drain_captures(&self) -> Vec<AudioChunk> {
        std::mem::take(&mut *self.capture_buffer.lock())
    }

    /// Enqueue a chunk for playback.
    pub fn enqueue_playback(&self, chunk: AudioChunk) {
        self.playback_buffer.lock().push(chunk);
    }

    /// Drain playback queue.
    pub fn drain_playback(&self) -> Vec<AudioChunk> {
        std::mem::take(&mut *self.playback_buffer.lock())
    }

    /// Get the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the number of channels.
    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(16_000, 1, AudioFormat::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_sine() {
        let engine = AudioEngine::new(16_000, 1, AudioFormat::F32);
        let chunk = engine.synthesize_sine(440.0, Duration::from_millis(100));
        assert_eq!(chunk.num_samples(), 1600);
        let samples = chunk.to_f32();
        assert!(samples.iter().any(|&s| s.abs() > 0.1));
    }

    #[test]
    fn chunk_duration() {
        let chunk = AudioChunk::new(16_000, 1, AudioFormat::F32, vec![0u8; 1600 * 4]);
        assert_eq!(chunk.duration().as_millis(), 100);
    }
}
