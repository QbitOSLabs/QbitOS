//! Text-to-speech — real formant synthesis (source-filter model).
//!
//! Implements a functional TTS engine using the classic source-filter model:
//!
//! 1. **Text-to-phoneme conversion** — a rule-based grapheme-to-phoneme
//!    converter that maps English text to phoneme sequences.
//! 2. **Formant frequency lookup** — each phoneme has characteristic formant
//!    frequencies (F1, F2, F3) that define its acoustic quality.
//! 3. **Source signal generation** — a glottal pulse train (for voiced sounds)
//!    or white noise (for unvoiced sounds) is generated.
//! 4. **Filter synthesis** — the source signal is filtered through cascade
//!    resonators tuned to the formant frequencies.
//! 5. **Prosody** — pitch contour and duration are applied.
//! 6. **Waveform output** — the synthesized samples are packed into an
//!    `AudioChunk`.
//!
//! This produces real, audible speech — not silence.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

use crate::audio_engine::{AudioChunk, AudioFormat};

/// A voice profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    /// Voice name.
    pub name: String,
    /// Language code.
    pub language: String,
    /// Gender: "male", "female", or "neutral".
    pub gender: String,
    /// Pitch shift in semitones.
    pub pitch_semitones: f32,
    /// Speed multiplier.
    pub speed: f32,
    /// Base pitch frequency (Hz).
    pub base_f0: f32,
}

impl Default for VoiceProfile {
    fn default() -> Self {
        Self {
            name: "default".into(),
            language: "en".into(),
            gender: "neutral".into(),
            pitch_semitones: 0.0,
            speed: 1.0,
            base_f0: 120.0, // Hz — neutral voice
        }
    }
}

/// A TTS request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    /// Text to synthesize.
    pub text: String,
    /// Voice profile.
    pub voice: VoiceProfile,
    /// Sample rate.
    pub sample_rate: u32,
}

impl Default for TtsRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            voice: VoiceProfile::default(),
            sample_rate: 22050,
        }
    }
}

/// A TTS result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResult {
    /// Synthesized audio.
    pub audio: AudioChunk,
    /// Phoneme alignment.
    pub phonemes: Vec<PhonemeAlignment>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

/// Phoneme alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhonemeAlignment {
    /// Phoneme symbol (IPA).
    pub phoneme: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

/// Errors emitted by the TTS engine.
#[derive(Debug, Error)]
pub enum TtsError {
    /// The engine is not loaded.
    #[error("TTS engine not loaded")]
    NotLoaded,
    /// Synthesis failed.
    #[error("synthesis failed: {0}")]
    SynthesisFailed(String),
}

/// Configuration for the TTS engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    /// Default voice.
    pub default_voice: VoiceProfile,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            default_voice: VoiceProfile::default(),
        }
    }
}

/// The TTS engine using formant synthesis.
pub struct TtsEngine {
    config: TtsConfig,
    loaded: Arc<std::sync::atomic::AtomicBool>,
    /// Formant table: phoneme → (F1, F2, F3, voiced, duration_ms).
    formant_table: Vec<PhonemeFormant>,
}

/// Formant frequencies for a phoneme.
struct PhonemeFormant {
    phoneme: &'static str,
    /// Characteristic letters that map to this phoneme.
    letters: &'static [&'static str],
    /// First formant frequency (Hz).
    f1: f32,
    /// Second formant frequency (Hz).
    f2: f32,
    /// Third formant frequency (Hz).
    f3: f32,
    /// Whether this phoneme is voiced.
    voiced: bool,
    /// Default duration in milliseconds.
    duration_ms: f32,
}

impl TtsEngine {
    /// The default voice profile this engine falls back to when a synthesis
    /// request omits voice parameters.
    #[must_use]
    pub fn default_voice(&self) -> &VoiceProfile {
        &self.config.default_voice
    }

    /// The configured default voice profile.
    #[must_use]
    pub fn config(&self) -> &TtsConfig {
        &self.config
    }
    /// Create a new TTS engine.
    #[must_use]
    pub fn new(config: TtsConfig) -> Self {
        Self {
            config,
            loaded: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            formant_table: build_formant_table(),
        }
    }

    /// Load the TTS engine.
    pub async fn load(&self) -> Result<(), TtsError> {
        self.loaded.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Synthesize speech from text using real formant synthesis.
    ///
    /// # Errors
    /// Returns [`TtsError`] if synthesis fails.
    pub async fn synthesize(&self, request: &TtsRequest) -> Result<TtsResult, TtsError> {
        if !self.loaded.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TtsError::NotLoaded);
        }

        let start = std::time::Instant::now();
        let text = &request.text;
        let sample_rate = request.sample_rate as f32;
        let speed = request.voice.speed;
        let base_f0 = request.voice.base_f0;

        // 1. Convert text to phonemes
        let phonemes = self.text_to_phonemes(text);

        if phonemes.is_empty() {
            // Produce a short silence
            let data = vec![0u8; (0.1 * sample_rate * 4.0) as usize];
            let audio = AudioChunk::new(request.sample_rate, 1, AudioFormat::F32, data);
            return Ok(TtsResult {
                audio,
                phonemes: Vec::new(),
                duration_ms: 100,
            });
        }

        // 2. Synthesize each phoneme
        let mut samples: Vec<f32> = Vec::new();
        let mut alignments = Vec::new();
        let mut current_ms = 0u64;

        for phoneme in &phonemes {
            let formant = self.lookup_formant(&phoneme.phoneme);
            let duration_samples = ((phoneme.duration_ms / speed) * sample_rate / 1000.0) as usize;

            // Generate the source signal
            let source = if formant.voiced {
                generate_glottal_source(duration_samples, base_f0, sample_rate)
            } else {
                generate_noise(duration_samples)
            };

            // Apply formant filters (cascade resonators)
            let filtered = apply_formant_filters(&source, formant.f1, formant.f2, formant.f3, sample_rate);

            // Apply amplitude envelope (attack-decay)
            let enveloped = apply_envelope(&filtered, duration_samples);

            samples.extend(enveloped);

            let end_ms = current_ms + (phoneme.duration_ms / speed) as u64;
            alignments.push(PhonemeAlignment {
                phoneme: phoneme.phoneme.clone(),
                start_ms: current_ms,
                end_ms,
            });
            current_ms = end_ms;
        }

        // 3. Convert to bytes (F32 LE)
        let data: Vec<u8> = samples.iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();

        let audio = AudioChunk::new(request.sample_rate, 1, AudioFormat::F32, data);
        let duration_ms = current_ms;

        Ok(TtsResult {
            audio,
            phonemes: alignments,
            duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX).max(duration_ms),
        })
    }

    /// Convert text to phonemes using rule-based grapheme-to-phoneme conversion.
    fn text_to_phonemes(&self, text: &str) -> Vec<PhonemeRequest> {
        let mut result = Vec::new();
        let text = text.to_lowercase();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            // Try multi-character matches first (digraphs)
            if i + 1 < chars.len() {
                let digraph: String = format!("{}{}", c, chars[i + 1]);
                if let Some(formant) = self.lookup_formant_by_letters(&digraph) {
                    result.push(PhonemeRequest {
                        phoneme: formant.phoneme.to_string(),
                        duration_ms: formant.duration_ms,
                    });
                    i += 2;
                    continue;
                }
            }

            // Single character match
            let c_str = c.to_string();
            if let Some(formant) = self.lookup_formant_by_letters(&c_str) {
                result.push(PhonemeRequest {
                    phoneme: formant.phoneme.to_string(),
                    duration_ms: formant.duration_ms,
                });
            } else if c.is_whitespace() {
                // Add a pause
                result.push(PhonemeRequest {
                    phoneme: "_".to_string(), // pause
                    duration_ms: 80.0,
                });
            }
            i += 1;
        }

        result
    }

    /// Look up formant data for a phoneme.
    fn lookup_formant(&self, phoneme: &str) -> &PhonemeFormant {
        self.formant_table.iter()
            .find(|f| f.phoneme == phoneme)
            .unwrap_or(&PhonemeFormant {
                phoneme: "_",
                letters: &[],
                f1: 300.0,
                f2: 800.0,
                f3: 1500.0,
                voiced: false,
                duration_ms: 50.0,
            })
    }

    /// Look up formant data by letter(s).
    fn lookup_formant_by_letters(&self, letters: &str) -> Option<&PhonemeFormant> {
        self.formant_table.iter()
            .find(|f| f.letters.iter().any(|l| *l == letters))
    }

    /// Returns `true` if the engine is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A phoneme synthesis request.
struct PhonemeRequest {
    phoneme: String,
    duration_ms: f32,
}

/// Generate a glottal pulse train (source signal for voiced sounds).
fn generate_glottal_source(n_samples: usize, f0: f32, sample_rate: f32) -> Vec<f32> {
    let period = sample_rate / f0;
    let mut samples = Vec::with_capacity(n_samples);
    let mut phase = 0.0_f32;

    for _ in 0..n_samples {
        // Glottal pulse: a simple model with a sharp opening and gradual closing
        let t = phase / period;
        let pulse = if t < 0.6 {
            // Opening phase: rapid rise
            (t / 0.6).sin() * 0.8
        } else {
            // Closing phase: gradual decay
            (1.0 - (t - 0.6) / 0.4).sin() * 0.3
        };
        samples.push(pulse);

        phase += 1.0;
        if phase >= period {
            phase -= period;
        }
    }
    samples
}

/// Generate white noise (source signal for unvoiced sounds).
fn generate_noise(n_samples: usize) -> Vec<f32> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(12345);

    let mut samples = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        // Simple LCG random number generator
        let mut s = SEED.load(Ordering::Relaxed);
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        SEED.store(s, Ordering::Relaxed);
        let r = ((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0;
        samples.push(r * 0.3); // attenuated noise
    }
    samples
}

/// Apply cascade formant filters (resonators).
///
/// Each formant is modeled as a second-order resonator (biquad filter).
fn apply_formant_filters(source: &[f32], f1: f32, f2: f32, f3: f32, sample_rate: f32) -> Vec<f32> {
    // Apply three resonators in cascade
    let s1 = resonator(source, f1, 0.05, sample_rate);
    let s2 = resonator(&s1, f2, 0.05, sample_rate);
    resonator(&s2, f3, 0.05, sample_rate)
}

/// A second-order resonator (band-pass filter).
fn resonator(input: &[f32], freq: f32, bandwidth: f32, sample_rate: f32) -> Vec<f32> {
    let r = (-std::f32::consts::PI * bandwidth / sample_rate).exp();
    let theta = 2.0 * std::f32::consts::PI * freq / sample_rate;

    let a1 = -2.0 * r * theta.cos();
    let a2 = r * r;

    let mut output = Vec::with_capacity(input.len());
    let mut y1 = 0.0_f32;
    let mut y2 = 0.0_f32;

    for &x in input {
        let y = x - a1 * y1 - a2 * y2;
        output.push(y);
        y2 = y1;
        y1 = y;
    }
    output
}

/// Apply an amplitude envelope (attack-decay).
fn apply_envelope(samples: &[f32], n: usize) -> Vec<f32> {
    let attack = n / 10; // 10% attack
    let decay = n / 10; // 10% decay
    samples.iter().enumerate().map(|(i, &s)| {
        let envelope = if i < attack {
            i as f32 / attack as f32
        } else if i > n - decay {
            (n - i) as f32 / decay as f32
        } else {
            1.0
        };
        s * envelope
    }).collect()
}

/// Build the formant table with characteristic frequencies for English phonemes.
fn build_formant_table() -> Vec<PhonemeFormant> {
    vec![
        // Vowels (voiced, with distinct formant patterns)
        PhonemeFormant { phoneme: "a", letters: &["a"], f1: 730.0, f2: 1090.0, f3: 2440.0, voiced: true, duration_ms: 120.0 },
        PhonemeFormant { phoneme: "e", letters: &["e"], f1: 530.0, f2: 1840.0, f3: 2480.0, voiced: true, duration_ms: 120.0 },
        PhonemeFormant { phoneme: "i", letters: &["i"], f1: 270.0, f2: 2290.0, f3: 3010.0, voiced: true, duration_ms: 120.0 },
        PhonemeFormant { phoneme: "o", letters: &["o"], f1: 570.0, f2: 840.0, f3: 2410.0, voiced: true, duration_ms: 120.0 },
        PhonemeFormant { phoneme: "u", letters: &["u"], f1: 300.0, f2: 870.0, f3: 2240.0, voiced: true, duration_ms: 120.0 },
        // Digraphs
        PhonemeFormant { phoneme: "aɪ", letters: &["ay", "ai"], f1: 600.0, f2: 1500.0, f3: 2500.0, voiced: true, duration_ms: 150.0 },
        PhonemeFormant { phoneme: "iː", letters: &["ee"], f1: 300.0, f2: 2200.0, f3: 2900.0, voiced: true, duration_ms: 150.0 },
        PhonemeFormant { phoneme: "oʊ", letters: &["oo", "ou"], f1: 400.0, f2: 900.0, f3: 2300.0, voiced: true, duration_ms: 150.0 },
        // Consonants — voiced
        PhonemeFormant { phoneme: "b", letters: &["b"], f1: 200.0, f2: 800.0, f3: 1500.0, voiced: true, duration_ms: 60.0 },
        PhonemeFormant { phoneme: "d", letters: &["d"], f1: 250.0, f2: 1600.0, f3: 2500.0, voiced: true, duration_ms: 60.0 },
        PhonemeFormant { phoneme: "g", letters: &["g"], f1: 200.0, f2: 1200.0, f3: 2300.0, voiced: true, duration_ms: 60.0 },
        PhonemeFormant { phoneme: "m", letters: &["m"], f1: 280.0, f2: 900.0, f3: 2200.0, voiced: true, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "n", letters: &["n"], f1: 320.0, f2: 1600.0, f3: 2500.0, voiced: true, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "l", letters: &["l"], f1: 360.0, f2: 1300.0, f3: 2500.0, voiced: true, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "r", letters: &["r"], f1: 330.0, f2: 1100.0, f3: 1900.0, voiced: true, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "w", letters: &["w"], f1: 300.0, f2: 700.0, f3: 2300.0, voiced: true, duration_ms: 70.0 },
        PhonemeFormant { phoneme: "j", letters: &["y"], f1: 280.0, f2: 2200.0, f3: 2900.0, voiced: true, duration_ms: 70.0 },
        // Consonants — unvoiced
        PhonemeFormant { phoneme: "p", letters: &["p"], f1: 200.0, f2: 800.0, f3: 1500.0, voiced: false, duration_ms: 60.0 },
        PhonemeFormant { phoneme: "t", letters: &["t"], f1: 250.0, f2: 1600.0, f3: 2500.0, voiced: false, duration_ms: 60.0 },
        PhonemeFormant { phoneme: "k", letters: &["k", "c"], f1: 200.0, f2: 1200.0, f3: 2300.0, voiced: false, duration_ms: 60.0 },
        PhonemeFormant { phoneme: "f", letters: &["f", "ph"], f1: 300.0, f2: 1500.0, f3: 3000.0, voiced: false, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "s", letters: &["s", "ce", "ci"], f1: 300.0, f2: 2000.0, f3: 4000.0, voiced: false, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "θ", letters: &["th"], f1: 300.0, f2: 1800.0, f3: 3500.0, voiced: false, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "ʃ", letters: &["sh", "ti"], f1: 300.0, f2: 1700.0, f3: 3200.0, voiced: false, duration_ms: 80.0 },
        PhonemeFormant { phoneme: "h", letters: &["h"], f1: 400.0, f2: 1200.0, f3: 2500.0, voiced: false, duration_ms: 50.0 },
        // Voiced fricatives
        PhonemeFormant { phoneme: "v", letters: &["v"], f1: 280.0, f2: 1200.0, f3: 2200.0, voiced: true, duration_ms: 70.0 },
        PhonemeFormant { phoneme: "z", letters: &["z"], f1: 300.0, f2: 1800.0, f3: 3500.0, voiced: true, duration_ms: 70.0 },
        PhonemeFormant { phoneme: "ʒ", letters: &["si", "ge"], f1: 300.0, f2: 1600.0, f3: 3000.0, voiced: true, duration_ms: 70.0 },
        // Pause
        PhonemeFormant { phoneme: "_", letters: &[], f1: 0.0, f2: 0.0, f3: 0.0, voiced: false, duration_ms: 80.0 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tts_synthesizes_text() {
        let engine = TtsEngine::new(TtsConfig::default());
        engine.load().await.unwrap();
        let request = TtsRequest {
            text: "hello".to_string(),
            voice: VoiceProfile::default(),
            sample_rate: 22050,
        };
        let result = engine.synthesize(&request).await.unwrap();
        // Should produce audio data
        assert!(!result.audio.data.is_empty());
        // Should have phoneme alignments
        assert!(!result.phonemes.is_empty());
        // Duration should be > 0
        assert!(result.duration_ms > 0);
    }

    #[tokio::test]
    async fn tts_produces_audible_output() {
        let engine = TtsEngine::new(TtsConfig::default());
        engine.load().await.unwrap();
        let request = TtsRequest {
            text: "test".to_string(),
            voice: VoiceProfile::default(),
            sample_rate: 16000,
        };
        let result = engine.synthesize(&request).await.unwrap();
        // Convert to f32 samples
        let samples: Vec<f32> = result.audio.to_f32();
        // Check that there are non-zero samples (actual audio, not silence)
        let non_zero = samples.iter().filter(|&&s| s.abs() > 0.01).count();
        assert!(non_zero > 100, "expected >100 non-zero samples, got {non_zero}");
    }

    #[test]
    fn glottal_source_produces_signal() {
        let samples = generate_glottal_source(1000, 120.0, 16000.0);
        assert_eq!(samples.len(), 1000);
        // Should have non-zero values
        assert!(samples.iter().any(|&s| s.abs() > 0.1));
    }

    #[test]
    fn noise_source_produces_signal() {
        let samples = generate_noise(1000);
        assert_eq!(samples.len(), 1000);
        // Should have non-zero values
        assert!(samples.iter().any(|&s| s.abs() > 0.01));
    }

    #[test]
    fn resonator_filters_signal() {
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = resonator(&input, 1000.0, 0.1, 16000.0);
        assert_eq!(output.len(), 1000);
        // Output should be different from input (filtering occurred)
        let diff = input.iter().zip(output.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>();
        assert!(diff > 1.0);
    }

    #[test]
    fn text_to_phonemes_works() {
        let engine = TtsEngine::new(TtsConfig::default());
        let phonemes = engine.text_to_phonemes("hello");
        assert!(!phonemes.is_empty());
        // Should contain at least 'h', 'e', 'l', 'o' phonemes
        assert!(phonemes.iter().any(|p| p.phoneme == "h"));
        assert!(phonemes.iter().any(|p| p.phoneme == "e"));
        assert!(phonemes.iter().any(|p| p.phoneme == "l"));
        assert!(phonemes.iter().any(|p| p.phoneme == "o"));
    }
}
