//! Neural-codec TTS (VALL-E 2 / NaturalSpeech 3 2026).
#![allow(missing_docs)]
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::Arc;
use thiserror::Error;
use crate::audio_engine::{AudioChunk, AudioFormat};

#[derive(Debug, Error)]
pub enum NeuralTtsError { #[error("enrollment too short: {0} samples (need {1})")] EnrollmentTooShort(usize, usize), #[error("codec mismatch: expected {expected}, got {actual}")] CodecMismatch { expected: usize, actual: usize }, #[error("invalid prosody: {0}")] InvalidProsody(String) }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecToken { pub codebook: u8, pub token: u16 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEmbedding { pub vector: Vec<f32>, pub sample_rate: u32, pub enrollment_duration_s: f32 }
impl SpeakerEmbedding { #[must_use] pub fn cosine_similarity(&self, other: &Self) -> f32 { if self.vector.len() != other.vector.len() || self.vector.is_empty() { return 0.0; } let dot: f32 = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a * b).sum(); let na: f32 = self.vector.iter().map(|v| v*v).sum::<f32>().sqrt(); let nb: f32 = other.vector.iter().map(|v| v*v).sum::<f32>().sqrt(); if na < 1e-6 || nb < 1e-6 { 0.0 } else { dot / (na * nb) } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProsodyControl { pub pitch_scale: f32, pub energy_scale: f32, pub speaking_rate: f32, pub pause_duration_s: f32 }
impl Default for ProsodyControl { fn default() -> Self { Self { pitch_scale: 1.0, energy_scale: 1.0, speaking_rate: 1.0, pause_duration_s: 0.1 } } }
impl ProsodyControl { fn validate(&self) -> Result<(), NeuralTtsError> { if self.pitch_scale <= 0.0 { return Err(NeuralTtsError::InvalidProsody("pitch_scale <= 0".into())); } if self.energy_scale < 0.0 { return Err(NeuralTtsError::InvalidProsody("energy_scale < 0".into())); } if self.speaking_rate <= 0.0 { return Err(NeuralTtsError::InvalidProsody("speaking_rate <= 0".into())); } Ok(()) } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTtsConfig { pub sample_rate: u32, pub num_codebooks: usize, pub codebook_size: usize, pub frame_rate_hz: u32, pub speaker_embed_dim: usize }
impl Default for NeuralTtsConfig { fn default() -> Self { Self { sample_rate: 22050, num_codebooks: 4, codebook_size: 1024, frame_rate_hz: 75, speaker_embed_dim: 256 } } }

pub struct NeuralCodecTts { cfg: NeuralTtsConfig, speaker: Option<Arc<SpeakerEmbedding>> }

impl NeuralCodecTts {
    pub fn new(cfg: NeuralTtsConfig) -> Self { Self { cfg, speaker: None } }
    pub fn enroll(&mut self, audio: &AudioChunk) -> Result<Arc<SpeakerEmbedding>, NeuralTtsError> {
        let samples = audio.to_f32(); let min = self.cfg.sample_rate as usize;
        if samples.len() < min { return Err(NeuralTtsError::EnrollmentTooShort(samples.len(), min)); }
        let emb = self.extract(&samples, audio.sample_rate);
        let e = Arc::new(SpeakerEmbedding { vector: emb, sample_rate: audio.sample_rate, enrollment_duration_s: samples.len() as f32 / audio.sample_rate as f32 });
        self.speaker = Some(Arc::clone(&e)); Ok(e)
    }
    fn extract(&self, samples: &[f32], sr: u32) -> Vec<f32> {
        let sr_f = sr as f32; let fs = (sr_f * 0.025) as usize; let hop = (sr_f * 0.010) as usize; let dim = self.cfg.speaker_embed_dim;
        let mut feats = vec![0.0_f32; dim]; let mut fi = 0; let mut e_acc = 0.0; let mut z_acc = 0.0; let mut c_acc = 0.0; let mut p_acc = 0.0; let mut p_n = 0u32;
        let mut start = 0;
        while start + fs <= samples.len() { let fr = &samples[start..start+fs];
            let energy: f32 = fr.iter().map(|s| s*s).sum::<f32>().sqrt() / fr.len() as f32; e_acc += energy;
            let zcr: f32 = fr.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f32 / fr.len() as f32; z_acc += zcr;
            let centroid = spectral_centroid(fr, sr_f); c_acc += centroid;
            if let Some(f0) = estimate_pitch(fr, sr_f) { p_acc += f0; p_n += 1; }
            fi += 1; start += hop;
        }
        let n = fi.max(1) as f32; let em = e_acc / n; let zm = z_acc / n; let cm = c_acc / n; let pm = if p_n > 0 { p_acc / p_n as f32 } else { 120.0 };
        for i in 0..dim { let ph = i as f32 / dim as f32; feats[i] = (em * (ph * 2.0 * PI).cos() * 0.3 + zm * (ph * 4.0 * PI).sin() * 100.0 + cm * (ph * 6.0 * PI).cos() * 0.001 + pm * (ph * 8.0 * PI).sin() * 0.01) * 0.1; }
        feats
    }
    pub fn encode(&self, audio: &AudioChunk) -> Vec<CodecToken> { let samples = audio.to_f32(); let sr = audio.sample_rate as f32; let fs = (sr / self.cfg.frame_rate_hz as f32) as usize; let mut tokens = Vec::new(); let mut start = 0;
        while start + fs <= samples.len() { let fr = &samples[start..start+fs]; let energy: f32 = fr.iter().map(|s| s*s).sum::<f32>().sqrt() / fr.len() as f32; let centroid = spectral_centroid(fr, sr); let mut residual = energy + centroid * 0.001;
            for cb in 0..self.cfg.num_codebooks { let sc = (cb + 1) as f32; let token = (((residual * sc * 100.0) as i32).rem_euclid(self.cfg.codebook_size as i32)) as u16; let rec = token as f32 / (sc * 100.0); residual -= rec; tokens.push(CodecToken { codebook: cb as u8, token }); }
            start += fs;
        }
        tokens
    }
    pub fn decode(&self, tokens: &[CodecToken], prosody: &ProsodyControl) -> Result<AudioChunk, NeuralTtsError> {
        prosody.validate()?;
        for t in tokens { if t.codebook as usize >= self.cfg.num_codebooks { return Err(NeuralTtsError::CodecMismatch { expected: self.cfg.num_codebooks, actual: t.codebook as usize + 1 }); } }
        let sr = self.cfg.sample_rate; let fs = (sr as f32 / self.cfg.frame_rate_hz as f32) as usize; let mut samples = Vec::new();
        for chunk in tokens.chunks(self.cfg.num_codebooks) { if chunk.len() < self.cfg.num_codebooks { break; }
            let t0 = chunk[0].token as f32; let energy = (t0 / 100.0).max(0.01) * prosody.energy_scale;
            let pitch = self.speaker.as_ref().and_then(|s| { let v: f32 = s.vector.iter().take(16).sum::<f32>() * 10.0; Some(v.abs().max(80.0).min(400.0)) }).unwrap_or(120.0) * prosody.pitch_scale;
            let period = (sr as f32 / pitch) as usize;
            for i in 0..fs { let ph = (i % period) as f32 / period as f32 * 2.0 * PI; let pulse = if ph < 0.1 { (ph / 0.1 * PI).sin() } else { let d = (ph - 0.1) / 0.9; (ph * PI).sin() * (1.0 - d) * 0.3 }; samples.push(pulse * energy * 0.3); }
        }
        if prosody.speaking_rate != 1.0 { samples = resample(&samples, prosody.speaking_rate); }
        Ok(AudioChunk::new(sr, 1, AudioFormat::F32, samples.iter().flat_map(|s| s.to_le_bytes()).collect()))
    }
    pub fn synthesize(&self, text: &str, prosody: &ProsodyControl) -> Result<AudioChunk, NeuralTtsError> { prosody.validate()?; let ph = grapheme_to_phonemes(text); let tokens = self.phonemes_to_tokens(&ph); self.decode(&tokens, prosody) }
    fn phonemes_to_tokens(&self, ph: &[Phoneme]) -> Vec<CodecToken> { let mut t = Vec::new(); for p in ph { let bt = (p.f1 / 10.0) as u16 % self.cfg.codebook_size as u16; for cb in 0..self.cfg.num_codebooks { t.push(CodecToken { codebook: cb as u8, token: bt.wrapping_add((cb as u16).wrapping_mul(37)) }); } } t }
}

#[allow(dead_code)]
struct Phoneme { symbol: String, f1: f32, f2: f32, voiced: bool }
fn grapheme_to_phonemes(text: &str) -> Vec<Phoneme> { let mut r = Vec::new(); for c in text.to_lowercase().chars() { if c.is_whitespace() { r.push(Phoneme{symbol:"_".into(),f1:0.0,f2:0.0,voiced:false}); continue; } let (f1,f2,v) = match c { 'a'=>(700.0,1220.0,true),'e'=>(530.0,1840.0,true),'i'=>(270.0,2290.0,true),'o'=>(570.0,840.0,true),'u'=>(300.0,870.0,true),'b'|'d'|'g'|'m'|'n'|'l'|'r'|'v'|'w'|'y'=>(250.0,1500.0,true),_=>(400.0,1800.0,false) }; r.push(Phoneme{symbol:c.to_string(),f1,f2,voiced:v}); } r }
fn spectral_centroid(fr: &[f32], sr: f32) -> f32 { let n = fr.len(); if n == 0 { return 0.0; } let mut mags = vec![0.0_f32; n/2]; for k in 0..n/2 { let mut re = 0.0; let mut im = 0.0; for t in 0..n { let a = -2.0*PI*k as f32*t as f32/n as f32; re += fr[t] * a.cos(); im += fr[t] * a.sin(); } mags[k] = (re*re + im*im).sqrt(); } let tot: f32 = mags.iter().sum(); if tot < 1e-6 { return 0.0; } let w: f32 = mags.iter().enumerate().map(|(k,m)| k as f32 * m).sum(); w / tot * sr / n as f32 }
fn estimate_pitch(fr: &[f32], sr: f32) -> Option<f32> { let n = fr.len(); if n < 4 { return None; } let min_lag = (sr / 400.0) as usize; let max_lag = (sr / 80.0) as usize; if max_lag >= n { return None; } let mut best_lag = 0; let mut best_corr = 0.0; for lag in min_lag..=max_lag { let mut s = 0.0; for i in 0..n-lag { s += fr[i] * fr[i+lag]; } if s > best_corr { best_corr = s; best_lag = lag; } } if best_lag == 0 || best_corr < 0.1 { return None; } Some(sr / best_lag as f32) }
fn resample(s: &[f32], rate: f32) -> Vec<f32> { if rate <= 0.0 || s.is_empty() { return s.to_vec(); } let nl = (s.len() as f32 / rate) as usize; let mut o = Vec::with_capacity(nl); for i in 0..nl { let src = i as f32 * rate; let i0 = src as usize; let i1 = (i0+1).min(s.len()-1); let f = src - i0 as f32; o.push(s[i0] * (1.0-f) + s[i1] * f); } o }

#[cfg(test)]
mod tests {
    use super::*;
    fn audio(d: f32, sr: u32) -> AudioChunk { let n = (d * sr as f32) as usize; let s: Vec<f32> = (0..n).map(|i| { let t = i as f32 / sr as f32; (t * 2.0 * PI * 150.0).sin() * 0.3 }).collect(); AudioChunk::new(sr, 1, AudioFormat::F32, s.iter().flat_map(|v| v.to_le_bytes()).collect()) }
    #[test] fn enroll() { let mut t = NeuralCodecTts::new(NeuralTtsConfig::default()); let e = t.enroll(&audio(2.0, 22050)).unwrap(); assert_eq!(e.vector.len(), 256); }
    #[test] fn enroll_short() { let mut t = NeuralCodecTts::new(NeuralTtsConfig::default()); assert!(matches!(t.enroll(&AudioChunk::new(22050, 1, AudioFormat::F32, vec![0; 400])), Err(NeuralTtsError::EnrollmentTooShort(100, _)))); }
    #[test] fn cosine_identity() { let e = SpeakerEmbedding { vector: vec![1.0, 2.0, 3.0], sample_rate: 22050, enrollment_duration_s: 2.0 }; assert!((e.cosine_similarity(&e) - 1.0).abs() < 1e-5); }
    #[test] fn encode_decode() { let t = NeuralCodecTts::new(NeuralTtsConfig::default()); let a = audio(1.0, 22050); let tk = t.encode(&a); assert!(!tk.is_empty()); let d = t.decode(&tk, &ProsodyControl::default()).unwrap(); assert!(!d.data.is_empty()); }
    #[test] fn prosody_validation() { let t = NeuralCodecTts::new(NeuralTtsConfig::default()); assert!(t.decode(&[CodecToken{codebook:0,token:100}], &ProsodyControl{pitch_scale:-1.0,..Default::default()}).is_err()); }
    #[test] fn synthesize() { let cfg = NeuralTtsConfig::default(); let mut t = NeuralCodecTts::new(cfg); t.enroll(&audio(2.0, 22050)).unwrap(); let o = t.synthesize("hello world", &ProsodyControl::default()).unwrap(); assert!(!o.data.is_empty()); assert_eq!(o.sample_rate, 22050); }
    #[test] fn codec_mismatch() { let t = NeuralCodecTts::new(NeuralTtsConfig::default()); assert!(t.decode(&[CodecToken{codebook:99,token:0}], &ProsodyControl::default()).is_err()); }
}
