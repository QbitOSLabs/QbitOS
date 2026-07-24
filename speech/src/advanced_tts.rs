//! Advanced text-to-speech with prosody control, SSML-like markup, and
//! multiple voice profiles.
//!
//! Features:
//! 1. **SSML parsing** — parse a subset of SSML tags (`<prosody>`, `<emphasis>`,
//!    `<break>`, `<say-as>`, `<phoneme>`).
//! 2. **Prosody control** — adjust pitch, rate, volume, and emphasis.
//! 3. **Multiple voice profiles** — different formant frequencies for male,
//!    female, child, and robotic voices.
//! 4. **Phoneme-level control** — specify exact formant frequencies per phoneme.

use serde::{Deserialize, Serialize};

/// A voice profile defining formant frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    /// Voice name.
    pub name: String,
    /// Base pitch (F0) in Hz.
    pub base_pitch: f32,
    /// Pitch range in Hz (how much pitch varies).
    pub pitch_range: f32,
    /// First formant (F1) base frequency in Hz.
    pub f1_base: f32,
    /// Second formant (F2) base frequency in Hz.
    pub f2_base: f32,
    /// Third formant (F3) base frequency in Hz.
    pub f3_base: f32,
    /// Default speaking rate in words per minute.
    pub wpm: u32,
    /// Breathiness (0.0 = none, 1.0 = very breathy).
    pub breathiness: f32,
    /// Vocal effort (0.0 = soft, 1.0 = loud).
    pub effort: f32,
}

impl VoiceProfile {
    /// Adult male voice (low pitch, low F1/F2).
    #[must_use]
    pub fn male() -> Self {
        Self {
            name: "male".into(),
            base_pitch: 120.0,
            pitch_range: 40.0,
            f1_base: 500.0,
            f2_base: 1500.0,
            f3_base: 2500.0,
            wpm: 150,
            breathiness: 0.1,
            effort: 0.5,
        }
    }

    /// Adult female voice (higher pitch, higher F1/F2).
    #[must_use]
    pub fn female() -> Self {
        Self {
            name: "female".into(),
            base_pitch: 220.0,
            pitch_range: 60.0,
            f1_base: 600.0,
            f2_base: 1800.0,
            f3_base: 2800.0,
            wpm: 160,
            breathiness: 0.15,
            effort: 0.5,
        }
    }

    /// Child voice (very high pitch).
    #[must_use]
    pub fn child() -> Self {
        Self {
            name: "child".into(),
            base_pitch: 300.0,
            pitch_range: 80.0,
            f1_base: 700.0,
            f2_base: 2000.0,
            f3_base: 3000.0,
            wpm: 140,
            breathiness: 0.2,
            effort: 0.6,
        }
    }

    /// Robotic voice (monotone, low formants).
    #[must_use]
    pub fn robotic() -> Self {
        Self {
            name: "robotic".into(),
            base_pitch: 100.0,
            pitch_range: 5.0,
            f1_base: 400.0,
            f2_base: 1200.0,
            f3_base: 2200.0,
            wpm: 130,
            breathiness: 0.0,
            effort: 0.8,
        }
    }

    /// Get a voice profile by name.
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "male" => Some(Self::male()),
            "female" => Some(Self::female()),
            "child" => Some(Self::child()),
            "robotic" | "robot" => Some(Self::robotic()),
            _ => None,
        }
    }
}

/// Prosody parameters that modify voice characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prosody {
    /// Pitch multiplier (1.0 = normal, 2.0 = one octave up).
    pub pitch: f32,
    /// Rate multiplier (1.0 = normal, 2.0 = twice as fast).
    pub rate: f32,
    /// Volume in [0.0, 1.0].
    pub volume: f32,
    /// Emphasis level.
    pub emphasis: Emphasis,
}

impl Default for Prosody {
    fn default() -> Self {
        Self {
            pitch: 1.0,
            rate: 1.0,
            volume: 1.0,
            emphasis: Emphasis::Normal,
        }
    }
}

/// Emphasis level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Emphasis {
    /// Reduced emphasis (quieter, lower pitch).
    Reduced,
    /// Normal emphasis.
    Normal,
    /// Moderate emphasis.
    Moderate,
    /// Strong emphasis (louder, higher pitch).
    Strong,
}

/// A parsed SSML document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsmlDocument {
    /// The sequence of SSML elements.
    pub elements: Vec<SsmlElement>,
}

/// An element in an SSML document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SsmlElement {
    /// Plain text.
    Text {
        /// The text content.
        content: String,
    },
    /// A prosody modifier.
    Prosody {
        /// Pitch multiplier.
        pitch: Option<f32>,
        /// Rate multiplier.
        rate: Option<f32>,
        /// Volume.
        volume: Option<f32>,
        /// Nested content.
        content: Vec<SsmlElement>,
    },
    /// An emphasis element.
    Emphasis {
        /// Emphasis level.
        level: Emphasis,
        /// Nested content.
        content: Vec<SsmlElement>,
    },
    /// A break (pause).
    Break {
        /// Duration in milliseconds.
        duration_ms: u32,
    },
    /// A say-as element (changes how text is interpreted).
    SayAs {
        /// Interpretation hint (e.g., "date", "number", "spell-out").
        interpret_as: String,
        /// The text.
        content: String,
    },
}

/// Parse SSML markup into a document.
///
/// Supports a subset of SSML:
/// - `<prosody pitch="1.5" rate="0.8" volume="0.9">text</prosody>`
/// - `<emphasis level="strong">text</emphasis>`
/// - `<break time="500ms"/>`
/// - `<say-as interpret-as="spell-out">ABC</say-as>`
#[must_use]
pub fn parse_ssml(input: &str) -> SsmlDocument {
    let mut elements = Vec::new();
    let mut remaining = input;
    while !remaining.is_empty() {
        if let Some(start) = remaining.find('<') {
            // Text before the tag.
            if start > 0 {
                let text = &remaining[..start];
                if !text.trim().is_empty() {
                    elements.push(SsmlElement::Text {
                        content: text.trim().to_string(),
                    });
                }
            }
            // Find the end of the tag.
            if let Some(end) = remaining[start..].find('>') {
                let tag = &remaining[start..=start + end];
                remaining = &remaining[start + end + 1..];
                if tag.starts_with("</") {
                    // Closing tag — we handle nesting by returning from recursion.
                    break;
                }
                if tag.ends_with("/>") {
                    // Self-closing tag.
                    if tag.starts_with("<break") {
                        let duration = extract_attr(tag, "time")
                            .and_then(|s| parse_duration(&s))
                            .unwrap_or(500);
                        elements.push(SsmlElement::Break {
                            duration_ms: duration,
                        });
                    }
                } else if tag.starts_with("<prosody") {
                    let pitch = extract_attr(tag, "pitch").and_then(|s| s.parse().ok());
                    let rate = extract_attr(tag, "rate").and_then(|s| s.parse().ok());
                    let volume = extract_attr(tag, "volume").and_then(|s| s.parse().ok());
                    let inner = parse_ssml(remaining);
                    remaining = skip_to_closing_tag(remaining, "prosody");
                    elements.push(SsmlElement::Prosody {
                        pitch,
                        rate,
                        volume,
                        content: inner.elements,
                    });
                } else if tag.starts_with("<emphasis") {
                    let level = extract_attr(tag, "level")
                        .and_then(|s| match s.as_str() {
                            "reduced" => Some(Emphasis::Reduced),
                            "moderate" => Some(Emphasis::Moderate),
                            "strong" => Some(Emphasis::Strong),
                            _ => Some(Emphasis::Normal),
                        })
                        .unwrap_or(Emphasis::Normal);
                    let inner = parse_ssml(remaining);
                    remaining = skip_to_closing_tag(remaining, "emphasis");
                    elements.push(SsmlElement::Emphasis {
                        level,
                        content: inner.elements,
                    });
                } else if tag.starts_with("<say-as") {
                    let interpret_as = extract_attr(tag, "interpret-as").unwrap_or_default();
                    if let Some(close) = remaining.find("</say-as>") {
                        let content = &remaining[..close];
                        remaining = &remaining[close + "</say-as>".len()..];
                        elements.push(SsmlElement::SayAs {
                            interpret_as,
                            content: content.trim().to_string(),
                        });
                    }
                }
            } else {
                // No closing '>', treat as text.
                elements.push(SsmlElement::Text {
                    content: remaining.to_string(),
                });
                break;
            }
        } else {
            // No more tags.
            if !remaining.trim().is_empty() {
                elements.push(SsmlElement::Text {
                    content: remaining.trim().to_string(),
                });
            }
            break;
        }
    }
    SsmlDocument { elements }
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')?;
    Some(tag[start..start + end].to_string())
}

fn parse_duration(s: &str) -> Option<u32> {
    if let Some(ms) = s.strip_suffix("ms") {
        ms.parse().ok()
    } else if let Some(s_val) = s.strip_suffix('s') {
        s_val.parse::<f32>().ok().map(|v| (v * 1000.0) as u32)
    } else {
        s.parse().ok()
    }
}

fn skip_to_closing_tag<'a>(input: &'a str, tag: &str) -> &'a str {
    let close = format!("</{tag}>");
    if let Some(idx) = input.find(&close) {
        &input[idx + close.len()..]
    } else {
        ""
    }
}

/// A synthesized speech sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSample {
    /// PCM audio samples (mono, 16-bit, 16kHz).
    pub samples: Vec<i16>,
    /// Sample rate.
    pub sample_rate: u32,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Voice profile used.
    pub voice: String,
    /// Average pitch.
    pub avg_pitch: f32,
}

/// Synthesize speech from SSML.
#[must_use]
pub fn synthesize_ssml(ssml: &SsmlDocument, voice: &VoiceProfile) -> SpeechSample {
    let sample_rate = 16000u32;
    let mut samples = Vec::new();
    let prosody = Prosody::default();
    for element in &ssml.elements {
        let element_samples = synthesize_element(element, voice, &prosody, sample_rate);
        samples.extend(element_samples);
    }
    let duration_ms = (samples.len() as f64 / sample_rate as f64 * 1000.0) as u64;
    SpeechSample {
        samples,
        sample_rate,
        duration_ms,
        voice: voice.name.clone(),
        avg_pitch: voice.base_pitch,
    }
}

fn synthesize_element(
    element: &SsmlElement,
    voice: &VoiceProfile,
    prosody: &Prosody,
    sample_rate: u32,
) -> Vec<i16> {
    match element {
        SsmlElement::Text { content } => {
            synthesize_text(content, voice, prosody, sample_rate)
        }
        SsmlElement::Prosody {
            pitch,
            rate,
            volume,
            content,
        } => {
            let mut p = prosody.clone();
            if let Some(pt) = pitch {
                p.pitch *= pt;
            }
            if let Some(rt) = rate {
                p.rate *= rt;
            }
            if let Some(vol) = volume {
                p.volume *= vol;
            }
            let mut samples = Vec::new();
            for child in content {
                samples.extend(synthesize_element(child, voice, &p, sample_rate));
            }
            samples
        }
        SsmlElement::Emphasis { level, content } => {
            let mut p = prosody.clone();
            p.emphasis = *level;
            match level {
                Emphasis::Reduced => {
                    p.volume *= 0.6;
                    p.pitch *= 0.9;
                }
                Emphasis::Moderate => {
                    p.volume *= 1.2;
                    p.pitch *= 1.1;
                }
                Emphasis::Strong => {
                    p.volume *= 1.5;
                    p.pitch *= 1.2;
                }
                Emphasis::Normal => {}
            }
            let mut samples = Vec::new();
            for child in content {
                samples.extend(synthesize_element(child, voice, &p, sample_rate));
            }
            samples
        }
        SsmlElement::Break { duration_ms } => {
            let num_samples = (*duration_ms as f64 * sample_rate as f64 / 1000.0) as usize;
            vec![0i16; num_samples]
        }
        SsmlElement::SayAs {
            interpret_as,
            content,
        } => match interpret_as.as_str() {
            "spell-out" => {
                let mut samples = Vec::new();
                for ch in content.chars() {
                    samples.extend(synthesize_text(&ch.to_string(), voice, prosody, sample_rate));
                    // Add a short pause between letters.
                    samples.extend(vec![0i16; sample_rate as usize / 20]);
                }
                samples
            }
            _ => synthesize_text(content, voice, prosody, sample_rate),
        },
    }
}

fn synthesize_text(text: &str, voice: &VoiceProfile, prosody: &Prosody, sample_rate: u32) -> Vec<i16> {
    // Simple formant synthesis: generate a glottal pulse train at the base pitch
    // and apply formant filters.
    let pitch = voice.base_pitch * prosody.pitch;
    let rate_multiplier = if prosody.rate > 0.0 { prosody.rate } else { 0.5 };
    let effective_wpm = (voice.wpm as f32 / rate_multiplier) as u32;
    let phoneme_duration = 60_000.0 / effective_wpm as f32 / 5.0; // ~5 phonemes per word
    let samples_per_phoneme = (sample_rate as f32 * phoneme_duration / 1000.0) as usize;
    let mut samples = Vec::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            // Short pause for spaces.
            samples.extend(vec![0i16; samples_per_phoneme / 4]);
            continue;
        }
        // Map character to formant frequencies based on phoneme lookup.
        let (f1, f2, f3) = char_to_formants(ch, voice);
        let period = sample_rate as f32 / pitch;
        for i in 0..samples_per_phoneme {
            let t = i as f32 / sample_rate as f32;
            // Glottal pulse (sawtooth approximation).
            let phase = (i as f32 % period) / period;
            let glottal = if phase < 0.1 { phase * 10.0 } else { 0.0 };
            // Formant resonances (sine waves at formant frequencies).
            let s1 = (2.0 * std::f32::consts::PI * f1 * t).sin() * 0.5;
            let s2 = (2.0 * std::f32::consts::PI * f2 * t).sin() * 0.3;
            let s3 = (2.0 * std::f32::consts::PI * f3 * t).sin() * 0.2;
            // Envelope (attack-sustain-release).
            let envelope = {
                let attack = (i as f32 / samples_per_phoneme as f32).min(0.1) * 10.0;
                let release = (1.0 - i as f32 / samples_per_phoneme as f32).min(0.1) * 10.0;
                attack.min(release).min(1.0)
            };
            let sample = (glottal * 0.4 + s1 * 0.3 + s2 * 0.2 + s3 * 0.1) * envelope * prosody.volume;
            samples.push((sample * 16000.0).clamp(-32768.0, 32767.0) as i16);
        }
    }
    samples
}

fn char_to_formants(ch: char, voice: &VoiceProfile) -> (f32, f32, f32) {
    let c = ch.to_ascii_lowercase();
    match c {
        'a' => (voice.f1_base + 100.0, voice.f2_base - 200.0, voice.f3_base),
        'e' => (voice.f1_base - 50.0, voice.f2_base + 200.0, voice.f3_base),
        'i' => (voice.f1_base - 150.0, voice.f2_base + 500.0, voice.f3_base),
        'o' => (voice.f1_base + 50.0, voice.f2_base - 400.0, voice.f3_base - 100.0),
        'u' => (voice.f1_base - 100.0, voice.f2_base - 600.0, voice.f3_base - 200.0),
        'b' | 'p' | 'm' => (voice.f1_base - 200.0, voice.f2_base - 300.0, voice.f3_base),
        'd' | 't' | 'n' => (voice.f1_base - 100.0, voice.f2_base + 100.0, voice.f3_base),
        'g' | 'k' => (voice.f1_base, voice.f2_base - 200.0, voice.f3_base),
        's' | 'z' => (voice.f1_base + 200.0, voice.f2_base + 800.0, voice.f3_base + 500.0),
        'f' | 'v' => (voice.f1_base, voice.f2_base + 400.0, voice.f3_base + 200.0),
        'l' | 'r' => (voice.f1_base + 50.0, voice.f2_base - 100.0, voice.f3_base),
        'w' | 'y' => (voice.f1_base - 50.0, voice.f2_base - 300.0, voice.f3_base - 100.0),
        _ => (voice.f1_base, voice.f2_base, voice.f3_base),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_profiles_exist() {
        assert_eq!(VoiceProfile::male().name, "male");
        assert_eq!(VoiceProfile::female().name, "female");
        assert_eq!(VoiceProfile::child().name, "child");
        assert_eq!(VoiceProfile::robotic().name, "robotic");
    }

    #[test]
    fn voice_by_name_works() {
        assert!(VoiceProfile::by_name("male").is_some());
        assert!(VoiceProfile::by_name("female").is_some());
        assert!(VoiceProfile::by_name("child").is_some());
        assert!(VoiceProfile::by_name("robot").is_some());
        assert!(VoiceProfile::by_name("unknown").is_none());
    }

    #[test]
    fn parse_ssml_text() {
        let doc = parse_ssml("Hello, World!");
        assert_eq!(doc.elements.len(), 1);
        match &doc.elements[0] {
            SsmlElement::Text { content } => assert_eq!(content, "Hello, World!"),
            _ => panic!("expected text element"),
        }
    }

    #[test]
    fn parse_ssml_break() {
        let doc = parse_ssml("Hello<break time=\"500ms\"/>World");
        assert_eq!(doc.elements.len(), 3);
        match &doc.elements[1] {
            SsmlElement::Break { duration_ms } => assert_eq!(*duration_ms, 500),
            _ => panic!("expected break element"),
        }
    }

    #[test]
    fn parse_ssml_emphasis() {
        let doc = parse_ssml("<emphasis level=\"strong\">important</emphasis>");
        assert_eq!(doc.elements.len(), 1);
        match &doc.elements[0] {
            SsmlElement::Emphasis { level, content } => {
                assert_eq!(*level, Emphasis::Strong);
                assert_eq!(content.len(), 1);
            }
            _ => panic!("expected emphasis element"),
        }
    }

    #[test]
    fn parse_ssml_prosody() {
        let doc = parse_ssml("<prosody pitch=\"1.5\" rate=\"0.8\">text</prosody>");
        assert_eq!(doc.elements.len(), 1);
        match &doc.elements[0] {
            SsmlElement::Prosody {
                pitch,
                rate,
                volume,
                content,
            } => {
                assert_eq!(*pitch, Some(1.5));
                assert_eq!(*rate, Some(0.8));
                assert_eq!(*volume, None);
                assert_eq!(content.len(), 1);
            }
            _ => panic!("expected prosody element"),
        }
    }

    #[test]
    fn parse_ssml_say_as_spell_out() {
        let doc = parse_ssml("<say-as interpret-as=\"spell-out\">ABC</say-as>");
        assert_eq!(doc.elements.len(), 1);
        match &doc.elements[0] {
            SsmlElement::SayAs {
                interpret_as,
                content,
            } => {
                assert_eq!(interpret_as, "spell-out");
                assert_eq!(content, "ABC");
            }
            _ => panic!("expected say-as element"),
        }
    }

    #[test]
    fn synthesize_ssml_produces_audio() {
        let doc = parse_ssml("Hello");
        let voice = VoiceProfile::male();
        let sample = synthesize_ssml(&doc, &voice);
        assert!(!sample.samples.is_empty());
        assert_eq!(sample.sample_rate, 16000);
        assert_eq!(sample.voice, "male");
    }

    #[test]
    fn synthesize_with_break_produces_silence() {
        let doc = parse_ssml("Hi<break time=\"100ms\"/>There");
        let voice = VoiceProfile::female();
        let sample = synthesize_ssml(&doc, &voice);
        assert!(sample.duration_ms >= 100);
    }

    #[test]
    fn prosody_default_is_normal() {
        let p = Prosody::default();
        assert_eq!(p.pitch, 1.0);
        assert_eq!(p.rate, 1.0);
        assert_eq!(p.volume, 1.0);
        assert_eq!(p.emphasis, Emphasis::Normal);
    }

    #[test]
    fn parse_duration_handles_units() {
        assert_eq!(parse_duration("500ms"), Some(500));
        assert_eq!(parse_duration("1.5s"), Some(1500));
        assert_eq!(parse_duration("1000"), Some(1000));
    }
}
