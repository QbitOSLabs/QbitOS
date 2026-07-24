# qbitos-speech

Speech-to-text (FFT+MFCC+VAD) and text-to-speech (formant synthesis).

> Part of the [QbitOS](../README.md) project · [Architecture](../docs/ARCHITECTURE.md) · [API](../docs/API.md) · [Deployment](../docs/DEPLOYMENT.md)

## Overview

A from-scratch speech pipeline.

## STT (Speech-to-Text)

- FFT (Fast Fourier Transform) for frequency analysis
- MFCC (Mel-Frequency Cepstral Coefficients) feature extraction
- VAD (Voice Activity Detection) to segment speech
- Template matching for recognition

## TTS (Text-to-Speech)

- Formant synthesis (models vocal tract resonances)
- Multiple voices (configurable)
- Adjustable speed (WPM)

## Usage

```rust
use qbitos_speech::SpeechModule;
let speech = SpeechModule::new(memory.embedder.clone());
let text = speech.stt(&audio_bytes, 16000).await?;
let audio = speech.tts("hello", "default", 150).await?;
```
