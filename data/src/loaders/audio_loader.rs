//! Audio loader (requires `audio-loader` feature).
//!
//! Loads audio files using `symphonia`. Emits one record per file with
//! the raw bytes and decoded metadata (sample rate, channels, duration).

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::loader::{
    read_file_to_bytes, DatasetError, DatasetLoader, DatasetRecord, DatasetRecordStream, LoadOptions,
};
use crate::streaming::MemoryRecordStream;

/// Audio loader.
pub struct AudioLoader;

impl Default for AudioLoader {
    fn default() -> Self {
        Self
    }
}

impl AudioLoader {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DatasetLoader for AudioLoader {
    fn id(&self) -> &str { "audio" }
    fn name(&self) -> &str { "Audio (.wav, .mp3, .flac, .ogg, .aac)" }
    fn extensions(&self) -> &[&str] { &["wav", "mp3", "flac", "ogg", "aac", "m4a", "opus", "wma"] }

    async fn load_with_options(
        &self,
        path: &Path,
        _options: &LoadOptions,
    ) -> Result<Box<dyn DatasetRecordStream>, DatasetError> {
        let bytes = read_file_to_bytes(path).await?;
        let source = path.to_string_lossy().to_string();
        let mime = mime_guess::from_path(path).first_or_octet_stream().to_string();

        // Probe to extract metadata
        let metadata = probe_audio_metadata(&bytes);

        let mut record = DatasetRecord::binary(source, mime, bytes);
        record.modality = crate::loader::Modality::Audio;
        for (k, v) in metadata {
            record.metadata.insert(k, v);
        }
        Ok(Box::new(MemoryRecordStream::new(vec![record])))
    }
}

/// Probe an audio file's metadata using symphonia.
fn probe_audio_metadata(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
    let mut out = Vec::new();
    let cursor = std::io::Cursor::new(bytes);
    let media_source = Box::new(cursor);
    let probed = match symphonia::default::get_probe().format(
        &Default::default(),
        media_source,
        &Default::default(),
        &Default::default(),
    ) {
        Ok(p) => p,
        Err(_) => return out,
    };
    let format = probed.format;
    let track = format.tracks().first().cloned();
    if let Some(track) = track {
        if let Some(params) = track.codec_params {
            if let Some(rate) = params.sample_rate {
                out.push(("sample_rate".into(), serde_json::Value::Number(serde_json::Number::from(rate))));
            }
            if let Some(channels) = params.channels {
                out.push(("channels".into(), serde_json::Value::Number(serde_json::Number::from(channels.count()))));
            }
            if let Some(n_frames) = params.n_frames {
                if let Some(rate) = params.sample_rate {
                    let duration_secs = n_frames as f64 / rate as f64;
                    out.push(("duration_secs".into(), serde_json::Number::from_f64(duration_secs).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)));
                }
            }
            out.push(("codec".into(), serde_json::Value::String(format!("{:?}", params.codec))));
        }
    }
    let _ = Arc::new(format); // keep alive
    out
}
