import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Mic, Square, Volume2, AudioWaveform, Play, Pause, Loader2 } from "lucide-react";

export default function VoiceView() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [synthesizing, setSynthesizing] = useState(false);
  const [ttsText, setTtsText] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  async function startRecording() {
    setTranscript("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const reader = new FileReader();
        reader.onload = async () => {
          const dataUrl = reader.result as string;
          try {
            const result = await invoke<string>("transcribe_audio", { audio: dataUrl });
            setTranscript(result);
          } catch {
            setTranscript("Error: STT backend unavailable. Ensure QbitOS is running with speech modules loaded.");
          } finally {
            setRecording(false);
          }
        };
        reader.readAsDataURL(blob);
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch {
      setTranscript("Error: Microphone access denied or unavailable.");
      setRecording(false);
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
  }

  async function synthesize() {
    if (!ttsText.trim()) return;
    setSynthesizing(true);
    try {
      const result = await invoke<number[]>("synthesize_speech", { text: ttsText });
      const bytes = new Uint8Array(result);
      const blob = new Blob([bytes], { type: "audio/wav" });
      setAudioUrl(URL.createObjectURL(blob));
    } catch {
      setAudioUrl(null);
    } finally {
      setSynthesizing(false);
    }
  }

  function togglePlay() {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setPlaying(!playing);
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Mic className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Voice I/O</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">Speech-to-text and text-to-speech</p>
        </div>
      </div>

      {/* STT Section */}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <AudioWaveform className="w-4 h-4 text-[var(--color-info)]" />
          Speech to Text
        </h3>
        <div className="flex flex-col items-center gap-4 py-4">
          <button
            onClick={recording ? stopRecording : startRecording}
            disabled={false}
            className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
              recording
                ? "bg-[var(--color-danger)] animate-pulse-glow"
                : "gradient-bg hover:shadow-[var(--shadow-glow)]"
            }`}
          >
            {recording ? <Square className="w-7 h-7 text-white" /> : <Mic className="w-8 h-8 text-white" />}
          </button>
          <p className="text-xs text-[var(--color-fg-dim)]">
            {recording ? "Recording... Click stop to transcribe" : "Click to start recording"}
          </p>
        </div>
        {transcript && (
          <div className="mt-4 p-4 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
            <p className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider mb-2">Transcript</p>
            <p className="text-sm leading-relaxed">{transcript}</p>
          </div>
        )}
      </div>

      {/* TTS Section */}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
          <Volume2 className="w-4 h-4 text-[var(--color-success)]" />
          Text to Speech
        </h3>
        <textarea
          value={ttsText}
          onChange={(e) => setTtsText(e.target.value)}
          placeholder="Enter text to synthesize..."
          rows={3}
          className="w-full rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-4 py-3 text-sm focus:outline-none focus:border-[var(--color-accent)] transition-all resize-none"
        />
        <button
          onClick={synthesize}
          disabled={!ttsText.trim() || synthesizing}
          className="mt-3 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[var(--shadow-glow)] transition-all"
        >
          {synthesizing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Synthesizing...
          </>
        ) : (
          "Synthesize"
        )}
        </button>
        {audioUrl && (
          <div className="mt-4 flex items-center gap-3 p-3 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
            <button
              onClick={togglePlay}
              className="w-10 h-10 rounded-lg bg-[var(--color-success)] flex items-center justify-center hover:opacity-80 transition-opacity"
            >
              {playing ? <Pause className="w-4 h-4 text-white" /> : <Play className="w-4 h-4 text-white" />}
            </button>
            <div className="flex-1">
              <p className="text-xs text-[var(--color-fg-muted)]">Synthesized audio</p>
              <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setPlaying(false)}
                className="hidden"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
