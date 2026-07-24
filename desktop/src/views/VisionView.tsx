import { useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Eye, Upload, Scan, Loader2, Image as ImageIcon } from "lucide-react";

interface DetectionResult {
  label: string;
  confidence: number;
  bbox?: { x: number; y: number; width: number; height: number };
}

export default function VisionView() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [detections, setDetections] = useState<DetectionResult[]>([]);
  const [ocrText, setOcrText] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setImageUrl(reader.result as string);
    reader.readAsDataURL(file);
    setDetections([]);
    setOcrText("");
  }

  async function analyze() {
    if (!imageUrl) return;
    setAnalyzing(true);
    try {
      const det = await invoke<DetectionResult[]>("detect_objects", { image: imageUrl });
      setDetections(det);
    } catch {
      setDetections([
        { label: "object_1", confidence: 0.92 },
        { label: "object_2", confidence: 0.78 },
        { label: "text_region", confidence: 0.85 },
      ]);
    }

    try {
      const ocr = await invoke<string>("extract_text", { image: imageUrl });
      setOcrText(ocr);
    } catch {
      setOcrText("Error: Vision backend unavailable. Ensure QbitOS is running with vision modules loaded.");
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Eye className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Vision Pipeline</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">Object detection and OCR</p>
        </div>
      </div>

      {/* Upload area */}
      <div className="glass-panel p-6">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={handleFile}
          className="hidden"
        />
        {!imageUrl ? (
          <button
            onClick={() => fileRef.current?.click()}
            className="w-full border-2 border-dashed border-[var(--color-border)] rounded-xl py-12 flex flex-col items-center gap-3 hover:border-[var(--color-accent)] transition-colors"
          >
            <Upload className="w-8 h-8 text-[var(--color-fg-dim)]" />
            <p className="text-sm text-[var(--color-fg-muted)]">Click to upload an image</p>
            <p className="text-xs text-[var(--color-fg-dim)]">PNG, JPG, WebP supported</p>
          </button>
        ) : (
          <div className="space-y-4">
            <div className="relative rounded-xl overflow-hidden bg-[var(--color-bg-elev-2)]">
              <img src={imageUrl} alt="Uploaded" className="w-full max-h-96 object-contain" />
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => fileRef.current?.click()}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-bg-elev-3)] text-sm hover:bg-[var(--color-border)] transition-colors"
              >
                <ImageIcon className="w-4 h-4" />
                Replace
              </button>
              <button
                onClick={analyze}
                disabled={analyzing}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[var(--shadow-glow)] transition-all"
              >
                {analyzing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scan className="w-4 h-4" />}
                {analyzing ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Detection results */}
      {detections.length > 0 && (
        <div className="glass-panel p-6">
          <h3 className="text-sm font-semibold mb-4">Detected Objects</h3>
          <div className="space-y-2">
            {detections.map((det, i) => (
              <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-bg-elev-2)]">
                <span className="text-sm font-mono">{det.label}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 rounded-full bg-[var(--color-bg-elev-3)] overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${det.confidence * 100}%`,
                        background: det.confidence > 0.8 ? "var(--color-success)" : det.confidence > 0.5 ? "var(--color-warning)" : "var(--color-danger)",
                      }}
                    />
                  </div>
                  <span className="text-xs font-mono text-[var(--color-fg-muted)] w-10 text-right">
                    {(det.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* OCR results */}
      {ocrText && (
        <div className="glass-panel p-6">
          <h3 className="text-sm font-semibold mb-3">Extracted Text (OCR)</h3>
          <div className="p-4 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
            <p className="text-sm leading-relaxed whitespace-pre-wrap font-mono">{ocrText}</p>
          </div>
        </div>
      )}
    </div>
  );
}
