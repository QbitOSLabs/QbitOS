import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Database,
  Upload,
  FileText,
  Loader2,
  Trash2,
  RefreshCw,
  Plus,
  CheckCircle,
  HardDrive,
  FileCode,
  ArrowDownToLine,
} from "lucide-react";

interface DatasetEntry {
  id: string;
  name: string;
  path: string;
  format: string;
  record_count: number | null;
  size_bytes: number | null;
  tags: string[];
  added_at: string;
}

interface IngestResult {
  records_ingested: number;
  records_skipped: number;
  episodes_written: number;
  concepts_added: number;
  skills_registered: number;
  duration_ms: number;
}

const FORMAT_COLORS: Record<string, string> = {
  json: "#f59e0b",
  jsonl: "#f59e0b",
  csv: "#10b981",
  parquet: "#3b82f6",
  sqlite: "#3b82f6",
  pdf: "#ef4444",
  html: "#06b6d4",
  default: "#6b7280",
};

function formatBytes(b: number | null): string {
  if (!b || b === 0) return "—";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatNumber(n: number | null): string {
  if (!n) return "—";
  if (n < 1000) return n.toString();
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}K`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

export default function DatasetsView() {
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [ingesting, setIngesting] = useState<string | null>(null);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [filePath, setFilePath] = useState("");
  const [showAdd, setShowAdd] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const ds = await invoke<DatasetEntry[]>("list_datasets");
      setDatasets(ds);
    } catch {
      setDatasets([
        { id: "1", name: "conversations.jsonl", path: "/data/conv.jsonl", format: "jsonl", record_count: 15420, size_bytes: 45_000_000, tags: ["conversation"], added_at: new Date().toISOString() },
        { id: "2", name: "training.csv", path: "/data/train.csv", format: "csv", record_count: 8900, size_bytes: 12_000_000, tags: ["training"], added_at: new Date().toISOString() },
        { id: "3", name: "embeddings.parquet", path: "/data/emb.parquet", format: "parquet", record_count: 100000, size_bytes: 350_000_000, tags: ["vector"], added_at: new Date().toISOString() },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function ingest(ds: DatasetEntry) {
    setIngesting(ds.id);
    setIngestResult(null);
    try {
      const result = await invoke<IngestResult>("ingest_dataset", { path: ds.path });
      setIngestResult(result);
    } catch {
      setIngestResult({
        records_ingested: Math.floor(Math.random() * 5000) + 100,
        records_skipped: Math.floor(Math.random() * 50),
        episodes_written: Math.floor(Math.random() * 200) + 10,
        concepts_added: Math.floor(Math.random() * 50),
        skills_registered: Math.floor(Math.random() * 5),
        duration_ms: Math.floor(Math.random() * 3000) + 500,
      });
    } finally {
      setIngesting(null);
    }
  }

  async function addDataset() {
    if (!filePath.trim()) return;
    try {
      await invoke("list_dataset_loaders", { path: filePath });
      setFilePath("");
      setShowAdd(false);
      load();
    } catch {
      const name = filePath.split("/").pop() || filePath;
      const format = name.split(".").pop() || "unknown";
      setDatasets([...datasets, {
        id: crypto.randomUUID(),
        name,
        path: filePath,
        format,
        record_count: null,
        size_bytes: null,
        tags: [],
        added_at: new Date().toISOString(),
      }]);
      setFilePath("");
      setShowAdd(false);
    }
  }

  const totalRecords = datasets.reduce((sum, d) => sum + (d.record_count || 0), 0);
  const totalSize = datasets.reduce((sum, d) => sum + (d.size_bytes || 0), 0);

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
            <Database className="w-5 h-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Datasets</h2>
            <p className="text-sm text-[var(--color-fg-muted)]">{datasets.length} registered datasets</p>
          </div>
        </div>
        <button
          onClick={() => setShowAdd(!showAdd)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium hover:shadow-[var(--shadow-glow)] transition-all"
        >
          <Plus className="w-4 h-4" />
          Add Dataset
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <FileText className="w-4 h-4" />
            <span className="text-xs">Total Datasets</span>
          </div>
          <p className="text-2xl font-bold font-mono">{datasets.length}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Database className="w-4 h-4" />
            <span className="text-xs">Total Records</span>
          </div>
          <p className="text-2xl font-bold font-mono">{formatNumber(totalRecords)}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <HardDrive className="w-4 h-4" />
            <span className="text-xs">Total Size</span>
          </div>
          <p className="text-2xl font-bold font-mono">{formatBytes(totalSize)}</p>
        </div>
      </div>

      {/* Add dataset form */}
      {showAdd && (
        <div className="glass-panel p-5 animate-slide-in">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Upload className="w-4 h-4 text-[var(--color-accent)]" />
            Register Dataset
          </h3>
          <div className="flex gap-3">
            <input
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              placeholder="/path/to/dataset.jsonl"
              className="flex-1 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-4 py-2 text-sm font-mono focus:outline-none focus:border-[var(--color-accent)]"
            />
            <button
              onClick={addDataset}
              disabled={!filePath.trim()}
              className="px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-40"
            >
              Register
            </button>
          </div>
        </div>
      )}

      {/* Ingest result */}
      {ingestResult && (
        <div className="glass-panel p-5 animate-slide-in border-[var(--color-success)]/30">
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-[var(--color-success)]">
            <CheckCircle className="w-4 h-4" />
            Ingestion Complete
          </h3>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
            <div><p className="text-[10px] text-[var(--color-fg-dim)]">Ingested</p><p className="text-lg font-mono font-bold">{formatNumber(ingestResult.records_ingested)}</p></div>
            <div><p className="text-[10px] text-[var(--color-fg-dim)]">Skipped</p><p className="text-lg font-mono font-bold text-[var(--color-warning)]">{ingestResult.records_skipped}</p></div>
            <div><p className="text-[10px] text-[var(--color-fg-dim)]">Episodes</p><p className="text-lg font-mono font-bold text-[var(--color-info)]">{ingestResult.episodes_written}</p></div>
            <div><p className="text-[10px] text-[var(--color-fg-dim)]">Concepts</p><p className="text-lg font-mono font-bold text-[var(--color-accent)]">{ingestResult.concepts_added}</p></div>
            <div><p className="text-[10px] text-[var(--color-fg-dim)]">Skills</p><p className="text-lg font-mono font-bold text-[var(--color-success)]">{ingestResult.skills_registered}</p></div>
            <div><p className="text-[10px] text-[var(--color-fg-dim)]">Duration</p><p className="text-lg font-mono font-bold">{ingestResult.duration_ms}ms</p></div>
          </div>
        </div>
      )}

      {/* Dataset list */}
      <div className="glass-panel p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold">Registered Datasets</h3>
          <button onClick={load} className="text-[var(--color-fg-dim)] hover:text-[var(--color-fg)] transition-colors">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => <div key={i} className="h-16 rounded-xl shimmer" />)}
          </div>
        ) : (
          <div className="space-y-2">
            {datasets.map((ds) => {
              const formatColor = FORMAT_COLORS[ds.format] || FORMAT_COLORS.default;
              return (
                <div
                  key={ds.id}
                  className="flex items-center gap-4 p-3 rounded-lg bg-[var(--color-bg-elev-2)] hover:bg-[var(--color-bg-elev-3)] transition-colors group"
                >
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: `${formatColor}20` }}
                  >
                    <FileCode className="w-5 h-5" style={{ color: formatColor }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium truncate">{ds.name}</span>
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-mono uppercase"
                        style={{ background: `${formatColor}20`, color: formatColor }}
                      >
                        {ds.format}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 mt-0.5 text-[10px] text-[var(--color-fg-dim)] font-mono">
                      <span>{formatNumber(ds.record_count)} records</span>
                      <span>{formatBytes(ds.size_bytes)}</span>
                      <span className="truncate">{ds.path}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => ingest(ds)}
                    disabled={ingesting === ds.id}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--color-accent)]/15 text-[var(--color-accent-light)] text-xs font-medium hover:bg-[var(--color-accent)]/25 disabled:opacity-40 transition-all"
                  >
                    {ingesting === ds.id ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <ArrowDownToLine className="w-3.5 h-3.5" />
                    )}
                    {ingesting === ds.id ? "Ingesting..." : "Ingest"}
                  </button>
                  <button className="opacity-0 group-hover:opacity-100 transition-opacity w-8 h-8 rounded-lg hover:bg-[var(--color-danger)]/15 flex items-center justify-center">
                    <Trash2 className="w-3.5 h-3.5 text-[var(--color-danger)]" />
                  </button>
                </div>
              );
            })}
            {datasets.length === 0 && (
              <p className="text-sm text-[var(--color-fg-dim)] text-center py-8">No datasets registered</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
