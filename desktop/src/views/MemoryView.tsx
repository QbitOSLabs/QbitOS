import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Database, Layers, TrendingUp, Clock } from "lucide-react";
import { useKernelStore } from "../store";

const LAYER_COLORS: Record<string, string> = {
  L0: "#ef4444",
  L1: "#f59e0b",
  L2: "#10b981",
  L3: "#3b82f6",
  L4: "#3b82f6",
  L5: "#3b82f6",
};

const LAYER_NAMES: Record<string, string> = {
  L0: "Sensory",
  L1: "Working",
  L2: "Episodic",
  L3: "Semantic",
  L4: "Procedural",
  L5: "Meta",
};

export default function MemoryView() {
  const status = useKernelStore((s) => s.status);
  const [backendCount, setBackendCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [backendNames, setBackendNames] = useState<string[]>([]);

  useEffect(() => {
    invoke<string[]>("list_tools")
      .then((names) => {
        setBackendNames(names);
        setBackendCount(names.length);
      })
      .catch(() => {
        setBackendCount(0);
        setBackendNames([]);
      })
      .finally(() => setLoading(false));
  }, []);

  const layers = status?.memory_layers;
  const totalMemories = layers
    ? Object.values(layers).reduce((a, b) => a + b, 0)
    : 0;

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Database className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Memory Fabric</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">L0-L5 biological memory hierarchy</p>
        </div>
      </div>

      {/* Stats grid */}
      {loading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-24 rounded-xl shimmer" />
          ))}
        </div>
      ) : (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Layers className="w-4 h-4" />
            <span className="text-xs">Total Memories</span>
          </div>
          <p className="text-2xl font-bold font-mono">{totalMemories}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Database className="w-4 h-4" />
            <span className="text-xs">Backends Attached</span>
          </div>
          <p className="text-2xl font-bold font-mono">{backendCount}</p>
          {backendNames.length > 0 && (
            <p className="text-[10px] text-[var(--color-fg-dim)] mt-1">{backendNames.join(", ")}</p>
          )}
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <TrendingUp className="w-4 h-4" />
            <span className="text-xs">Active Layers</span>
          </div>
          <p className="text-2xl font-bold font-mono">
            {layers ? Object.values(layers).filter((v) => v > 0).length : 0}
            <span className="text-sm text-[var(--color-fg-dim)]">/6</span>
          </p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Clock className="w-4 h-4" />
            <span className="text-xs">Cycles Completed</span>
          </div>
          <p className="text-2xl font-bold font-mono">{status?.cycles_completed ?? 0}</p>
        </div>
      </div>
      )}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold mb-4">Memory Layers</h3>
        <div className="space-y-3">
          {layers &&
            Object.entries(layers).map(([key, value]) => {
              const layerName = key.replace("l", "L").replace("_", " ").split(" ")[0].toUpperCase();
              const label = key.replace("l", "L").replace("_", " ").split("_")[0].toUpperCase();
              const displayName = LAYER_NAMES[label] || label;
              const color = LAYER_COLORS[label] || "#666";
              const pct = totalMemories > 0 ? (value / totalMemories) * 100 : 0;
              return (
                <div key={key} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                      <span className="font-medium">{label}</span>
                      <span className="text-[var(--color-fg-dim)]">{displayName}</span>
                    </div>
                    <span className="font-mono text-[var(--color-fg-muted)]">{value}</span>
                  </div>
                  <div className="h-2 rounded-full bg-[var(--color-bg-elev-3)] overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${pct}%`, background: color }}
                    />
                  </div>
                </div>
              );
            })}
          {!layers && (
            <p className="text-sm text-[var(--color-fg-dim)] text-center py-8">
              No memory data available. Start the kernel to populate memory layers.
            </p>
          )}
        </div>
      </div>

      {/* Backends */}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold mb-4">Database Backends</h3>
        {backendNames.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {backendNames.map((name) => (
              <span
                key={name}
                className="px-3 py-1.5 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] text-sm font-mono text-[var(--color-success)]"
              >
                {name}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-[var(--color-fg-dim)]">
            No backends attached. Set DATABASE_URL, QDRANT_URL, REDIS_URL, or QBITOS_ROCKSDB_PATH to connect.
          </p>
        )}
      </div>
    </div>
  );
}
