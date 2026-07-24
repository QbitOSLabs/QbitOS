import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Globe, Network, Plus, Search, GitBranch, ArrowRight } from "lucide-react";

interface Triple {
  subject: string;
  predicate: string;
  object: string;
}

export default function WorldModelView() {
  const [triples, setTriples] = useState<Triple[]>([]);
  const [entityCount, setEntityCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [newTriple, setNewTriple] = useState({ subject: "", predicate: "", object: "" });

  useEffect(() => {
    invoke<{ triples: Triple[]; entities: number }>("memory_layers")
      .then((data) => {
        setTriples(data.triples || []);
        setEntityCount(data.entities || 0);
      })
      .catch(() => {
        setTriples([
          { subject: "QbitOS", predicate: "is_a", object: "Cognitive OS" },
          { subject: "QbitOS", predicate: "written_in", object: "Rust" },
          { subject: "QbitOS", predicate: "has", object: "25 research modules" },
          { subject: "Memory", predicate: "has_layers", object: "L0-L5" },
          { subject: "Kernel", predicate: "runs", object: "8-stage loop" },
        ]);
        setEntityCount(5);
      })
      .finally(() => setLoading(false));
  }, []);

  function addTriple() {
    if (!newTriple.subject || !newTriple.predicate || !newTriple.object) return;
    setTriples([...triples, { ...newTriple }]);
    setNewTriple({ subject: "", predicate: "", object: "" });
  }

  const filtered = triples.filter(
    (t) =>
      t.subject.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.predicate.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.object.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Globe className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">World Model</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">Knowledge graph and causal reasoning</p>
        </div>
      </div>

      {/* Stats */}
      {loading ? (
        <div className="grid grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-20 rounded-xl shimmer" />
          ))}
        </div>
      ) : (
      <div className="grid grid-cols-3 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Network className="w-4 h-4" />
            <span className="text-xs">Entities</span>
          </div>
          <p className="text-2xl font-bold font-mono">{entityCount}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <GitBranch className="w-4 h-4" />
            <span className="text-xs">Triples</span>
          </div>
          <p className="text-2xl font-bold font-mono">{triples.length}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <ArrowRight className="w-4 h-4" />
            <span className="text-xs">Relations</span>
          </div>
          <p className="text-2xl font-bold font-mono">
            {new Set(triples.map((t) => t.predicate)).size}
          </p>
        </div>
      </div>
      )}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <Plus className="w-4 h-4 text-[var(--color-accent)]" />
          Add Knowledge Triple
        </h3>
        <div className="grid grid-cols-3 gap-3">
          <input
            value={newTriple.subject}
            onChange={(e) => setNewTriple({ ...newTriple, subject: e.target.value })}
            placeholder="Subject"
            className="rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-3 py-2 text-sm focus:outline-none focus:border-[var(--color-accent)]"
          />
          <input
            value={newTriple.predicate}
            onChange={(e) => setNewTriple({ ...newTriple, predicate: e.target.value })}
            placeholder="Predicate"
            className="rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-3 py-2 text-sm focus:outline-none focus:border-[var(--color-accent)]"
          />
          <input
            value={newTriple.object}
            onChange={(e) => setNewTriple({ ...newTriple, object: e.target.value })}
            placeholder="Object"
            className="rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-3 py-2 text-sm focus:outline-none focus:border-[var(--color-accent)]"
          />
        </div>
        <button
          onClick={addTriple}
          disabled={!newTriple.subject || !newTriple.predicate || !newTriple.object}
          className="mt-3 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[var(--shadow-glow)] transition-all"
        >
          Add Triple
        </button>
      </div>

      {/* Search & list */}
      <div className="glass-panel p-5">
        <div className="flex items-center gap-2 mb-4">
          <Search className="w-4 h-4 text-[var(--color-fg-dim)]" />
          <input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search triples..."
            className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-[var(--color-fg-dim)]"
          />
        </div>
        <div className="space-y-2">
          {filtered.map((t, i) => (
            <div
              key={i}
              className="flex items-center gap-3 p-3 rounded-lg bg-[var(--color-bg-elev-2)] hover:bg-[var(--color-bg-elev-3)] transition-colors animate-slide-in"
            >
              <span className="px-2.5 py-1 rounded-md bg-[var(--color-accent)]/15 text-[var(--color-accent-light)] text-xs font-mono">
                {t.subject}
              </span>
              <ArrowRight className="w-3 h-3 text-[var(--color-fg-dim)] flex-shrink-0" />
              <span className="px-2.5 py-1 rounded-md bg-[var(--color-info)]/15 text-[var(--color-info)] text-xs font-mono">
                {t.predicate}
              </span>
              <ArrowRight className="w-3 h-3 text-[var(--color-fg-dim)] flex-shrink-0" />
              <span className="px-2.5 py-1 rounded-md bg-[var(--color-success)]/15 text-[var(--color-success)] text-xs font-mono">
                {t.object}
              </span>
            </div>
          ))}
          {filtered.length === 0 && (
            <p className="text-sm text-[var(--color-fg-dim)] text-center py-4">No triples found</p>
          )}
        </div>
      </div>
    </div>
  );
}
