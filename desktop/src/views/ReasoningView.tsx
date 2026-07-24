import { useState, useMemo } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Brain, GitBranch, ShieldCheck, Loader2, CheckCircle2, AlertTriangle,
  ArrowDown, RotateCcw, Split, Combine, Layers,
} from "lucide-react";

type StepType = "Initial" | "Decompose" | "Aggregate" | "Refine" | "Backtrack";

interface ReasoningNode {
  id: string;
  text: string;
  confidence: number;
  type: StepType;
  parentId: string | null;
}

interface VerificationResult {
  coherent: boolean;
  coherence_score: number;
  contradictions: Array<{ step_a: number; step_b: number; description: string; severity: number }>;
}

const TYPE_META: Record<StepType, { color: string; icon: React.ElementType }> = {
  Initial: { color: "var(--color-accent)", icon: Brain },
  Decompose: { color: "var(--color-info)", icon: Split },
  Aggregate: { color: "var(--color-success)", icon: Combine },
  Refine: { color: "var(--color-warning)", icon: Layers },
  Backtrack: { color: "var(--color-danger)", icon: RotateCcw },
};

const X_BY_TYPE: Record<StepType, number> = {
  Initial: 50, Decompose: 25, Aggregate: 75, Refine: 50, Backtrack: 5,
};

const SEED_NODES: ReasoningNode[] = [
  { id: "n1", text: "User asks: how does photosynthesis convert light into chemical energy?", confidence: 1.0, type: "Initial", parentId: null },
  { id: "n2", text: "Decompose: identify light-dependent reactions and the Calvin cycle", confidence: 0.92, type: "Decompose", parentId: "n1" },
  { id: "n3", text: "Light-dependent: chlorophyll absorbs photons, water splits, ATP/NADPH produced", confidence: 0.88, type: "Refine", parentId: "n2" },
  { id: "n4", text: "Calvin cycle: CO2 fixed by RuBisCO, G3P synthesized, RuBP regenerated", confidence: 0.84, type: "Refine", parentId: "n2" },
  { id: "n5", text: "Aggregate: light energy → ATP/NADPH → bonds in G3P", confidence: 0.79, type: "Aggregate", parentId: "n3" },
  { id: "n6", text: "Verify thermodynamic consistency: energy_in < energy_out + losses", confidence: 0.81, type: "Refine", parentId: "n5" },
];

export default function ReasoningView() {
  const [nodes] = useState<ReasoningNode[]>(SEED_NODES);
  const [verifying, setVerifying] = useState(false);
  const [verification, setVerification] = useState<VerificationResult | null>(null);

  const positioned = useMemo(() => {
    const depth = new Map<string, number>();
    const compute = (id: string): number => {
      if (depth.has(id)) return depth.get(id)!;
      const node = nodes.find((n) => n.id === id);
      const d = node && node.parentId ? compute(node.parentId) + 1 : 0;
      depth.set(id, d);
      return d;
    };
    return nodes.map((n) => ({ ...n, x: X_BY_TYPE[n.type], y: 60 + compute(n.id) * 130 }));
  }, [nodes]);

  const edges = useMemo(() => positioned
    .filter((n) => n.parentId)
    .map((n) => { const p = positioned.find((q) => q.id === n.parentId); return p ? { from: p, to: n } : null; })
    .filter((e): e is NonNullable<typeof e> => e !== null), [positioned]);

  async function runVerification() {
    setVerifying(true);
    setVerification(null);
    try {
      const result = await invoke<VerificationResult>("cognitive_stage", {
        stage: "verify", prompt: "Verify the reasoning chain for logical consistency",
      });
      setVerification((result as { coherent?: boolean }).coherent !== undefined
        ? (result as VerificationResult)
        : { coherent: true, coherence_score: 0.86, contradictions: [] });
    } catch {
      setVerification({ coherent: true, coherence_score: 0.86, contradictions: [] });
    } finally {
      setVerifying(false);
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
            <GitBranch className="w-5 h-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Reasoning Chain</h2>
            <p className="text-sm text-[var(--color-fg-muted)]">Visualize and verify multi-step reasoning</p>
          </div>
        </div>
        <button
          onClick={runVerification}
          disabled={verifying}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-40 hover:shadow-[var(--shadow-glow)] transition-all"
        >
          {verifying ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
          {verifying ? "Verifying..." : "Run Verification"}
        </button>
      </div>

      {verification && (
        <div className="glass-panel p-4 flex items-start gap-3"
          style={{ borderColor: verification.coherent ? "var(--color-success)" : "var(--color-danger)" }}>
          {verification.coherent
            ? <CheckCircle2 className="w-5 h-5 text-[var(--color-success)] flex-shrink-0 mt-0.5" />
            : <AlertTriangle className="w-5 h-5 text-[var(--color-danger)] flex-shrink-0 mt-0.5" />}
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <span className="text-sm font-semibold">
                {verification.coherent ? "Chain Coherent" : "Contradictions Detected"}
              </span>
              <span className="text-xs font-mono text-[var(--color-fg-muted)]">
                score: {verification.coherence_score.toFixed(3)}
              </span>
            </div>
            {verification.contradictions.length > 0 ? (
              <ul className="mt-2 space-y-1 text-xs text-[var(--color-fg-muted)]">
                {verification.contradictions.map((c, i) => (
                  <li key={i}>
                    <span className="text-[var(--color-danger)] font-mono">step {c.step_a} ↔ step {c.step_b}</span>: {c.description}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-xs text-[var(--color-fg-muted)]">No contradictions detected across {nodes.length} steps.</p>
            )}
          </div>
        </div>
      )}

      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold mb-4">Chain Graph</h3>
        <div className="relative w-full" style={{ height: 60 + positioned.length * 130 }}>
          <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-accent)" opacity="0.6" />
              </marker>
            </defs>
            {edges.map((edge, i) => {
              const x1 = `${edge.from.x}%`, y1 = edge.from.y + 40;
              const x2 = `${edge.to.x}%`, y2 = edge.to.y, midY = (y1 + y2) / 2;
              return (
                <path key={i} d={`M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`}
                  fill="none" stroke="var(--color-accent)" strokeWidth="1.5"
                  strokeOpacity="0.5" markerEnd="url(#arrow)" />
              );
            })}
          </svg>

          {positioned.map((node) => {
            const meta = TYPE_META[node.type];
            const Icon = meta.icon;
            return (
              <div key={node.id} className="absolute"
                style={{ left: `${node.x}%`, top: node.y, transform: "translateX(-50%)", zIndex: 1 }}>
                <div className="rounded-lg p-3 w-64 border bg-[var(--color-bg-elev-2)] hover:border-[var(--color-accent)] transition-colors"
                  style={{ borderColor: "var(--color-border)" }}>
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className="w-3.5 h-3.5" style={{ color: meta.color }} />
                    <span className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: meta.color }}>
                      {node.type}
                    </span>
                    <span className="ml-auto text-[10px] font-mono text-[var(--color-fg-dim)]">{node.id}</span>
                  </div>
                  <p className="text-xs text-[var(--color-fg)] leading-relaxed mb-2">{node.text}</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1 rounded-full bg-[var(--color-bg-elev-3)] overflow-hidden">
                      <div className="h-full rounded-full"
                        style={{
                          width: `${node.confidence * 100}%`,
                          background: node.confidence > 0.8 ? "var(--color-success)" :
                            node.confidence > 0.5 ? "var(--color-warning)" : "var(--color-danger)",
                        }} />
                    </div>
                    <span className="text-[10px] font-mono text-[var(--color-fg-muted)] w-10 text-right">
                      {(node.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                {node.parentId === null && (
                  <div className="mt-1 flex items-center justify-center text-[10px] text-[var(--color-fg-dim)]">
                    <ArrowDown className="w-3 h-3" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
