import {
  Brain,
  Mic,
  Eye,
  Database,
  Sparkles,
  Globe,
  Activity,
  Cpu,
  HardDrive,
  Server,
  Zap,
  GitBranch,
} from "lucide-react";
import type { ViewKind, KernelStatus } from "../App";

interface Props {
  current: ViewKind;
  onSelect: (v: ViewKind) => void;
  status: KernelStatus | null;
  connected: boolean;
}

const NAV_ITEMS: { id: ViewKind; label: string; icon: React.ElementType; group: string }[] = [
  { id: "chat", label: "Chat", icon: Brain, group: "Interaction" },
  { id: "voice", label: "Voice", icon: Mic, group: "Interaction" },
  { id: "vision", label: "Vision", icon: Eye, group: "Interaction" },
  { id: "memory", label: "Memory", icon: Database, group: "Cognitive" },
  { id: "skills", label: "Skills", icon: Sparkles, group: "Cognitive" },
  { id: "world_model", label: "World Model", icon: Globe, group: "Cognitive" },
  { id: "reasoning", label: "Reasoning", icon: GitBranch, group: "Cognitive" },
  { id: "agents", label: "Agents", icon: Activity, group: "System" },
  { id: "datasets", label: "Datasets", icon: HardDrive, group: "System" },
  { id: "service", label: "Service", icon: Server, group: "System" },
];

function StatBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] text-[var(--color-fg-dim)]">
        <span>{label}</span>
        <span className="font-mono">{pct.toFixed(0)}%</span>
      </div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function Sidebar({ current, onSelect, status, connected }: Props) {
  const groups = ["Interaction", "Cognitive", "System"];
  const memUsed = status?.memory_used_bytes ?? 0;
  const memTotal = status?.memory_total_bytes ?? 1;
  const cpu = (status?.cpu_utilization ?? 0) * 100;

  return (
    <aside className="w-64 flex flex-col bg-[var(--color-bg-elev)] border-r border-[var(--color-border)] flex-shrink-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl gradient-bg flex items-center justify-center shadow-[var(--shadow-glow)]">
            <Zap className="w-5 h-5 text-white" fill="white" />
          </div>
          <div>
            <h1 className="text-base font-bold gradient-text">QbitOS</h1>
            <p className="text-[10px] text-[var(--color-fg-dim)] font-mono">v0.1.0</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-4">
        {groups.map((group) => (
          <div key={group}>
            <p className="px-3 mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-fg-dim)]">
              {group}
            </p>
            <div className="space-y-0.5">
              {NAV_ITEMS.filter((item) => item.group === group).map((item) => {
                const Icon = item.icon;
                const active = current === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => onSelect(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-150 ${
                      active
                        ? "bg-[var(--color-accent)]/15 text-[var(--color-accent-light)] font-medium"
                        : "text-[var(--color-fg-muted)] hover:bg-[var(--color-bg-elev-2)] hover:text-[var(--color-fg)]"
                    }`}
                  >
                    <Icon className="w-4 h-4 flex-shrink-0" />
                    <span>{item.label}</span>
                    {active && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-[var(--color-accent)] animate-pulse-glow" />}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* System Status */}
      <div className="border-t border-[var(--color-border)] p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-fg-dim)]">
            System
          </span>
          <span className="flex items-center gap-1.5 text-[10px]">
            <span className={`status-dot ${connected ? "status-dot-online" : "status-dot-warning"}`} />
            <span className="text-[var(--color-fg-muted)]">{connected ? "Online" : "Dev"}</span>
          </span>
        </div>

        {status && (
          <>
            <StatBar label="CPU" value={cpu} max={100} color="var(--color-accent)" />
            <StatBar label="Memory" value={memUsed} max={memTotal} color="var(--color-accent-light)" />

            {status.memory_layers && (
              <div className="space-y-1 pt-2 border-t border-[var(--color-border)]">
                <p className="text-[10px] text-[var(--color-fg-dim)]">Memory Layers</p>
                <div className="grid grid-cols-3 gap-1 text-[10px] font-mono text-[var(--color-fg-muted)]">
                  <span>L0: {status.memory_layers.l0_sensory}</span>
                  <span>L1: {status.memory_layers.l1_working}</span>
                  <span>L2: {status.memory_layers.l2_episodic}</span>
                  <span>L3: {status.memory_layers.l3_semantic}</span>
                  <span>L4: {status.memory_layers.l4_procedural}</span>
                  <span>L5: {status.memory_layers.l5_meta_strategies}</span>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between text-[10px] text-[var(--color-fg-dim)]">
              <span className="flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                {status.running_tasks} tasks
              </span>
              <span className="flex items-center gap-1">
                <Activity className="w-3 h-3" />
                {status.cycles_completed} cycles
              </span>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
