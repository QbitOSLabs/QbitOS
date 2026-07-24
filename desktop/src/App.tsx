import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatView from "./views/ChatView";
import VoiceView from "./views/VoiceView";
import VisionView from "./views/VisionView";
import MemoryView from "./views/MemoryView";
import SkillsView from "./views/SkillsView";
import WorldModelView from "./views/WorldModelView";
import AgentMonitorView from "./views/AgentMonitorView";
import DatasetsView from "./views/DatasetsView";
import ServiceView from "./views/ServiceView";
import ReasoningView from "./views/ReasoningView";
import { TooltipProvider } from "./components/ui/tooltip";
import { useKernelStore } from "./store";

export type ViewKind =
  | "chat"
  | "voice"
  | "vision"
  | "memory"
  | "skills"
  | "world_model"
  | "reasoning"
  | "agents"
  | "datasets"
  | "service";

export interface KernelStatus {
  boot_time: string;
  cycles_completed: number;
  running_tasks: number;
  memory_used_bytes: number;
  memory_total_bytes: number;
  cpu_utilization: number;
  stage: string;
  memory_layers?: {
    l0_sensory: number;
    l1_working: number;
    l2_episodic: number;
    l3_semantic: number;
    l4_procedural: number;
    l5_meta_strategies: number;
  };
}

export default function App() {
  const [view, setView] = useState<ViewKind>("chat");
  const [status, setStatus] = useState<KernelStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const setKernelStatus = useKernelStore((s) => s.setKernelStatus);

  const pollStatus = useCallback(async () => {
    try {
      const s = await invoke<KernelStatus>("kernel_status");
      setStatus(s);
      setKernelStatus(s);
      setConnected(true);
    } catch {
      // When no backend is reachable, display a neutral zero-state instead of
      // fake metrics. The header shows "Disconnected" rather than "Dev Mode"
      // to make the failure explicit to the operator.
      const disconnected_default: KernelStatus = {
        boot_time: new Date().toISOString(),
        cycles_completed: 0,
        running_tasks: 0,
        memory_used_bytes: 0,
        memory_total_bytes: 8 * 1024 * 1024 * 1024,
        cpu_utilization: 0,
        stage: "disconnected",
        memory_layers: {
          l0_sensory: 0,
          l1_working: 0,
          l2_episodic: 0,
          l3_semantic: 0,
          l4_procedural: 0,
          l5_meta_strategies: 0,
        },
      };
      setStatus(disconnected_default);
      setConnected(false);
    }
  }, [setKernelStatus]);

  useEffect(() => {
    pollStatus();
    const interval = setInterval(pollStatus, 3000);
    return () => clearInterval(interval);
  }, [pollStatus]);

  const renderView = () => {
    switch (view) {
      case "chat": return <ChatView />;
      case "voice": return <VoiceView />;
      case "vision": return <VisionView />;
      case "memory": return <MemoryView />;
      case "skills": return <SkillsView />;
      case "world_model": return <WorldModelView />;
      case "reasoning": return <ReasoningView />;
      case "agents": return <AgentMonitorView />;
      case "datasets": return <DatasetsView />;
      case "service": return <ServiceView />;
      default: return <ChatView />;
    }
  };

  return (
    <TooltipProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-[var(--color-bg)]">
        <Sidebar current={view} onSelect={setView} status={status} connected={connected} />
        <main className="flex-1 flex flex-col overflow-hidden">
          <header className="flex items-center justify-between px-6 py-3 border-b border-[var(--color-border)] bg-[var(--color-bg-elev)]">
            <div className="flex items-center gap-3">
              <h1 className="text-sm font-medium text-[var(--color-fg-muted)] capitalize">
                {view.replace("_", " ")}
              </h1>
            </div>
            <div className="flex items-center gap-4 text-xs text-[var(--color-fg-dim)]">
              {connected ? (
                <span className="flex items-center gap-1.5">
                  <span className="status-dot status-dot-online" />
                  Connected
                </span>
              ) : (
                <span className="flex items-center gap-1.5">
                  <span className="status-dot status-dot-warning" />
                  Disconnected
                </span>
              )}
              {status && (
                <span className="font-mono">
                  CPU: {(status.cpu_utilization * 100).toFixed(1)}%
                </span>
              )}
              {status && (
                <span className="font-mono">
                  RAM: {(status.memory_used_bytes / 1024 / 1024).toFixed(0)}MB
                </span>
              )}
              {status && status.running_tasks > 0 && (
                <span className="font-mono text-[var(--color-warning)]">
                  Tasks: {status.running_tasks}
                </span>
              )}
            </div>
          </header>
          <div className="flex-1 overflow-auto">{renderView()}</div>
        </main>
      </div>
    </TooltipProvider>
  );
}
