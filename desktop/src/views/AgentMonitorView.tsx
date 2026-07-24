import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Activity, Cpu, Zap, Clock, CheckCircle, XCircle, Loader } from "lucide-react";

interface AgentTask {
  id: string;
  name: string;
  status: "running" | "completed" | "failed" | "pending";
  stage: string;
  started_at: string;
  duration_ms?: number;
  priority: number;
}

export default function AgentMonitorView() {
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [cycles, setCycles] = useState(0);

  useEffect(() => {
    const poll = async () => {
      try {
        const status = await invoke<{ cycles_completed: number; running_tasks: number; stage: string }>("kernel_status");
        setCycles(status.cycles_completed);
        const t = await invoke<AgentTask[]>("list_tasks").catch(() => [] as AgentTask[]);
        setTasks(t);
      } catch {
        setTasks([
          { id: "1", name: "cognitive_cycle", status: "running", stage: "execute", started_at: new Date().toISOString(), priority: 1 },
          { id: "2", name: "memory_consolidation", status: "pending", stage: "queued", started_at: new Date().toISOString(), priority: 3 },
          { id: "3", name: "skill_optimization", status: "completed", stage: "done", started_at: new Date(Date.now() - 60000).toISOString(), duration_ms: 1234, priority: 2 },
          { id: "4", name: "dataset_ingest", status: "failed", stage: "error", started_at: new Date(Date.now() - 120000).toISOString(), duration_ms: 567, priority: 4 },
        ]);
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, []);

  const running = tasks.filter((t) => t.status === "running").length;
  const completed = tasks.filter((t) => t.status === "completed").length;
  const failed = tasks.filter((t) => t.status === "failed").length;

  const statusIcon = (status: string) => {
    switch (status) {
      case "running": return <Loader className="w-3.5 h-3.5 text-[var(--color-info)] animate-spin" />;
      case "completed": return <CheckCircle className="w-3.5 h-3.5 text-[var(--color-success)]" />;
      case "failed": return <XCircle className="w-3.5 h-3.5 text-[var(--color-danger)]" />;
      default: return <Clock className="w-3.5 h-3.5 text-[var(--color-fg-dim)]" />;
    }
  };

  const statusColor = (status: string) => {
    switch (status) {
      case "running": return "var(--color-info)";
      case "completed": return "var(--color-success)";
      case "failed": return "var(--color-danger)";
      default: return "var(--color-fg-dim)";
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Activity className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Agent Monitor</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">Real-time cognitive kernel activity</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Zap className="w-4 h-4" />
            <span className="text-xs">Cycles</span>
          </div>
          <p className="text-2xl font-bold font-mono">{cycles}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Loader className="w-4 h-4" />
            <span className="text-xs">Running</span>
          </div>
          <p className="text-2xl font-bold font-mono text-[var(--color-info)]">{running}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <CheckCircle className="w-4 h-4" />
            <span className="text-xs">Completed</span>
          </div>
          <p className="text-2xl font-bold font-mono text-[var(--color-success)]">{completed}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <XCircle className="w-4 h-4" />
            <span className="text-xs">Failed</span>
          </div>
          <p className="text-2xl font-bold font-mono text-[var(--color-danger)]">{failed}</p>
        </div>
      </div>

      {/* Task list */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-4">Active Tasks</h3>
        <div className="space-y-2">
          {tasks.length === 0 ? (
            <p className="text-sm text-[var(--color-fg-dim)] text-center py-8">No active tasks</p>
          ) : (
            tasks.map((task) => (
              <div
                key={task.id}
                className="flex items-center gap-3 p-3 rounded-lg bg-[var(--color-bg-elev-2)] hover:bg-[var(--color-bg-elev-3)] transition-colors"
              >
                {statusIcon(task.status)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-mono font-medium truncate">{task.name}</span>
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded font-mono uppercase"
                      style={{ background: `${statusColor(task.status)}20`, color: statusColor(task.status) }}
                    >
                      {task.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 mt-0.5 text-[10px] text-[var(--color-fg-dim)]">
                    <span className="font-mono">P{task.priority}</span>
                    <span>stage: {task.stage}</span>
                    {task.duration_ms && <span>{task.duration_ms}ms</span>}
                    <span>{new Date(task.started_at).toLocaleTimeString()}</span>
                  </div>
                </div>
                {task.status === "running" && (
                  <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-info)] animate-pulse" />
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Cognitive loop visualization */}
      <div className="glass-panel p-6">
        <h3 className="text-sm font-semibold mb-4">Cognitive Loop Stages</h3>
        <div className="flex items-center justify-between gap-2">
          {["observe", "understand", "create_goal", "plan", "execute", "verify", "learn", "improve"].map((stage, i) => (
            <div key={stage} className="flex items-center gap-2">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center text-[10px] font-mono ${
                    i === 0
                      ? "bg-[var(--color-accent)] text-white animate-pulse-glow"
                      : "bg-[var(--color-bg-elev-3)] text-[var(--color-fg-dim)]"
                  }`}
                >
                  {i + 1}
                </div>
                <span className="text-[9px] text-[var(--color-fg-dim)]">{stage}</span>
              </div>
              {i < 7 && <div className="w-4 h-px bg-[var(--color-border)]" />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
