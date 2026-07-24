import { useState, useEffect, useRef, useCallback } from "react";
import {
  Radio,
  Server,
  Activity,
  Send,
  Pause,
  Play,
  Trash2,
  CheckCircle2,
  XCircle,
  Loader2,
  Zap,
  Wifi,
  WifiOff,
} from "lucide-react";

interface ServiceEvent {
  id: { "0": number };
  timestamp: string;
  event:
    | { kind: "cognitive_cycle_completed"; cycle_id: string; phase: string; duration_ms: number; summary: string }
    | { kind: "memory_stored"; memory_id: string; layer: string; content_preview: string }
    | { kind: "skill_executed"; skill_id: string; success: boolean; duration_ms: number }
    | { kind: "tool_invoked"; tool_name: string; success: boolean }
    | { kind: "reasoning_conclusion"; engine: string; conclusion: string; confidence: number }
    | { kind: "vision_perception"; objects_detected: number; ocr_chars: number }
    | { kind: "speech_event"; modality: string; duration_ms: number }
    | { kind: "learning_event"; event_type: string; metric: string; value: number }
    | { kind: "safety_event"; rule: string; severity: string; message: string }
    | { kind: "subsystem_started"; name: string }
    | { kind: "subsystem_stopped"; name: string }
    | { kind: "log"; level: string; message: string }
    | { kind: "custom"; subsystem: string; payload: unknown };
}

interface HealthResponse {
  ok: boolean;
  data: {
    instance_id: string;
    version: string;
    uptime_seconds: number;
    cycles_completed: number;
    backends: string[];
  };
}

const EVENT_ICONS: Record<string, React.ElementType> = {
  cognitive_cycle_completed: Zap,
  memory_stored: Server,
  skill_executed: CheckCircle2,
  tool_invoked: Activity,
  reasoning_conclusion: Activity,
  vision_perception: Activity,
  speech_event: Activity,
  learning_event: Activity,
  safety_event: Activity,
  subsystem_started: CheckCircle2,
  subsystem_stopped: XCircle,
  log: Activity,
  custom: Activity,
};

const EVENT_COLORS: Record<string, string> = {
  cognitive_cycle_completed: "var(--color-accent)",
  memory_stored: "var(--color-info)",
  skill_executed: "var(--color-success)",
  tool_invoked: "var(--color-warning)",
  safety_event: "var(--color-danger)",
  subsystem_started: "var(--color-success)",
  subsystem_stopped: "var(--color-danger)",
  log: "var(--color-fg-dim)",
};

function getEventKind(event: ServiceEvent["event"]): string {
  return event.kind;
}

function getEventDescription(event: ServiceEvent["event"]): string {
  switch (event.kind) {
    case "cognitive_cycle_completed":
      return `Cycle ${event.cycle_id} phase ${event.phase} (${event.duration_ms}ms): ${event.summary}`;
    case "memory_stored":
      return `[${event.layer}] ${event.content_preview}`;
    case "skill_executed":
      return `Skill ${event.skill_id} ${event.success ? "succeeded" : "failed"} (${event.duration_ms}ms)`;
    case "tool_invoked":
      return `Tool ${event.tool_name} ${event.success ? "succeeded" : "failed"}`;
    case "reasoning_conclusion":
      return `[${event.engine}] ${event.conclusion} (conf: ${(event.confidence * 100).toFixed(0)}%)`;
    case "vision_perception":
      return `${event.objects_detected} objects, ${event.ocr_chars} OCR chars`;
    case "speech_event":
      return `${event.modality} (${event.duration_ms}ms)`;
    case "learning_event":
      return `${event.event_type}: ${event.metric}=${event.value}`;
    case "safety_event":
      return `[${event.severity}] ${event.rule}: ${event.message}`;
    case "subsystem_started":
      return `${event.name} started`;
    case "subsystem_stopped":
      return `${event.name} stopped`;
    case "log":
      return `[${event.level}] ${event.message}`;
    case "custom":
      return `[${event.subsystem}] ${JSON.stringify(event.payload)}`;
  }
}

export default function ServiceView() {
  const [connected, setConnected] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [events, setEvents] = useState<ServiceEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [wsUrl, setWsUrl] = useState("ws://localhost:8080/api/v1/events/ws");
  const [command, setCommand] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  const checkHealth = useCallback(async () => {
    try {
      const resp = await fetch("http://localhost:8080/api/v1/health");
      const data = await resp.json();
      setHealth(data);
      setConnected(true);
    } catch {
      setConnected(false);
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  function connect() {
    if (streaming) {
      wsRef.current?.close();
      setStreaming(false);
      return;
    }

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStreaming(true);
        setEvents((prev) => [
          ...prev,
          {
            id: { "0": Date.now() },
            timestamp: new Date().toISOString(),
            event: { kind: "log", level: "info", message: "WebSocket connected" },
          },
        ]);
      };

      ws.onmessage = (msg) => {
        try {
          const event = JSON.parse(msg.data);
          setEvents((prev) => [...prev.slice(-200), event]);
        } catch {
          // Non-JSON message
        }
      };

      ws.onclose = () => {
        setStreaming(false);
        setEvents((prev) => [
          ...prev,
          {
            id: { "0": Date.now() },
            timestamp: new Date().toISOString(),
            event: { kind: "log", level: "warn", message: "WebSocket disconnected" },
          },
        ]);
      };

      ws.onerror = () => {
        setStreaming(false);
      };
    } catch {
      setStreaming(false);
    }
  }

  function sendCommand() {
    if (!command.trim() || !wsRef.current) return;
    wsRef.current.send(command);
    setEvents((prev) => [
      ...prev,
      {
        id: { "0": Date.now() },
        timestamp: new Date().toISOString(),
        event: { kind: "log", level: "info", message: `→ ${command}` },
      },
    ]);
    setCommand("");
  }

  function clearEvents() {
    setEvents([]);
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Server className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Service Layer</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">HTTP API, WebSocket events, and JSON-RPC</p>
        </div>
      </div>

      {/* Connection status */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            {connected ? <Wifi className="w-4 h-4 text-[var(--color-success)]" /> : <WifiOff className="w-4 h-4 text-[var(--color-danger)]" />}
            <span className="text-xs">API Status</span>
          </div>
          <p className={`text-lg font-bold ${connected ? "text-[var(--color-success)]" : "text-[var(--color-danger)]"}`}>
            {connected ? "Online" : "Offline"}
          </p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Activity className="w-4 h-4" />
            <span className="text-xs">Cycles</span>
          </div>
          <p className="text-lg font-bold font-mono">{health?.data?.cycles_completed ?? 0}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Server className="w-4 h-4" />
            <span className="text-xs">Backends</span>
          </div>
          <p className="text-lg font-bold font-mono">{health?.data?.backends?.length ?? 0}</p>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 text-[var(--color-fg-dim)] mb-2">
            <Zap className="w-4 h-4" />
            <span className="text-xs">Uptime</span>
          </div>
          <p className="text-lg font-bold font-mono">
            {health?.data?.uptime_seconds ? `${Math.floor(health.data.uptime_seconds / 60)}m` : "—"}
          </p>
        </div>
      </div>

      {/* WebSocket controls */}
      <div className="glass-panel p-5">
        <div className="flex items-center gap-3 mb-4">
          <Radio className={`w-5 h-5 ${streaming ? "text-[var(--color-success)] animate-pulse" : "text-[var(--color-fg-dim)]"}`} />
          <h3 className="text-sm font-semibold">Event Stream</h3>
          <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${streaming ? "bg-[var(--color-success)]/15 text-[var(--color-success)]" : "bg-[var(--color-bg-elev-3)] text-[var(--color-fg-dim)]"}`}>
            {streaming ? "LIVE" : "IDLE"}
          </span>
          <span className="text-xs text-[var(--color-fg-dim)] ml-auto">{events.length} events</span>
        </div>

        <div className="flex gap-2 mb-3">
          <input
            value={wsUrl}
            onChange={(e) => setWsUrl(e.target.value)}
            disabled={streaming}
            placeholder="ws://localhost:8080/api/v1/events/ws"
            className="flex-1 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-3 py-2 text-sm font-mono focus:outline-none focus:border-[var(--color-accent)] disabled:opacity-50"
          />
          <button
            onClick={connect}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              streaming
                ? "bg-[var(--color-danger)] text-white hover:opacity-80"
                : "bg-[var(--color-success)] text-white hover:shadow-[var(--shadow-glow)]"
            }`}
          >
            {streaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {streaming ? "Stop" : "Connect"}
          </button>
          <button
            onClick={clearEvents}
            className="px-3 py-2 rounded-lg bg-[var(--color-bg-elev-3)] hover:bg-[var(--color-border)] transition-colors"
          >
            <Trash2 className="w-4 h-4 text-[var(--color-fg-muted)]" />
          </button>
        </div>

        {/* Command input */}
        {streaming && (
          <div className="flex gap-2 mb-3 animate-slide-in">
            <input
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendCommand()}
              placeholder="Send command..."
              className="flex-1 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-3 py-2 text-sm font-mono focus:outline-none focus:border-[var(--color-accent)]"
            />
            <button
              onClick={sendCommand}
              disabled={!command.trim()}
              className="flex items-center gap-1 px-3 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm disabled:opacity-40"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Event log */}
        <div className="h-80 overflow-y-auto rounded-lg bg-[var(--color-bg)] border border-[var(--color-border)] p-3 space-y-1">
          {events.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Radio className="w-8 h-8 text-[var(--color-fg-dim)] mb-2" />
              <p className="text-sm text-[var(--color-fg-dim)]">
                {streaming ? "Waiting for events..." : "Connect to start streaming events"}
              </p>
            </div>
          ) : (
            events.map((event, i) => {
              const kind = getEventKind(event.event);
              const desc = getEventDescription(event.event);
              const Icon = EVENT_ICONS[kind] || Activity;
              const color = EVENT_COLORS[kind] || "var(--color-fg-muted)";
              return (
                <div
                  key={i}
                  className="flex items-start gap-2 p-2 rounded hover:bg-[var(--color-bg-elev)] transition-colors animate-slide-in"
                >
                  <Icon className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: `${color}20`, color }}>
                        {kind}
                      </span>
                      <span className="text-[10px] text-[var(--color-fg-dim)] font-mono">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-xs text-[var(--color-fg-muted)] mt-0.5 truncate">{desc}</p>
                  </div>
                </div>
              );
            })
          )}
          <div ref={eventsEndRef} />
        </div>
      </div>

      {/* API endpoints reference */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-3">API Endpoints</h3>
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          {[
            ["GET", "/api/v1/health", "var(--color-success)"],
            ["GET", "/api/v1/version", "var(--color-info)"],
            ["GET", "/api/v1/stats", "var(--color-info)"],
            ["POST", "/api/v1/cognitive/cycle", "var(--color-warning)"],
            ["GET", "/api/v1/memory/stats", "var(--color-info)"],
            ["POST", "/api/v1/memory/store", "var(--color-warning)"],
            ["GET", "/api/v1/skills", "var(--color-info)"],
            ["POST", "/api/v1/skills/execute", "var(--color-warning)"],
            ["GET", "/api/v1/world/stats", "var(--color-info)"],
            ["WS", "/api/v1/events/ws", "var(--color-accent)"],
          ].map(([method, path, color]) => (
            <div key={path} className="flex items-center gap-2 p-2 rounded bg-[var(--color-bg-elev-2)]">
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold" style={{ background: `${color}20`, color }}>
                {method}
              </span>
              <span className="text-[var(--color-fg-muted)]">{path}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
