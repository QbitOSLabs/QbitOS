import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  Shield, ShieldCheck, AlertTriangle, Loader2, CheckCircle2,
  Activity, Lock, FileWarning, CircuitBoard,
} from "lucide-react";
import { Button, Progress } from "../components/ui";
import { useKernelStore } from "../store";
import ReactECharts from "echarts-for-react";

interface HallucinationResult {
  overall_score: number;
  claims_checked: number;
  unsupported_claims: string[];
}

export default function SafetyView() {
  const safetyStats = useKernelStore((s) => s.safetyStats);
  const addToast = useKernelStore((s) => s.addToast);
  const [checkText, setCheckText] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<HallucinationResult | null>(null);

  async function checkResponse() {
    if (!checkText.trim()) return;
    setChecking(true);
    try {
      const response = await invoke<unknown>("cognitive_stage", {
        stage: "verify",
        prompt: checkText,
      });
      const parsed = response as HallucinationResult;
      setResult({
        overall_score: parsed.overall_score ?? 0,
        claims_checked: parsed.claims_checked ?? 0,
        unsupported_claims: parsed.unsupported_claims ?? [],
      });
    } catch (err) {
      addToast({
        type: "error",
        title: "Verification Failed",
        message: String(err),
      });
    } finally {
      setChecking(false);
    }
  }

  const guardrails = [
    { label: "Safety Enforced", enabled: safetyStats?.safetyEnforced ?? true, icon: ShieldCheck },
    { label: "Audit Logging", enabled: safetyStats?.auditLogEnabled ?? true, icon: FileWarning },
    { label: "Self-Improvement", enabled: safetyStats?.selfImproveAllowed ?? false, icon: Activity },
    { label: "Circuit Breakers", enabled: safetyStats?.circuitBreakersActive ?? true, icon: CircuitBoard },
  ];

  const circuitBreakers = safetyStats?.circuitBreakers ?? [
    { tool: "filesystem", state: "closed", failures: 0 },
    { tool: "terminal", state: "closed", failures: 0 },
    { tool: "http", state: "closed", failures: 0 },
    { tool: "calculator", state: "closed", failures: 0 },
  ];

  const radarOption = {
    tooltip: {},
    radar: {
      indicator: [
        { name: "Hallucination Guard", max: 100 },
        { name: "Fact Verification", max: 100 },
        { name: "Circuit Breakers", max: 100 },
        { name: "Audit Coverage", max: 100 },
        { name: "Safety Enforcement", max: 100 },
      ],
      axisName: { color: "#94a3b8", fontSize: 10 },
      splitArea: { areaStyle: { color: ["rgba(59,130,246,0.05)", "rgba(59,130,246,0.1)"] } },
      splitLine: { lineStyle: { color: "#1e293b" } },
      axisLine: { lineStyle: { color: "#334155" } },
    },
    series: [{
      type: "radar" as const,
      data: [{
        value: [
          (safetyStats?.hallucinationScore ?? 0) * 100,
          (result?.overall_score ?? 0) * 100,
          circuitBreakers.filter(cb => cb.state === "closed").length / circuitBreakers.length * 100,
          safetyStats?.auditLogEnabled ? 100 : 0,
          safetyStats?.safetyEnforced ? 100 : 0,
        ],
        name: "Safety Score",
        areaStyle: { color: "rgba(59,130,246,0.2)" },
        lineStyle: { color: "#3b82f6", width: 2 },
        itemStyle: { color: "#3b82f6" },
      }],
    }],
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
          <Shield className="w-5 h-5 text-[var(--color-accent)]" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">Safety & Hallucination Guard</h2>
          <p className="text-sm text-[var(--color-fg-muted)]">
            Real-time safety monitoring and verification
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-[var(--color-warning)]" />
            <span className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Hallucination Score</span>
          </div>
          <span className="text-2xl font-mono" style={{
            color: (safetyStats?.hallucinationScore ?? 1) > 0.7 ? "var(--color-success)" : "var(--color-warning)"
          }}>
            {((safetyStats?.hallucinationScore ?? 1) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle2 className="w-4 h-4 text-[var(--color-success)]" />
            <span className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Claims Checked</span>
          </div>
          <span className="text-2xl font-mono text-[var(--color-success)]">
            {safetyStats?.claimsChecked ?? 0}
          </span>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <FileWarning className="w-4 h-4 text-[var(--color-danger)]" />
            <span className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Unsupported Claims</span>
          </div>
          <span className="text-2xl font-mono text-[var(--color-danger)]">
            {safetyStats?.unsupportedClaims ?? 0}
          </span>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center gap-2 mb-2">
            <Lock className="w-4 h-4 text-[var(--color-accent)]" />
            <span className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Guardrails Active</span>
          </div>
          <span className="text-2xl font-mono text-[var(--color-accent)]">
            {guardrails.filter(g => g.enabled).length}/{guardrails.length}
          </span>
        </div>
      </div>

      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-4">Safety Radar</h3>
        <ReactECharts option={radarOption} style={{ height: 300, width: "100%" }} />
      </div>

      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-4">Guardrail Status</h3>
        <div className="grid grid-cols-2 gap-3">
          {guardrails.map((g) => (
            <div key={g.label} className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
              <div className="flex items-center gap-2">
                <g.icon className="w-4 h-4 text-[var(--color-fg-muted)]" />
                <span className="text-sm">{g.label}</span>
              </div>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                g.enabled
                  ? "bg-[var(--color-success)]/20 text-[var(--color-success)]"
                  : "bg-[var(--color-danger)]/20 text-[var(--color-danger)]"
              }`}>
                {g.enabled ? "Enabled" : "Disabled"}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-4">Circuit Breaker Status</h3>
        <div className="space-y-2">
          {circuitBreakers.map((cb) => (
            <div key={cb.tool} className="flex items-center justify-between p-3 rounded-lg bg-[var(--color-bg-elev-2)]">
              <div className="flex items-center gap-2">
                <CircuitBoard className="w-4 h-4 text-[var(--color-fg-muted)]" />
                <span className="text-sm font-mono">{cb.tool}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-[var(--color-fg-dim)]">{cb.failures} failures</span>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  cb.state === "closed" ? "bg-[var(--color-success)]/20 text-[var(--color-success)]" :
                  cb.state === "open" ? "bg-[var(--color-danger)]/20 text-[var(--color-danger)]" :
                  "bg-[var(--color-warning)]/20 text-[var(--color-warning)]"
                }`}>
                  {cb.state.toUpperCase()}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold mb-3">Check Response for Hallucinations</h3>
        <textarea
          value={checkText}
          onChange={(e) => setCheckText(e.target.value)}
          placeholder="Enter text to verify against knowledge base..."
          rows={3}
          className="w-full rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] px-4 py-3 text-sm focus:outline-none focus:border-[var(--color-accent)] transition-all resize-none"
        />
        <Button onClick={checkResponse} disabled={!checkText.trim() || checking} className="mt-3">
          {checking ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
          {checking ? "Checking..." : "Verify Response"}
        </Button>
        {result && (
          <div className="mt-4 p-4 rounded-lg bg-[var(--color-bg-elev-2)] border border-[var(--color-border)]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Verification Result</span>
              <span className="text-sm font-mono" style={{
                color: result.overall_score > 0.7 ? "var(--color-success)" : "var(--color-warning)"
              }}>
                {(result.overall_score * 100).toFixed(0)}% confidence
              </span>
            </div>
            <div className="text-xs text-[var(--color-fg-muted)]">
              {result.claims_checked} claims checked · {result.unsupported_claims.length} unsupported
            </div>
            {result.unsupported_claims.length > 0 && (
              <div className="mt-2 space-y-1">
                {result.unsupported_claims.map((claim, i) => (
                  <div key={i} className="text-xs text-[var(--color-danger)] flex items-start gap-1">
                    <AlertTriangle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                    <span>{claim}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
