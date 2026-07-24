import { useQuery } from "@tanstack/react-query";
import { invoke } from "@tauri-apps/api/core";
import ReactECharts from "echarts-for-react";
import {
  Sparkles,
  Plus,
  Play,
  TrendingUp,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { Button, Progress } from "../components/ui";

interface SkillInfo {
  id: string;
  name: string;
  description: string;
  confidence: number;
  success_rate: number;
  execution_count: number;
  tags: string[];
}



export default function SkillsView() {
  const { data: skills = [], isLoading: loading } = useQuery<SkillInfo[]>({
    queryKey: ["skills"],
    queryFn: async () => {
      const result = await invoke<SkillInfo[]>("list_skills");
      if (!result || result.length === 0) {
        throw new Error("No skills registered");
      }
      return result;
    },
    refetchInterval: 5000,
    staleTime: 4000,
    retry: 1,
  });

  function confColor(c: number): string {
    if (c >= 0.8) return "var(--color-success)";
    if (c >= 0.5) return "var(--color-warning)";
    return "var(--color-danger)";
  }

  const chartOption = {
    tooltip: { trigger: "axis" },
    legend: { data: ["Confidence", "Success Rate"], textStyle: { color: "#94a3b8" } },
    xAxis: { type: "category" as const, data: skills.map((s) => s.name), axisLabel: { color: "#94a3b8", fontSize: 11, rotate: 30 }, axisLine: { lineStyle: { color: "#334155" } } },
    yAxis: { type: "value" as const, max: 100, axisLabel: { color: "#94a3b8", fontSize: 11, formatter: "{value}%" }, splitLine: { lineStyle: { color: "#1e293b" } } },
    series: [
      { name: "Confidence", type: "bar" as const, data: skills.map((s) => Math.round(s.confidence * 100)), itemStyle: { color: "#3b82f6", borderRadius: [4, 4, 0, 0] } },
      { name: "Success Rate", type: "bar" as const, data: skills.map((s) => Math.round(s.success_rate * 100)), itemStyle: { color: "#10b981", borderRadius: [4, 4, 0, 0] } },
    ],
    grid: { left: "8%", right: "5%", bottom: "15%", top: "10%" },
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elev-2)] flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-[var(--color-accent)]" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Skill Genome</h2>
            <p className="text-sm text-[var(--color-fg-muted)]">
              {skills.length} registered skills
            </p>
          </div>
        </div>
        <Button>
          <Plus className="w-4 h-4" />
          New Skill
        </Button>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 rounded-xl shimmer" />
          ))}
        </div>
      ) : (
        <>
          {/* Confidence vs Success Rate chart */}
          {chartData.length > 0 && (
            <div className="glass-panel p-5">
              <h3 className="text-sm font-semibold mb-4">
                Confidence vs Success Rate
              </h3>
              <div style={{ width: "100%", height: 220 }}>
                <ResponsiveContainer>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }}
                      stroke="var(--color-border)"
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fill: "var(--color-fg-muted)", fontSize: 11 }}
                      stroke="var(--color-border)"
                    />
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: "var(--color-bg-elev-2)",
                        border: "1px solid var(--color-border)",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <Bar dataKey="confidence" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="success" fill="var(--color-success)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div className="grid gap-4 md:grid-cols-2">
            {skills.map((skill) => (
              <div
                key={skill.id}
                className="glass-panel p-5 hover:border-[var(--color-accent)] transition-all duration-200 group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-sm group-hover:text-[var(--color-accent-light)] transition-colors">
                      {skill.name}
                    </h3>
                    <p className="text-xs text-[var(--color-fg-muted)] mt-0.5">
                      {skill.description}
                    </p>
                  </div>
                  <Button variant="ghost" size="icon" className="opacity-0 group-hover:opacity-100 transition-opacity">
                    <Play className="w-3.5 h-3.5" />
                  </Button>
                </div>

                <div className="space-y-2 mt-4">
                  <div>
                    <div className="flex justify-between text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider mb-1">
                      <span>Confidence</span>
                      <span className="font-mono">{(skill.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <Progress
                      value={skill.confidence * 100}
                      indicatorClassName={confColor(skill.confidence)}
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider mb-1">
                      <span>Success Rate</span>
                      <span className="font-mono">{(skill.success_rate * 100).toFixed(0)}%</span>
                    </div>
                    <Progress
                      value={skill.success_rate * 100}
                      indicatorClassName={confColor(skill.success_rate)}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3 mt-4 pt-3 border-t border-[var(--color-border)]">
                  <div>
                    <p className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Confidence</p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <div className="w-2 h-2 rounded-full" style={{ background: confColor(skill.confidence) }} />
                      <span className="text-sm font-mono">{(skill.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Success</p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {skill.success_rate >= 0.8 ? (
                        <CheckCircle className="w-3.5 h-3.5 text-[var(--color-success)]" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-[var(--color-warning)]" />
                      )}
                      <span className="text-sm font-mono">{(skill.success_rate * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">Runs</p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <TrendingUp className="w-3.5 h-3.5 text-[var(--color-info)]" />
                      <span className="text-sm font-mono">{skill.execution_count}</span>
                    </div>
                  </div>
                </div>

                {skill.tags.length > 0 && (
                  <div className="flex gap-1.5 mt-3 pt-3 border-t border-[var(--color-border)]">
                    {skill.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 rounded text-[10px] bg-[var(--color-bg-elev-3)] text-[var(--color-fg-dim)] font-mono"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
