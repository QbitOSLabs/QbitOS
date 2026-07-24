import type { LucideIcon } from "lucide-react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { cn } from "../lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon?: LucideIcon;
  /** Signed percentage or delta value; positive => up, negative => down. */
  trend?: number;
  trendLabel?: string;
  /** Override the left accent bar color (defaults to --color-accent). */
  color?: string;
  className?: string;
}

/**
 * Glass-panel metric card with optional trend arrow.
 * Designed for dashboards: training loss, reward, safety scores, etc.
 */
export function MetricCard({
  title,
  value,
  unit,
  icon: Icon,
  trend,
  trendLabel,
  color,
  className,
}: MetricCardProps) {
  const trendUp = (trend ?? 0) >= 0;
  return (
    <div className={cn("metric-card", className)} style={color ? { ["--color-accent" as string]: color } : undefined}>
      <div className="flex items-start justify-between gap-2">
        <p className="metric-card-label">{title}</p>
        {Icon && <Icon className="w-4 h-4 text-[var(--color-fg-dim)] flex-shrink-0" />}
      </div>
      <div className="flex items-end gap-2 mt-1">
        <p className="metric-card-value">
          {value}
          {unit && <span className="metric-card-unit">{unit}</span>}
        </p>
        {trend !== undefined && (
          <span className={cn("metric-card-trend mb-1", trendUp ? "metric-card-trend-up" : "metric-card-trend-down")}>
            {trendUp ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
            {Math.abs(trend).toFixed(1)}
            {trendLabel ? ` ${trendLabel}` : "%"}
          </span>
        )}
      </div>
    </div>
  );
}

export default MetricCard;
