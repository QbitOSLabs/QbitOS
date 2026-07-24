import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/utils";

export type StatusKind = "online" | "offline" | "warning" | "error" | "idle";

interface StatusBadgeProps {
  status: StatusKind;
  label?: string;
  icon?: LucideIcon;
  className?: string;
}

const DEFAULT_LABELS: Record<StatusKind, string> = {
  online: "Online",
  offline: "Offline",
  warning: "Warning",
  error: "Error",
  idle: "Idle",
};

/**
 * Reusable status badge: colored dot + text label, with an optional icon.
 * Pulses softly when status is "online" (see .status-badge-online in index.css).
 */
export function StatusBadge({ status, label, icon: Icon, className }: StatusBadgeProps) {
  const text = label ?? DEFAULT_LABELS[status];
  return (
    <span className={cn("status-badge", `status-badge-${status}`, className)}>
      <span className="status-badge-dot" />
      {Icon && <Icon className="w-3 h-3" />}
      <span>{text}</span>
    </span>
  );
}

export default StatusBadge;
