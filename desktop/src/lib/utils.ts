import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS classes with proper conflict resolution.
 * Combines clsx (conditional classes) with tailwind-merge (dedup).
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format bytes to human-readable string.
 */
export function formatBytes(bytes: number): string {
  if (!bytes || bytes === 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

/**
 * Format a large number with K/M suffixes.
 */
export function formatNumber(n: number): string {
  if (!n) return "0";
  if (n < 1000) return n.toString();
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}K`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

/**
 * Format a percentage from a 0-1 float.
 */
export function formatPct(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

/**
 * Format a duration in milliseconds.
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

/**
 * Get a color for a confidence/score value.
 */
export function scoreColor(v: number): string {
  if (v >= 0.8) return "var(--color-success)";
  if (v >= 0.5) return "var(--color-warning)";
  return "var(--color-danger)";
}

/**
 * Format an ISO timestamp to a locale time string.
 */
export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}
