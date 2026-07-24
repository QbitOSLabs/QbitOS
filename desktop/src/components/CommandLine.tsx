import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Brain,
  Mic,
  Eye,
  Database,
  Sparkles,
  Globe,
  Activity,
  HardDrive,
  Server,
  GitBranch,
  GraduationCap,
  Shield,
  Search,
  CornerDownLeft,
  ArrowUp,
  ArrowDown,
  Command as CommandIcon,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import type { ViewKind } from "../App";
import { useToast } from "../hooks/useToast";
import { useKernelStore } from "../store";
import { cn } from "../lib/utils";

interface Command {
  id: string;
  label: string;
  hint?: string;
  shortcut?: string;
  icon: React.ElementType;
  group: "Navigate" | "Action";
  run: () => void;
}

interface CommandLineProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (view: ViewKind) => void;
}

/** Simple fuzzy match: every char of the query must appear in order (case-insensitive). */
function fuzzyMatch(query: string, target: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) qi++;
  }
  return qi === q.length;
}

/**
 * Command palette modal. Press Ctrl+K (or Cmd+K on macOS) to open.
 * Supports fuzzy search, arrow-key navigation, enter to run, escape to close.
 */
export function CommandLine({ open, onOpenChange, onNavigate }: CommandLineProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const { success: toastSuccess, info: toastInfo } = useToast();
  const pushMessage = useKernelStore((s) => s.pushMessage);

  const commands = useMemo<Command[]>(() => {
    const nav: Array<{ id: ViewKind; label: string; icon: React.ElementType }> = [
      { id: "chat", label: "Chat", icon: Brain },
      { id: "voice", label: "Voice", icon: Mic },
      { id: "vision", label: "Vision", icon: Eye },
      { id: "memory", label: "Memory", icon: Database },
      { id: "skills", label: "Skills", icon: Sparkles },
      { id: "world_model", label: "World Model", icon: Globe },
      { id: "reasoning", label: "Reasoning", icon: GitBranch },
      { id: "training", label: "Training", icon: GraduationCap },
      { id: "safety", label: "Safety", icon: Shield },
      { id: "agents", label: "Agents", icon: Activity },
      { id: "datasets", label: "Datasets", icon: HardDrive },
      { id: "service", label: "Service", icon: Server },
    ];
    return [
      ...nav.map((n) => ({
        id: `nav-${n.id}`,
        label: `Go to ${n.label}`,
        hint: "Navigate",
        icon: n.icon,
        group: "Navigate" as const,
        run: () => {
          onNavigate(n.id);
          onOpenChange(false);
        },
      })),
      {
        id: "action-clear-chat",
        label: "Clear chat history",
        hint: "Action",
        icon: Sparkles,
        group: "Action",
        run: () => {
          pushMessage({
            id: `cmd-${Date.now()}`,
            role: "system",
            content: "Chat history cleared via command palette.",
            timestamp: new Date().toISOString(),
          });
          toastSuccess("Chat cleared", "A fresh conversation has started.");
          onOpenChange(false);
        },
      },
      {
        id: "action-run-training",
        label: "Run training step",
        hint: "Action",
        icon: GraduationCap,
        shortcut: "T",
        group: "Action",
        run: () => {
          onNavigate("training");
          onOpenChange(false);
          toastInfo("Opening Training", "Run a learning step from the dashboard.");
        },
      },
      {
        id: "action-check-safety",
        label: "Open safety guard",
        hint: "Action",
        icon: Shield,
        shortcut: "S",
        group: "Action",
        run: () => {
          onNavigate("safety");
          onOpenChange(false);
        },
      },
    ];
  }, [onNavigate, onOpenChange, pushMessage, toastSuccess, toastInfo]);

  const filtered = useMemo(
    () =>
      commands.filter(
        (c) =>
          fuzzyMatch(query, c.label) ||
          (c.hint ? fuzzyMatch(query, c.hint) : false),
      ),
    [commands, query],
  );

  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      // Focus input on next tick after dialog opens
      const t = window.setTimeout(() => inputRef.current?.focus(), 50);
      return () => window.clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter") {
        e.preventDefault();
        const cmd = filtered[activeIndex];
        if (cmd) cmd.run();
      }
    },
    [filtered, activeIndex],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="command-palette p-0 gap-0 max-w-xl">
        <DialogTitle className="sr-only">Command Palette</DialogTitle>
        <DialogDescription className="sr-only">
          Search and run commands across the QbitOS desktop.
        </DialogDescription>
        <div className="flex items-center gap-3 px-4">
          <Search className="w-4 h-4 text-[var(--color-fg-dim)] flex-shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a command or search views…"
            className="command-palette-input"
            style={{ borderBottom: "none", padding: "14px 0" }}
          />
          <kbd className="command-palette-item-shortcut">ESC</kbd>
        </div>
        <div className="command-palette-list">
          {filtered.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-[var(--color-fg-dim)]">
              No matching commands.
            </div>
          ) : (
            filtered.map((cmd, i) => {
              const Icon = cmd.icon;
              const active = i === activeIndex;
              return (
                <div
                  key={cmd.id}
                  role="option"
                  aria-selected={active}
                  className={cn("command-palette-item", active && "active")}
                  onMouseMove={() => setActiveIndex(i)}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    cmd.run();
                  }}
                >
                  <Icon className="command-palette-item-icon" />
                  <span>{cmd.label}</span>
                  <span className="ml-2 text-[10px] text-[var(--color-fg-dim)] uppercase tracking-wider">
                    {cmd.group}
                  </span>
                  {cmd.shortcut && (
                    <span className="command-palette-item-shortcut">
                      {cmd.shortcut}
                    </span>
                  )}
                </div>
              );
            })
          )}
        </div>
        <div className="command-palette-footer">
          <span className="flex items-center gap-1">
            <CommandIcon className="w-3 h-3" /> QbitOS
          </span>
          <span className="flex items-center gap-2">
            <span className="flex items-center gap-0.5">
              <ArrowUp className="w-2.5 h-2.5" />
              <ArrowDown className="w-2.5 h-2.5" /> navigate
            </span>
            <span className="flex items-center gap-0.5">
              <CornerDownLeft className="w-2.5 h-2.5" /> select
            </span>
          </span>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Hook that registers a global Ctrl+K / Cmd+K shortcut to open the command palette.
 * Returns [open, setOpen] for controlled usage.
 */
export function useCommandLineShortcut(): [boolean, (v: boolean) => void] {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  return [open, setOpen];
}

export default CommandLine;
