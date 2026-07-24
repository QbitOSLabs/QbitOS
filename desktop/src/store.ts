import { create } from "zustand";
import type { KernelStatus } from "./App";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  stage?: string;
}

export interface MemoryEntry {
  id: string;
  layer: string;
  content: string;
  importance: number;
  created_at: string;
}

export interface SkillEntry {
  id: string;
  name: string;
  description: string;
  confidence: number;
  success_rate: number;
  execution_count: number;
}

interface KernelStore {
  status: KernelStatus | null;
  setKernelStatus: (s: KernelStatus) => void;
  chatHistory: ChatMessage[];
  pushMessage: (m: ChatMessage) => void;
  clearChat: () => void;
  memories: MemoryEntry[];
  setMemories: (m: MemoryEntry[]) => void;
  skills: SkillEntry[];
  setSkills: (s: SkillEntry[]) => void;
}

export const useKernelStore = create<KernelStore>((set) => ({
  status: null,
  setKernelStatus: (s) => set({ status: s }),
  chatHistory: [
    {
      id: "init",
      role: "system",
      content: "QbitOS cognitive kernel initialized. The 8-stage loop (observe → understand → create_goal → plan → execute → verify → learn → improve) is ready.",
      timestamp: new Date().toISOString(),
    },
  ],
  pushMessage: (m) => set((s) => ({ chatHistory: [...s.chatHistory, m] })),
  clearChat: () => set({ chatHistory: [] }),
  memories: [],
  setMemories: (m) => set({ memories: m }),
  skills: [],
  setSkills: (s) => set({ skills: s }),
}));
