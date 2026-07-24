import { invoke } from "@tauri-apps/api/core";
import { useState, useRef, useEffect } from "react";
import { Send, Brain, ArrowDown, Loader2, User, Bot } from "lucide-react";
import { useKernelStore } from "../store";

const COGNITIVE_STAGES = [
  "observe",
  "understand",
  "create_goal",
  "plan",
  "execute",
  "verify",
  "learn",
  "improve",
] as const;

type Stage = (typeof COGNITIVE_STAGES)[number];

export default function ChatView() {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const messages = useKernelStore((s) => s.chatHistory);
  const pushMessage = useKernelStore((s) => s.pushMessage);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    if (isAtBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      setShowScrollBtn(el.scrollHeight - el.scrollTop - el.clientHeight > 200);
    };
    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!busy) inputRef.current?.focus();
  }, [busy]);

  async function send() {
    if (!input.trim() || busy) return;
    const prompt = input.trim();
    setInput("");
    setBusy(true);
    setStage("observe");

    pushMessage({
      id: crypto.randomUUID(),
      role: "user",
      content: prompt,
      timestamp: new Date().toISOString(),
    });

    let stageIdx = 0;
    const stageInterval = setInterval(() => {
      stageIdx++;
      if (stageIdx < COGNITIVE_STAGES.length) {
        setStage(COGNITIVE_STAGES[stageIdx]);
      }
    }, 400);

    try {
      const response = await invoke<string>("generate", { prompt });
      clearInterval(stageInterval);
      setStage(null);

      pushMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: response,
        timestamp: new Date().toISOString(),
        stage: "complete",
      });
    } catch {
      clearInterval(stageInterval);
      setStage(null);

      pushMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: `Error: Cognitive kernel unreachable. Ensure QbitOS backend is running (qbitos serve). Response for: "${prompt}"`,
        timestamp: new Date().toISOString(),
        stage: "complete",
      });
    } finally {
      setBusy(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-3">
            <div className="w-16 h-16 rounded-2xl gradient-bg flex items-center justify-center shadow-[var(--shadow-glow)]">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold gradient-text">QbitOS Cognitive Agent</h2>
              <p className="text-sm text-[var(--color-fg-muted)] mt-1">
                Ask anything. The 8-stage cognitive loop will process your request.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 animate-slide-in ${msg.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                msg.role === "user"
                  ? "bg-[var(--color-bg-elev-3)]"
                  : "gradient-bg"
              }`}
            >
              {msg.role === "user" ? (
                <User className="w-4 h-4 text-[var(--color-fg-muted)]" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-[var(--color-accent)] text-white rounded-tr-sm"
                  : "glass-panel rounded-tl-sm"
              }`}
            >
              {msg.role === "system" && (
                <p className="text-xs text-[var(--color-fg-dim)] mb-1">System</p>
              )}
              <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
              <p className="text-[10px] text-[var(--color-fg-dim)] mt-1.5 font-mono">
                {new Date(msg.timestamp).toLocaleTimeString()}
                {msg.stage && ` · ${msg.stage}`}
              </p>
            </div>
          </div>
        ))}

        {/* Cognitive stage indicator */}
        {busy && stage && (
          <div className="flex items-center gap-3 pl-11 animate-slide-in">
            <Loader2 className="w-4 h-4 text-[var(--color-accent)] animate-spin" />
            <div className="flex items-center gap-1.5">
              {COGNITIVE_STAGES.map((s, i) => {
                const currentIdx = COGNITIVE_STAGES.indexOf(stage as Stage);
                return (
                  <span
                    key={s}
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition-all duration-300 ${
                      i <= currentIdx
                        ? "bg-[var(--color-accent)]/20 text-[var(--color-accent-light)]"
                        : "text-[var(--color-fg-dim)]"
                    }`}
                  >
                    {s}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Scroll to bottom button */}
      {showScrollBtn && (
        <button
          onClick={() => scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })}
          className="absolute bottom-24 right-8 w-9 h-9 rounded-full bg-[var(--color-bg-elev-2)] border border-[var(--color-border)] flex items-center justify-center hover:bg-[var(--color-bg-elev-3)] transition-colors shadow-[var(--shadow-card)]"
        >
          <ArrowDown className="w-4 h-4 text-[var(--color-fg-muted)]" />
        </button>
      )}

      {/* Input area */}
      <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-elev)] px-6 py-4">
        <div className="flex items-end gap-3 max-w-4xl mx-auto">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask QbitOS anything..."
              rows={1}
              className="input-field w-full resize-none px-4 py-3 text-sm"
              style={{ minHeight: "48px", maxHeight: "200px" }}
              disabled={busy}
            />
          </div>
          <button
            onClick={send}
            disabled={!input.trim() || busy}
            className="btn-primary w-12 h-12 rounded-xl flex items-center justify-center hover:shadow-[var(--shadow-glow)] transition-all flex-shrink-0"
          >
            {busy ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </div>
        <p className="text-[10px] text-[var(--color-fg-dim)] text-center mt-2">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
