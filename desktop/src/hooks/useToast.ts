import { useCallback } from "react";
import { useKernelStore, type ToastEntry } from "../store";

type ToastVariant = ToastEntry["variant"];

interface UseToastReturn {
  toasts: ToastEntry[];
  toast: (t: { variant?: ToastVariant; title: string; description?: string }) => string;
  success: (title: string, description?: string) => string;
  error: (title: string, description?: string) => string;
  warning: (title: string, description?: string) => string;
  info: (title: string, description?: string) => string;
  dismiss: (id: string) => void;
}

/** Auto-dismiss delay (ms). */
const AUTO_DISMISS_MS = 5000;

/**
 * Toast notification hook backed by the Zustand kernel store.
 * Every toast is auto-dismissed after 5 seconds.
 */
export function useToast(): UseToastReturn {
  const toasts = useKernelStore((s) => s.toasts);
  const addToast = useKernelStore((s) => s.addToast);
  const removeToast = useKernelStore((s) => s.removeToast);

  const dismiss = useCallback(
    (id: string) => {
      removeToast(id);
    },
    [removeToast],
  );

  const toast = useCallback<UseToastReturn["toast"]>(
    ({ variant = "default", title, description }) => {
      const id = addToast({ variant, title, description });
      if (typeof window !== "undefined") {
        window.setTimeout(() => removeToast(id), AUTO_DISMISS_MS);
      }
      return id;
    },
    [addToast, removeToast],
  );

  const success = useCallback(
    (title: string, description?: string) =>
      toast({ variant: "success", title, description }),
    [toast],
  );

  const error = useCallback(
    (title: string, description?: string) =>
      toast({ variant: "error", title, description }),
    [toast],
  );

  const warning = useCallback(
    (title: string, description?: string) =>
      toast({ variant: "warning", title, description }),
    [toast],
  );

  const info = useCallback(
    (title: string, description?: string) =>
      toast({ variant: "info", title, description }),
    [toast],
  );

  return { toasts, toast, success, error, warning, info, dismiss };
}

export default useToast;
