import { useQuery } from "@tanstack/react-query";
import { invoke } from "@tauri-apps/api/core";
import type { KernelStatus } from "../App";

/**
 * Polls the kernel status endpoint at a configurable interval.
 * Returns { status, loading, error, connected }.
 * Throws on backend unreachable — caller should handle error state.
 */
export function useKernelPoll(intervalMs: number = 3000) {
  const query = useQuery<KernelStatus>({
    queryKey: ["kernel_status"],
    queryFn: async () => {
      return await invoke<KernelStatus>("kernel_status");
    },
    refetchInterval: intervalMs,
    staleTime: intervalMs - 500,
    retry: 2,
    retryDelay: 1000,
  });

  return {
    status: query.data ?? null,
    loading: query.isLoading,
    error: query.error,
    connected: query.isSuccess,
  };
}
