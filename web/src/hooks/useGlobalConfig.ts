import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/apiClient";
import type {
  GlobalConfig,
  UpdateGlobalConfigRequest,
  UpdateGlobalConfigResponse,
} from "@/lib/api/models";

type UpdateGlobalConfigArgs = {
  defaultModel?: string;
  defaultThinking?: boolean;
  restartRunningSessions?: boolean;
  forceRestartBusySessions?: boolean;
};

export type UseGlobalConfigReturn = {
  config: GlobalConfig | null;
  isLoading: boolean;
  isUpdating: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  update: (args: UpdateGlobalConfigArgs) => Promise<UpdateGlobalConfigResponse>;
};

export function useGlobalConfig(): UseGlobalConfigReturn {
  const [config, setConfig] = useState<GlobalConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isInitializedRef = useRef(false);
  const lastRefreshAtRef = useRef(0);

  const refresh = useCallback(async (options?: { quiet?: boolean }) => {
    const quiet = options?.quiet ?? false;
    lastRefreshAtRef.current = Date.now();
    if (!quiet) {
      setIsLoading(true);
    }
    setError(null);
    try {
      const nextConfig = await apiClient.config.getGlobalConfigApiConfigGet();
      setConfig(nextConfig);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load global config";
      setError(message);
      console.error("[useGlobalConfig] Failed to load global config:", err);
    } finally {
      if (!quiet) {
        setIsLoading(false);
      }
    }
  }, []);

  const update = useCallback(
    async (
      args: UpdateGlobalConfigArgs,
    ): Promise<UpdateGlobalConfigResponse> => {
      setIsUpdating(true);
      setError(null);
      try {
        const body: UpdateGlobalConfigRequest = {
          defaultModel: args.defaultModel ?? undefined,
          defaultThinking: args.defaultThinking ?? undefined,
          restartRunningSessions: args.restartRunningSessions ?? undefined,
          forceRestartBusySessions: args.forceRestartBusySessions ?? undefined,
        };

        const resp = await apiClient.config.updateGlobalConfigApiConfigPatch({
          updateGlobalConfigRequest: body,
        });
        setConfig(resp.config);
        return resp;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to update global config";
        setError(message);
        console.error("[useGlobalConfig] Failed to update global config:", err);
        throw err;
      } finally {
        setIsUpdating(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (isInitializedRef.current) {
      return;
    }
    isInitializedRef.current = true;
    refresh();
  }, [refresh]);

  // Re-fetch config when another tab/session changes it (broadcast via custom event)
  useEffect(() => {
    const handler = () => {
      refresh();
    };
    window.addEventListener("kimi:config-update", handler);
    return () => window.removeEventListener("kimi:config-update", handler);
  }, [refresh]);

  // Re-fetch config when the window regains focus or becomes visible again.
  // In the desktop app, switching back from the native Settings window
  // triggers focus; this picks up config changes made outside this page
  // (e.g. Settings saved -> backend restarted). Throttled so bursts of
  // focus/visibility events only trigger one request.
  useEffect(() => {
    const MIN_INTERVAL_MS = 2000;
    const maybeRefresh = () => {
      if (document.visibilityState !== "visible") {
        return;
      }
      if (Date.now() - lastRefreshAtRef.current < MIN_INTERVAL_MS) {
        return;
      }
      refresh({ quiet: true });
    };
    window.addEventListener("focus", maybeRefresh);
    document.addEventListener("visibilitychange", maybeRefresh);
    return () => {
      window.removeEventListener("focus", maybeRefresh);
      document.removeEventListener("visibilitychange", maybeRefresh);
    };
  }, [refresh]);

  return {
    config,
    isLoading,
    isUpdating,
    error,
    refresh,
    update,
  };
}
