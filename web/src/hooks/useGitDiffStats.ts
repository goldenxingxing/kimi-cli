import { useState, useCallback, useEffect, useRef } from "react";
import type { GitDiffStats } from "../lib/api/models";
import { getAuthHeader } from "../lib/auth";
import { getApiBaseUrl } from "./utils";

type UseGitDiffStatsReturn = {
  stats: GitDiffStats | null;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
};

const CACHE_TTL_MS = 10000; // 10 seconds cache
const POLL_INTERVAL_MS = 30000; // 30 seconds polling

/**
 * Hook for fetching git diff stats for a session
 */
export function useGitDiffStats(sessionId: string | null): UseGitDiffStatsReturn {
  const [stats, setStats] = useState<GitDiffStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Cache to avoid unnecessary requests
  const cacheRef = useRef<{
    sessionId: string;
    stats: GitDiffStats;
    timestamp: number;
  } | null>(null);

  // The session this hook is currently *for*. A request started for one
  // session and answered after the reader has moved to another used to write
  // both setStats and the cache under the old session's name — the panel then
  // showed the previous session's counts, and the "clear on session change"
  // effect had already run, so nothing corrected it until the next poll.
  const requestedSessionRef = useRef(sessionId);
  requestedSessionRef.current = sessionId;

  const fetchStats = useCallback(async (forceRefresh = false) => {
    if (!sessionId) {
      setStats(null);
      return;
    }

    // Check cache
    const now = Date.now();
    if (
      !forceRefresh &&
      cacheRef.current &&
      cacheRef.current.sessionId === sessionId &&
      now - cacheRef.current.timestamp < CACHE_TTL_MS
    ) {
      setStats(cacheRef.current.stats);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const basePath = getApiBaseUrl();
      const response = await fetch(
        `${basePath}/api/sessions/${encodeURIComponent(sessionId)}/git-diff`,
        { headers: getAuthHeader() }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch git diff stats");
      }

      const data = await response.json();
      // Convert snake_case to camelCase
      const gitDiffStats: GitDiffStats = {
        isGitRepo: data.is_git_repo,
        hasChanges: data.has_changes ?? false,
        totalAdditions: data.total_additions ?? 0,
        totalDeletions: data.total_deletions ?? 0,
        files: (data.files ?? []).map((f: Record<string, unknown>) => ({
          path: f.path,
          additions: f.additions,
          deletions: f.deletions,
          status: f.status,
        })),
        error: data.error ?? null,
      };

      if (requestedSessionRef.current !== sessionId) {
        // Answered for a session nobody is looking at any more.
        return;
      }

      // Update cache
      cacheRef.current = {
        sessionId,
        stats: gitDiffStats,
        timestamp: now,
      };

      setStats(gitDiffStats);
    } catch (err) {
      if (requestedSessionRef.current !== sessionId) {
        return;
      }
      const message =
        err instanceof Error ? err.message : "Failed to fetch git diff stats";
      setError(message);
      setStats(null);
    } finally {
      if (requestedSessionRef.current === sessionId) {
        setIsLoading(false);
      }
    }
  }, [sessionId]);

  // Initial fetch and polling
  useEffect(() => {
    fetchStats();

    const interval = setInterval(() => {
      fetchStats();
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [fetchStats]);

  // Clear cache and stats when session changes
  useEffect(() => {
    if (cacheRef.current && cacheRef.current.sessionId !== sessionId) {
      cacheRef.current = null;
      setStats(null);
    }
  }, [sessionId]);

  const refresh = useCallback(async () => {
    await fetchStats(true);
  }, [fetchStats]);

  return {
    stats,
    isLoading,
    error,
    refresh,
  };
}
