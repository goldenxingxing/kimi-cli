import { useState, useEffect, useCallback, useRef } from "react";
import { login as apiLogin, logout as apiLogout, getMe } from "../lib/api/apis/AuthApi";
import type { UserInfo } from "../lib/api/apis/AuthApi";
import { onSessionExpired } from "../lib/session-expiry";

const CURRENT_USER_KEY = "kimi_current_user";

/** Don't re-ask the server on every tab switch; a session cannot lapse that fast. */
const REVALIDATE_INTERVAL_MS = 30_000;

function loadCachedUser(): UserInfo | null {
  try {
    const raw = localStorage.getItem(CURRENT_USER_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as UserInfo;
  } catch {
    return null;
  }
}

function saveUser(user: UserInfo | null): void {
  if (user) {
    localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
  } else {
    localStorage.removeItem(CURRENT_USER_KEY);
  }
}

type UseAuthReturn = {
  currentUser: UserInfo | null;
  isLoading: boolean;
  isAdmin: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
};

export function useAuth(): UseAuthReturn {
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(() => loadCachedUser());
  const [isLoading, setIsLoading] = useState(true);

  const lastCheckRef = useRef(0);

  // On mount, verify session with backend (use cache optimistically while verifying)
  useEffect(() => {
    let cancelled = false;

    lastCheckRef.current = Date.now();
    getMe()
      .then((user) => {
        if (cancelled) return;
        setCurrentUser(user);
        saveUser(user);
      })
      .catch(() => {
        if (cancelled) return;
        // Network error - keep cached user to avoid unnecessary logout
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // A session that lapsed while the tab sat in the background used to show up
  // as a workspace that quietly did nothing until someone reloaded. Re-check
  // when the tab comes back, and drop the user the moment the server says it
  // no longer knows them — the app renders the login page off `currentUser`.
  useEffect(() => {
    let cancelled = false;

    const clear = () => {
      if (cancelled) return;
      setCurrentUser(null);
      saveUser(null);
    };

    const revalidate = () => {
      if (document.visibilityState !== "visible") return;
      const now = Date.now();
      if (now - lastCheckRef.current < REVALIDATE_INTERVAL_MS) return;
      lastCheckRef.current = now;
      getMe()
        .then((user) => {
          if (cancelled) return;
          if (user) {
            setCurrentUser(user);
            saveUser(user);
          } else {
            clear();
          }
        })
        .catch(() => {
          // Offline or the server is restarting: keep the cached user rather
          // than logging someone out over a blip.
        });
    };

    const unsubscribe = onSessionExpired(clear);
    window.addEventListener("focus", revalidate);
    document.addEventListener("visibilitychange", revalidate);

    return () => {
      cancelled = true;
      unsubscribe();
      window.removeEventListener("focus", revalidate);
      document.removeEventListener("visibilitychange", revalidate);
    };
  }, []);

  const login = useCallback(async (username: string, password: string): Promise<void> => {
    const user = await apiLogin(username, password);
    setCurrentUser(user);
    saveUser(user);
  }, []);

  const logout = useCallback(async (): Promise<void> => {
    try {
      await apiLogout();
    } finally {
      setCurrentUser(null);
      saveUser(null);
    }
  }, []);

  const isAdmin = currentUser?.role === "admin";

  return { currentUser, isLoading, isAdmin, login, logout };
}
