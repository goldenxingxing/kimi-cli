/**
 * One place to say "the server no longer knows who you are".
 *
 * Both halves of the app can find this out independently: an HTTP call comes
 * back 401, or the session socket closes with 4401. Before this, each half
 * logged its own message and left the page sitting there — the only way back
 * was for the reader to guess that a reload would show a login form. Now
 * either signal routes to the same place: `useAuth` drops the cached user and
 * the app renders the login page on the spot.
 */

const EVENT = "kimi:session-expired";

/** Announce that the current session is no longer valid. Safe to call twice. */
export function notifySessionExpired(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(EVENT));
}

/**
 * Notice a 401 wherever it happens, without touching a dozen call sites.
 *
 * API calls in this app go through at least four different helpers — the
 * generated client, three hand-written `fetch` wrappers, and a pile of inline
 * `fetch` calls in hooks. Hooking each one is churn that drifts out of date
 * the next time someone adds a fifth. One wrapper around `fetch` catches all
 * of them and cannot be forgotten.
 *
 * `/api/auth/*` is left alone: a 401 from the login route means "wrong
 * password", and the caller already renders that.
 */
export function installSessionExpiryInterceptor(): void {
  if (installed || typeof window === "undefined" || typeof window.fetch !== "function") {
    return;
  }
  installed = true;
  const original = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const response = await original(input, init);
    if (response.status === 401 && isOwnApiCall(input)) {
      notifySessionExpired();
    }
    return response;
  };
}

let installed = false;

function isOwnApiCall(input: RequestInfo | URL): boolean {
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  try {
    const url = new URL(raw, window.location.href);
    if (url.origin !== window.location.origin) return false;
    return url.pathname.includes("/api/") && !url.pathname.includes("/api/auth/");
  } catch {
    return false;
  }
}

/** Subscribe to expiry. Returns an unsubscribe function. */
export function onSessionExpired(handler: () => void): () => void {
  if (typeof window === "undefined") {
    return () => {
      /* nothing was subscribed */
    };
  }
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}
