/**
 * When a dropped session socket should be retried, and when the reader should
 * be told about it.
 *
 * A WebSocket `error` event carries no detail by design, so reporting one
 * verbatim tells the reader nothing they can act on. Worse, it usually fires
 * for blips the client recovers from within a few hundred milliseconds, and a
 * toast cannot be retracted once shown — so the interface ends up permanently
 * displaying an error about a connection that is currently fine.
 *
 * The rule here: recover silently, and only speak up once recovery has
 * actually failed. Silence alone would be worse than the noise it replaces,
 * which is why retrying and reporting are decided together.
 */

/** Codes the server uses to refuse a session outright. Retrying cannot help. */
export const TERMINAL_CLOSE_CODES: Record<number, string> = {
  4004: "sessionNotFound",
  4029: "tooManySessions",
  // The server has stated a reason a retry cannot change. Leaving these out
  // spent five reconnects to arrive at "lost connection", which describes a
  // flaky network — and sent people looking at the wrong thing entirely.
  4401: "unauthorized",
  4403: "forbidden",
};

/** A clean, intentional close. Nothing to recover from. */
const NORMAL_CLOSE_CODES = new Set([1000, 1005]);

export const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 300;
const MAX_RECONNECT_DELAY_MS = 5_000;

export type CloseDecision =
  | { action: "ignore" }
  | { action: "report"; reason: string }
  | { action: "retry"; delayMs: number; attempt: number };

/**
 * Decide what to do about a socket that just closed.
 *
 * @param code       the WebSocket close code
 * @param attempt    how many consecutive reconnects have already been tried
 * @param intentional whether the client asked for this close
 */
export function decideOnClose(
  code: number,
  attempt: number,
  intentional: boolean,
): CloseDecision {
  if (intentional || NORMAL_CLOSE_CODES.has(code)) {
    return { action: "ignore" };
  }
  const terminal = TERMINAL_CLOSE_CODES[code];
  if (terminal) {
    // The server has told us why; retrying would only repeat it.
    return { action: "report", reason: terminal };
  }
  if (attempt >= MAX_RECONNECT_ATTEMPTS) {
    return { action: "report", reason: "unreachable" };
  }
  return {
    action: "retry",
    attempt: attempt + 1,
    delayMs: reconnectDelayMs(attempt),
  };
}

/** Exponential backoff, capped, so a server that is down is not hammered. */
export function reconnectDelayMs(attempt: number): number {
  const delay = BASE_RECONNECT_DELAY_MS * 2 ** attempt;
  return Math.min(delay, MAX_RECONNECT_DELAY_MS);
}

/**
 * Whether a raw `error` event should reach the reader.
 *
 * Never on its own: the event is opaque, and the close that follows carries
 * the code that actually decides. Reporting here is what produced an error
 * toast for every recoverable blip.
 */
export function shouldReportErrorEvent(): boolean {
  return false;
}
