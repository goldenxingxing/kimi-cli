/**
 * Queued messages, kept per session.
 *
 * The queue sits in the composer's toolbar, so it belongs to the conversation
 * it was typed for. Holding one global list meant a session switch had to throw
 * the queue away to avoid sending it to the wrong session; keyed by session,
 * each queue simply waits where it was left.
 */

export interface QueuedItem {
  id: string;
  text: string;
}

export type QueuesBySession = Record<string, QueuedItem[]>;

export type QueueState = {
  activeSessionId: string;
  queues: QueuesBySession;
  /** The active session's queue, kept in sync so consumers need no session id. */
  queue: QueuedItem[];
};

export const NO_SESSION = "";

export const initialQueueState: QueueState = {
  activeSessionId: NO_SESSION,
  queues: {},
  queue: [],
};

const queueFor = (queues: QueuesBySession, sessionId: string): QueuedItem[] =>
  queues[sessionId] ?? [];

/** Point the store at another session, leaving every queue where it is. */
export function selectSession(state: QueueState, sessionId: string): QueueState {
  if (sessionId === state.activeSessionId) {
    return state;
  }
  return {
    activeSessionId: sessionId,
    queues: state.queues,
    queue: queueFor(state.queues, sessionId),
  };
}

/** Replace the active session's queue, dropping the entry once it empties. */
export function withActiveQueue(
  state: QueueState,
  next: (current: QueuedItem[]) => QueuedItem[],
): QueueState {
  const queue = next(queueFor(state.queues, state.activeSessionId));
  const queues = { ...state.queues };
  if (queue.length === 0) {
    delete queues[state.activeSessionId];
  } else {
    queues[state.activeSessionId] = queue;
  }
  return { activeSessionId: state.activeSessionId, queues, queue };
}

/** Forget one session's queue entirely, e.g. once it is deleted. */
export function forgetSession(state: QueueState, sessionId: string): QueueState {
  if (!(sessionId in state.queues)) {
    return state;
  }
  const queues = { ...state.queues };
  delete queues[sessionId];
  return {
    activeSessionId: state.activeSessionId,
    queues,
    queue: queueFor(queues, state.activeSessionId),
  };
}
