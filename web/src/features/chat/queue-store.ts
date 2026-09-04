import { create } from "zustand";

import {
  type QueueState,
  type QueuedItem,
  forgetSession,
  initialQueueState,
  selectSession,
  withActiveQueue,
} from "./queue-state";

export type { QueuedItem };

type QueueStore = QueueState & {
  /** Switch which session's queue every other method acts on. */
  setActiveSession: (sessionId: string) => void;
  enqueue: (text: string) => void;
  removeFromQueue: (id: string) => void;
  editQueueItem: (id: string, text: string) => void;
  moveQueueItemUp: (id: string) => void;
  dequeue: () => QueuedItem | undefined;
  clearQueue: () => void;
  forgetSessionQueue: (sessionId: string) => void;
};

export const useQueueStore = create<QueueStore>((set, get) => ({
  ...initialQueueState,
  setActiveSession: (sessionId) => set((s) => selectSession(s, sessionId)),
  enqueue: (text) =>
    set((s) =>
      withActiveQueue(s, (queue) => [...queue, { id: crypto.randomUUID(), text }]),
    ),
  removeFromQueue: (id) =>
    set((s) => withActiveQueue(s, (queue) => queue.filter((q) => q.id !== id))),
  editQueueItem: (id, text) =>
    set((s) =>
      withActiveQueue(s, (queue) =>
        queue.map((q) => (q.id === id ? { ...q, text } : q)),
      ),
    ),
  moveQueueItemUp: (id) =>
    set((s) =>
      withActiveQueue(s, (queue) => {
        const idx = queue.findIndex((q) => q.id === id);
        if (idx <= 0) return queue;
        const next = [...queue];
        [next[idx - 1], next[idx]] = [next[idx], next[idx - 1]];
        return next;
      }),
    ),
  dequeue: () => {
    const [first] = get().queue;
    if (!first) return undefined;
    set((s) => withActiveQueue(s, (queue) => queue.slice(1)));
    return first;
  },
  clearQueue: () => set((s) => withActiveQueue(s, () => [])),
  forgetSessionQueue: (sessionId) => set((s) => forgetSession(s, sessionId)),
}));
