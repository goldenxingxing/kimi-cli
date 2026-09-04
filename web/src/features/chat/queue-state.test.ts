import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  type QueueState,
  type QueuedItem,
  forgetSession,
  initialQueueState,
  selectSession,
  withActiveQueue,
} from "./queue-state.ts";

const item = (id: string): QueuedItem => ({ id, text: `message ${id}` });

const queued = (sessionId: string, ...ids: string[]): QueueState =>
  withActiveQueue(selectSession(initialQueueState, sessionId), () => ids.map(item));

describe("per-session message queue", () => {
  it("leaves a queue waiting in the session it was typed for", () => {
    const inA = queued("a", "1", "2");

    const inB = selectSession(inA, "b");
    assert.deepEqual(inB.queue, []);

    const backToA = selectSession(inB, "a");
    assert.deepEqual(
      backToA.queue.map((q) => q.id),
      ["1", "2"],
    );
  });

  it("edits only the active session's queue", () => {
    const inA = queued("a", "1");
    const inB = withActiveQueue(selectSession(inA, "b"), (queue) => [...queue, item("2")]);

    assert.deepEqual(
      selectSession(inB, "a").queue.map((q) => q.id),
      ["1"],
    );
    assert.deepEqual(
      selectSession(inB, "b").queue.map((q) => q.id),
      ["2"],
    );
  });

  it("drops the entry once a queue is emptied", () => {
    const drained = withActiveQueue(queued("a", "1"), (queue) => queue.slice(1));

    assert.deepEqual(drained.queue, []);
    assert.equal("a" in drained.queues, false);
  });

  it("forgets a deleted session's queue without disturbing the open one", () => {
    const inB = selectSession(queued("a", "1"), "b");
    const withB = withActiveQueue(inB, () => [item("2")]);

    const afterDelete = forgetSession(withB, "a");

    assert.equal("a" in afterDelete.queues, false);
    assert.deepEqual(
      afterDelete.queue.map((q) => q.id),
      ["2"],
    );
  });

  it("re-selecting the open session changes nothing", () => {
    const inA = queued("a", "1");

    assert.equal(selectSession(inA, "a"), inA);
  });
});
