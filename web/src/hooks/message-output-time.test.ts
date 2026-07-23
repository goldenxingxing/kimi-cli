import assert from "node:assert/strict";
import test from "node:test";
import { setMessageOutputTime } from "./message-output-time.ts";
import type { LiveMessage } from "./types.ts";

test("sets completion time only on the selected assistant text message", () => {
  const messages: LiveMessage[] = [
    {
      id: "a1",
      role: "assistant",
      variant: "text",
      content: "hi",
    },
    {
      id: "tool",
      role: "assistant",
      variant: "tool",
    },
  ];

  const updated = setMessageOutputTime(messages, "a1", 1_721_234_567_000);

  assert.equal(updated[0]?.completedAt, 1_721_234_567_000);
  assert.equal(updated[1]?.completedAt, undefined);
  assert.notEqual(updated, messages);
});

test("leaves messages unchanged when timestamp or id is missing", () => {
  const messages: LiveMessage[] = [
    {
      id: "a1",
      role: "assistant",
      variant: "text",
    },
  ];

  assert.equal(setMessageOutputTime(messages, null, 1000), messages);
  assert.equal(setMessageOutputTime(messages, "a1", undefined), messages);
});
