import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { startsIncomingConversation } from "./conversation-reset.ts";

describe("deferring a reconnect's conversation reset", () => {
  it("holds the old conversation while only bookkeeping has arrived", () => {
    assert.equal(
      startsIncomingConversation({ method: "session_status" }, "init-1"),
      false,
    );
    assert.equal(startsIncomingConversation({ id: "init-1" }, "init-1"), false);
  });

  it("clears once replayed content starts arriving", () => {
    assert.equal(startsIncomingConversation({ method: "event" }, "init-1"), true);
  });

  it("clears on history_complete, so an emptied session does empty", () => {
    assert.equal(
      startsIncomingConversation({ method: "history_complete" }, "init-1"),
      true,
    );
  });

  it("does not mistake another response for the initialize reply", () => {
    assert.equal(startsIncomingConversation({ id: "other" }, "init-1"), true);
    assert.equal(startsIncomingConversation({ id: "init-1" }, null), true);
  });
});
