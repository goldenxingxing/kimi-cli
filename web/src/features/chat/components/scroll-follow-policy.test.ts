import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  FOLLOW_GAP_TOLERANCE,
  decideFollowOutput,
  nextCatchUp,
  type FollowOutputState,
} from "./scroll-follow-policy.ts";

function state(overrides: Partial<FollowOutputState> = {}): FollowOutputState {
  return {
    isAtBottom: false,
    isReplayingHistory: false,
    isCatchingUp: false,
    gapToBottom: null,
    ...overrides,
  };
}

describe("decideFollowOutput", () => {
  it("follows while history is replaying, however far the content has grown", () => {
    const decision = decideFollowOutput(
      state({ isReplayingHistory: true, gapToBottom: 500_000 }),
    );

    assert.equal(decision, "auto");
  });

  it("follows while catching up after a rebuild, however far the content has grown", () => {
    const decision = decideFollowOutput(
      state({ isCatchingUp: true, gapToBottom: 500_000 }),
    );

    assert.equal(decision, "auto");
  });

  it("follows when the viewport is already at the bottom", () => {
    assert.equal(decideFollowOutput(state({ isAtBottom: true })), "auto");
  });

  it("follows when the reader is near enough to the bottom", () => {
    assert.equal(
      decideFollowOutput(state({ gapToBottom: FOLLOW_GAP_TOLERANCE })),
      "auto",
    );
  });

  it("leaves the reader alone once they are well above the bottom", () => {
    assert.equal(
      decideFollowOutput(state({ gapToBottom: FOLLOW_GAP_TOLERANCE + 1 })),
      false,
    );
  });

  it("leaves the reader alone when the gap cannot be measured", () => {
    assert.equal(decideFollowOutput(state({ gapToBottom: null })), false);
  });
});

describe("nextCatchUp", () => {
  it("starts catching up whenever the list is rebuilt from empty", () => {
    assert.equal(nextCatchUp(false, { type: "list-rebuilt" }), true);
  });

  it("keeps catching up when the viewport merely reaches the bottom", () => {
    // The regression: early in a replay only a few messages exist, so the
    // viewport is trivially at the bottom. Ending catch-up there strands the
    // reader at the top once the rest of the history arrives.
    assert.equal(nextCatchUp(true, { type: "reached-bottom" }), true);
  });

  it("stops catching up only when the reader takes control", () => {
    assert.equal(nextCatchUp(true, { type: "reader-took-control" }), false);
  });

  it("does not resume catching up just because the reader returns to the bottom", () => {
    assert.equal(nextCatchUp(false, { type: "reached-bottom" }), false);
  });
});

describe("the reconnect replay sequence", () => {
  it("keeps the newest message in view for the whole replay", () => {
    // Reconnect empties the list, then history streams back in chunks.
    let catchingUp = nextCatchUp(false, { type: "list-rebuilt" });

    // A couple of messages land: the viewport is at the bottom trivially.
    catchingUp = nextCatchUp(catchingUp, { type: "reached-bottom" });
    assert.equal(catchingUp, true);

    // Hundreds more arrive, pushing the content far beyond the tolerance.
    const decision = decideFollowOutput(
      state({ isCatchingUp: catchingUp, gapToBottom: 42_000 }),
    );

    assert.equal(decision, "auto", "must not strand the reader at the top");
  });

  it("hands control back as soon as the reader scrolls away", () => {
    let catchingUp = nextCatchUp(false, { type: "list-rebuilt" });
    catchingUp = nextCatchUp(catchingUp, { type: "reader-took-control" });

    const decision = decideFollowOutput(
      state({ isCatchingUp: catchingUp, gapToBottom: 42_000 }),
    );

    assert.equal(decision, false, "the reader's position is theirs to keep");
  });
});
