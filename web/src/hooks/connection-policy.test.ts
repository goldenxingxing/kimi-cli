import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  MAX_RECONNECT_ATTEMPTS,
  decideOnClose,
  reconnectDelayMs,
  shouldReportErrorEvent,
} from "./connection-policy.ts";

describe("decideOnClose", () => {
  it("ignores a close the client asked for", () => {
    assert.deepEqual(decideOnClose(1006, 0, true), { action: "ignore" });
  });

  it("ignores a clean close", () => {
    assert.deepEqual(decideOnClose(1000, 0, false), { action: "ignore" });
    assert.deepEqual(decideOnClose(1005, 0, false), { action: "ignore" });
  });

  it("retries an abnormal close instead of reporting it", () => {
    // 1006 is what a dropped connection looks like, and it is exactly the case
    // that used to produce an error toast for a blip the client recovered from.
    const decision = decideOnClose(1006, 0, false);

    assert.equal(decision.action, "retry");
    assert.equal(decision.action === "retry" && decision.attempt, 1);
  });

  it("keeps retrying up to the cap", () => {
    for (let attempt = 0; attempt < MAX_RECONNECT_ATTEMPTS; attempt++) {
      assert.equal(decideOnClose(1006, attempt, false).action, "retry");
    }
  });

  it("reports only once recovery has actually failed", () => {
    const decision = decideOnClose(1006, MAX_RECONNECT_ATTEMPTS, false);

    assert.deepEqual(decision, { action: "report", reason: "unreachable" });
  });

  it("reports a refusal immediately rather than retrying it", () => {
    // The server has already said why; retrying would only repeat it.
    assert.deepEqual(decideOnClose(4004, 0, false), {
      action: "report",
      reason: "sessionNotFound",
    });
    assert.deepEqual(decideOnClose(4029, 0, false), {
      action: "report",
      reason: "tooManySessions",
    });
  });

  it("does not let a refusal be retried away by a high attempt count", () => {
    assert.equal(decideOnClose(4004, MAX_RECONNECT_ATTEMPTS + 10, false).action, "report");
  });
});

describe("reconnectDelayMs", () => {
  it("backs off exponentially", () => {
    assert.ok(reconnectDelayMs(1) > reconnectDelayMs(0));
    assert.ok(reconnectDelayMs(2) > reconnectDelayMs(1));
  });

  it("starts fast enough to be invisible for a blip", () => {
    assert.ok(reconnectDelayMs(0) <= 500);
  });

  it("caps so a downed server is not hammered", () => {
    assert.equal(reconnectDelayMs(50), reconnectDelayMs(10));
    assert.ok(reconnectDelayMs(50) <= 5_000);
  });
});

describe("shouldReportErrorEvent", () => {
  it("never reports the raw error event", () => {
    // It carries no detail by design; the close that follows decides.
    assert.equal(shouldReportErrorEvent(), false);
  });
});

describe("the blip that produced the complaint", () => {
  it("recovers silently and says nothing", () => {
    const reports: string[] = [];
    let attempt = 0;

    // Socket drops...
    assert.equal(shouldReportErrorEvent(), false, "error event must stay quiet");
    const first = decideOnClose(1006, attempt, false);
    assert.equal(first.action, "retry");
    attempt = first.action === "retry" ? first.attempt : attempt;

    // ...and the retry succeeds, which resets the counter.
    attempt = 0;

    assert.deepEqual(reports, []);
    assert.equal(attempt, 0);
  });

  it("speaks up when the server really is gone", () => {
    let attempt = 0;
    let reported: string | null = null;

    for (let i = 0; i < 20 && reported === null; i++) {
      const decision = decideOnClose(1006, attempt, false);
      if (decision.action === "retry") {
        attempt = decision.attempt;
      } else if (decision.action === "report") {
        reported = decision.reason;
      }
    }

    assert.equal(reported, "unreachable");
    assert.equal(attempt, MAX_RECONNECT_ATTEMPTS);
  });
});
