import assert from "node:assert/strict";
import test from "node:test";
import { reconcileApprovalRequestIds } from "./approval-snapshot.ts";

test("restores server approvals and identifies stale local approvals", () => {
  const result = reconcileApprovalRequestIds(
    new Set(["stale", "confirmed"]),
    new Set(["confirmed", "lost"]),
  );

  assert.deepEqual([...result.confirmed].sort(), ["confirmed", "lost"]);
  assert.deepEqual([...result.stale], ["stale"]);
});
