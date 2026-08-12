import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { summarizeApproval } from "./approval-summary.ts";

describe("summarizeApproval", () => {
  it("summarizes a shell request by its first meaningful line", () => {
    const summary = summarizeApproval({
      action: "run command",
      display: [
        { type: "shell", language: "bash", command: "\n  git push origin main\n" },
      ],
    });
    assert.deepEqual(summary, {
      kind: "command",
      command: "git push origin main",
      extraLines: 0,
    });
  });

  it("counts the remaining lines of a multi-line command", () => {
    const summary = summarizeApproval({
      display: [
        { type: "shell", command: "cd build\nmake -j8\n\nmake install\n" },
      ],
    });
    assert.deepEqual(summary, {
      kind: "command",
      command: "cd build",
      // The blank line is not hidden content and must not be advertised.
      extraLines: 2,
    });
  });

  it("summarizes an edit by path and size, aggregating multiple diffs", () => {
    const summary = summarizeApproval({
      display: [
        { type: "diff", path: "src/a.ts", old_text: "a\nb\n", new_text: "a\nb\nc\n" },
        { type: "diff", path: "src/b.ts", old_text: "x\n", new_text: "" },
      ],
    });
    assert.deepEqual(summary, {
      kind: "diff",
      path: "src/a.ts",
      files: 2,
      before: 3,
      after: 3,
    });
  });

  it("prefers the edit over a command when a request carries both", () => {
    const summary = summarizeApproval({
      display: [
        { type: "shell", command: "true" },
        { type: "diff", path: "p.ts", old_text: "", new_text: "a\n" },
      ],
    });
    assert.equal(summary?.kind, "diff");
  });

  it("reads blocks whose payload is nested under data", () => {
    const summary = summarizeApproval({
      display: [{ type: "shell", data: { command: "rm -rf build" } }],
    });
    assert.deepEqual(summary, {
      kind: "command",
      command: "rm -rf build",
      extraLines: 0,
    });
  });

  it("summarizes wiki and todo blocks by their counts", () => {
    assert.deepEqual(
      summarizeApproval({
        display: [
          { type: "wiki", summary: "Record 2 findings", pages: ["a", "b"] },
        ],
      }),
      { kind: "wiki", pages: 2, summary: "Record 2 findings" },
    );
    assert.deepEqual(
      summarizeApproval({
        display: [
          {
            type: "todo",
            items: [{ status: "completed" }, { status: "pending" }],
          },
        ],
      }),
      { kind: "todo", total: 2, done: 1 },
    );
  });

  it("falls back to the description, then to the action", () => {
    assert.deepEqual(
      summarizeApproval({
        action: "write file",
        sender: "Write",
        description: "Write to /tmp/x\nwith 40 lines",
      }),
      { kind: "generic", text: "Write to /tmp/x" },
    );
    assert.deepEqual(
      summarizeApproval({ action: "write file", sender: "Write" }),
      { kind: "generic", text: "Write · write file" },
    );
  });

  it("returns nothing when there is nothing to say", () => {
    assert.equal(summarizeApproval({}), null);
    assert.equal(summarizeApproval({ description: "  \n\n" }), null);
  });

  it("ignores a display block with no usable content", () => {
    assert.equal(
      summarizeApproval({ display: [{ type: "shell", command: "   " }] }),
      null,
    );
  });
});
