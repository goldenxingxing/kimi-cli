import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  nextExpandAll,
  readExpandedCategories,
  toggleCategory,
  writeExpandedCategories,
} from "./skill-section-state.ts";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
    read: () => data,
  };
}

const KEY = "kimi.admin.skills.expandedCategories";

describe("remembering which sections are open", () => {
  it("starts with everything closed", () => {
    assert.deepEqual([...readExpandedCategories(fakeStorage())], []);
  });

  it("round-trips through storage", () => {
    const storage = fakeStorage();
    writeExpandedCategories(new Set(["engineering", "data"]), storage);

    assert.deepEqual(
      [...readExpandedCategories(storage)].sort(),
      ["data", "engineering"],
    );
  });

  it("treats an unreadable value as nothing open rather than throwing", () => {
    // A half-written entry, or something from a future version of the panel.
    assert.deepEqual([...readExpandedCategories(fakeStorage({ [KEY]: "{oops" }))], []);
    assert.deepEqual([...readExpandedCategories(fakeStorage({ [KEY]: '"a"' }))], []);
    assert.deepEqual(
      [...readExpandedCategories(fakeStorage({ [KEY]: '["ok", 7, null]' }))],
      ["ok"],
    );
  });

  it("does not fail a render when storage is unavailable", () => {
    assert.deepEqual([...readExpandedCategories(undefined)], []);
    writeExpandedCategories(new Set(["x"]), undefined); // must not throw
  });

  it("does not fail when storage refuses to write", () => {
    const refusing = {
      setItem: () => {
        throw new Error("quota exceeded");
      },
    };
    writeExpandedCategories(new Set(["x"]), refusing);
  });
});

describe("toggleCategory", () => {
  it("opens what is closed and closes what is open", () => {
    const opened = toggleCategory(new Set(), "data");
    assert.deepEqual([...opened], ["data"]);
    assert.deepEqual([...toggleCategory(opened, "data")], []);
  });

  it("does not mutate the set it was given", () => {
    const current = new Set(["data"]);
    toggleCategory(current, "engineering");
    assert.deepEqual([...current], ["data"]);
  });
});

describe("nextExpandAll", () => {
  const ids = ["engineering", "data", "design"];

  it("opens everything while anything is still closed", () => {
    const { expanded, action } = nextExpandAll(new Set(["data"]), ids);
    assert.equal(action, "expand");
    assert.deepEqual([...expanded].sort(), [...ids].sort());
  });

  it("closes everything only once all of it is open", () => {
    const { expanded, action } = nextExpandAll(new Set(ids), ids);
    assert.equal(action, "collapse");
    assert.deepEqual([...expanded], []);
  });

  it("offers to expand when there is nothing to expand yet", () => {
    // No categories rendered (an empty install): the control must not claim
    // everything is already open.
    assert.equal(nextExpandAll(new Set(), []).action, "expand");
  });
});
