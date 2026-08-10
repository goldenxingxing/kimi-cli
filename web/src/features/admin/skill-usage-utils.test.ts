import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type {
  ManagedSkill,
  SkillUsageEntry,
  SkillUsageReport,
} from "../../lib/api/apis/AdminSkillsApi.ts";
import {
  bulkTargets,
  formatCount,
  indexUsage,
  sortSkills,
  sparkBars,
} from "./skill-usage-utils.ts";

function entry(partial: Partial<SkillUsageEntry>): SkillUsageEntry {
  return {
    name: "alpha",
    key: "alpha",
    installed: true,
    source: "builtin",
    count: 0,
    by_origin: {},
    read_count: 0,
    slash_count: 0,
    flow_count: 0,
    error_count: 0,
    resource_read_count: 0,
    session_count: 0,
    user_count: 0,
    first_used: null,
    last_used: null,
    daily: [],
    paths: [],
    ...partial,
  };
}

function report(skills: SkillUsageEntry[]): SkillUsageReport {
  return {
    generated_at: 0,
    window_days: 30,
    scanned: {
      sessions: 0,
      sessions_read: 0,
      duration_ms: 0,
      truncated: false,
      cached: false,
    },
    totals: {
      invocations: 0,
      read_invocations: 0,
      slash_invocations: 0,
      unmatched_slash_invocations: 0,
      distinct_skills: skills.length,
      distinct_sessions: 0,
      distinct_users: 0,
      by_origin: {},
    },
    dates: [],
    daily: [],
    skills,
    top_users: [],
  };
}

function skill(name: string): ManagedSkill {
  return {
    name,
    description: "",
    origin: "builtin",
    enabled: true,
    deleted: false,
    modified: false,
    files: [],
  };
}

describe("indexUsage", () => {
  it("returns an empty map for a null report", () => {
    assert.equal(indexUsage(null).size, 0);
  });

  it("indexes case-insensitively so cards join regardless of casing", () => {
    const index = indexUsage(report([entry({ name: "DOCX Pro", count: 3 })]));
    assert.equal(index.get("docx pro")?.count, 3);
  });
});

describe("sparkBars", () => {
  it("returns nothing for an empty series", () => {
    assert.deepEqual(sparkBars([], 100, 10), []);
  });

  it("does not divide by zero on an all-zero series", () => {
    const bars = sparkBars([0, 0, 0], 90, 10);
    assert.equal(bars.length, 3);
    for (const bar of bars) {
      assert.ok(Number.isFinite(bar.height), "height must be finite");
      assert.ok(bar.height <= 10);
      assert.ok(bar.width > 0);
    }
  });

  it("scales the tallest bar to the full height", () => {
    const bars = sparkBars([1, 4], 100, 20);
    assert.equal(bars[1].height, 20);
    assert.equal(bars[1].y, 0);
    assert.ok(bars[0].height < bars[1].height);
  });

  it("handles a single data point", () => {
    const bars = sparkBars([7], 50, 10);
    assert.equal(bars.length, 1);
    assert.equal(bars[0].height, 10);
  });

  it("keeps a visible baseline for empty days between busy ones", () => {
    const bars = sparkBars([5, 0], 100, 20);
    assert.ok(bars[1].height > 0, "zero days still need a baseline");
    assert.ok(bars[1].height < bars[0].height);
  });
});

describe("sortSkills", () => {
  const skills = [skill("beta"), skill("alpha"), skill("gamma")];
  const usage = indexUsage(
    report([
      entry({ name: "alpha", count: 1, last_used: 300 }),
      entry({ name: "beta", count: 9, last_used: 100 }),
    ]),
  );

  it("sorts by name", () => {
    assert.deepEqual(
      sortSkills(skills, usage, "name").map((s) => s.name),
      ["alpha", "beta", "gamma"],
    );
  });

  it("sorts by count, tie-breaking by name", () => {
    assert.deepEqual(
      sortSkills(skills, usage, "most-used").map((s) => s.name),
      ["beta", "alpha", "gamma"],
    );
  });

  it("sorts by recency, treating unused skills as oldest", () => {
    assert.deepEqual(
      sortSkills(skills, usage, "recent").map((s) => s.name),
      ["alpha", "beta", "gamma"],
    );
  });

  it("sorts by category, most-used first inside each section", () => {
    const categorized = [
      skill("sql-insight"),
      skill("k8s-cluster-ops"),
      skill("secure-code-review"),
      skill("using-git-worktrees"),
    ];
    const counts = indexUsage(
      report([
        entry({ name: "using-git-worktrees", count: 5 }),
        entry({ name: "secure-code-review", count: 2 }),
      ]),
    );
    assert.deepEqual(
      sortSkills(categorized, counts, "category").map((s) => s.name),
      // Engineering, then DevOps & Infra, then Data & Analytics.
      ["using-git-worktrees", "secure-code-review", "k8s-cluster-ops", "sql-insight"],
    );
  });

  it("does not mutate the input array", () => {
    const original = [...skills];
    sortSkills(skills, usage, "most-used");
    assert.deepEqual(skills, original);
  });

  it("tolerates skills with no usage entry", () => {
    const result = sortSkills([skill("zeta")], indexUsage(null), "most-used");
    assert.deepEqual(
      result.map((s) => s.name),
      ["zeta"],
    );
  });
});

describe("bulkTargets", () => {
  const on = skill("on");
  const off = { ...skill("off"), enabled: false };
  const gone = { ...skill("gone"), enabled: false, deleted: true };
  const all = [on, off, gone];

  it("enables only what is disabled, never what is deleted", () => {
    assert.deepEqual(
      bulkTargets(all, "enable").map((s) => s.name),
      ["off"],
    );
  });

  it("disables only what is currently enabled", () => {
    assert.deepEqual(
      bulkTargets(all, "disable").map((s) => s.name),
      ["on"],
    );
  });

  it("deletes everything not already deleted", () => {
    assert.deepEqual(
      bulkTargets(all, "delete").map((s) => s.name),
      ["on", "off"],
    );
  });

  it("returns nothing when the action is a no-op", () => {
    assert.deepEqual(bulkTargets([gone], "delete"), []);
    assert.deepEqual(bulkTargets([on], "enable"), []);
  });
});

describe("formatCount", () => {
  it("compacts large numbers", () => {
    assert.equal(formatCount(1234), "1.2K");
    assert.equal(formatCount(7), "7");
  });
});
