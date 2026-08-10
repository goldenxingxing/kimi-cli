import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ManagedSkill } from "../../lib/api/apis/AdminSkillsApi.ts";
import {
  SKILL_CATEGORIES,
  categorizeSkill,
  categoryLabel,
  categoryRank,
  groupByCategory,
  resolveDeclaredCategory,
} from "./skill-categories.ts";

function skill(
  name: string,
  description = "",
  category: string | null = null,
): ManagedSkill {
  return {
    name,
    description,
    category,
    origin: "builtin",
    enabled: true,
    deleted: false,
    modified: false,
    files: [],
  };
}

describe("categorizeSkill", () => {
  it("classifies from the name across the bundled skill library", () => {
    const cases: [string, string][] = [
      ["secure-code-review", "engineering"],
      ["using-git-worktrees", "engineering"],
      ["k8s-cluster-ops", "devops"],
      ["terraform-deploy-traps", "devops"],
      ["sql-insight", "data"],
      ["dataset-health-audit", "data"],
      ["music-to-video", "design"],
      ["hyperframes-cli", "design"],
      ["idea-to-prd", "product"],
      ["okr-strategist", "product"],
      ["seo-content-writer", "marketing"],
      ["ecom-listing-copywriter", "marketing"],
      ["discounted-cashflow-model", "finance"],
      ["equity-research-report", "finance"],
      ["academic-paper-reviewer", "research"],
      ["competitor-analysis", "research"],
      ["xindaya-translator", "writing"],
      ["pro-email-composer", "comms"],
      ["structured-minutes", "comms"],
      ["docx-media-aware", "docs"],
      ["tos-clause-scanner", "legal"],
      ["nmpa-medical-device-registration", "legal"],
      ["mock-interview-drill", "learning"],
      ["sql-tutor", "learning"],
      ["skill-creator", "agent"],
      ["kimi-help-center", "agent"],
    ];
    for (const [name, expected] of cases) {
      assert.equal(categorizeSkill(skill(name)), expected, name);
    }
  });

  it("resolves overlapping vocabularies by rule precedence", () => {
    // "research" and "writer" both appear, but the finance rule runs first.
    assert.equal(categorizeSkill(skill("equity-researcher")), "finance");
    // "seo" wins over "content-writer".
    assert.equal(categorizeSkill(skill("seo-copywriting-guide")), "marketing");
  });

  it("falls back to the description when the name says nothing", () => {
    assert.equal(
      categorizeSkill(skill("acme-helper", "Reviews Terraform plans before deploy")),
      "devops",
    );
  });

  it("prefers the name over an incidental mention in the description", () => {
    assert.equal(
      categorizeSkill(
        skill("k8s-cluster-ops", "Writes a report about the cluster for the team"),
      ),
      "devops",
    );
  });

  it("does not read 'how-tos' as a terms-of-service skill", () => {
    assert.equal(
      categorizeSkill(skill("faceless-explainer", "Topic explainers and how-tos")),
      "design",
    );
  });

  it("does not read 'ad hoc' as advertising", () => {
    assert.notEqual(
      categorizeSkill(skill("ad-hoc-fixture", "Runs ad hoc checks")),
      "marketing",
    );
  });

  it("classifies Chinese-only descriptions", () => {
    assert.equal(categorizeSkill(skill("chrono-flow", "生成交互式时间线HTML页面")), "design");
  });

  it("falls back to 'other' when nothing matches", () => {
    assert.equal(categorizeSkill(skill("zzz", "")), "other");
  });
});

describe("declared categories", () => {
  it("beats the name-based guess", () => {
    // Would be "engineering" on its name alone.
    assert.equal(
      categorizeSkill(skill("code-to-diagram", "", "design")),
      "design",
    );
  });

  it("accepts ids, labels, and common aliases, case- and spacing-insensitively", () => {
    assert.equal(resolveDeclaredCategory("devops"), "devops");
    assert.equal(resolveDeclaredCategory("DevOps & Infra"), "devops");
    assert.equal(resolveDeclaredCategory("Data & Analytics"), "data");
    assert.equal(resolveDeclaredCategory("  Education "), "learning");
    assert.equal(resolveDeclaredCategory("data_science"), "data");
  });

  it("keeps an unrecognized category as a section of its own", () => {
    const id = resolveDeclaredCategory("Customer Ops");
    assert.equal(id, "custom:customer-ops");
    assert.equal(categoryLabel(id as string), "Customer Ops");
    // Above the catch-all, below every curated section.
    assert.ok(categoryRank(id as string) < categoryRank("other"));
    assert.ok(categoryRank(id as string) > categoryRank("agent"));
  });

  it("ignores an empty declaration and falls back to the guess", () => {
    assert.equal(resolveDeclaredCategory("   "), null);
    assert.equal(categorizeSkill(skill("k8s-cluster-ops", "", "  ")), "devops");
  });
});

describe("categoryRank", () => {
  it("keeps 'other' last", () => {
    const others = categoryRank("other");
    for (const category of SKILL_CATEGORIES) {
      if (category.id !== "other") assert.ok(categoryRank(category.id) < others);
    }
  });
});

describe("groupByCategory", () => {
  it("slices a sorted list into labelled runs", () => {
    const groups = groupByCategory([
      skill("secure-code-review"),
      skill("using-git-worktrees"),
      skill("k8s-cluster-ops"),
    ]);
    assert.deepEqual(
      groups.map((group) => [group.id, group.skills.length]),
      [
        ["engineering", 2],
        ["devops", 1],
      ],
    );
    assert.equal(groups[0].label, "Engineering");
  });

  it("returns nothing for an empty list", () => {
    assert.deepEqual(groupByCategory([]), []);
  });
});
