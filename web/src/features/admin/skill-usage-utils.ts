import type {
  ManagedSkill,
  SkillUsageEntry,
  SkillUsageReport,
} from "../../lib/api/apis/AdminSkillsApi";
// Explicit extension: this module is exercised by `node --test
// --experimental-strip-types`, whose ESM resolver does not guess one.
import { categorizeSkill, categoryRank } from "./skill-categories.ts";

export type SkillSortMode = "most-used" | "recent" | "name" | "category";

/**
 * Index usage entries by lowercased skill name.
 *
 * The backend already resolves directory-name vs frontmatter-name aliasing, so
 * `SkillUsageEntry.name` matches `ManagedSkill.name` for installed skills and a
 * plain case-insensitive compare is enough here.
 */
export function indexUsage(
  report: SkillUsageReport | null,
): Map<string, SkillUsageEntry> {
  const index = new Map<string, SkillUsageEntry>();
  if (!report) return index;
  for (const entry of report.skills) {
    index.set(entry.name.toLowerCase(), entry);
  }
  return index;
}

export interface Bar {
  x: number;
  y: number;
  width: number;
  height: number;
  value: number;
}

/**
 * Lay out bar geometry for a sparkline in a `viewBox` of `width` x `height`.
 *
 * Zero-valued slots still get a 1-unit baseline so the axis stays readable, and
 * an all-zero series does not divide by zero.
 */
export function sparkBars(
  values: number[],
  width: number,
  height: number,
  gap = 0.15,
): Bar[] {
  if (values.length === 0) return [];
  const slot = width / values.length;
  const barWidth = Math.max(slot * (1 - gap), 0.01);
  const max = Math.max(...values, 0);
  return values.map((value, i) => {
    const h = max > 0 ? (value / max) * height : 0;
    // Keep a hairline for empty days rather than rendering nothing.
    const drawn = value > 0 ? Math.max(h, height * 0.04) : Math.min(1, height);
    return {
      x: i * slot + (slot - barWidth) / 2,
      y: height - drawn,
      width: barWidth,
      height: drawn,
      value,
    };
  });
}

/** Order skill cards. Skills with no usage sort last within each mode. */
export function sortSkills(
  skills: ManagedSkill[],
  usage: Map<string, SkillUsageEntry>,
  mode: SkillSortMode,
): ManagedSkill[] {
  const get = (s: ManagedSkill) => usage.get(s.name.toLowerCase());
  const byName = (a: ManagedSkill, b: ManagedSkill) =>
    a.name.localeCompare(b.name);
  const byUsage = (a: ManagedSkill, b: ManagedSkill) => {
    const diff = (get(b)?.count ?? 0) - (get(a)?.count ?? 0);
    return diff !== 0 ? diff : byName(a, b);
  };
  const copy = [...skills];
  if (mode === "name") return copy.sort(byName);
  if (mode === "most-used") return copy.sort(byUsage);
  if (mode === "category") {
    // Classify once per skill rather than once per comparison.
    const category = new Map(skills.map((s) => [s, categorizeSkill(s)]));
    // Sections in their declared order; the most-used skill leads each one, so
    // the grouped view stays useful without a second sort control.
    return copy.sort((a, b) => {
      const left = category.get(a) ?? "other";
      const right = category.get(b) ?? "other";
      const diff = categoryRank(left) - categoryRank(right);
      if (diff !== 0) return diff;
      // Self-declared categories share a rank, so keep each one contiguous —
      // `groupByCategory` slices consecutive runs and would otherwise split
      // two custom sections into a mess of alternating fragments.
      if (left !== right) return left.localeCompare(right);
      return byUsage(a, b);
    });
  }
  return copy.sort((a, b) => {
    const diff = (get(b)?.last_used ?? 0) - (get(a)?.last_used ?? 0);
    return diff !== 0 ? diff : byName(a, b);
  });
}

export type SkillBulkAction = "enable" | "disable" | "delete";

/**
 * Which skills a bulk action would actually touch.
 *
 * Bulk enable deliberately skips deleted skills: a tombstoned built-in is not
 * merely "disabled", and resurrecting one from a category-wide button would
 * surprise. Per-card Restore stays the way back.
 */
export function bulkTargets(
  skills: ManagedSkill[],
  action: SkillBulkAction,
): ManagedSkill[] {
  if (action === "enable") {
    return skills.filter((skill) => !(skill.enabled || skill.deleted));
  }
  if (action === "disable") return skills.filter((skill) => skill.enabled);
  return skills.filter((skill) => !skill.deleted);
}

/** Compact number formatting: 1234 -> "1.2K". */
export function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}
