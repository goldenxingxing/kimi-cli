import { getAuthHeader } from "../../auth";
import { getApiBaseUrl } from "../../../hooks/utils";

export interface ManagedSkill {
  name: string;
  description: string;
  origin: "builtin" | "user";
  enabled: boolean;
  deleted: boolean;
  modified: boolean;
  files: string[];
  /**
   * `category` as declared in SKILL.md frontmatter (top level or under
   * `metadata`), verbatim. Null when the skill declares none — the panel then
   * classifies it from its name and description.
   */
  category?: string | null;
}

/** Per-skill usage, reconstructed server-side from session wire logs. */
export interface SkillUsageEntry {
  /** Matches ManagedSkill.name whenever `installed` is true. */
  name: string;
  /** Normalized directory key — stable join handle across renames. */
  key: string;
  installed: boolean;
  /**
   * Which skills root the reads came from.
   * - `extra`   — the user-picked global library (CUSTOM_SKILLS_HOST_PATH /
   *   extra_skill_dirs). On most installs this, not `builtin`, is what actually
   *   gets read; it also shadows same-named bundled skills at runtime.
   * - `builtin` — bundled with kimi-cli.
   * - `managed` — the admin panel's writable layer.
   * - `external` — some other root, or a root the reporting process could not
   *   resolve (e.g. a server started without CUSTOM_SKILLS_HOST_PATH reading a
   *   historic session). Only the label degrades; counts are unaffected.
   */
  source: "builtin" | "managed" | "extra" | "external" | "unknown";
  count: number;
  /**
   * Invocations split by who issued them: "main" for the top-level agent, or a
   * subagent type ("explore", "coder", …). Sums to `count`. Subagents drive most
   * tool traffic, so a skill can look popular without any user reaching for it
   * directly. Available from the API; not surfaced in the UI yet.
   */
  by_origin: Record<string, number>;
  read_count: number;
  slash_count: number;
  flow_count: number;
  error_count: number;
  resource_read_count: number;
  session_count: number;
  user_count: number;
  /** Epoch **seconds** — multiply by 1000 before constructing a Date. */
  first_used: number | null;
  last_used: number | null;
  /** Parallel to the report's `dates` array. */
  daily: number[];
  paths: string[];
}

export interface SkillUsageReport {
  generated_at: number;
  window_days: number;
  scanned: {
    sessions: number;
    sessions_read: number;
    duration_ms: number;
    truncated: boolean;
    cached: boolean;
  };
  totals: {
    invocations: number;
    read_invocations: number;
    slash_invocations: number;
    unmatched_slash_invocations: number;
    distinct_skills: number;
    distinct_sessions: number;
    distinct_users: number;
    /** Same split as SkillUsageEntry.by_origin, summed across all skills. */
    by_origin: Record<string, number>;
  };
  dates: string[];
  daily: number[];
  skills: SkillUsageEntry[];
  top_users: { user_id: string; username: string; count: number }[];
}

function url(path: string): string {
  return `${getApiBaseUrl()}${path}`;
}

async function response<T>(value: Response): Promise<T> {
  if (!value.ok) {
    let message = `Request failed (${value.status})`;
    try {
      const body = (await value.json()) as { detail?: string };
      message = body.detail || message;
    } catch {
      // Keep the status-based fallback.
    }
    throw new Error(message);
  }
  return value.json() as Promise<T>;
}

function request(path: string, init?: RequestInit): Promise<Response> {
  return fetch(url(path), {
    credentials: "include",
    ...init,
    headers: { ...getAuthHeader(), ...init?.headers },
  });
}

export async function listSkills(): Promise<ManagedSkill[]> {
  return response(await request("/api/admin/skills"));
}

export async function getSkillUsage(days = 30): Promise<SkillUsageReport> {
  return response(await request(`/api/admin/skills/usage?days=${days}`));
}

export async function uploadSkill(
  file: File,
  replace = false,
): Promise<ManagedSkill> {
  const body = new FormData();
  body.append("file", file);
  return response(
    await request(`/api/admin/skills/upload?replace=${replace}`, {
      method: "POST",
      body,
    }),
  );
}

export async function readSkillMd(name: string): Promise<string> {
  const result = await response<{ content: string }>(
    await request(
      `/api/admin/skills/${encodeURIComponent(name)}/files/SKILL.md`,
    ),
  );
  return result.content;
}

export async function updateSkillMd(
  name: string,
  content: string,
): Promise<ManagedSkill> {
  return response(
    await request(`/api/admin/skills/${encodeURIComponent(name)}/skill-md`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    }),
  );
}

export async function skillAction(
  name: string,
  action: "enable" | "disable" | "restore",
): Promise<ManagedSkill> {
  return response(
    await request(
      `/api/admin/skills/${encodeURIComponent(name)}/${action}`,
      { method: "POST" },
    ),
  );
}

export interface SkillBulkResult {
  /** Normalized names the action was applied to. */
  applied: string[];
  /** Names that resolved to no installed skill; the rest still went through. */
  missing: string[];
  /** The list as it stands after the action — no follow-up GET needed. */
  skills: ManagedSkill[];
}

/**
 * Apply one action to many skills in a single request.
 *
 * The server writes the skill state once for the whole batch, so a category
 * sweep either lands or does not — looping the per-skill routes client-side
 * would leave every intermediate state observable.
 */
export async function bulkSkillAction(
  names: string[],
  action: "enable" | "disable" | "delete",
): Promise<SkillBulkResult> {
  return response(
    await request("/api/admin/skills/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names, action }),
    }),
  );
}

export async function deleteSkill(name: string): Promise<void> {
  const result = await request(`/api/admin/skills/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!result.ok) {
    await response(result);
  }
}
