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

export async function deleteSkill(name: string): Promise<void> {
  const result = await request(`/api/admin/skills/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!result.ok) {
    await response(result);
  }
}
