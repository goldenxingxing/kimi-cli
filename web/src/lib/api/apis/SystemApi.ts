import { getAuthHeader } from "../../auth";
import { getApiBaseUrl } from "../../../hooks/utils";

export interface SystemCapabilities {
  platform: string;
  /** Whether Git for Windows itself resolved. Not the same as "can run commands". */
  git_bash: boolean;
  git_bash_install_url: string;
  /**
   * Whether *any* shell is usable, including the portable one bundled with the
   * app. Older servers omit it; treat a missing value as "fall back to
   * git_bash" so the banner still behaves on a stale backend.
   */
  shell_available?: boolean;
}

function apiUrl(path: string): string {
  return `${getApiBaseUrl()}${path}`;
}

async function handleResponse<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let message = `Request failed (${resp.status})`;
    try {
      const data = await resp.json();
      if (typeof data.detail === "string") {
        message = data.detail;
      } else if (typeof data.msg === "string") {
        message = data.msg;
      }
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(message);
  }
  return resp.json() as Promise<T>;
}

export async function getSystemCapabilities(): Promise<SystemCapabilities> {
  const resp = await fetch(apiUrl("/api/system/capabilities"), {
    method: "GET",
    headers: { ...getAuthHeader() },
    credentials: "include",
  });
  return handleResponse<SystemCapabilities>(resp);
}
