import { getAuthHeader } from "../../auth";
import { getApiBaseUrl } from "../../../hooks/utils";

export interface UserInfo {
  user_id: string;
  username: string;
  role: string;
}

function apiUrl(path: string): string {
  return `${getApiBaseUrl()}${path}`;
}

export async function login(username: string, password: string): Promise<UserInfo> {
  const resp = await fetch(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeader(),
    },
    credentials: "include",
    body: JSON.stringify({ username, password }),
  });

  if (!resp.ok) {
    let message = "Login failed";
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

  return resp.json() as Promise<UserInfo>;
}

export async function logout(): Promise<void> {
  await fetch(apiUrl("/api/auth/logout"), {
    method: "POST",
    headers: {
      ...getAuthHeader(),
    },
    credentials: "include",
  });
}

export async function getMe(): Promise<UserInfo | null> {
  const resp = await fetch(apiUrl("/api/auth/me"), {
    method: "GET",
    headers: {
      ...getAuthHeader(),
    },
    credentials: "include",
  });

  // null means "not logged in" — and only that. Everything else is "cannot
  // tell", which has to be an error: useAuth treats null as a definitive
  // answer and clears the session, so a 502 from a restarting backend logged
  // the user out and wiped their cached user, which its own catch block was
  // written to prevent.
  if (resp.status === 401 || resp.status === 404) {
    return null;
  }

  if (!resp.ok) {
    throw new Error(`Could not verify the session: HTTP ${resp.status}`);
  }

  return resp.json() as Promise<UserInfo>;
}
