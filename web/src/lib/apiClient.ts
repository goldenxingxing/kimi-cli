import {
  Configuration,
  ConfigApi,
  SessionsApi,
  type RequestContext,
  type ResponseContext,
} from "./api";
import { getApiBaseUrl } from "../hooks/utils";
import { getAuthHeader } from "./auth";

/**
 * Format validation errors from FastAPI into a readable string.
 * FastAPI returns validation errors as:
 * { "detail": [{ "loc": ["body", "llm", "model"], "msg": "Field required", "type": "missing" }] }
 */
function formatValidationError(detail: unknown): string {
  if (Array.isArray(detail)) {
    return detail
      .map((err) => {
        if (err && typeof err === "object" && "msg" in err) {
          const loc = Array.isArray(err.loc) ? err.loc.slice(1).join(".") : "";
          return loc ? `${loc}: ${err.msg}` : err.msg;
        }
        return String(err);
      })
      .join("; ");
  }
  if (typeof detail === "string") {
    return detail;
  }
  return "Validation failed";
}

/**
 * Create API configuration with the current base URL.
 * Lazily evaluated to support runtime base URL changes.
 */
function createConfig(): Configuration {
  return new Configuration({
    basePath: getApiBaseUrl(),
    middleware: [
      {
        pre: async (context: RequestContext) => {
          context.init.headers = {
            ...context.init.headers,
            ...getAuthHeader(),
          };
          return context;
        },
        post: async (context: ResponseContext) => {
          if (!context.response.ok) {
            // Defensively: an error body is not always JSON. A proxy's HTML
            // 502, an empty 413, a gateway timeout — .json() throws on those,
            // and the SyntaxError replaced the real HTTP failure with
            // "Unexpected token <" everywhere the message is shown.
            let data: Record<string, unknown> = {};
            try {
              data = (await context.response.json()) as Record<string, unknown>;
            } catch {
              data = {};
            }
            let message: string;

            if (context.response.status === 422 && data.detail) {
              // FastAPI validation error
              message = formatValidationError(data.detail);
            } else if (typeof data.detail === "string") {
              message = data.detail;
            } else if (typeof data.msg === "string") {
              message = data.msg;
            } else {
              message =
                `Request failed: HTTP ${context.response.status}` +
                (context.response.statusText ? ` ${context.response.statusText}` : "");
            }

            switch (context.response.status) {
              case 401:
                // Handled globally by the fetch interceptor in
                // lib/session-expiry.ts, which also covers the hand-written
                // fetch helpers this client does not go through.
                break;
              case 403:
                console.error(message);
                break;
              case 404:
                console.error("The requested resource was not found.");
                break;
              default:
                console.error(message);
            }

            throw new Error(message);
          }
          return context.response;
        },
      },
    ],
  });
}

// A fresh Configuration per access, deliberately: getApiBaseUrl() can change
// at runtime and the client has to follow it. (There was a `_apiClient` cache
// variable here that nothing ever read or assigned.)
export const apiClient = {
  get config() {
    return new ConfigApi(createConfig());
  },
  get sessions() {
    return new SessionsApi(createConfig());
  },
};
