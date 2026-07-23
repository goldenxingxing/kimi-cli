# Session List Network Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry a transient network failure once when refreshing the active session list and show a localized local-service error only if the retry also fails.

**Architecture:** Add a framework-independent retry helper beside the handwritten API client and leave generated OpenAPI files untouched. `useSessions.refreshSessions` will wrap only its full-list GET with the helper, classify a final `FetchError`, and translate that failure through the existing i18n singleton.

**Tech Stack:** TypeScript 5.9, React 19, Node 26 built-in test runner, i18next, generated OpenAPI Fetch client.

## Global Constraints

- Retry only full active-session list reads performed by `refreshSessions`.
- Do not retry pagination, archived-session reads, or any mutation.
- Retry only `FetchError`; HTTP, business, parsing, and other errors fail immediately.
- Wait 1,000 milliseconds before one retry.
- Preserve the final thrown error for console diagnostics.
- Do not edit `web/src/lib/api/runtime.ts`.
- Add no test-framework dependency.

---

### Task 1: Tested Session List Retry Helper

**Files:**
- Create: `web/src/lib/session-list-request.ts`
- Create: `web/tests/session-list-request.test.ts`

**Interfaces:**
- Consumes: the type-only `FetchError` shape from `web/src/lib/api/runtime.ts`
- Produces: `isFetchError(error: unknown): error is FetchError`
- Produces: `retrySessionListRequest<T>(operation: () => Promise<T>, delay?: (milliseconds: number) => Promise<void>): Promise<T>`

- [ ] **Step 1: Write tests for one retry, immediate non-network failure, and final failure**

```ts
import assert from "node:assert/strict";
import test from "node:test";
import {
  isFetchError,
  retrySessionListRequest,
} from "../src/lib/session-list-request.ts";

const VALIDATION_FAILED_REGEX = /validation failed/;

function createFetchError(message: string): Error {
  return Object.assign(new Error(message), {
    name: "FetchError",
    cause: new TypeError("Failed to fetch"),
  });
}

test("retries one FetchError after a one-second delay", async () => {
  let attempts = 0;
  const delays: number[] = [];

  const result = await retrySessionListRequest(
    async () => {
      attempts += 1;
      if (attempts === 1) {
        throw createFetchError("request failed");
      }
      return ["session"];
    },
    async (milliseconds) => {
      delays.push(milliseconds);
    },
  );

  assert.deepEqual(result, ["session"]);
  assert.equal(attempts, 2);
  assert.deepEqual(delays, [1_000]);
});

test("does not retry a non-FetchError", async () => {
  let attempts = 0;

  await assert.rejects(
    retrySessionListRequest(
      async () => {
        attempts += 1;
        throw new Error("validation failed");
      },
      async () => {
        assert.fail("delay must not run");
      },
    ),
    VALIDATION_FAILED_REGEX,
  );

  assert.equal(attempts, 1);
});

test("rethrows the final error after one retry", async () => {
  let attempts = 0;
  const finalError = createFetchError("second failure");

  await assert.rejects(
    retrySessionListRequest(
      async () => {
        attempts += 1;
        if (attempts === 1) {
          throw createFetchError("first failure");
        }
        throw finalError;
      },
      async () => undefined,
    ),
    (error) => error === finalError,
  );

  assert.equal(attempts, 2);
  assert.equal(isFetchError(finalError), true);
});
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd web
node --test tests/session-list-request.test.ts
```

Expected: FAIL with `ERR_MODULE_NOT_FOUND` for `src/lib/session-list-request.ts`.

- [ ] **Step 3: Implement the minimal retry helper**

```ts
import type { FetchError } from "./api/runtime.ts";

const SESSION_LIST_RETRY_DELAY_MS = 1_000;

type Delay = (milliseconds: number) => Promise<void>;

const wait: Delay = (milliseconds) =>
  new Promise((resolve) => {
    globalThis.setTimeout(resolve, milliseconds);
  });

export function isFetchError(error: unknown): error is FetchError {
  return (
    error instanceof Error && error.name === "FetchError" && "cause" in error
  );
}

export async function retrySessionListRequest<T>(
  operation: () => Promise<T>,
  delay: Delay = wait,
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (!isFetchError(error)) {
      throw error;
    }
  }

  await delay(SESSION_LIST_RETRY_DELAY_MS);
  return operation();
}
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
cd web
node --test tests/session-list-request.test.ts
```

Expected: three tests pass with no failures.

- [ ] **Step 5: Commit the helper and tests**

```bash
git add web/src/lib/session-list-request.ts web/tests/session-list-request.test.ts
git commit -m "test(web): cover session list network retry"
```

### Task 2: Integrate Retry and Localized Error

**Files:**
- Modify: `web/src/hooks/useSessions.ts:1-10,167-190`
- Modify: `web/src/i18n/locales/en/toasts.json:17-20`
- Modify: `web/src/i18n/locales/zh-CN/toasts.json:17-20`

**Interfaces:**
- Consumes: `retrySessionListRequest` and `isFetchError` from Task 1
- Consumes: `i18n.t(key)` from `web/src/i18n/index.ts`
- Produces: unchanged `UseSessionsReturn.error: string | null`

- [ ] **Step 1: Add localized network-error copy**

Update the session entries to:

```json
"session": {
  "errorTitle": "Session Error",
  "networkError": "Unable to connect to the local service. It may be restarting; please try again shortly."
}
```

and:

```json
"session": {
  "errorTitle": "会话错误",
  "networkError": "暂时无法连接本地服务，服务可能正在重启，请稍后重试。"
}
```

- [ ] **Step 2: Wrap only the full-list request and classify the final error**

Add these imports to `useSessions.ts`:

```ts
import i18n from "../i18n";
import {
  isFetchError,
  retrySessionListRequest,
} from "../lib/session-list-request";
```

Replace the direct list call in `refreshSessions` with:

```ts
const sessionsList = await retrySessionListRequest(() =>
  apiClient.sessions.listSessionsApiSessionsGet({
    limit: PAGE_SIZE,
    offset: 0,
    q: searchQuery.trim() || undefined,
  }),
);
```

Replace its catch-message construction with:

```ts
const message = isFetchError(err)
  ? i18n.t("toasts:session.networkError")
  : err instanceof Error
    ? err.message
    : "Failed to load sessions";
```

Leave `loadMoreSessions` and every mutation unchanged.

- [ ] **Step 3: Run focused and static verification**

Run:

```bash
cd web
node --test tests/session-list-request.test.ts
npm run typecheck
npm run lint
```

Expected: three tests pass; TypeScript and Biome exit with status 0.

- [ ] **Step 4: Run the production build**

Run:

```bash
cd web
npm run build
```

Expected: TypeScript build and Vite production bundle complete with status 0.

- [ ] **Step 5: Confirm scope and generated-source integrity**

Run:

```bash
git diff --check
git diff -- web/src/lib/api/runtime.ts
git status --short
```

Expected: no whitespace errors; no diff for generated runtime; only the planned helper, test,
hook, locale, and plan files are changed.

- [ ] **Step 6: Commit the integration**

```bash
git add web/src/hooks/useSessions.ts web/src/i18n/locales/en/toasts.json web/src/i18n/locales/zh-CN/toasts.json
git commit -m "fix(web): retry transient session list failures"
```
