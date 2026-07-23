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
