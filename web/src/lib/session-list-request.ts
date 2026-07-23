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
