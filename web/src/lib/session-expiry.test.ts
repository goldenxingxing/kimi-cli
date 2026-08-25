/**
 * A 401 means "we no longer know you" everywhere except the login form, where
 * it means "wrong password". Bouncing the reader out of the form they are
 * typing in would be worse than the problem this interceptor solves.
 */

import assert from "node:assert/strict";
import { after, beforeEach, describe, it } from "node:test";

type FakeWindow = {
  fetch: (input: unknown, init?: unknown) => Promise<{ status: number }>;
  addEventListener: (type: string, handler: () => void) => void;
  removeEventListener: (type: string, handler: () => void) => void;
  dispatchEvent: (event: Event) => boolean;
  location: { href: string; origin: string };
};

const listeners = new Map<string, Set<() => void>>();

function makeWindow(status: number): FakeWindow {
  return {
    fetch: async () => ({ status }),
    addEventListener: (type, handler) => {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)?.add(handler);
    },
    removeEventListener: (type, handler) => {
      listeners.get(type)?.delete(handler);
    },
    dispatchEvent: (event: Event) => {
      for (const handler of listeners.get(event.type) ?? []) handler();
      return true;
    },
    location: { href: "http://localhost:5494/", origin: "http://localhost:5494" },
  };
}

const originalWindow = (globalThis as { window?: unknown }).window;

after(() => {
  (globalThis as { window?: unknown }).window = originalWindow;
});

async function loadFresh(status: number) {
  listeners.clear();
  (globalThis as { window?: unknown }).window = makeWindow(status);
  // A fresh module instance per case: the interceptor installs exactly once.
  return (await import(`./session-expiry.ts?case=${status}-${Math.random()}`)) as typeof import(
    "./session-expiry"
  );
}

describe("session expiry interceptor", () => {
  beforeEach(() => listeners.clear());

  it("announces expiry when an API call comes back 401", async () => {
    const mod = await loadFresh(401);
    let fired = 0;
    mod.onSessionExpired(() => {
      fired += 1;
    });
    mod.installSessionExpiryInterceptor();

    await (globalThis as unknown as { window: FakeWindow }).window.fetch("/api/sessions/?limit=100");

    assert.equal(fired, 1);
  });

  it("stays quiet for the login route, where 401 means wrong password", async () => {
    const mod = await loadFresh(401);
    let fired = 0;
    mod.onSessionExpired(() => {
      fired += 1;
    });
    mod.installSessionExpiryInterceptor();

    await (globalThis as unknown as { window: FakeWindow }).window.fetch("/api/auth/login");

    assert.equal(fired, 0);
  });

  it("stays quiet for another origin's 401", async () => {
    const mod = await loadFresh(401);
    let fired = 0;
    mod.onSessionExpired(() => {
      fired += 1;
    });
    mod.installSessionExpiryInterceptor();

    await (globalThis as unknown as { window: FakeWindow }).window.fetch(
      "https://api.example.com/api/whatever",
    );

    assert.equal(fired, 0);
  });

  it("passes successful responses straight through", async () => {
    const mod = await loadFresh(200);
    let fired = 0;
    mod.onSessionExpired(() => {
      fired += 1;
    });
    mod.installSessionExpiryInterceptor();

    const response = await (globalThis as unknown as { window: FakeWindow }).window.fetch(
      "/api/sessions/",
    );

    assert.equal(response.status, 200);
    assert.equal(fired, 0);
  });
});
