import { createElement, useEffect, useState } from "react";
import type { LiveMessage } from "../../hooks/types";

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;
const RELATIVE_DAY_LIMIT = 7;

export function formatMessageOutputTime(
  completedAt: number,
  now = Date.now(),
): string {
  const elapsed = Math.max(0, now - completedAt);

  if (elapsed < MINUTE_MS) {
    return "刚刚";
  }
  if (elapsed < HOUR_MS) {
    return `${Math.floor(elapsed / MINUTE_MS)}分钟前`;
  }
  if (elapsed < DAY_MS) {
    return `${Math.floor(elapsed / HOUR_MS)}小时前`;
  }
  if (elapsed < RELATIVE_DAY_LIMIT * DAY_MS) {
    return `${Math.floor(elapsed / DAY_MS)}天前`;
  }
  return new Date(completedAt).toLocaleDateString();
}

export function MessageOutputTime({
  completedAt,
  now,
}: {
  completedAt: number;
  now?: number;
}) {
  const [currentTime, setCurrentTime] = useState(() => now ?? Date.now());

  useEffect(() => {
    if (now !== undefined) {
      setCurrentTime(now);
      return;
    }

    const interval = window.setInterval(() => {
      setCurrentTime(Date.now());
    }, MINUTE_MS);
    return () => window.clearInterval(interval);
  }, [now]);

  const date = new Date(completedAt);
  return createElement(
    "time",
    {
      className: "ml-1 text-[11px] text-muted-foreground/70 tabular-nums",
      dateTime: date.toISOString(),
      title: date.toLocaleString(),
    },
    formatMessageOutputTime(completedAt, currentTime),
  );
}

export function shouldShowMessageOutputTime(
  message: LiveMessage,
): boolean {
  return (
    message.role === "assistant" &&
    !message.isStreaming &&
    (!message.variant || message.variant === "text") &&
    message.completedAt !== undefined
  );
}
