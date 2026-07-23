import type { LiveMessage } from "./types";

export function setMessageOutputTime(
  messages: LiveMessage[],
  messageId: string | null,
  completedAt: number | undefined,
): LiveMessage[] {
  if (!(messageId && completedAt !== undefined)) {
    return messages;
  }

  return messages.map((message) =>
    message.id === messageId &&
    message.role === "assistant" &&
    (!message.variant || message.variant === "text")
      ? { ...message, completedAt }
      : message,
  );
}
