/**
 * Turns the agent starts for itself.
 *
 * When a background task finishes while the session is idle, the server runs a
 * turn whose input is a `<system-reminder>` — the nudge that makes the agent
 * read the result and report it. That input is not something the user said, so
 * the transcript must not show it as their message; the report the agent then
 * writes is the part worth seeing.
 */

const SYSTEM_REMINDER_ONLY =
  /^\s*(?:<system-reminder>[\s\S]*?<\/system-reminder>\s*)+$/;

/** Whether this turn's input is nothing but system reminders. */
export function isSystemReminderOnly(text: string): boolean {
  if (!text.trim()) return false;
  return SYSTEM_REMINDER_ONLY.test(text);
}
