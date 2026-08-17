import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { isSystemReminderOnly } from "./system-reminder.ts";

describe("isSystemReminderOnly", () => {
  it("recognises the nudge a self-started turn carries", () => {
    assert.equal(
      isSystemReminderOnly(
        "<system-reminder>Background tasks completed while you were idle.</system-reminder>",
      ),
      true,
    );
  });

  it("tolerates surrounding whitespace and several reminders", () => {
    assert.equal(
      isSystemReminderOnly("  <system-reminder>a</system-reminder>\n<system-reminder>b</system-reminder> "),
      true,
    );
  });

  it("leaves a real message alone even when it carries a reminder", () => {
    // The user is still speaking here; hiding this would lose their words.
    assert.equal(
      isSystemReminderOnly("run the tests <system-reminder>be careful</system-reminder>"),
      false,
    );
    assert.equal(isSystemReminderOnly("what happened to the download?"), false);
  });

  it("is false for nothing at all", () => {
    assert.equal(isSystemReminderOnly(""), false);
    assert.equal(isSystemReminderOnly("   \n "), false);
  });
});
