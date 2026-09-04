import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  type DraftBook,
  emptyDraft,
  isEmptyDraft,
  switchDraft,
} from "./prompt-input-drafts.ts";

type File = { id: string };

const draft = (text: string, files: File[] = []) => ({ text, files });

describe("composer drafts", () => {
  it("a draft stays with the scope it was typed in", () => {
    const book: DraftBook<File> = new Map();

    const afterSwitch = switchDraft(book, "a", draft("half a question"), "b");
    assert.deepEqual(afterSwitch, emptyDraft());

    const backToA = switchDraft(book, "b", emptyDraft(), "a");
    assert.equal(backToA.text, "half a question");
  });

  it("attachments follow the scope too", () => {
    const book: DraftBook<File> = new Map();
    const file = { id: "f1" };

    switchDraft(book, "a", draft("", [file]), "b");
    const backToA = switchDraft(book, "b", emptyDraft(), "a");

    assert.deepEqual(backToA.files, [file]);
  });

  it("switching away twice from the same state restores the same draft", () => {
    // React may run a render more than once, so the swap must not consume a draft.
    const book: DraftBook<File> = new Map();
    book.set("b", draft("waiting in b"));

    const first = switchDraft(book, "a", draft("typed in a"), "b");
    const second = switchDraft(book, "a", draft("typed in a"), "b");

    assert.deepEqual(first, second);
    assert.equal(book.get("a")?.text, "typed in a");
  });

  it("a sent draft leaves nothing behind", () => {
    const book: DraftBook<File> = new Map();
    switchDraft(book, "a", draft("to send"), "b");
    switchDraft(book, "b", emptyDraft(), "a");

    // Sent from a, so a is now empty; leaving must clear its entry.
    switchDraft(book, "a", emptyDraft(), "b");

    assert.equal(book.has("a"), false);
    assert.equal(isEmptyDraft(switchDraft(book, "b", emptyDraft(), "a")), true);
  });

  it("staying in the same scope keeps the live draft untouched", () => {
    const book: DraftBook<File> = new Map();
    const current = draft("still typing");

    assert.equal(switchDraft(book, "a", current, "a"), current);
    assert.equal(book.size, 0);
  });
});
