/**
 * Per-scope storage for unsent composer input.
 *
 * The composer's text and attachments live in one provider above the session
 * switcher, so without a scope they are shared by every session: type into one,
 * switch, and the draft follows you into a conversation it was never meant for.
 * A draft belongs to the session it was typed in, so it is stashed under that
 * session's key and restored when you come back.
 */

export type ComposerDraft<TFile> = {
  text: string;
  files: TFile[];
};

export type DraftBook<TFile> = Map<string, ComposerDraft<TFile>>;

export const emptyDraft = <TFile>(): ComposerDraft<TFile> => ({
  text: "",
  files: [],
});

export const isEmptyDraft = <TFile>(draft: ComposerDraft<TFile>): boolean =>
  draft.text === "" && draft.files.length === 0;

/**
 * Move the composer from one scope to another, returning what to show next.
 *
 * Idempotent for a given (from, current, to): the outgoing draft is written
 * over its own entry and the incoming one is only read, so React re-running a
 * render cannot consume a draft twice.
 */
export function switchDraft<TFile>(
  book: DraftBook<TFile>,
  fromKey: string,
  current: ComposerDraft<TFile>,
  toKey: string,
): ComposerDraft<TFile> {
  if (fromKey === toKey) {
    return current;
  }
  // Empty drafts are dropped rather than stored, so the book only ever holds
  // what someone actually typed.
  if (isEmptyDraft(current)) {
    book.delete(fromKey);
  } else {
    book.set(fromKey, current);
  }
  const restored = book.get(toKey);
  return restored ? { text: restored.text, files: [...restored.files] } : emptyDraft();
}
