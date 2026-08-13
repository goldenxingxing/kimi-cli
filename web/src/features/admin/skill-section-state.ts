/**
 * Which category sections are open, remembered across visits.
 *
 * With fifteen categories over a few hundred skills, an all-open list means
 * scrolling past everything to reach anything. Sections therefore start closed
 * — the headers become a table of contents, each with its count — and whatever
 * the reader opens stays open next time, because the set of categories someone
 * works in does not change from one visit to the next.
 */

const STORAGE_KEY = "kimi.admin.skills.expandedCategories";

/** Read the remembered set. Any unusable value is treated as "none open". */
export function readExpandedCategories(
  storage: Pick<Storage, "getItem"> | undefined = safeStorage(),
): Set<string> {
  if (!storage) return new Set();
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((id): id is string => typeof id === "string"));
  } catch {
    return new Set();
  }
}

/** Persist the set. Storage being unavailable is not worth failing a render. */
export function writeExpandedCategories(
  ids: Set<string>,
  storage: Pick<Storage, "setItem"> | undefined = safeStorage(),
): void {
  if (!storage) return;
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify([...ids]));
  } catch {
    // Private mode, quota, a locked-down browser — none of it matters enough
    // to interrupt the page.
  }
}

export function toggleCategory(current: Set<string>, id: string): Set<string> {
  const next = new Set(current);
  if (!next.delete(id)) next.add(id);
  return next;
}

/**
 * What the expand/collapse-all control should do next.
 *
 * Anything still closed means the useful action is to open everything; only
 * when all of them are open does the button become "collapse".
 */
export function nextExpandAll(
  current: Set<string>,
  ids: string[],
): { expanded: Set<string>; action: "expand" | "collapse" } {
  const allOpen = ids.length > 0 && ids.every((id) => current.has(id));
  return allOpen
    ? { expanded: new Set(), action: "collapse" }
    : { expanded: new Set(ids), action: "expand" };
}

function safeStorage(): Pick<Storage, "getItem" | "setItem"> | undefined {
  try {
    return typeof window === "undefined" ? undefined : window.localStorage;
  } catch {
    return undefined;
  }
}
