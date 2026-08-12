/**
 * Distil an approval request into one line.
 *
 * The dialog shows the request in full — the whole command, the whole
 * before/after text of an edit — which is what you want *after* you know what
 * you are being asked. It is a poor way to find that out: the first thing on
 * screen should be the shortest true statement of what will happen.
 *
 * Display blocks arrive flattened (`{type: "shell", command, language}`), but
 * some producers nest their payload under `data`, so every read goes through
 * `field` and tolerates both.
 */

/** A display block as it arrives: `type` plus the producer's own fields. */
export type DisplayItem = {
  type: string;
  data?: unknown;
  [key: string]: unknown;
};

export type ApprovalSummary =
  | { kind: "command"; command: string; extraLines: number }
  | { kind: "diff"; path: string; files: number; before: number; after: number }
  | { kind: "todo"; total: number; done: number }
  | { kind: "task"; description: string }
  | { kind: "wiki"; pages: number; summary: string }
  | { kind: "generic"; text: string };

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Read a field from a block, whether it sits at the top level or under `data`. */
function field(item: DisplayItem, name: string): unknown {
  const top = record(item);
  if (top && name in top) return top[name];
  const nested = record(item.data);
  return nested ? nested[name] : undefined;
}

function text(item: DisplayItem, name: string): string {
  const value = field(item, name);
  return typeof value === "string" ? value : "";
}

/** Lines in a text block, ignoring one trailing newline. */
function countLines(value: string): number {
  if (!value) return 0;
  const body = value.endsWith("\n") ? value.slice(0, -1) : value;
  return body.split("\n").length;
}

function firstLine(value: string): { line: string; rest: number } {
  const lines = value.split("\n");
  const firstIndex = lines.findIndex((line) => line.trim().length > 0);
  if (firstIndex === -1) return { line: "", rest: 0 };
  return {
    line: lines[firstIndex].trim(),
    // Only count lines that carry something, so a trailing blank line does not
    // advertise itself as hidden content.
    rest: lines.slice(firstIndex + 1).filter((line) => line.trim().length > 0)
      .length,
  };
}

function blocks(display: DisplayItem[] | undefined, type: string): DisplayItem[] {
  return (display ?? []).filter((item) => item?.type === type);
}

export function summarizeApproval(input: {
  action?: string;
  sender?: string;
  description?: string;
  display?: DisplayItem[];
}): ApprovalSummary | null {
  const { display } = input;

  // Order is by how much the block tells you, not by how the request was
  // built: an edit's path beats the generic action word every time.
  const diffs = blocks(display, "diff");
  if (diffs.length > 0) {
    const paths = new Set(diffs.map((item) => text(item, "path")).filter(Boolean));
    let before = 0;
    let after = 0;
    for (const item of diffs) {
      before += countLines(text(item, "old_text"));
      after += countLines(text(item, "new_text"));
    }
    return {
      kind: "diff",
      path: [...paths][0] ?? "",
      files: paths.size,
      before,
      after,
    };
  }

  const shell = blocks(display, "shell")[0];
  if (shell) {
    const { line, rest } = firstLine(text(shell, "command"));
    if (line) return { kind: "command", command: line, extraLines: rest };
  }

  const wiki = blocks(display, "wiki")[0];
  if (wiki) {
    const pages = field(wiki, "pages");
    return {
      kind: "wiki",
      pages: Array.isArray(pages) ? pages.length : 0,
      summary: text(wiki, "summary"),
    };
  }

  const todo = blocks(display, "todo")[0];
  if (todo) {
    const items = field(todo, "items");
    const list = Array.isArray(items) ? items : [];
    const done = list.filter((entry) => {
      const row = record(entry);
      return row?.status === "completed" || row?.status === "done";
    }).length;
    return { kind: "todo", total: list.length, done };
  }

  const task = blocks(display, "background_task")[0];
  if (task) {
    const description = text(task, "description") || text(task, "kind");
    if (description) return { kind: "task", description };
  }

  // Nothing structured to lean on: the description's first line still beats
  // showing the caller a wall of prose.
  const { line } = firstLine(input.description ?? "");
  if (line) return { kind: "generic", text: line };

  const action = (input.action ?? "").trim();
  if (action) {
    const sender = (input.sender ?? "").trim();
    return { kind: "generic", text: sender ? `${sender} · ${action}` : action };
  }
  return null;
}
