import { useCallback, useEffect, useState } from "react";

import { Loader2, Plus, RefreshCcw, Save, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  type KnowledgeFile,
  type MemoryKind,
  readKnowledge,
  useKnowledgeBase,
  usePersistentMemory,
  useRecentSummaries,
  writeKnowledge,
  deleteKnowledge,
  useCandidates,
} from "@/hooks/useMemory";

type Tab = "knowledge" | "persistent" | "recent" | "suggested";

const KIND_LABELS: Record<MemoryKind, string> = {
  user: "User",
  feedback: "Feedback",
  project: "Project",
  reference: "Reference",
};

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString();
}

export function MemoryPanel({ sessionId }: { sessionId: string | null }) {
  const [tab, setTab] = useState<Tab>("persistent");
  // Counted here rather than inside the tab so the number shows while another
  // tab is open. A queue nobody is told about is the reason this tab exists.
  const { candidates } = useCandidates();

  return (
    <div className="flex flex-col gap-3 p-3">
      <div className="flex gap-2">
        <Button
          size="sm"
          variant={tab === "persistent" ? "default" : "outline"}
          onClick={() => setTab("persistent")}
        >
          Persistent
        </Button>
        <Button
          size="sm"
          variant={tab === "suggested" ? "default" : "outline"}
          onClick={() => setTab("suggested")}
        >
          Suggested
          {candidates.length > 0 && (
            <span className="ml-1.5 rounded-full bg-primary/15 px-1.5 text-xs font-medium">
              {candidates.length}
            </span>
          )}
        </Button>
        <Button
          size="sm"
          variant={tab === "recent" ? "default" : "outline"}
          onClick={() => setTab("recent")}
        >
          Recent
        </Button>
        <Button
          size="sm"
          variant={tab === "knowledge" ? "default" : "outline"}
          onClick={() => setTab("knowledge")}
        >
          Knowledge
        </Button>
      </div>
      {tab === "persistent" && <PersistentTab />}
      {tab === "suggested" && <SuggestedTab />}
      {tab === "recent" && <RecentTab />}
      {tab === "knowledge" && <KnowledgeTab sessionId={sessionId} />}
    </div>
  );
}

function SuggestedTab() {
  const { candidates, loading, error, promote, dismiss } = useCandidates();

  const decide = async (action: () => Promise<void>, done: string) => {
    try {
      await action();
      toast.success(done);
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  if (loading) return <p className="text-sm text-muted-foreground">Loading…</p>;
  if (error) return <p className="text-sm text-destructive">{error}</p>;
  if (candidates.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Nothing waiting. Facts noticed at the end of a session appear here until you
        keep or discard them; unanswered, they expire after a fortnight.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {candidates.map((c) => (
        <Card key={c.id}>
          <CardHeader className="flex flex-row items-center justify-between">
            <span className="text-xs text-muted-foreground">
              {KIND_LABELS[c.kind]} · {fmtTime(c.created_at)}
            </span>
            <div className="flex gap-2">
              <Button size="sm" onClick={() => decide(() => promote(c.id), "Kept")}>
                Keep
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => decide(() => dismiss(c.id), "Discarded")}
              >
                Discard
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap text-sm">{c.content}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function PersistentTab() {
  const { entries, loading, error, refresh, add, update, remove } = usePersistentMemory();
  const [newKind, setNewKind] = useState<MemoryKind>("user");
  const [newContent, setNewContent] = useState("");

  const handleAdd = async () => {
    const trimmed = newContent.trim();
    if (!trimmed) return;
    try {
      await add(newKind, trimmed);
      setNewContent("");
      toast.success("Added");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <div className="flex flex-col gap-3">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-sm">Add a persistent memory</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          <select
            className="rounded border px-2 py-1 text-sm"
            value={newKind}
            onChange={(e) => setNewKind(e.target.value as MemoryKind)}
          >
            {Object.entries(KIND_LABELS).map(([k, label]) => (
              <option key={k} value={k}>
                {label}
              </option>
            ))}
          </select>
          <Textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            placeholder="What should the agent remember across sessions?"
            rows={3}
          />
          <div className="flex justify-end">
            <Button size="sm" onClick={handleAdd} disabled={!newContent.trim()}>
              <Plus className="mr-1 h-3.5 w-3.5" /> Add
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">
          {loading ? "Loading…" : `${entries.length} entries`}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            refresh();
          }}
        >
          <RefreshCcw className="mr-1 h-3.5 w-3.5" /> Refresh
        </Button>
      </div>
      {error && <div className="text-sm text-red-500">{error}</div>}

      <ul className="flex flex-col gap-2">
        {entries.map((e) => (
          <li key={e.id}>
            <PersistentEntryItem
              entry={e}
              onSave={(content) => update(e.id, content)}
              onDelete={() => remove(e.id)}
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

function PersistentEntryItem({
  entry,
  onSave,
  onDelete,
}: {
  entry: { id: string; kind: MemoryKind; content: string; created_at: number };
  onSave: (content: string) => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [content, setContent] = useState(entry.content);
  const [busy, setBusy] = useState(false);
  const dirty = content !== entry.content;

  const handleSave = async () => {
    setBusy(true);
    try {
      await onSave(content);
      toast.success("Updated");
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    setBusy(true);
    try {
      await onDelete();
      toast.success("Deleted");
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardContent className="flex flex-col gap-2 p-3">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span className="flex items-center gap-2">
            <Badge variant="outline">{KIND_LABELS[entry.kind]}</Badge>
            <span>{fmtTime(entry.created_at)}</span>
          </span>
          <span className="font-mono">{entry.id.slice(0, 8)}</span>
        </div>
        <Textarea value={content} onChange={(e) => setContent(e.target.value)} rows={3} />
        <div className="flex justify-end gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleDelete}
            disabled={busy}
          >
            <Trash2 className="mr-1 h-3.5 w-3.5" /> Delete
          </Button>
          <Button size="sm" onClick={handleSave} disabled={!dirty || busy}>
            {busy ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : <Save className="mr-1 h-3.5 w-3.5" />}
            Save
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function RecentTab() {
  const { items, loading, error, refresh } = useRecentSummaries(50);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">
          {loading ? "Loading…" : `${items.length} summaries`}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            refresh();
          }}
        >
          <RefreshCcw className="mr-1 h-3.5 w-3.5" /> Refresh
        </Button>
      </div>
      {error && <div className="text-sm text-red-500">{error}</div>}
      <ul className="flex flex-col gap-2">
        {items.map((s) => (
          <li key={s.id}>
            <Card>
              <CardContent className="p-3 text-sm">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="flex items-center gap-2">
                    <Badge variant="outline">{s.trigger}</Badge>
                    <span>{fmtTime(s.created_at)}</span>
                  </span>
                  <span className="font-mono">{s.session_id.slice(0, 8)}</span>
                </div>
                <pre className="mt-1 whitespace-pre-wrap font-sans">{s.summary}</pre>
              </CardContent>
            </Card>
          </li>
        ))}
      </ul>
    </div>
  );
}

function KnowledgeTab({ sessionId }: { sessionId: string | null }) {
  const { files, loading, error, refresh } = useKnowledgeBase(sessionId);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [originalContent, setOriginalContent] = useState("");
  const [busy, setBusy] = useState(false);
  const [newName, setNewName] = useState("");

  useEffect(() => {
    if (!(selected && sessionId)) return;
    readKnowledge(sessionId, selected).then((f) => {
      setContent(f.content);
      setOriginalContent(f.content);
    });
  }, [selected, sessionId]);

  const handleSave = useCallback(async () => {
    if (!sessionId) return;
    const target = selected ?? newName.trim();
    if (!target) return;
    setBusy(true);
    try {
      await writeKnowledge(sessionId, target, content);
      setOriginalContent(content);
      setSelected(target);
      setNewName("");
      await refresh();
      toast.success("Saved");
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [sessionId, selected, newName, content, refresh]);

  const handleDelete = useCallback(async () => {
    if (!(sessionId && selected)) return;
    setBusy(true);
    try {
      await deleteKnowledge(sessionId, selected);
      setSelected(null);
      setContent("");
      setOriginalContent("");
      await refresh();
      toast.success("Deleted");
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [sessionId, selected, refresh]);

  if (!sessionId) {
    return <div className="text-sm text-muted-foreground">Select a session first.</div>;
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">
          {loading ? "Loading…" : `${files.length} files`}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            refresh();
          }}
        >
          <RefreshCcw className="mr-1 h-3.5 w-3.5" /> Refresh
        </Button>
      </div>
      {error && <div className="text-sm text-red-500">{error}</div>}

      <div className="flex flex-col gap-2">
        {files.map((f) => (
          <FileRow
            key={f.name}
            file={f}
            active={f.name === selected}
            onClick={() => setSelected(f.name)}
          />
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">
            {selected ? `Edit ${selected}` : "Create new file"}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {!selected && (
            <Input
              placeholder="filename.md"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
            />
          )}
          <Textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={12}
            placeholder="Markdown content for the shared knowledge base"
          />
          <div className="flex justify-end gap-2">
            {selected && (
              <Button size="sm" variant="outline" onClick={handleDelete} disabled={busy}>
                <Trash2 className="mr-1 h-3.5 w-3.5" /> Delete
              </Button>
            )}
            <Button
              size="sm"
              onClick={handleSave}
              disabled={
                busy ||
                !(selected || newName.trim()) ||
                (!!selected && content === originalContent)
              }
            >
              {busy ? (
                <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Save className="mr-1 h-3.5 w-3.5" />
              )}
              Save
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function FileRow({
  file,
  active,
  onClick,
}: {
  file: KnowledgeFile;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`flex items-center justify-between rounded border px-3 py-2 text-left text-sm transition ${
        active ? "border-primary bg-primary/5" : "hover:bg-muted/40"
      }`}
      onClick={onClick}
    >
      <span className="font-mono">{file.name}</span>
      <span className="text-xs text-muted-foreground">
        {(file.size / 1024).toFixed(1)} KB · {fmtTime(file.mtime)}
      </span>
    </button>
  );
}
