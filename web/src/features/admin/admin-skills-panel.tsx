import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  bulkSkillAction,
  deleteSkill,
  getSkillUsage,
  listSkills,
  readSkillMd,
  skillAction,
  updateSkillMd,
  uploadSkill,
  type ManagedSkill,
  type SkillUsageReport,
} from "@/lib/api/apis/AdminSkillsApi";
import {
  bulkTargets,
  formatCount,
  indexUsage,
  sortSkills,
  type SkillBulkAction,
  type SkillSortMode,
} from "./skill-usage-utils";
import { groupByCategory, type SkillGroup } from "./skill-categories";
import {
  nextExpandAll,
  readExpandedCategories,
  toggleCategory,
  writeExpandedCategories,
} from "./skill-section-state";
import { UsageSparkline } from "./usage-sparkline";
import { formatRelativeTime } from "@/hooks/utils";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FilePenLine,
  Loader2,
  RefreshCw,
  RotateCcw,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";

const RANGES = [7, 30, 90] as const;

const SORTS: { value: SkillSortMode; label: string }[] = [
  { value: "most-used", label: "Most used" },
  { value: "recent", label: "Recent" },
  { value: "name", label: "Name" },
  { value: "category", label: "Category" },
];

const BULK_ACTIONS: { action: SkillBulkAction; label: string }[] = [
  { action: "enable", label: "Enable" },
  { action: "disable", label: "Disable" },
  { action: "delete", label: "Delete" },
];

// Static list on purpose: Tailwind's scanner cannot see interpolated class
// names, so `bg-chart-${i}` would emit no CSS at all.
const CHART_BG = [
  "bg-chart-1",
  "bg-chart-2",
  "bg-chart-3",
  "bg-chart-4",
  "bg-chart-5",
] as const;

export function AdminSkillsPanel() {
  const [skills, setSkills] = useState<ManagedSkill[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [editing, setEditing] = useState<ManagedSkill | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ManagedSkill | null>(null);
  const [bulkBusy, setBulkBusy] = useState<string | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    readExpandedCategories,
  );
  const [bulkDeleteTarget, setBulkDeleteTarget] = useState<SkillGroup | null>(null);
  const [replaceFile, setReplaceFile] = useState<File | null>(null);
  const [content, setContent] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const [usage, setUsage] = useState<SkillUsageReport | null>(null);
  const [usageLoading, setUsageLoading] = useState(true);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [days, setDays] = useState<number>(30);
  const [sortMode, setSortMode] = useState<SkillSortMode>("most-used");

  useEffect(() => {
    writeExpandedCategories(expandedCategories);
  }, [expandedCategories]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setSkills(await listSkills());
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load skills");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadUsage = useCallback(async () => {
    setUsageLoading(true);
    setUsageError(null);
    try {
      setUsage(await getSkillUsage(days));
    } catch (error) {
      // Deliberately an inline banner rather than toast.error (the convention
      // elsewhere in this panel): a stats scan that keeps failing must not spam
      // toasts or get in the way of managing skills.
      setUsageError(
        error instanceof Error ? error.message : "Failed to load usage statistics",
      );
    } finally {
      setUsageLoading(false);
    }
  }, [days]);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

  useEffect(() => {
    loadUsage().catch(() => undefined);
  }, [loadUsage]);

  const refreshAll = useCallback(() => {
    load().catch(() => undefined);
    loadUsage().catch(() => undefined);
  }, [load, loadUsage]);

  const usageByName = useMemo(() => indexUsage(usage), [usage]);
  const orderedSkills = useMemo(
    () => sortSkills(skills, usageByName, sortMode),
    [skills, usageByName, sortMode],
  );
  // One headed section per category; the other sort modes keep a single
  // unlabelled grid.
  const groups = useMemo(
    () =>
      sortMode === "category"
        ? groupByCategory(orderedSkills)
        : [{ id: "other" as const, label: "", skills: orderedSkills }],
    [orderedSkills, sortMode],
  );
  const topSkills = useMemo(() => (usage?.skills ?? []).slice(0, 10), [usage]);
  const maxCount = topSkills[0]?.count ?? 0;
  const orphaned = useMemo(
    () => (usage?.skills ?? []).filter((s) => !s.installed),
    [usage],
  );

  const runAction = useCallback(
    async (skill: ManagedSkill, action: "enable" | "disable" | "restore") => {
      setBusy(skill.name);
      try {
        await skillAction(skill.name, action);
        await load();
        toast.success(`${skill.name} ${action}d`);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Skill update failed");
      } finally {
        setBusy(null);
      }
    },
    [load],
  );

  const runBulk = useCallback(
    async (group: SkillGroup, action: SkillBulkAction) => {
      const names = bulkTargets(group.skills, action).map((skill) => skill.name);
      if (names.length === 0) return;
      setBulkBusy(group.id);
      try {
        const result = await bulkSkillAction(names, action);
        // The response carries the refreshed list, so no reload round-trip.
        setSkills(result.skills);
        setBulkDeleteTarget(null);
        toast.success(
          `${result.applied.length} ${group.label} skill${
            result.applied.length === 1 ? "" : "s"
          } ${action}d`,
        );
        if (result.missing.length > 0) {
          toast.warning(`Skipped ${result.missing.length} unknown skill(s)`);
        }
      } catch (error) {
        toast.error(error instanceof Error ? error.message : `Bulk ${action} failed`);
      } finally {
        setBulkBusy(null);
      }
    },
    [],
  );

  const beginEdit = useCallback(async (skill: ManagedSkill) => {
    setBusy(skill.name);
    try {
      setContent(await readSkillMd(skill.name));
      setEditing(skill);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to read skill");
    } finally {
      setBusy(null);
    }
  }, []);

  const saveEdit = useCallback(async () => {
    if (!editing) return;
    setBusy(editing.name);
    try {
      await updateSkillMd(editing.name, content);
      setEditing(null);
      await load();
      toast.success(`${editing.name} updated`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to update skill");
    } finally {
      setBusy(null);
    }
  }, [content, editing, load]);

  const remove = useCallback(
    async (skill: ManagedSkill) => {
      setBusy(skill.name);
      try {
        await deleteSkill(skill.name);
        await load();
        toast.success(`${skill.name} deleted`);
        setDeleteTarget(null);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : "Failed to delete skill");
      } finally {
        setBusy(null);
      }
    },
    [load],
  );

  const upload = useCallback(
    async (file: File) => {
      setBusy("__upload__");
      try {
        await uploadSkill(file);
        await load();
        toast.success(`${file.name} installed`);
      } catch (error) {
        if (
          error instanceof Error &&
          error.message.includes("already exists")
        ) {
          setReplaceFile(file);
        } else {
          toast.error(error instanceof Error ? error.message : "Upload failed");
        }
      } finally {
        setBusy(null);
        if (fileInput.current) fileInput.current.value = "";
      }
    },
    [load],
  );

  const replace = useCallback(async () => {
    if (!replaceFile) return;
    setBusy("__upload__");
    try {
      await uploadSkill(replaceFile, true);
      await load();
      toast.success(`${replaceFile.name} replaced`);
      setReplaceFile(null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Replacement failed");
    } finally {
      setBusy(null);
    }
  }, [load, replaceFile]);

  const totals = usage?.totals;

  // Only the category view has sections to fold; every other sort mode renders
  // one unlabelled group, which is always open.
  const grouped = sortMode === "category";
  const isExpanded = (id: string) => !grouped || expandedCategories.has(id);
  const allExpanded =
    grouped && groups.length > 0 && groups.every((group) => expandedCategories.has(group.id));

  const renderSkillCard = (skill: ManagedSkill) => {
    const stats = usageByName.get(skill.name.toLowerCase());
    const count = stats?.count ?? 0;
    return (
      <Card key={skill.name} className={!skill.enabled ? "opacity-65" : ""}>
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-3">
            <CardTitle className="text-base">{skill.name}</CardTitle>
            <div className="flex flex-wrap justify-end gap-1">
              {count > 0 && stats && (
                <Badge
                  variant="secondary"
                  title={`${stats.read_count} SKILL.md reads, ${stats.slash_count} explicit calls, across ${stats.session_count} session(s)`}
                >
                  {formatCount(count)}×
                </Badge>
              )}
              <Badge variant={skill.origin === "builtin" ? "secondary" : "outline"}>
                {skill.origin === "builtin" ? "Built-in" : "User installed"}
              </Badge>
              {skill.modified && <Badge variant="outline">Modified</Badge>}
              {skill.deleted && <Badge variant="destructive">Deleted</Badge>}
              {!(skill.enabled || skill.deleted) && <Badge variant="outline">Disabled</Badge>}
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <p className="min-h-10 text-sm text-muted-foreground">{skill.description}</p>
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              {skill.files.length} file{skill.files.length === 1 ? "" : "s"}
              {" · "}
              {count} invocation{count === 1 ? "" : "s"}
              {" · "}
              {/* last_used is epoch SECONDS from the API. */}
              {stats?.last_used
                ? formatRelativeTime(new Date(stats.last_used * 1000))
                : "never used"}
              {stats && stats.error_count > 0 && (
                <span className="text-destructive">
                  {" · "}
                  {stats.error_count} error{stats.error_count === 1 ? "" : "s"}
                </span>
              )}
            </p>
            {stats && count > 0 && (
              <UsageSparkline
                values={stats.daily.slice(-14)}
                labels={usage?.dates.slice(-14)}
                className="h-5 w-20 shrink-0"
                ariaLabel={`Recent usage of ${skill.name}`}
              />
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {!skill.deleted && (
              <Button
                variant="outline"
                size="sm"
                disabled={busy === skill.name}
                onClick={() => beginEdit(skill)}
              >
                <FilePenLine className="mr-1 size-3.5" />Edit
              </Button>
            )}
            {!skill.deleted && (
              <Button
                variant="outline"
                size="sm"
                disabled={busy === skill.name}
                onClick={() => runAction(skill, skill.enabled ? "disable" : "enable")}
              >
                {skill.enabled ? "Disable" : "Enable"}
              </Button>
            )}
            {skill.origin === "builtin" && (skill.modified || skill.deleted) && (
              <Button
                variant="outline"
                size="sm"
                disabled={busy === skill.name}
                onClick={() => runAction(skill, "restore")}
              >
                <RotateCcw className="mr-1 size-3.5" />Restore
              </Button>
            )}
            {!skill.deleted && (
              <Button
                variant="destructive"
                size="sm"
                disabled={busy === skill.name}
                onClick={() => setDeleteTarget(skill)}
              >
                <Trash2 className="mr-1 size-3.5" />Delete
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="flex flex-col gap-5">
      {/* ---------------------------------------------------------------- */}
      {/* Usage statistics                                                  */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Usage</h3>
          {usageLoading && <Loader2 className="size-3.5 animate-spin text-muted-foreground" />}
          {usage?.scanned.truncated && (
            <Badge
              variant="outline"
              title="The scan hit its time budget; numbers are a lower bound."
            >
              Partial
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-1 rounded-lg border bg-muted/40 p-1">
          {RANGES.map((range) => (
            <button
              key={range}
              type="button"
              onClick={() => setDays(range)}
              className={cn(
                "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                days === range
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {range}d
            </button>
          ))}
        </div>
      </div>

      {usageError && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {usageError}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Invocations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{formatCount(totals?.invocations ?? 0)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Skills used
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{totals?.distinct_skills ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Sessions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{totals?.distinct_sessions ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle
              className="text-sm font-medium text-muted-foreground"
              title="Explicit /skill: and /flow: commands. Most usage is the model reading SKILL.md on its own."
            >
              Explicit calls
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{totals?.slash_invocations ?? 0}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Daily invocations
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-1">
            <UsageSparkline
              values={usage?.daily ?? []}
              labels={usage?.dates}
              className="h-16"
              ariaLabel="Daily skill invocations"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{usage?.dates[0] ?? ""}</span>
              <span>{usage?.dates[usage.dates.length - 1] ?? ""}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Top skills
            </CardTitle>
          </CardHeader>
          <CardContent>
            {topSkills.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                {usageLoading ? "Loading…" : "No skill usage recorded in this window."}
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                {topSkills.map((entry, index) => (
                  <div
                    key={entry.key}
                    className={cn(
                      "grid grid-cols-[minmax(0,9rem)_1fr_3rem] items-center gap-3",
                      !entry.installed && "opacity-60",
                    )}
                  >
                    <span className="flex items-center gap-1 truncate text-sm" title={entry.paths[0]}>
                      <span className="truncate">{entry.name}</span>
                      {!entry.installed && (
                        <Badge variant="outline" className="shrink-0 text-[10px]">
                          Not installed
                        </Badge>
                      )}
                    </span>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                      <div
                        className={cn("h-2 rounded-full", CHART_BG[index % CHART_BG.length])}
                        style={{
                          width: `${maxCount > 0 ? (entry.count / maxCount) * 100 : 0}%`,
                        }}
                      />
                    </div>
                    <span className="text-right text-sm tabular-nums text-muted-foreground">
                      {formatCount(entry.count)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {orphaned.length > 0 && (
        <details className="rounded-md border bg-card px-4 py-3">
          <summary className="cursor-pointer text-sm font-medium">
            Used but not installed ({orphaned.length})
          </summary>
          <p className="mt-2 text-xs text-muted-foreground">
            These were invoked in past sessions but no longer resolve to an installed
            skill — renamed, deleted, or defined in a project scope this panel does not
            manage.
          </p>
          <ul className="mt-2 flex flex-col gap-1">
            {orphaned.map((entry) => (
              <li
                key={entry.key}
                className="flex items-center justify-between gap-3 text-sm"
                title={entry.paths[0]}
              >
                <span className="truncate text-muted-foreground">{entry.name}</span>
                <span className="tabular-nums text-muted-foreground">{entry.count}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Skill management                                                  */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={refreshAll}>
          <RefreshCw className="mr-2 size-4" />Refresh
        </Button>
        <Button
          size="sm"
          disabled={busy === "__upload__"}
          onClick={() => fileInput.current?.click()}
        >
          {busy === "__upload__" ? (
            <Loader2 className="mr-2 size-4 animate-spin" />
          ) : (
            <Upload className="mr-2 size-4" />
          )}
          Upload Skill
        </Button>
        <input
          ref={fileInput}
          type="file"
          accept=".zip,.md,application/zip,text/markdown"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) upload(file).catch(() => undefined);
          }}
        />
        <p className="text-sm text-muted-foreground">
          Upload a ZIP containing one skill, or a standalone SKILL.md. Declare{" "}
          <code className="text-xs">category:</code> in its frontmatter to choose the
          section it lands in; otherwise it is classified from its name and
          description.
        </p>
        {grouped && (
          <Button
            variant="outline"
            size="sm"
            className="ml-auto"
            onClick={() =>
              setExpandedCategories(
                (current) =>
                  nextExpandAll(
                    current,
                    groups.map((group) => group.id),
                  ).expanded,
              )
            }
          >
            {allExpanded ? "Collapse all" : "Expand all"}
          </Button>
        )}
        <div
          className={cn(
            "flex items-center gap-1 rounded-lg border bg-muted/40 p-1",
            !grouped && "ml-auto",
          )}
        >
          {SORTS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => setSortMode(option.value)}
              className={cn(
                "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                sortMode === option.value
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-8 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />Loading skills…
        </div>
      ) : (
        <div
          // Tighter when grouped: collapsed sections are single rows, and the
          // point of closing them is to see many at once.
          className={cn("flex flex-col", grouped ? "gap-3" : "gap-6")}
        >
          {groups.map((group) => (
            <section key={group.id} className="flex flex-col gap-2">
              {group.label && (
                <div className="flex flex-wrap items-center gap-2 border-b pb-1">
                  {/* Only the label is the toggle. The bulk buttons sit beside
                      it rather than inside it: nesting them would be invalid
                      markup, and clicking Delete must not also open a section. */}
                  <button
                    type="button"
                    className="flex items-center gap-2 rounded-sm text-left"
                    aria-expanded={isExpanded(group.id)}
                    onClick={() =>
                      setExpandedCategories((current) =>
                        toggleCategory(current, group.id),
                      )
                    }
                  >
                    {isExpanded(group.id) ? (
                      <ChevronDown className="size-3.5 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="size-3.5 text-muted-foreground" />
                    )}
                    <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      {group.label}
                    </h4>
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {group.skills.length}
                    </span>
                  </button>
                  <div className="ml-auto flex items-center gap-1">
                    {bulkBusy === group.id && (
                      <Loader2 className="size-3.5 animate-spin text-muted-foreground" />
                    )}
                    {BULK_ACTIONS.map(({ action, label }) => {
                      const affected = bulkTargets(group.skills, action).length;
                      return (
                        <Button
                          key={action}
                          variant="ghost"
                          size="sm"
                          className={cn(
                            "h-6 px-2 text-xs",
                            action === "delete" &&
                              "text-destructive hover:text-destructive",
                          )}
                          disabled={bulkBusy !== null || affected === 0}
                          title={`${label} the ${affected} affected skill(s) in ${group.label}`}
                          onClick={() =>
                            action === "delete"
                              ? setBulkDeleteTarget(group)
                              : runBulk(group, action)
                          }
                        >
                          {label} all{affected > 0 && ` (${affected})`}
                        </Button>
                      );
                    })}
                  </div>
                </div>
              )}
              {isExpanded(group.id) && (
                <div className="grid gap-3 md:grid-cols-2">
                  {group.skills.map(renderSkillCard)}
                </div>
              )}
            </section>
          ))}
        </div>
      )}

      <Dialog open={editing !== null} onOpenChange={(open) => !open && setEditing(null)}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Edit {editing?.name}</DialogTitle>
            <DialogDescription>
              Editing a built-in skill creates a managed override. Restore removes it.
            </DialogDescription>
          </DialogHeader>
          <textarea
            aria-label="SKILL.md content"
            className="min-h-96 w-full resize-y rounded-md border bg-background p-3 font-mono text-sm"
            value={content}
            onChange={(event) => setContent(event.target.value)}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setEditing(null)}>Cancel</Button>
            <Button onClick={saveEdit} disabled={busy !== null}>
              <Download className="mr-2 size-4" />Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteTarget !== null} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {deleteTarget?.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              Built-in skills remain recoverable with Restore. User-installed skills are removed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteTarget && remove(deleteTarget)}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={bulkDeleteTarget !== null}
        onOpenChange={(open) => !open && setBulkDeleteTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete all {bulkDeleteTarget?.label} skills?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This removes{" "}
              {bulkDeleteTarget
                ? bulkTargets(bulkDeleteTarget.skills, "delete").length
                : 0}{" "}
              skill(s) in one go. Built-in skills remain recoverable with Restore;
              user-installed ones are removed for good.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => bulkDeleteTarget && runBulk(bulkDeleteTarget, "delete")}
            >
              Delete all
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={replaceFile !== null} onOpenChange={(open) => !open && setReplaceFile(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Replace existing skill?</AlertDialogTitle>
            <AlertDialogDescription>
              The managed version from {replaceFile?.name} will replace the current override.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={replace}>Replace</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
