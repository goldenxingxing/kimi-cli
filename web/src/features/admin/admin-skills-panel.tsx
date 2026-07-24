import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteSkill,
  listSkills,
  readSkillMd,
  skillAction,
  updateSkillMd,
  uploadSkill,
  type ManagedSkill,
} from "@/lib/api/apis/AdminSkillsApi";
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
  Download,
  FilePenLine,
  Loader2,
  RefreshCw,
  RotateCcw,
  Trash2,
  Upload,
} from "lucide-react";
import { toast } from "sonner";

export function AdminSkillsPanel() {
  const [skills, setSkills] = useState<ManagedSkill[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [editing, setEditing] = useState<ManagedSkill | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ManagedSkill | null>(null);
  const [replaceFile, setReplaceFile] = useState<File | null>(null);
  const [content, setContent] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

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

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

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

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={load}>
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
          Upload a ZIP containing one skill, or a standalone SKILL.md.
        </p>
      </div>

      {loading ? (
        <div className="flex items-center gap-2 py-8 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />Loading skills…
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {skills.map((skill) => (
            <Card key={skill.name} className={!skill.enabled ? "opacity-65" : ""}>
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-3">
                  <CardTitle className="text-base">{skill.name}</CardTitle>
                  <div className="flex flex-wrap justify-end gap-1">
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
                <p className="text-xs text-muted-foreground">
                  {skill.files.length} file{skill.files.length === 1 ? "" : "s"}
                </p>
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
                      onClick={() =>
                        runAction(skill, skill.enabled ? "disable" : "enable")
                      }
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
