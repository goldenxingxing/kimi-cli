import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { cn } from "@/lib/utils";
import type { ApprovalResponseDecision } from "@/hooks/wireTypes";
import type { LiveMessage } from "@/hooks/types";
import { translateBackendMessage } from "@/lib/translate-backend";

type ApprovalDialogProps = {
  messages: LiveMessage[];
  onApprovalResponse?: (
    requestId: string,
    decision: ApprovalResponseDecision,
    reason?: string,
  ) => Promise<void>;
  pendingApprovalMap: Record<string, boolean>;
  canRespondToApproval: boolean;
};

type WikiApprovalDisplay = {
  type: "wiki";
  summary: string;
  pages: string[];
  sources: string[];
  duplicate_pages: string[];
  conflict_pages: string[];
  workspace_id: string | null;
  session_id: string;
  details: string[];
  omitted: {
    pages: number;
    sources: number;
    duplicates: number;
    conflicts: number;
  };
};

function asWikiApprovalDisplay(
  item: { type: string; data: unknown },
): WikiApprovalDisplay | null {
  if (item.type === "wiki") {
    const value =
      item.data && typeof item.data === "object"
        ? { type: "wiki", ...item.data }
        : item;
    const candidate = value as Partial<WikiApprovalDisplay>;
    if (
      typeof candidate.summary === "string" &&
      Array.isArray(candidate.pages) &&
      candidate.pages.every((page) => typeof page === "string") &&
      Array.isArray(candidate.details) &&
      candidate.details.every((detail) => typeof detail === "string")
    ) {
      const stringList = (items: unknown): string[] =>
        Array.isArray(items) &&
        items.every((entry) => typeof entry === "string")
          ? items
          : [];
      const omitted =
        candidate.omitted && typeof candidate.omitted === "object"
          ? candidate.omitted
          : {};
      const omittedCount = (key: keyof WikiApprovalDisplay["omitted"]) => {
        const value = (omitted as Record<string, unknown>)[key];
        return typeof value === "number" && value >= 0 ? value : 0;
      };
      return {
        type: "wiki",
        summary: candidate.summary,
        pages: candidate.pages,
        sources: stringList(candidate.sources),
        duplicate_pages: stringList(candidate.duplicate_pages),
        conflict_pages: stringList(candidate.conflict_pages),
        workspace_id:
          typeof candidate.workspace_id === "string"
            ? candidate.workspace_id
            : null,
        session_id:
          typeof candidate.session_id === "string" ? candidate.session_id : "",
        details: candidate.details,
        omitted: {
          pages: omittedCount("pages"),
          sources: omittedCount("sources"),
          duplicates: omittedCount("duplicates"),
          conflicts: omittedCount("conflicts"),
        },
      };
    }
  }
  return null;
}

export function ApprovalDialog({
  messages,
  onApprovalResponse,
  pendingApprovalMap,
  canRespondToApproval,
}: ApprovalDialogProps) {
  const { t } = useTranslation(["chat"]);
  const [feedbackMode, setFeedbackMode] = useState(false);
  const [feedbackText, setFeedbackText] = useState("");
  const feedbackInputRef = useRef<HTMLTextAreaElement>(null);

  // from messages, extract the pending approval request
  const pendingApproval = useMemo(() => {
    for (const message of messages) {
      if (
        message.variant === "tool" &&
        message.toolCall?.approval &&
        message.toolCall.state === "approval-requested" &&
        !message.toolCall.approval.submitted
      ) {
        return {
          message,
          approval: message.toolCall.approval,
          toolCall: message.toolCall,
        };
      }
    }
    return null;
  }, [messages]);

  // Reset feedback state when the pending approval changes
  const currentApprovalId = pendingApproval?.approval?.id;
  const prevApprovalIdRef = useRef(currentApprovalId);
  if (prevApprovalIdRef.current !== currentApprovalId) {
    prevApprovalIdRef.current = currentApprovalId;
    // Always clear stale feedback text, not just when feedbackMode is active.
    // Otherwise old text leaks into the next approval's feedback input.
    if (feedbackMode || feedbackText) {
      setFeedbackMode(false);
      setFeedbackText("");
    }
  }

  const handleResponse = useCallback(
    async (decision: ApprovalResponseDecision, reason?: string) => {
      if (!(pendingApproval && onApprovalResponse)) return;

      const { approval } = pendingApproval;
      if (!approval.id) return;

      try {
        await onApprovalResponse(approval.id, decision, reason);
      } catch (error) {
        console.error("[ApprovalDialog] Failed to respond", error);
      }
    },
    [pendingApproval, onApprovalResponse],
  );

  const handleFeedbackSubmit = useCallback(() => {
    const trimmed = feedbackText.trim();
    if (!trimmed) return;
    setFeedbackMode(false);
    setFeedbackText("");
    handleResponse("reject", trimmed);
  }, [feedbackText, handleResponse]);

  // Compute disable state before early return (hooks must run unconditionally)
  const approvalId = pendingApproval?.approval?.id;
  const approvalPending = approvalId
    ? pendingApprovalMap[approvalId] === true
    : false;
  const disableActions =
    !(canRespondToApproval && onApprovalResponse) || approvalPending;

  // Focus the feedback input when feedback mode is activated
  useEffect(() => {
    if (feedbackMode) {
      // Use rAF to wait for the DOM to be ready after state update
      requestAnimationFrame(() => {
        feedbackInputRef.current?.focus();
      });
    }
  }, [feedbackMode]);

  // Keyboard shortcuts: 1=Approve, 2=Approve for session, 3=Decline, 4=Decline with feedback
  useEffect(() => {
    if (!pendingApproval || disableActions) return;
    // When in feedback mode, don't handle number shortcuts
    if (feedbackMode) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.repeat) return;
      if (event.isComposing) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      // Skip when any input element is focused
      const el = document.activeElement;
      if (el) {
        const tag = el.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        if ((el as HTMLElement).isContentEditable) return;
      }

      if (event.key === "4") {
        event.preventDefault();
        setFeedbackMode(true);
        return;
      }

      const keyMap: Record<string, ApprovalResponseDecision> = {
        "1": "approve",
        "2": "approve_for_session",
        "3": "reject",
      };
      const decision = keyMap[event.key];
      if (decision) {
        event.preventDefault();
        handleResponse(decision);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [pendingApproval, disableActions, handleResponse, feedbackMode]);

  // if no pending approval request, do not render anything
  if (!pendingApproval) return null;

  const { approval, toolCall } = pendingApproval;
  const wikiApproval =
    toolCall.display
      ?.map(asWikiApprovalDisplay)
      .find((item): item is WikiApprovalDisplay => item !== null) ?? null;
  const otherDisplay = toolCall.display?.filter(
    (item) => item.type !== "wiki",
  );

  const sourceLabel = (() => {
    if (approval.sourceDescription) return approval.sourceDescription;
    const agentType = toolCall.subagentType;
    const agentId = toolCall.subagentAgentId;
    const idSuffix = agentId ? ` (${agentId})` : "";
    if (approval.sourceKind === "background_agent") {
      return agentType
        ? `Background · ${agentType}${idSuffix}`
        : `Background agent${idSuffix}`;
    }
    // Foreground sub-agent approvals (isSubagentOrigin)
    if (toolCall.isSubagentOrigin) {
      return agentType
        ? `${agentType}${idSuffix}`
        : `Sub-agent${idSuffix}`;
    }
    return null;
  })();

  return (
    <div className="px-3 pb-2 w-full">
      <div
        role="alert"
        className={cn(
          "relative w-full border border-border/60 shadow-xs",
          "border-l border-l-blue-400/50",
          "rounded-lg px-4 py-3",
          "transition-all duration-200",
          "max-h-[70vh]",
          "overflow-hidden",
        )}
      >
        <div className="flex flex-col gap-2.5">
          {/* Header */}
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-blue-400 animate-pulse flex-shrink-0" />
            <div className="font-semibold text-sm text-foreground">
              {wikiApproval
                ? t("chat:wikiApproval.title")
                : t("chat:approval.allowAction", { action: approval.action })}
            </div>
            {approval.sender && (
              <span className="text-xs text-muted-foreground">
                · {approval.sender}
              </span>
            )}
            {sourceLabel && (
              <span className="text-xs text-muted-foreground/70 bg-muted/50 px-1.5 py-0.5 rounded">
                {sourceLabel}
              </span>
            )}
          </div>

          {/* Description */}
          {approval.description && (
            <div className="rounded-md bg-muted/50 px-3 py-2 w-full max-h-44 overflow-auto">
              <pre className="font-mono text-xs whitespace-pre-wrap text-foreground/90">
                {translateBackendMessage(approval.description, t)}
              </pre>
            </div>
          )}

          {/* Display blocks (if any) */}
          {wikiApproval && (
            <details className="rounded-md bg-muted/30 px-3 py-2 text-xs">
              <summary className="cursor-pointer text-muted-foreground">
                {t("chat:wikiApproval.details")}
              </summary>
              <div className="mt-2 max-h-40 space-y-2 overflow-auto text-foreground/80">
                <div>
                  <div className="font-medium">
                    {t("chat:wikiApproval.paths")}
                    {wikiApproval.omitted.pages > 0 &&
                      ` · ${t("chat:wikiApproval.omitted", {
                        count: wikiApproval.omitted.pages,
                      })}`}
                  </div>
                  <ul className="mt-1 list-disc pl-4 font-mono">
                    {wikiApproval.pages.map((page) => (
                      <li key={page}>{page}</li>
                    ))}
                  </ul>
                </div>
                {(wikiApproval.sources.length > 0 ||
                  wikiApproval.omitted.sources > 0) && (
                  <div>
                    <div className="font-medium">
                      {t("chat:wikiApproval.sources")}
                      {wikiApproval.omitted.sources > 0 &&
                        ` · ${t("chat:wikiApproval.omitted", {
                          count: wikiApproval.omitted.sources,
                        })}`}
                    </div>
                    <ul className="mt-1 list-disc pl-4 font-mono">
                      {wikiApproval.sources.map((source) => (
                        <li key={source}>{source}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {(wikiApproval.duplicate_pages.length > 0 ||
                  wikiApproval.omitted.duplicates > 0) && (
                  <div>
                    <div className="font-medium">
                      {t("chat:wikiApproval.duplicates")}
                      {wikiApproval.omitted.duplicates > 0 &&
                        ` · ${t("chat:wikiApproval.omitted", {
                          count: wikiApproval.omitted.duplicates,
                        })}`}
                    </div>
                    <ul className="mt-1 list-disc pl-4 font-mono">
                      {wikiApproval.duplicate_pages.map((page) => (
                        <li key={page}>{page}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {(wikiApproval.conflict_pages.length > 0 ||
                  wikiApproval.omitted.conflicts > 0) && (
                  <div>
                    <div className="font-medium">
                      {t("chat:wikiApproval.conflicts")}
                      {wikiApproval.omitted.conflicts > 0 &&
                        ` · ${t("chat:wikiApproval.omitted", {
                          count: wikiApproval.omitted.conflicts,
                        })}`}
                    </div>
                    <ul className="mt-1 list-disc pl-4 font-mono">
                      {wikiApproval.conflict_pages.map((page) => (
                        <li key={page}>{page}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {wikiApproval.details.length > 0 && (
                  <ul className="list-disc space-y-1 pl-4">
                    {wikiApproval.details.map((detail) => (
                      <li key={detail}>{detail}</li>
                    ))}
                  </ul>
                )}
              </div>
            </details>
          )}

          {/* Non-Wiki display blocks retain their existing presentation. */}
          {otherDisplay && otherDisplay.length > 0 && (
            <div className="rounded-md bg-muted/30 px-3 py-2 text-sm max-h-40 overflow-auto">
              {otherDisplay.map((item) => {
                const displayKeyBase =
                  typeof item.data === "string" ||
                  typeof item.data === "number" ||
                  typeof item.data === "boolean"
                    ? `${item.type}:${item.data}`
                    : item.data == null
                      ? `${item.type}:null`
                      : (() => {
                          try {
                            return `${item.type}:${JSON.stringify(item.data)}`;
                          } catch {
                            return `${item.type}:unserializable`;
                          }
                        })();
                const displayKey = `${toolCall.toolCallId ?? toolCall.title}:${displayKeyBase}`;

                return (
                  <div key={displayKey} className="font-mono text-xs">
                    {JSON.stringify(item, null, 2)}
                  </div>
                );
              })}
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={disableActions}
              onClick={() => handleResponse("approve")}
              className="transition-all"
            >
              {approvalPending
                ? t("chat:approval.approving")
                : wikiApproval
                  ? t("chat:wikiApproval.approveOnce")
                  : t("chat:approval.approve")}
              {!approvalPending && <Kbd className="ml-1.5">1</Kbd>}
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={disableActions}
              onClick={() => handleResponse("approve_for_session")}
              className="transition-all"
            >
              {approvalPending
                ? t("chat:approval.approving")
                : wikiApproval
                  ? t("chat:wikiApproval.approveForSession")
                  : t("chat:approval.approveForSession")}
              {!approvalPending && <Kbd className="ml-1.5">2</Kbd>}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={disableActions}
              onClick={() => handleResponse("reject")}
              className={cn(
                "transition-all",
                "text-muted-foreground hover:text-destructive hover:bg-destructive/10",
              )}
            >
              {approvalPending
                ? t("chat:approval.declining")
                : wikiApproval
                  ? t("chat:wikiApproval.decline")
                  : t("chat:approval.decline")}
              {!approvalPending && <Kbd className="ml-1.5">3</Kbd>}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              disabled={disableActions}
              onClick={() => setFeedbackMode(!feedbackMode)}
              className={cn(
                "transition-all",
                feedbackMode
                  ? "text-foreground bg-muted"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {feedbackMode
                ? t("chat:approval.cancelFeedback")
                : t("chat:approval.declineWithFeedback")}
              {!(feedbackMode || approvalPending) && (
                <Kbd className="ml-1.5">4</Kbd>
              )}
            </Button>
          </div>

          {/* Feedback input */}
          {feedbackMode && (
            <div className="flex flex-col gap-1.5">
              <textarea
                ref={feedbackInputRef}
                value={feedbackText}
                onChange={(e) => setFeedbackText(e.target.value)}
                onKeyDown={(e) => {
                  // Guard against IME composition (e.g. Chinese input)
                  if (e.nativeEvent.isComposing) return;
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleFeedbackSubmit();
                  }
                  if (e.key === "Escape") {
                    e.preventDefault();
                    setFeedbackMode(false);
                    setFeedbackText("");
                  }
                }}
                placeholder={t("chat:approval.feedbackPlaceholder")}
                className={cn(
                  "w-full rounded-md border border-border/60 bg-muted/30",
                  "px-3 py-2 text-sm text-foreground",
                  "placeholder:text-muted-foreground/50",
                  "focus:outline-none focus:ring-1 focus:ring-ring",
                  "resize-none",
                )}
                rows={2}
              />
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {t("chat:approval.feedbackHint")}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!feedbackText.trim()}
                  onClick={handleFeedbackSubmit}
                  className="text-xs"
                >
                  {t("chat:approval.submitFeedback")}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
