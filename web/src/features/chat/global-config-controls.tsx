import { useCallback, useState, type ReactElement } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Check, Cpu, Paperclip, RefreshCcw } from "lucide-react";
import { usePromptInputAttachments } from "@ai-elements";
import { useGlobalConfig } from "@/hooks/useGlobalConfig";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Loader } from "@/components/ai-elements/loader";
import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorEmpty,
  ModelSelectorGroup,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "@/components/ai-elements/model-selector";
import { cn } from "@/lib/utils";

/** Long enough that passing over a control is not a request to read about it. */
const TOOLTIP_HOVER_DELAY_MS = 500;

/**
 * Keep focus from opening a tooltip.
 *
 * Radix opens on focus *immediately* — the hover delay does not apply — and
 * because the trigger wraps the switch rather than being it, focusin bubbles
 * up from the control and counts as focusing the trigger. So clicking one of
 * these switches, or anything that lands focus on it, popped the explanation
 * with no pointer involved: selecting a different session was enough.
 *
 * Capture phase, so the event is swallowed on the way down, before Radix's
 * handler on the trigger sees it. Hover is untouched, and the switches carry
 * aria-labels, so nothing is lost for screen readers.
 */
function swallowFocus(event: React.FocusEvent): void {
  event.stopPropagation();
}

export type GlobalConfigControlsProps = {
  className?: string;
  planMode?: boolean;
  onPlanModeChange?: (enabled: boolean) => void;
  yolo?: boolean;
  onYoloChange?: (enabled: boolean) => void;
  /** Current session ID; undefined = draft composer (no session yet) */
  sessionId?: string;
  /** Per-session model override; null/undefined = follow global default */
  sessionModel?: string | null;
  /** Whether the session is currently generating a response */
  sessionBusy?: boolean;
  /** Persist a per-session model selection; resolves true on success */
  onSelectSessionModel?: (sessionId: string, model: string) => Promise<boolean>;
  /** Staged draft model shown when no session exists yet */
  draftModel?: string | null;
  /** Called when the draft (pre-session) model selection changes */
  onDraftModelChange?: (model: string) => void;
};

export function GlobalConfigControls({
  className,
  planMode = false,
  onPlanModeChange,
  yolo = false,
  onYoloChange,
  sessionId,
  sessionModel,
  sessionBusy = false,
  onSelectSessionModel,
  draftModel,
  onDraftModelChange,
}: GlobalConfigControlsProps): ReactElement {
  const { config, isLoading, error, refresh } = useGlobalConfig();
  const { t } = useTranslation(["toasts", "config", "chat"]);

  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [isSwitching, setIsSwitching] = useState(false);
  // Local fallback for the draft selection when the parent does not control it.
  const [localDraftModel, setLocalDraftModel] = useState<string | null>(null);

  const effectiveDraftModel = draftModel !== undefined ? draftModel : localDraftModel;

  // The model this selector currently shows: per-session override (or staged
  // draft) wins, otherwise the global default.
  const effectiveModelName = sessionId
    ? (sessionModel ?? config?.defaultModel)
    : (effectiveDraftModel ?? config?.defaultModel);

  const handleSelectModel = useCallback(
    async (modelKey: string) => {
      setIsSelectorOpen(false);
      if (!config || modelKey === effectiveModelName) {
        return;
      }

      if (sessionId && onSelectSessionModel) {
        if (sessionBusy) {
          toast.message(t("toasts:sessionModel.busyTitle"), {
            description: t("toasts:sessionModel.busyDesc"),
          });
          return;
        }
        setIsSwitching(true);
        try {
          const ok = await onSelectSessionModel(sessionId, modelKey);
          if (ok) {
            toast.success(t("toasts:sessionModel.successTitle"));
          }
        } finally {
          setIsSwitching(false);
        }
        return;
      }

      // Draft state (no session yet): stage locally; the selection is sent
      // along when the session is created.
      setLocalDraftModel(modelKey);
      onDraftModelChange?.(modelKey);
    },
    [
      config,
      effectiveModelName,
      sessionId,
      sessionBusy,
      onSelectSessionModel,
      onDraftModelChange,
      t,
    ],
  );

  const attachments = usePromptInputAttachments();

  // Resolve the active model entry: try model-key match first (current
  // behaviour), then fall back to a provider-name match so configs whose
  // ``default_model`` is actually a provider name (the new ``LLM_PROVIDERS``
  // shape) still surface the underlying model id rather than the provider.
  const activeModel =
    config?.models.find((m) => m.name === effectiveModelName) ??
    config?.models.find((m) => m.provider === effectiveModelName);
  const triggerLabel = activeModel?.name ?? effectiveModelName;

  return (
    <div className={cn("flex items-center gap-1", className)}>
      <Button
        variant="ghost"
        size="icon"
        className="size-9 border-0"
        aria-label={t("chat:attachFiles")}
        type="button"
        onClick={() => attachments.openFileDialog()}
      >
        <Paperclip className="size-4" />
      </Button>

      <div className="mx-0 h-4 w-px bg-border/70" />

      <ModelSelector open={isSelectorOpen} onOpenChange={setIsSelectorOpen}>
        <ModelSelectorTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="h-9 max-w-[160px] justify-start gap-2 border-0"
            aria-label={t("config:model.changeAria")}
            type="button"
            disabled={isLoading || isSwitching || !config}
          >
            <Cpu className="size-4 shrink-0" />
            <span className="truncate">
              {config ? triggerLabel : t("config:model.fallback")}
            </span>
            {(isLoading || isSwitching) && (
              <Loader className="ml-auto shrink-0" size={14} />
            )}
          </Button>
        </ModelSelectorTrigger>
        <ModelSelectorContent title={t("config:model.title")}>
          <ModelSelectorInput placeholder={t("config:model.placeholder")} />
          <ModelSelectorList>
            <ModelSelectorEmpty>{t("config:model.empty")}</ModelSelectorEmpty>
            <ModelSelectorGroup heading={t("config:model.heading")}>
              {(config?.models ?? []).map((m) => {
                const isSelected = m.name === effectiveModelName;
                const label = `${m.name} (${m.model})`;
                return (
                  <ModelSelectorItem
                    key={m.name}
                    value={`${m.name} ${m.model} ${m.provider}`}
                    onSelect={(_value) => handleSelectModel(m.name)}
                    className="flex items-center gap-2"
                  >
                    {isSelected ? (
                      <Check className="size-4 text-foreground" />
                    ) : (
                      <span className="size-4" />
                    )}
                    <ModelSelectorName title={label}>
                      {m.name}
                    </ModelSelectorName>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {m.model}
                    </span>
                  </ModelSelectorItem>
                );
              })}
            </ModelSelectorGroup>
          </ModelSelectorList>
        </ModelSelectorContent>
      </ModelSelector>

      {onPlanModeChange && (
        <>
          <div className="mx-0 h-4 w-px bg-border/70" />
          {/* These two sit between the composer and the session list, so the
              pointer crosses them on the way to switching sessions. At the
              zero delay the shared Tooltip defaults to, merely passing over
              popped the explanation — often enough to read as a glitch. Long
              enough that only a deliberate hover asks for it. */}
          <span onFocusCapture={swallowFocus}>
          <Tooltip delayDuration={TOOLTIP_HOVER_DELAY_MS}>
            <TooltipTrigger asChild>
              <div className="flex h-9 items-center gap-2 rounded-md px-2">
                <span className="text-xs text-muted-foreground">
                  {t("chat:planMode.label")}
                </span>
                <Switch
                  aria-label={t("chat:planMode.toggle")}
                  checked={planMode}
                  onCheckedChange={onPlanModeChange}
                />
              </div>
            </TooltipTrigger>
            <TooltipContent sideOffset={8}>
              {planMode
                ? t("chat:planMode.active")
                : t("chat:planMode.enable")}
            </TooltipContent>
          </Tooltip>
          </span>
        </>
      )}

      {onYoloChange && (
        <>
          <div className="mx-0 h-4 w-px bg-border/70" />
          <span onFocusCapture={swallowFocus}>
          <Tooltip delayDuration={TOOLTIP_HOVER_DELAY_MS}>
            <TooltipTrigger asChild>
              <div className="flex h-9 items-center gap-2 rounded-md px-2">
                <span
                  className={cn(
                    "text-xs",
                    // Auto-approval is the one setting whose cost is paid
                    // silently, so an active YOLO says so at a glance instead
                    // of looking like every other muted label.
                    yolo
                      ? "font-medium text-amber-600 dark:text-amber-400"
                      : "text-muted-foreground",
                  )}
                >
                  {t("chat:yoloMode.label")}
                </span>
                <Switch
                  aria-label={t("chat:yoloMode.toggle")}
                  checked={yolo}
                  onCheckedChange={onYoloChange}
                />
              </div>
            </TooltipTrigger>
            <TooltipContent sideOffset={8} className="max-w-72">
              {yolo ? t("chat:yoloMode.active") : t("chat:yoloMode.enable")}
            </TooltipContent>
          </Tooltip>
          </span>
        </>
      )}

      {error ? (
        <>
          <div className="mx-1.5 h-4 w-px bg-border/70" />
          <Button
            variant="outline"
            size="icon"
            className="size-9"
            aria-label={t("config:model.reload")}
            title={t("config:model.reload")}
            type="button"
            onClick={() => {
              refresh();
            }}
          >
            <RefreshCcw className="size-4" />
          </Button>
        </>
      ) : null}
    </div>
  );
}
