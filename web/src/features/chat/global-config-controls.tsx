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

export type GlobalConfigControlsProps = {
  className?: string;
  planMode?: boolean;
  onPlanModeChange?: (enabled: boolean) => void;
  yolo?: boolean;
  onYoloChange?: (enabled: boolean) => void;
};

export function GlobalConfigControls({
  className,
  planMode = false,
  onPlanModeChange,
  yolo = false,
  onYoloChange,
}: GlobalConfigControlsProps): ReactElement {
  const { config, isLoading, isUpdating, error, refresh, update } =
    useGlobalConfig();
  const { t } = useTranslation(["toasts", "config", "chat"]);

  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [lastBusySkip, setLastBusySkip] = useState<string[] | null>(null);

  const handleSelectModel = useCallback(
    async (modelKey: string) => {
      setIsSelectorOpen(false);
      if (!config || modelKey === config.defaultModel) {
        return;
      }

      try {
        const resp = await update({ defaultModel: modelKey });
        const restarted = resp.restartedSessionIds ?? [];
        const skippedBusy = resp.skippedBusySessionIds ?? [];

        if (restarted.length > 0) {
          toast.success(t("toasts:globalModel.successTitle"), {
            description: t("toasts:globalModel.successDesc", {
              count: restarted.length,
            }),
          });
        } else {
          toast.success(t("toasts:globalModel.successTitle"));
        }

        if (skippedBusy.length > 0) {
          setLastBusySkip(skippedBusy);
          toast.message(t("toasts:globalModel.busyTitle"), {
            description: t("toasts:globalModel.busyDesc", {
              count: skippedBusy.length,
            }),
          });
        } else {
          setLastBusySkip(null);
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : t("toasts:globalModel.fallbackError");
        toast.error(t("toasts:globalModel.errorTitle"), { description: message });
      }
    },
    [config, update, t],
  );

  const handleForceRestartBusy = useCallback(async () => {
    if (!lastBusySkip || lastBusySkip.length === 0) {
      return;
    }
    try {
      const resp = await update({ forceRestartBusySessions: true });
      const restarted = resp.restartedSessionIds ?? [];
      const skippedBusy = resp.skippedBusySessionIds ?? [];

      if (skippedBusy.length === 0) {
        setLastBusySkip(null);
      } else {
        setLastBusySkip(skippedBusy);
      }

      toast.success(t("toasts:restartBusy.successTitle"), {
        description:
          restarted.length > 0
            ? t("toasts:restartBusy.successDesc", { count: restarted.length })
            : t("toasts:restartBusy.successDescNone"),
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : t("toasts:restartBusy.fallbackError");
      toast.error(t("toasts:restartBusy.errorTitle"), { description: message });
    }
  }, [lastBusySkip, update, t]);

  const attachments = usePromptInputAttachments();

  // Resolve the active model entry: try model-key match first (current
  // behaviour), then fall back to a provider-name match so configs whose
  // ``default_model`` is actually a provider name (the new ``LLM_PROVIDERS``
  // shape) still surface the underlying model id rather than the provider.
  const activeModel =
    config?.models.find((m) => m.name === config?.defaultModel) ??
    config?.models.find((m) => m.provider === config?.defaultModel);
  const triggerLabel = activeModel?.name ?? config?.defaultModel;

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
            disabled={isLoading || isUpdating || !config}
          >
            <Cpu className="size-4 shrink-0" />
            <span className="truncate">
              {config ? triggerLabel : t("config:model.fallback")}
            </span>
            {(isLoading || isUpdating) && (
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
                const isSelected = m.name === config?.defaultModel;
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
        </>
      )}

      {onYoloChange && (
        <>
          <div className="mx-0 h-4 w-px bg-border/70" />
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
        </>
      )}

      {(lastBusySkip && lastBusySkip.length > 0) || error ? (
        <div className="mx-1.5 h-4 w-px bg-border/70" />
      ) : null}

      {lastBusySkip && lastBusySkip.length > 0 ? (
        <Button
          variant="outline"
          size="icon"
          className="size-9"
          aria-label={t("config:model.forceRestart")}
          title={t("config:model.forceRestart")}
          type="button"
          onClick={handleForceRestartBusy}
          disabled={isUpdating}
        >
          <RefreshCcw className="size-4" />
        </Button>
      ) : null}

      {error ? (
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
      ) : null}
    </div>
  );
}
