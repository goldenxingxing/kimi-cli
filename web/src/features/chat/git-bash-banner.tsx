import { useCallback, useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { AlertTriangleIcon, RefreshCwIcon } from "lucide-react";
import {
  getSystemCapabilities,
  type SystemCapabilities,
} from "@/lib/api/apis/SystemApi";

/**
 * Banner shown on Windows when Git for Windows (and its bundled bash.exe)
 * cannot be located. Without it, kimi-cli's Shell tool cannot execute
 * commands. Renders nothing on other platforms or once git-bash is detected.
 */
export function GitBashBanner() {
  const { t } = useTranslation("chat");
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(
    null,
  );
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchCapabilities = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const data = await getSystemCapabilities();
      setCapabilities(data);
    } catch (err) {
      // Don't show the banner on transient fetch errors — the worst case
      // is the user finds the shell tool fails and we surface the error
      // from the tool itself.
      console.error("[GitBashBanner] Failed to load capabilities:", err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchCapabilities();
  }, [fetchCapabilities]);

  if (!capabilities) return null;
  if (capabilities.platform !== "win32") return null;
  if (capabilities.git_bash) return null;

  const installUrl = capabilities.git_bash_install_url;

  return (
    <div
      role="alert"
      className="mx-3 mt-2 flex items-start gap-3 rounded-xl border border-amber-300/60 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-500/40 dark:bg-amber-950/40 dark:text-amber-100 sm:mx-5"
    >
      <AlertTriangleIcon className="mt-0.5 size-4 shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="font-medium">{t("system.gitBashMissingTitle")}</div>
        <p className="mt-1 leading-relaxed">
          <Trans
            i18nKey="chat:system.gitBashMissing"
            values={{ link: installUrl }}
            components={{
              installLink: (
                // biome-ignore lint/a11y/useAnchorContent: link text is injected by <Trans> from the i18n message at runtime
                <a
                  href={installUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={installUrl}
                  className="font-medium underline underline-offset-2 hover:text-amber-700 dark:hover:text-amber-300"
                />
              ),
            }}
          />
        </p>
      </div>
      <button
        type="button"
        onClick={fetchCapabilities}
        disabled={isRefreshing}
        aria-label={t("system.gitBashRefresh")}
        className="inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-md border border-amber-300/60 bg-amber-100/60 px-2 py-1 text-xs font-medium text-amber-900 transition-colors hover:bg-amber-200/70 disabled:cursor-wait disabled:opacity-60 dark:border-amber-500/40 dark:bg-amber-900/40 dark:text-amber-100 dark:hover:bg-amber-900/60"
      >
        <RefreshCwIcon
          className={`size-3 ${isRefreshing ? "animate-spin" : ""}`}
        />
        {t("system.gitBashRefresh")}
      </button>
    </div>
  );
}
