import { cn } from "@/lib/utils";
import { sparkBars } from "./skill-usage-utils";

const VIEW_W = 300;
const VIEW_H = 40;

interface UsageSparklineProps {
  values: number[];
  /** Parallel to `values`; used for the native hover tooltip. */
  labels?: string[];
  className?: string;
  ariaLabel?: string;
}

/**
 * Dependency-free bar chart.
 *
 * Notes for anyone editing this:
 * - `preserveAspectRatio="none"` stretches the viewBox to the element box, which
 *   also non-uniformly scales strokes and rounded corners. So: fill only, no
 *   `stroke`, no `rx`.
 * - `var(--chart-1)` / `var(--border)` are declared for both light and dark in
 *   index.css, so theme switching needs no JS.
 * - `<title>` gives a native tooltip without wiring up a positioned component.
 * - Axis labels belong outside the SVG; `<text>` would be stretched too.
 */
export function UsageSparkline({
  values,
  labels,
  className,
  ariaLabel = "Usage over time",
}: UsageSparklineProps) {
  const bars = sparkBars(values, VIEW_W, VIEW_H);
  const empty = values.every((v) => v === 0);
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      preserveAspectRatio="none"
      className={cn("w-full", className)}
      role="img"
      aria-label={ariaLabel}
    >
      {bars.map((bar, i) => (
        <rect
          // Index is the identity here: bars are a fixed-length day series.
          key={`${labels?.[i] ?? i}`}
          x={bar.x}
          y={bar.y}
          width={bar.width}
          height={bar.height}
          fill={bar.value > 0 ? "var(--chart-1)" : "var(--border)"}
        >
          <title>
            {labels?.[i] ? `${labels[i]}: ${bar.value}` : String(bar.value)}
          </title>
        </rect>
      ))}
      {empty && <title>No usage in this window</title>}
    </svg>
  );
}
