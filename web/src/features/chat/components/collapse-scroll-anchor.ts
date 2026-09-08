/**
 * Keep a collapsible block where the reader left it while it opens or closes.
 *
 * The message list is virtualized: it measures items and compensates the
 * scroll position when one of them changes height. A thinking block holding
 * thousands of pixels of text is the pathological case — collapsing it while
 * reading inside it moves the viewport far from where the reader was, usually
 * back into earlier messages, because the compensation is applied against the
 * block the reader is looking at rather than around it.
 *
 * So the toggle pins its own block instead. Two things make that reliable:
 *
 * - The block is measured on the click that toggles it, in the capture phase,
 *   before anything has moved. Radix reports an uncontrolled open/close from an
 *   effect, which runs after the new DOM is committed and after the virtual
 *   list's ResizeObserver has already had its say — measuring there would be
 *   measuring the jump, not preventing it. A toggle that no click preceded
 *   (nothing does this today; a programmatic open would) falls back to
 *   measuring at toggle time, which is the old, weaker behaviour rather than
 *   a wrong one.
 * - The correction runs only while the block's own height is actually changing,
 *   plus a few frames after it settles. The list keeps following new output
 *   while the agent is still talking, and that pull is legitimate; the pinning
 *   is here to survive one height change, not to own the viewport.
 *
 * It also lets go early when the reader takes over (wheel, touch, key, or a
 * scrollbar drag) and when the scroller is already at its end, where the
 * correction cannot be applied and holding on would just burn frames.
 */

import { useCallback, useEffect, useRef } from "react";

/** Longest a toggle is held in place; the collapse animation is ~150ms. */
export const ANCHOR_WINDOW_MS = 500;

/** How long a click's measurement stays usable if no toggle follows it. */
const MEASUREMENT_TTL_MS = 1000;

/** Sub-pixel drift is layout noise, not a jump. */
const DRIFT_EPSILON = 0.5;

/** Frames of unchanging block height that end the pinning. */
const STABLE_FRAMES = 3;

/** Scroll positions are fractional; a pixel of slack is not a refused write. */
const CLAMP_TOLERANCE = 1;

/** What a collapse trigger looks like: Radix's, and anything marked like it. */
const TRIGGER_SELECTOR = '[data-slot="collapsible-trigger"]';

/**
 * Pixels to add to the scroller's scrollTop to put the block back where it was.
 *
 * Kept separate from the DOM so the rule can be read and tested on its own.
 */
export function scrollCorrection(
  anchorTopBefore: number,
  anchorTopNow: number,
): number {
  const drift = anchorTopNow - anchorTopBefore;
  return Math.abs(drift) < DRIFT_EPSILON ? 0 : drift;
}

/**
 * Whether the scroller actually went where it was told.
 *
 * Collapsing a tall block shortens the document, so holding the block at its
 * old offset can ask for a scrollTop past the end. The browser clamps that
 * write, the block never comes back, and every further frame would repeat a
 * forced layout for nothing — so a refused write ends the pinning.
 */
export function wasScrollApplied(target: number, achieved: number): boolean {
  return Math.abs(achieved - target) <= CLAMP_TOLERANCE;
}

function findScrollParent(element: HTMLElement): HTMLElement | null {
  let node = element.parentElement;
  while (node) {
    const overflowY = window.getComputedStyle(node).overflowY;
    if (
      (overflowY === "auto" || overflowY === "scroll") &&
      node.scrollHeight > node.clientHeight
    ) {
      return node;
    }
    node = node.parentElement;
  }
  return null;
}

type Measurement = {
  top: number;
  height: number;
  takenAt: number;
};

export function useCollapseScrollAnchor<
  T extends HTMLElement = HTMLDivElement,
>() {
  const anchorRef = useRef<T | null>(null);
  const measurementRef = useRef<Measurement | null>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => () => stopRef.current?.(), []);

  const measure = useCallback((): Measurement | null => {
    const element = anchorRef.current;
    if (!element) return null;
    const measurement: Measurement = {
      top: element.getBoundingClientRect().top,
      height: element.offsetHeight,
      takenAt: performance.now(),
    };
    measurementRef.current = measurement;
    return measurement;
  }, []);

  /**
   * Take the reading for the toggle that is about to happen. Goes on the block
   * in the capture phase, so it runs before the trigger's own handler — and
   * before the state change, which is the whole point.
   *
   * Only clicks on the block's own collapse trigger are read. A block holds
   * plenty of other things to click — approve buttons, links in tool output —
   * and a reading taken at one of those describes a moment that no toggle
   * followed.
   */
  const measureAnchor = useCallback(
    (event: { target: EventTarget | null }) => {
      const target = event.target;
      if (!(target instanceof Element && target.closest(TRIGGER_SELECTOR))) {
        return;
      }
      measure();
    },
    [measure],
  );

  const anchorToggle = useCallback(() => {
    stopRef.current?.();

    const element = anchorRef.current;
    const scroller = element ? findScrollParent(element) : null;
    if (!(element && scroller)) return;

    const clicked = measurementRef.current;
    const measurement =
      clicked && performance.now() - clicked.takenAt <= MEASUREMENT_TTL_MS
        ? clicked
        : measure();
    if (!measurement) return;
    measurementRef.current = null;

    const deadline = performance.now() + ANCHOR_WINDOW_MS;
    let frame = 0;

    const stop = () => {
      cancelAnimationFrame(frame);
      scroller.removeEventListener("wheel", stop);
      scroller.removeEventListener("touchmove", stop);
      scroller.removeEventListener("keydown", stop);
      scroller.removeEventListener("pointerdown", stop);
      stopRef.current = null;
    };
    stopRef.current = stop;

    const options = { passive: true } as const;
    scroller.addEventListener("wheel", stop, options);
    scroller.addEventListener("touchmove", stop, options);
    scroller.addEventListener("keydown", stop, options);
    scroller.addEventListener("pointerdown", stop, options);

    let height = measurement.height;
    let resized = false;
    let stillFrames = 0;

    const tick = () => {
      const node = anchorRef.current;
      if (!node) {
        stop();
        return;
      }

      const nextHeight = node.offsetHeight;
      if (nextHeight === height) {
        stillFrames += 1;
      } else {
        height = nextHeight;
        resized = true;
        stillFrames = 0;
      }

      // Before the block's height moves there is nothing to compensate for,
      // and pulling on the viewport would only fight whatever else is moving
      // it — following the agent's output, most likely.
      if (resized) {
        const correction = scrollCorrection(
          measurement.top,
          node.getBoundingClientRect().top,
        );
        if (correction !== 0) {
          const target = scroller.scrollTop + correction;
          scroller.scrollTop = target;
          if (!wasScrollApplied(target, scroller.scrollTop)) {
            stop();
            return;
          }
        }
      }

      const settled = resized && stillFrames >= STABLE_FRAMES;
      if (settled || performance.now() >= deadline) {
        stop();
        return;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }, [measure]);

  return { anchorRef, anchorToggle, measureAnchor };
}
