/**
 * When the message list should keep following new output.
 *
 * The list is rebuilt from empty whenever the socket reconnects or the worker
 * restarts, and history then streams back in over many frames. While that is
 * happening the reader has expressed no intent, so the viewport must stay at
 * the newest message; once the reader scrolls away deliberately, their position
 * is theirs to keep.
 *
 * This is kept as plain data-in/data-out so the policy can be tested without a
 * DOM or a virtual list.
 */

export type FollowOutputDecision = "auto" | false;

export type FollowOutputState = {
  /** The virtual list's own at-bottom reading. */
  isAtBottom: boolean;
  /** History is being replayed into a list that was emptied. */
  isReplayingHistory: boolean;
  /** The list was rebuilt and the reader has not taken control since. */
  isCatchingUp: boolean;
  /** Pixels between the viewport bottom and the content bottom, if measurable. */
  gapToBottom: number | null;
};

/**
 * Height estimates for collapsed blocks are far off once blocks expand, so the
 * "close enough to the bottom to keep following" test is deliberately loose.
 */
export const FOLLOW_GAP_TOLERANCE = 1500;

export function decideFollowOutput(
  state: FollowOutputState,
): FollowOutputDecision {
  if (state.isReplayingHistory || state.isCatchingUp) {
    return "auto";
  }
  if (state.isAtBottom) {
    return "auto";
  }
  if (state.gapToBottom !== null && state.gapToBottom <= FOLLOW_GAP_TOLERANCE) {
    return "auto";
  }
  return false;
}

export type CatchUpEvent =
  /** The list went from empty to populated — a reconnect or worker restart. */
  | { type: "list-rebuilt" }
  /** The reader scrolled, dragged, or keyed the viewport themselves. */
  | { type: "reader-took-control" }
  /** The viewport happened to reach the bottom. */
  | { type: "reached-bottom" };

/**
 * Advance the catch-up flag.
 *
 * `reached-bottom` deliberately does not end catch-up. Early in a replay only a
 * handful of messages exist, so the viewport sits at the bottom trivially; if
 * that ended catch-up, the hundreds of messages still arriving would push the
 * content far past the gap tolerance and the reader would be left staring at
 * the top of their own history.
 */
export function nextCatchUp(current: boolean, event: CatchUpEvent): boolean {
  switch (event.type) {
    case "list-rebuilt":
      return true;
    case "reader-took-control":
      return false;
    case "reached-bottom":
      return current;
  }
}
