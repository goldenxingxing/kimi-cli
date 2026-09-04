/**
 * When a reconnecting session may clear the conversation on screen.
 *
 * Connecting used to empty the message list immediately. For a session switch
 * that is right -- the old conversation is gone. For a *re*connect it means the
 * reader watches their conversation vanish, a spinner take its place (the list
 * is swapped out entirely while `messages` is empty), and the whole thing get
 * rebuilt as history replays. Every blip repaints the entire message area.
 *
 * So the clear is deferred until its replacement is actually arriving. The
 * server sends replayed content first and `history_complete` after it, so the
 * first message that carries conversation content clears the old list in the
 * same tick that appends the new one -- React batches them and no empty frame
 * is ever painted. A session whose history really is empty sends no content at
 * all, and `history_complete` clears it instead.
 */

/** The shape this decision needs; the wire type carries much more. */
export type ResetProbe = {
  method?: string;
  id?: string | number | null;
};

/**
 * Whether this message means the incoming conversation has started arriving.
 *
 * Status updates and the reply to `initialize` are bookkeeping: they say
 * nothing about the conversation, and clearing on them would paint the empty
 * frame this exists to avoid.
 */
export function startsIncomingConversation(
  message: ResetProbe,
  initializeId: string | number | null,
): boolean {
  if (message.method === "session_status") {
    return false;
  }
  if (initializeId !== null && message.id != null && message.id === initializeId) {
    return false;
  }
  return true;
}
