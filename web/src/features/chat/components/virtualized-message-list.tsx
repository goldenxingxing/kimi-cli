import type { LiveMessage } from "@/hooks/types";
import {
  Message,
  MessageActions,
  MessageAttachment,
  MessageAttachments,
  MessageContent,
  MessageCopyButton,
  MessageForkButton,
  UserMessageContent,
} from "@ai-elements";
import {
  AssistantMessage,
  type AssistantApprovalHandler,
} from "./assistant-message";

import type React from "react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  type ComponentPropsWithoutRef,
} from "react";
import { Virtuoso, type VirtuosoHandle } from "react-virtuoso";
import { cn } from "@/lib/utils";
import {
  MessageOutputTime,
  shouldShowMessageOutputTime,
} from "../message-output-time";
import { decideFollowOutput, nextCatchUp } from "./scroll-follow-policy";

export type VirtualizedMessageListProps = {
  messages: LiveMessage[];
  conversationKey: string;
  pendingApprovalMap: Record<string, boolean>;
  onApprovalAction?: AssistantApprovalHandler;
  canRespondToApproval: boolean;
  blocksExpanded: boolean;
  /** Index of message to highlight (for search) */
  highlightedMessageIndex?: number;
  /** Callback when scroll position changes */
  onAtBottomChange?: (atBottom: boolean) => void;
  /** Callback to fork session from before a specific turn */
  onForkSession?: (turnIndex: number) => void;
  /** True while history is being replayed into the list */
  isReplayingHistory?: boolean;
};

export type VirtualizedMessageListHandle = {
  scrollToIndex: (index: number, behavior?: "auto" | "smooth") => void;
  scrollToBottom: () => void;
};

type ConversationListItem = {
  message: LiveMessage;
  index: number;
};

function VirtuosoScrollerComponent(
  props: ComponentPropsWithoutRef<"div">,
  ref: React.Ref<HTMLDivElement>,
) {
  const { className, ...rest } = props;
  return (
    <div
      ref={ref}
      className={cn(
        "flex-1 overflow-y-auto overflow-x-hidden pr-1 sm:pr-2",
        className,
      )}
      {...rest}
    />
  );
}

const VirtuosoScroller = forwardRef(VirtuosoScrollerComponent);

function VirtuosoListComponent(
  props: ComponentPropsWithoutRef<"div">,
  ref: React.Ref<HTMLDivElement>,
) {
  const { className, ...rest } = props;
  return (
    <div
      ref={ref}
      className={cn("flex flex-col px-3 py-4 sm:px-6 lg:px-8", className)}
      {...rest}
    />
  );
}

const VirtuosoList = forwardRef(VirtuosoListComponent);

VirtuosoScroller.displayName = "VirtuosoScroller";
VirtuosoList.displayName = "VirtuosoList";

function getMessageSpacingClass(
  message: LiveMessage,
  index: number,
  allMessages: LiveMessage[],
): string | undefined {
  // Terminal-style message spacing - more compact
  // 1. User messages get breathing room (`mt-3`) from previous content
  // 2. Assistant messages flow naturally with minimal spacing
  // 3. Tool calls have subtle spacing to group related operations
  const previousMessage = index > 0 ? allMessages[index - 1] : undefined;
  const nextMessage =
    index < allMessages.length - 1 ? allMessages[index + 1] : undefined;

  const classes: string[] = [];

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const isToolMessage = isAssistant && message.variant === "tool";
  const isThinkingMessage = isAssistant && message.variant === "thinking";
  const previousIsUser = previousMessage?.role === "user";
  const previousIsAssistant = previousMessage?.role === "assistant";
  const previousIsTool =
    previousIsAssistant && previousMessage?.variant === "tool";

  if (index > 0) {
    if (isUser) {
      // User messages get more space from previous content
      classes.push("mt-4");
    } else if (isAssistant) {
      if (isToolMessage) {
        // Tool calls: slightly more breathing room between consecutive calls
        classes.push(previousIsUser ? "mt-2" : "mt-1.5");
      } else if (isThinkingMessage) {
        // Thinking blocks have minimal spacing
        classes.push(previousIsUser ? "mt-2" : "mt-1");
      } else if (previousIsTool) {
        // Text after tool gets slight spacing
        classes.push("mt-2");
      } else if (previousIsAssistant) {
        // Consecutive assistant messages flow together
        classes.push("mt-1");
      } else {
        // After user message
        classes.push("mt-2");
      }
    }
  }

  // Add bottom margin for the last message to avoid clashing with UI below
  if (!nextMessage) {
    classes.push("mb-30");
  }

  return classes.length > 0 ? classes.join(" ") : undefined;
}

function VirtualizedMessageListComponent(
  {
    messages,
    conversationKey,
    pendingApprovalMap,
    onApprovalAction,
    canRespondToApproval,
    blocksExpanded,
    highlightedMessageIndex = -1,
    onAtBottomChange,
    onForkSession,
    isReplayingHistory = false,
  }: VirtualizedMessageListProps,
  ref: React.Ref<VirtualizedMessageListHandle>,
) {
  const virtuosoRef = useRef<VirtuosoHandle | null>(null);
  const scrollerRef = useRef<HTMLElement | null>(null);

  // When the list is rebuilt from empty (WS reconnect / worker restart
  // clears messages, then history streams back in), stick to the bottom
  // until the viewport has settled there once.  This is independent of
  // the isReplayingHistory flag, which a session_status event can clear
  // prematurely while replay events are still in flight — without this
  // guard the 1500px gap rule below would pin the viewport to the top.
  const catchUpRef = useRef(true);
  const prevItemCountRef = useRef(0);

  // Filtered messages list (excluding message-id) aligned with listItems indices
  const filteredMessages = useMemo(
    () => messages.filter((m) => m.variant !== "message-id"),
    [messages],
  );

  const listItems = useMemo<ConversationListItem[]>(
    () =>
      filteredMessages.map((message, index) => ({ message, index })),
    [filteredMessages],
  );

  // Detect an empty -> non-empty transition (rebuild from scratch).
  // Assigned during render so the flag is set before Virtuoso invokes
  // followOutput for the incoming items (a useEffect would run after
  // Virtuoso's own effects — too late).
  if (prevItemCountRef.current === 0 && listItems.length > 0) {
    catchUpRef.current = nextCatchUp(catchUpRef.current, {
      type: "list-rebuilt",
    });
  }
  prevItemCountRef.current = listItems.length;

  const handleAtBottomChange = useCallback(
    (atBottom: boolean) => {
      // Deliberately does not end catch-up. Early in a replay only a few
      // messages exist, so the viewport reaches the bottom trivially; ending
      // catch-up there strands the reader at the top once the rest of the
      // history arrives. Only the reader's own scrolling ends it — see
      // nextCatchUp() and its tests.
      onAtBottomChange?.(atBottom);
    },
    [onAtBottomChange],
  );

  // Catch-up ends when the reader moves the viewport themselves. Wheel, touch,
  // and key events are the intent signals; programmatic scrolling (ours) is
  // not, which is why this listens for input rather than for scroll events.
  const detachReaderIntentRef = useRef<(() => void) | null>(null);
  const handleScrollerRef = useCallback(
    (ref: HTMLElement | Window | null) => {
      detachReaderIntentRef.current?.();
      detachReaderIntentRef.current = null;

      const element = ref instanceof HTMLElement ? ref : null;
      scrollerRef.current = element;
      if (!element) return;

      const takeControl = () => {
        catchUpRef.current = nextCatchUp(catchUpRef.current, {
          type: "reader-took-control",
        });
      };
      const options = { passive: true } as const;
      element.addEventListener("wheel", takeControl, options);
      element.addEventListener("touchmove", takeControl, options);
      element.addEventListener("keydown", takeControl, options);
      detachReaderIntentRef.current = () => {
        element.removeEventListener("wheel", takeControl);
        element.removeEventListener("touchmove", takeControl);
        element.removeEventListener("keydown", takeControl);
      };
    },
    [],
  );

  useEffect(
    () => () => {
      detachReaderIntentRef.current?.();
      detachReaderIntentRef.current = null;
    },
    [],
  );

  // Use a generous threshold to tolerate height estimation mismatches
  // when blocks are expanded (actual heights >> defaultItemHeight).
  // This is decoupled from atBottomStateChange which uses Virtuoso's
  // default tight threshold for the scroll-to-bottom button.
  const handleFollowOutput = useCallback(
    (isAtBottom: boolean) => {
      const scroller = scrollerRef.current;
      return decideFollowOutput({
        isAtBottom,
        isReplayingHistory,
        isCatchingUp: catchUpRef.current,
        gapToBottom: scroller
          ? scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight
          : null,
      });
    },
    [isReplayingHistory],
  );

  useImperativeHandle(
    ref,
    () => ({
      scrollToIndex: (
        index: number,
        behavior: "auto" | "smooth" = "smooth",
      ) => {
        virtuosoRef.current?.scrollToIndex({
          index,
          align: "center",
          behavior,
        });
      },
      scrollToBottom: () => {
        if (listItems.length > 0) {
          virtuosoRef.current?.scrollToIndex({
            index: listItems.length - 1,
            align: "end",
            behavior: "auto",
          });
        }
      },
    }),
    [listItems.length],
  );

  return (
    <Virtuoso
      key={conversationKey}
      ref={virtuosoRef}
      data={listItems}
      className="h-full"
      scrollerRef={handleScrollerRef}
      followOutput={handleFollowOutput}
      defaultItemHeight={160}
      increaseViewportBy={{ top: 400, bottom: 400 }}
      overscan={200}
      minOverscanItemCount={4}
      atBottomStateChange={handleAtBottomChange}
      initialTopMostItemIndex={{
        index: Math.max(0, listItems.length - 1),
        align: "end",
      }}
      components={{
        Scroller: VirtuosoScroller,
        List: VirtuosoList,
      }}
      computeItemKey={(_index: number, item: ConversationListItem) =>
        item.message.id
      }
      itemContent={(_index, item) => {
        const message = item.message;

        if (message.variant === "status") {
          return (
            <Message
              className={messages.length > 0 ? "mt-2" : undefined}
              from="assistant"
            >
              <MessageContent className="text-xs text-muted-foreground">
                {message.content}
              </MessageContent>
            </Message>
          );
        }

        const spacingClass = getMessageSpacingClass(
          message,
          item.index,
          filteredMessages,
        );

        const isHighlighted = item.index === highlightedMessageIndex;

        return (
          <Message
            className={cn(
              spacingClass,
              isHighlighted && "rounded-lg ring-2 ring-primary/50",
            )}
            from={message.role}
          >
            {message.role === "user" ? (
              message.content ? (
                <UserMessageContent>{message.content}</UserMessageContent>
              ) : null
            ) : (
              <>
                <AssistantMessage
                  message={message}
                  pendingApprovalMap={pendingApprovalMap}
                  onApprovalAction={onApprovalAction}
                  canRespondToApproval={canRespondToApproval}
                  blocksExpanded={blocksExpanded}
                />
                {!message.isStreaming &&
                  (!message.variant || message.variant === "text") &&
                  (message.content || (onForkSession && message.turnIndex !== undefined)) && (
                  <MessageActions className="
                  hover-reveal
                   opacity-0 group-hover:opacity-100 transition-opacity mt-1">
                    {message.content && <MessageCopyButton content={message.content} />}
                    {onForkSession && message.turnIndex !== undefined && (
                      <MessageForkButton onFork={() => onForkSession(message.turnIndex!)} />
                    )}
                    {shouldShowMessageOutputTime(message) && (
                      <MessageOutputTime completedAt={message.completedAt!} />
                    )}
                  </MessageActions>
                )}
              </>
            )}
            {message.attachments && message.attachments.length > 0 ? (
              <MessageAttachments>
                {message.attachments.map((attachment, attIdx) => {
                  const key =
                    "kind" in attachment
                      ? attachment.filename
                      : (attachment.filename ??
                        attachment.url ??
                        `${message.id}-${attIdx}`);
                  return (
                    <MessageAttachment
                      className="size-28 sm:size-32 lg:size-40"
                      data={attachment}
                      key={key}
                    />
                  );
                })}
              </MessageAttachments>
            ) : null}
          </Message>
        );
      }}
    />
  );
}

export const VirtualizedMessageList = forwardRef(
  VirtualizedMessageListComponent,
);
VirtualizedMessageList.displayName = "VirtualizedMessageList";
