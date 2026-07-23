export function reconcileApprovalRequestIds(
  localIds: ReadonlySet<string>,
  serverIds: ReadonlySet<string>,
): { confirmed: Set<string>; stale: Set<string> } {
  return {
    confirmed: new Set(serverIds),
    stale: new Set([...localIds].filter((id) => !serverIds.has(id))),
  };
}
