Manage your todo list for tracking task progress.

Todo list is a simple yet powerful tool to help you get things done. You typically want to use this tool when the given task involves multiple subtasks/milestones, or, multiple tasks are given in a single request. This tool can help you to break down the task and track the progress.

**Usage modes:**

- **Update mode**: Pass `todos` to set the entire todo list. The previous list is replaced.
- **Query mode**: Omit `todos` (or pass null) to retrieve the current todo list without changes.
- **Clear mode**: Pass an empty array `[]` to clear all todos.

This is the only todo list tool available to you. That said, each time you want to update the todo list, you need to provide the whole list. Make sure to maintain the todo items and their statuses properly.

The list is shown to the user as live progress, so it has to match reality at every moment, not just at the end. Keep it honest by updating it *at each transition* rather than in batches:

- Mark an item `in_progress` before you begin it, and keep exactly one item `in_progress` at a time.
- The moment an item is finished, mark it `done` — in the same call that marks the next item `in_progress`, so one transition is one call.
- Never leave a finished item sitting at `pending` or `in_progress` while you work on something else. A stale list is worse than no list: the user reads it as your current state and cannot tell that you have already moved on.

That rhythm is one call per completed item, which is real progress between calls. Also, you can give yourself a self-encouragement to keep you motivated.

Abusing this tool to track too small steps will just waste your time and make your context messy. For example, here are some cases you should not use this tool:

- When the user just simply ask you a question. E.g. "What language and framework is used in the project?", "What is the best practice for x?"
- When it only takes a few steps/tool calls to complete the task. E.g. "Fix the unit test function 'test_xxx'", "Refactor the function 'xxx' to make it more solid."
- When the user prompt is very specific and the only thing you need to do is brainlessly following the instructions. E.g. "Replace xxx to yyy in the file zzz", "Create a file xxx with content yyy."

However, do not get stuck in a rut. Be flexible. Sometimes, you may try to use todo list at first, then realize the task is too simple and you can simply stop using it; or, sometimes, you may realize the task is complex after a few steps and then you can start using todo list to break it down.

IMPORTANT: Do not call this tool repeatedly without making real progress on at least one task between calls. If you are unsure about the current state, use Query mode (omit `todos`) to check before updating. If you find yourself unable to advance any task with your available tools, inform the user about what is blocking you instead of replanning. Repeatedly updating the todo list without doing actual work is counterproductive.
