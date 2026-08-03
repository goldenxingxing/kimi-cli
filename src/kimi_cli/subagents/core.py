"""Shared core logic for preparing a subagent soul.

Both ``ForegroundSubagentRunner`` and ``BackgroundAgentRunner`` delegate
the repetitive build-restore-prompt pipeline to :func:`prepare_soul` so
that prompt enhancements (e.g. git context injection) only need to be
implemented once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.subagents.builder import SubagentBuilder
from kimi_cli.subagents.models import AgentLaunchSpec, AgentTypeDefinition
from kimi_cli.subagents.store import SubagentStore
from kimi_cli.utils.logging import logger
from kimi_cli.wiki.schema import content_hash
from kimi_cli.wiki.triggers import PersistedEvidenceRef

if TYPE_CHECKING:
    from kimi_cli.soul.agent import Runtime


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentRunSpec:
    """Everything needed to prepare a soul, without lifecycle concerns."""

    agent_id: str
    type_def: AgentTypeDefinition
    launch_spec: AgentLaunchSpec
    prompt: str
    resumed: bool
    run_generation: int = 0


async def prepare_soul(
    spec: SubagentRunSpec,
    runtime: Runtime,
    builder: SubagentBuilder,
    store: SubagentStore,
    on_stage: Callable[[str], None] | None = None,
) -> tuple[KimiSoul, str]:
    """Build agent, restore context, handle system prompt, write prompt file.

    Returns ``(soul, final_prompt)`` ready for execution via
    :func:`run_with_summary_continuation`.
    """

    # 1. Build agent from type definition
    agent = await builder.build_builtin_instance(
        agent_id=spec.agent_id,
        type_def=spec.type_def,
        launch_spec=spec.launch_spec,
        run_generation=spec.run_generation,
    )
    if on_stage:
        on_stage("agent_built")

    # 2. Restore conversation context
    context = Context(store.context_path(spec.agent_id))
    await context.restore()
    if on_stage:
        on_stage("context_restored")

    # 3. System prompt: reuse persisted prompt on resume, persist on first run
    if context.system_prompt is not None:
        agent = replace(agent, system_prompt=context.system_prompt)
    else:
        await context.write_system_prompt(agent.system_prompt)
    if on_stage:
        on_stage("context_ready")

    # 4. For new (non-resumed) explore agents, prepend git context to the prompt
    prompt = spec.prompt
    if spec.type_def.name == "explore" and not spec.resumed:
        from kimi_cli.subagents.git_context import collect_git_context

        git_ctx = await collect_git_context(runtime.builtin_args.KIMI_WORK_DIR)
        if git_ctx:
            prompt = f"{git_ctx}\n\n{prompt}"

    # 5. Write prompt snapshot (debugging aid)
    store.prompt_path(spec.agent_id).write_text(prompt, encoding="utf-8")

    # 6. Create soul
    soul = KimiSoul(agent, context=context)
    return soul, prompt


def sealed_subagent_evidence(subagent_runtime: Runtime) -> tuple[PersistedEvidenceRef, ...]:
    """Return one finished subagent run's hash-only grounding evidence."""
    reporter = subagent_runtime.wiki_evidence_reporter
    if reporter is None:
        return ()
    try:
        return reporter.seal_subagent_run()
    except Exception:
        logger.warning("Failed to seal subagent Wiki evidence; continuing without it")
        return ()


async def deliver_subagent_checkpoint(
    root_runtime: Runtime,
    subagent_runtime: Runtime,
    *,
    agent_id: str,
    run_generation: int,
    summary: str,
) -> str:
    """Hand a finished foreground run's evidence to the root as one checkpoint.

    Returns the managed block to append after the summary, or an empty string
    when there is nothing to hand over.  Every failure is silent: a subagent
    result must still reach the user when Wiki bookkeeping cannot.
    """
    coordinator = root_runtime.wiki_coordinator
    normalized = summary.strip()
    if coordinator is None or not normalized:
        return ""
    evidence = sealed_subagent_evidence(subagent_runtime)
    if not evidence:
        return ""
    receiving_root_turn_id = coordinator.active_turn_id
    if receiving_root_turn_id is None:
        return ""
    try:
        checkpoint = await coordinator.accept_subagent_result(
            agent_id=agent_id,
            run_generation=run_generation,
            summary_hash=content_hash(normalized.encode("utf-8")),
            evidence=evidence,
            receiving_root_turn_id=receiving_root_turn_id,
        )
    except Exception:
        logger.warning("Failed to attach subagent Wiki checkpoint; returning the summary as-is")
        return ""
    if checkpoint is None:
        return ""
    return await coordinator.render_checkpoints((checkpoint,))
