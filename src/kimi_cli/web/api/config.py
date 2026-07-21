"""Config API routes."""

from __future__ import annotations

import os
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from kimi_cli import logger
from kimi_cli.config import LLMModel, get_config_file, load_config, save_config
from kimi_cli.llm import (
    ALL_MODEL_CAPABILITIES,
    ModelCapability,
    ProviderType,
    derive_model_capabilities,
    parse_llm_providers_env,
)
from kimi_cli.web.runner.process import KimiCLIRunner

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigModel(LLMModel):
    """Model configuration for frontend."""

    name: str = Field(description="Model key in kimi-cli config (Config.models)")
    provider_type: ProviderType = Field(description="Provider type (LLMProvider.type)")


class GlobalConfig(BaseModel):
    """Global configuration snapshot for frontend."""

    default_model: str = Field(description="Current default model key")
    default_thinking: bool = Field(description="Current default thinking mode")
    models: list[ConfigModel] = Field(description="All configured models")


class UpdateGlobalConfigRequest(BaseModel):
    """Request to update global config."""

    default_model: str | None = Field(default=None, description="New default model key")
    default_thinking: bool | None = Field(default=None, description="New default thinking mode")
    restart_running_sessions: bool | None = Field(
        default=None, description="Whether to restart running sessions"
    )
    force_restart_busy_sessions: bool | None = Field(
        default=None, description="Whether to force restart busy sessions"
    )


class UpdateGlobalConfigResponse(BaseModel):
    """Response after updating global config."""

    config: GlobalConfig = Field(description="Updated config snapshot")
    restarted_session_ids: list[str] | None = Field(
        default=None, description="IDs of restarted sessions"
    )
    skipped_busy_session_ids: list[str] | None = Field(
        default=None, description="IDs of busy sessions that were skipped"
    )


class ConfigToml(BaseModel):
    """Raw config.toml content."""

    content: str = Field(description="Raw TOML content")
    path: str = Field(description="Path to config file")


class UpdateConfigTomlRequest(BaseModel):
    """Request to update config.toml."""

    content: str = Field(description="New TOML content")


class UpdateConfigTomlResponse(BaseModel):
    """Response after updating config.toml."""

    success: bool = Field(description="Whether the update was successful")
    error: str | None = Field(default=None, description="Error message if failed")


def _build_global_config() -> GlobalConfig:
    """Build GlobalConfig from kimi-cli config.

    Environment variables (``LLM_PROVIDERS`` and legacy vendor-prefixed keys)
    are **merged** into ``config.toml``: any provider/model defined in the
    environment but absent from the config file is appended automatically so
    that the frontend can see and switch between them.
    """
    from pydantic import SecretStr

    from kimi_cli.config import LLMProvider

    config = load_config()
    env_models_added = False

    # ------------------------------------------------------------------
    # 1. Merge LLM_PROVIDERS (YAML) into config.toml
    # ------------------------------------------------------------------
    providers_list = parse_llm_providers_env()
    if providers_list:
        for p in providers_list:
            name = p.get("name")
            provider_type = p.get("type")
            api_key = p.get("api_key", "")
            base_url = p.get("base_url", "")
            model_name = p.get("model")
            max_ctx = p.get("max_context_size", 262144)
            caps_raw: Any = p.get("capabilities", [])

            if not name or not provider_type or not model_name:
                logger.warning(
                    "Skipping LLM_PROVIDERS entry missing required fields: {entry}",
                    entry=p,
                )
                continue

            try:
                max_ctx_int = int(max_ctx)
            except (TypeError, ValueError):
                max_ctx_int = 262144

            caps_set: set[str] = set()
            if isinstance(caps_raw, list):
                caps_list = cast(list[Any], caps_raw)
                caps_set = {str(c).strip().lower() for c in caps_list if c}
            elif isinstance(caps_raw, str):
                caps_set = {c.strip().lower() for c in caps_raw.split(",") if c.strip()}

            model_caps: set[ModelCapability] = {
                cap for cap in ALL_MODEL_CAPABILITIES if cap in caps_set
            }

            # Add provider if not already present
            if name not in config.providers:
                config.providers[name] = LLMProvider(
                    type=provider_type,
                    base_url=base_url,
                    api_key=SecretStr(api_key),
                )
                env_models_added = True

            # Add model if not already present
            if name not in config.models:
                config.models[name] = LLMModel(
                    provider=name,
                    model=model_name,
                    max_context_size=max_ctx_int,
                    capabilities=model_caps or None,
                )
                env_models_added = True

        if not config.default_model:
            default_provider = os.environ.get("LLM_DEFAULT_PROVIDER", "").lower()
            if default_provider:
                for model_name in config.models:
                    if model_name.lower() == default_provider:
                        config.default_model = model_name
                        break
            if not config.default_model and config.models:
                config.default_model = next(iter(config.models.keys()))

    # ------------------------------------------------------------------
    # 2. Legacy vendor-prefixed env vars (only if no LLM_PROVIDERS)
    # ------------------------------------------------------------------
    if not providers_list:
        # Kimi / Moonshot
        kimi_key = os.environ.get("KIMI_API_KEY")
        if kimi_key:
            model_name = os.environ.get("KIMI_MODEL_NAME", "kimi-k2")
            max_ctx = int(os.environ.get("KIMI_MODEL_MAX_CONTEXT_SIZE", "262144"))
            if model_name not in config.models:
                config.models[model_name] = LLMModel(
                    provider="kimi",
                    model=model_name,
                    max_context_size=max_ctx,
                )
                env_models_added = True
            if "kimi" not in config.providers:
                config.providers["kimi"] = LLMProvider(
                    type="kimi",
                    base_url=os.environ.get("KIMI_BASE_URL", ""),
                    api_key=SecretStr(kimi_key),
                )
                env_models_added = True

        # OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
            max_ctx = int(os.environ.get("OPENAI_MODEL_MAX_CONTEXT_SIZE", "128000"))
            if model_name not in config.models:
                config.models[model_name] = LLMModel(
                    provider="openai",
                    model=model_name,
                    max_context_size=max_ctx,
                )
                env_models_added = True
            if "openai" not in config.providers:
                config.providers["openai"] = LLMProvider(
                    type="openai_legacy",
                    base_url=os.environ.get("OPENAI_BASE_URL", ""),
                    api_key=SecretStr(openai_key),
                )
                env_models_added = True

        # Anthropic
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            model_name = os.environ.get("ANTHROPIC_MODEL_NAME", "claude-3-5-sonnet-20241022")
            max_ctx = int(os.environ.get("ANTHROPIC_MODEL_MAX_CONTEXT_SIZE", "200000"))
            if model_name not in config.models:
                config.models[model_name] = LLMModel(
                    provider="anthropic",
                    model=model_name,
                    max_context_size=max_ctx,
                )
                env_models_added = True
            if "anthropic" not in config.providers:
                config.providers["anthropic"] = LLMProvider(
                    type="anthropic",
                    base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
                    api_key=SecretStr(anthropic_key),
                )
                env_models_added = True

        if not config.default_model and config.models:
            llm_provider = os.environ.get("LLM_PROVIDER", "kimi").lower()
            for m_name, m in config.models.items():
                if m.provider == llm_provider:
                    config.default_model = m_name
                    break
            if not config.default_model:
                config.default_model = next(iter(config.models.keys()))

    # Persist merged providers/models back to config.toml
    if env_models_added:
        save_config(config)
        logger.info("Merged environment LLM providers into config.toml")

    # ------------------------------------------------------------------
    # 3. Build frontend-facing GlobalConfig
    # ------------------------------------------------------------------
    models: list[ConfigModel] = []
    for model_name, model in config.models.items():
        provider = config.providers.get(model.provider)
        if provider is None:
            continue

        derived_caps = derive_model_capabilities(model)
        capabilities = derived_caps or None

        models.append(
            ConfigModel(
                name=model_name,
                model=model.model,
                provider=model.provider,
                provider_type=provider.type,
                max_context_size=model.max_context_size,
                capabilities=capabilities,
            )
        )

    default_thinking = config.default_thinking

    return GlobalConfig(
        default_model=config.default_model,
        default_thinking=default_thinking,
        models=models,
    )


def _get_runner(req: Request) -> KimiCLIRunner:
    """Get KimiCLIRunner from FastAPI app state."""
    return req.app.state.runner


def _ensure_sensitive_apis_allowed(request: Request) -> None:
    """Block sensitive config writes when restricted."""
    if getattr(request.app.state, "restrict_sensitive_apis", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sensitive config APIs are disabled in this mode.",
        )


@router.get("/", summary="Get global (kimi-cli) config snapshot")
async def get_global_config() -> GlobalConfig:
    """Get global (kimi-cli) config snapshot."""
    return _build_global_config()


@router.patch("/", summary="Update global (kimi-cli) default model/thinking")
async def update_global_config(
    request: UpdateGlobalConfigRequest,
    http_request: Request,
    runner: KimiCLIRunner = Depends(_get_runner),
) -> UpdateGlobalConfigResponse:
    """Update global (kimi-cli) default model/thinking."""
    _ensure_sensitive_apis_allowed(http_request)
    config = load_config()

    # Build effective model list (includes env-var fallback models)
    global_config = _build_global_config()
    effective_model_names = {m.name for m in global_config.models}

    # Validate and update default_model
    if request.default_model is not None:
        if request.default_model not in effective_model_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.default_model}' not found in config",
            )
        config.default_model = request.default_model

    # Update default_thinking
    if request.default_thinking is not None:
        config.default_thinking = request.default_thinking

    # Save config.  If models are currently sourced from environment
    # variables (config.toml has no model entries) we only persist
    # default_thinking so that we don't write a default_model that
    # references a model absent from config.toml, which would break
    # Pydantic validation on the next load.
    had_models_before = bool(config.models)
    if had_models_before:
        save_config(config)
    elif request.default_thinking is not None:
        # Preserve original default_model (likely empty) and only update thinking
        save_config(config)

    # Restart running workers to apply config changes
    restarted: list[str] = []
    skipped_busy: list[str] = []

    restart_running = request.restart_running_sessions
    if restart_running is None:
        restart_running = True  # Default to restarting sessions

    if restart_running:
        summary = await runner.restart_running_workers(
            reason="config_update",
            force=request.force_restart_busy_sessions or False,
        )
        restarted = [str(sid) for sid in summary.restarted_session_ids]
        skipped_busy = [str(sid) for sid in summary.skipped_busy_session_ids]

    return UpdateGlobalConfigResponse(
        config=_build_global_config(),
        restarted_session_ids=restarted if restarted else None,
        skipped_busy_session_ids=skipped_busy if skipped_busy else None,
    )


@router.get("/toml", summary="Get kimi-cli config.toml")
async def get_config_toml(http_request: Request) -> ConfigToml:
    """Get kimi-cli config.toml."""
    _ensure_sensitive_apis_allowed(http_request)
    config_file = get_config_file()
    if not config_file.exists():
        return ConfigToml(content="", path=str(config_file))
    return ConfigToml(content=config_file.read_text(encoding="utf-8"), path=str(config_file))


@router.put("/toml", summary="Update kimi-cli config.toml")
async def update_config_toml(
    request: UpdateConfigTomlRequest,
    http_request: Request,
    runner: KimiCLIRunner = Depends(_get_runner),
) -> UpdateConfigTomlResponse:
    """Update kimi-cli config.toml."""
    from kimi_cli.config import load_config_from_string

    _ensure_sensitive_apis_allowed(http_request)
    try:
        # Validate the config first
        load_config_from_string(request.content)

        # Write to file
        config_file = get_config_file()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(request.content, encoding="utf-8")

        # Config only reaches a session worker when it (re)starts; restart
        # idle workers so provider edits apply to live sessions instead of
        # silently staying stale until the next server restart.
        try:
            summary = await runner.restart_running_workers(
                reason="config_update", force=False
            )
            if summary.skipped_busy_session_ids:
                logger.info(
                    "config.toml updated; {n} busy session(s) keep the old "
                    "config until their next restart",
                    n=len(summary.skipped_busy_session_ids),
                )
        except Exception as e:
            logger.warning(f"Failed to restart workers after config.toml update: {e}")

        return UpdateConfigTomlResponse(success=True)
    except Exception as e:
        logger.warning(f"Failed to update config.toml: {e}")
        return UpdateConfigTomlResponse(success=False, error=str(e))
