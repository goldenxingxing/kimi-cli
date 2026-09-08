"""A default model that config.toml does not know must not be written to it.

`effective_model_names` merges config.toml with models built from environment
variables, so a model can be selectable without existing in the file. Config's
own validator refuses a `default_model` that is not in `models` — so persisting
an env-only name leaves a config.toml that raises on the next `load_config()`,
which is every worker restart and every start of the server after it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from kimi_cli.config import Config, LLMModel, LLMProvider
from kimi_cli.web.api import config as config_api


class _FakeRunner:
    async def restart_running_workers(self, **_kwargs):  # noqa: ANN003
        raise AssertionError("must not get as far as restarting workers")

    async def apply_compaction_ratio(self, _ratio):  # noqa: ANN001
        raise AssertionError("not part of this test")


def _config_with_one_model() -> Config:
    return Config(
        providers={
            "kimi": LLMProvider(type="kimi", base_url="https://api.example.invalid", api_key="k")
        },
        models={"kimi": LLMModel(provider="kimi", model="kimi-latest", max_context_size=1024)},
        default_model="kimi",
    )


def _http_request() -> SimpleNamespace:
    """A caller with no cookie and no bearer header: nobody is logged in.

    The handler reads `app.state.restrict_sensitive_apis` and resolves the
    caller to check their role; with no user, the role check does not apply.
    """
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(restrict_sensitive_apis=False)),
        cookies={},
        headers={},
    )


@pytest.fixture
def stub_config(monkeypatch: pytest.MonkeyPatch):
    saved: list[Config] = []
    monkeypatch.setattr(config_api, "load_config", _config_with_one_model)
    monkeypatch.setattr(config_api, "save_config", saved.append)
    # config.toml has "kimi"; "from-env" exists only because LLM_PROVIDERS
    # named it, which is exactly the case that used to be written to disk.
    monkeypatch.setattr(config_api, "get_effective_model_names", lambda: {"kimi", "from-env"})
    return saved


@pytest.mark.anyio
async def test_an_env_only_model_is_refused_rather_than_persisted(stub_config) -> None:
    with pytest.raises(HTTPException) as raised:
        await config_api.update_global_config(
            config_api.UpdateGlobalConfigRequest(default_model="from-env"),
            _http_request(),
            runner=_FakeRunner(),
        )

    assert raised.value.status_code == 400
    assert "config.toml" in raised.value.detail
    assert stub_config == []


@pytest.mark.anyio
async def test_an_env_only_model_is_refused_when_the_file_has_no_models_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-models case is the one the first version of this guard skipped.

    `config.models and ...` made it its own exception: with nothing in
    config.toml the check never ran, the assignment went through, and the
    default_thinking branch below saved the file that no longer loads.
    """
    saved: list[Config] = []
    empty = Config(providers={}, models={}, default_model="")
    monkeypatch.setattr(config_api, "load_config", lambda: empty)
    monkeypatch.setattr(config_api, "save_config", saved.append)
    monkeypatch.setattr(config_api, "get_effective_model_names", lambda: {"from-env"})

    with pytest.raises(HTTPException) as raised:
        await config_api.update_global_config(
            config_api.UpdateGlobalConfigRequest(default_model="from-env", default_thinking=True),
            _http_request(),
            runner=_FakeRunner(),
        )

    assert raised.value.status_code == 400
    assert saved == []


@pytest.mark.anyio
async def test_a_model_that_is_in_the_file_is_still_written(stub_config) -> None:
    with pytest.raises(AssertionError, match="restarting workers"):
        await config_api.update_global_config(
            config_api.UpdateGlobalConfigRequest(default_model="kimi"),
            _http_request(),
            runner=_FakeRunner(),
        )

    assert [c.default_model for c in stub_config] == ["kimi"]
