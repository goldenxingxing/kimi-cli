from types import SimpleNamespace

import pytest

from kimi_cli.tools.skill_install import InstallSkill, Params, validate_skill_url


@pytest.mark.parametrize(
    "url",
    ["http://example.com/skill.zip", "file:///tmp/skill.zip", "ftp://example.com/a"],
)
def test_validate_skill_url_requires_https(url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        validate_skill_url(url)


@pytest.mark.anyio
async def test_install_skill_requests_approval_before_install(monkeypatch) -> None:
    calls: list[str] = []

    class Approval:
        async def request(self, tool, action, description):
            calls.append(f"approve:{tool}:{action}")
            return SimpleNamespace(
                __bool__=lambda self: True,
                rejection_error=lambda: None,
            )

    class Manager:
        def install_archive(self, data, replace=False):
            calls.append("install")
            return SimpleNamespace(name="demo", description="Demo")

    runtime = SimpleNamespace(approval=Approval())
    monkeypatch.setattr(
        "kimi_cli.tools.skill_install.download_skill_archive",
        lambda url: b"archive",
    )
    monkeypatch.setattr(
        "kimi_cli.tools.skill_install.inspect_archive_name",
        lambda data: "demo",
    )
    monkeypatch.setattr(
        "kimi_cli.tools.skill_install.SkillManager",
        lambda: Manager(),
    )

    result = await InstallSkill(runtime)(Params(source_url="https://example.com/demo.zip"))

    assert calls == ["approve:InstallSkill:skill.install", "install"]
    assert result.is_error is False
