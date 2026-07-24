import io
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from kimi_cli.skill.manager import SkillManager
from kimi_cli.web.api import admin
from kimi_cli.web.user_auth import require_admin


def _archive(name: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            f"{name}/SKILL.md",
            f"---\nname: {name}\ndescription: Uploaded\n---\n",
        )
    return stream.getvalue()


def test_admin_can_upload_and_manage_skills(monkeypatch, tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    writable = tmp_path / "skill"
    builtin.mkdir()
    factory = builtin / "factory"
    factory.mkdir()
    (factory / "SKILL.md").write_text(
        "---\nname: factory\ndescription: Factory\n---\n", encoding="utf-8"
    )
    monkeypatch.setattr(admin, "_skill_manager", lambda: SkillManager(builtin, writable))

    app = FastAPI()
    app.include_router(admin.router)
    app.dependency_overrides[require_admin] = lambda: {"id": "admin", "role": "admin"}
    client = TestClient(app)

    listed = client.get("/api/admin/skills")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "factory"
    assert "path" not in listed.json()[0]

    uploaded = client.post(
        "/api/admin/skills/upload",
        files={"file": ("custom.zip", _archive("custom"), "application/zip")},
    )
    assert uploaded.status_code == 201
    assert uploaded.json()["origin"] == "user"

    markdown = client.post(
        "/api/admin/skills/upload",
        files={
            "file": (
                "single.md",
                b"---\nname: single\ndescription: Single file\n---\n",
                "text/markdown",
            )
        },
    )
    assert markdown.status_code == 201
    assert markdown.json()["name"] == "single"

    disabled = client.post("/api/admin/skills/factory/disable")
    assert disabled.status_code == 200
    assert disabled.json()["enabled"] is False

    restored = client.post("/api/admin/skills/factory/restore")
    assert restored.status_code == 200
    assert restored.json()["enabled"] is True

    removed = client.delete("/api/admin/skills/custom")
    assert removed.status_code == 204
    assert all(item["name"] != "custom" for item in client.get("/api/admin/skills").json())
