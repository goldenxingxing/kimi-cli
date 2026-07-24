"""User-approved installation of a skill into OpenKimo's managed layer."""

from __future__ import annotations

import asyncio
import http.client
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

from kosong.tooling import BriefDisplayBlock, CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.skill.archive import ArchiveLimits, extract_skill_archive
from kimi_cli.skill.manager import SkillManager
from kimi_cli.soul.agent import Runtime


class Params(BaseModel):
    source_url: str = Field(description="HTTPS URL of a ZIP containing one skill.")
    replace: bool = Field(
        default=False,
        description="Replace an existing managed skill with the same name.",
    )


def validate_skill_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Skill source must be an HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("Skill source URL must not contain credentials")
    return url


class _HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_skill_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_skill_archive(url: str) -> bytes:
    validate_skill_url(url)
    opener = urllib.request.build_opener(_HTTPSRedirectHandler())
    request = urllib.request.Request(url, headers={"User-Agent": "OpenKimo/skill-installer"})
    maximum = ArchiveLimits().max_archive_bytes
    with opener.open(request, timeout=20) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Skill archive exceeds size limit")
    return data


def inspect_archive_name(data: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="openkimo-skill-inspect-") as directory:
        return extract_skill_archive(data, Path(directory)).name


class InstallSkill(CallableTool2[Params]):
    name: str = "InstallSkill"
    description: str = (
        "Install one skill from an HTTPS ZIP URL into OpenKimo's shared managed "
        "skill library. Always requires user approval before changing the library."
    )
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    async def __call__(self, params: Params) -> ToolReturnValue:
        try:
            validate_skill_url(params.source_url)
            data = await asyncio.to_thread(download_skill_archive, params.source_url)
            skill_name = await asyncio.to_thread(inspect_archive_name, data)
        except Exception as exc:
            return ToolError(message=str(exc), brief="Skill validation failed")

        result = await self._runtime.approval.request(
            self.name,
            "skill.install",
            (
                f"Install skill '{skill_name}' from {params.source_url} into "
                "OpenKimo's managed skill library"
            ),
        )
        if not result:
            return result.rejection_error()

        try:
            installed = await asyncio.to_thread(
                SkillManager().install_archive,
                data,
                replace=params.replace,
            )
        except FileExistsError:
            return ToolError(
                message=f"Skill '{skill_name}' already exists. Retry with replace=true.",
                brief="Skill already exists",
            )
        except Exception as exc:
            return ToolError(message=str(exc), brief="Skill installation failed")

        return ToolReturnValue(
            is_error=False,
            output=json.dumps({"name": installed.name, "installed": True}),
            message="",
            display=[BriefDisplayBlock(text=f"Installed skill: {installed.name}")],
        )
