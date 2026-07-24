"""Strict data models for authoritative global Wiki Markdown."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal
from urllib.parse import parse_qsl, urlsplit
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    field_validator,
    model_validator,
)

from kimi_cli.utils.sensitive import is_sensitive_file

_SENSITIVE_URL_PARAMETERS = frozenset(
    {
        "apikey",
        "accesstoken",
        "auth",
        "authorization",
        "authtoken",
        "clientsecret",
        "credential",
        "credentials",
        "cookie",
        "cookies",
        "key",
        "password",
        "idtoken",
        "privatekey",
        "refreshtoken",
        "secret",
        "secretkey",
        "session",
        "sessionid",
        "sessiontoken",
        "signature",
        "sig",
        "token",
        "usertoken",
        "userpassword",
        "xamzcredential",
        "xamzsecuritytoken",
        "xamzsignature",
        "xgoogsignature",
    }
)


class UnsafeWikiPath(ValueError):
    """Raised when a Wiki logical or resolved path escapes its managed root."""


class UnsafeWikiPage(ValueError):
    """Raised when Wiki content cannot be stored safely."""


def validate_relative_source_path(value: str) -> str:
    """Validate portable provenance without allowing host-specific paths."""
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or windows_path.drive
        or windows_path.is_absolute()
        or value.startswith(("//", "\\\\"))
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
        or is_sensitive_file(value)
    ):
        raise ValueError("source paths must be safe relative POSIX paths")
    return value


def _normalized_query_name(value: str) -> str:
    """Canonicalize decoded query parameter names before secret-family matching."""
    return "".join(character for character in value.casefold() if character.isalnum())


def is_sensitive_url_parameter(value: str) -> bool:
    """Return whether a decoded URL parameter name is a credential-bearing alias."""
    normalized = _normalized_query_name(value)
    if normalized in _SENSITIVE_URL_PARAMETERS:
        return True
    camel_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    components = tuple(part.casefold() for part in re.findall(r"[A-Za-z0-9]+", camel_boundaries))
    if components[:2] not in {("x", "amz"), ("x", "goog")}:
        return False
    return "".join(components[2:]) in {"credential", "signature", "securitytoken"}


def has_sensitive_url_parameters(url: str) -> bool:
    """Inspect both query and fragment parameters using the shared alias rules."""
    parts = urlsplit(url)
    return any(
        is_sensitive_url_parameter(name)
        for component in (parts.query, parts.fragment)
        for name, _ in parse_qsl(component, keep_blank_values=True)
    )


def has_url_credentials(url: str) -> bool:
    """Return whether a URL has userinfo or sensitive query/fragment parameters."""
    parts = urlsplit(url)
    return (
        parts.username is not None
        or parts.password is not None
        or has_sensitive_url_parameters(url)
    )


class SourceRef(BaseModel):
    """Portable provenance for information recorded in a Wiki page."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["workspace-file", "conversation", "web"]
    workspace_id: UUID | None = None
    path: str | None = None
    session_id: UUID | None = None
    url: HttpUrl | None = None
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return validate_relative_source_path(value) if value is not None else None

    @model_validator(mode="after")
    def require_kind_specific_provenance(self) -> SourceRef:
        workspace = self.workspace_id is not None and self.path is not None
        if self.kind == "workspace-file":
            if not workspace or self.session_id is not None or self.url is not None:
                raise ValueError("workspace-file sources require only workspace_id and path")
        elif self.kind == "conversation":
            if self.session_id is None or any(
                value is not None for value in (self.workspace_id, self.path, self.url)
            ):
                raise ValueError("conversation sources require only session_id")
        else:
            if self.url is None or any(
                value is not None for value in (self.workspace_id, self.path, self.session_id)
            ):
                raise ValueError("web sources require only url")
            if has_url_credentials(str(self.url)):
                raise ValueError("web source URLs cannot contain credentials")
        return self


class CurrentSource(BaseModel):
    """Content supplied in the active session and eligible for controlled ingest."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline", "workspace-file"]
    content: str | None = None
    workspace_id: UUID | None = None
    relative_path: str | None = None

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        return validate_relative_source_path(value) if value is not None else None

    @model_validator(mode="after")
    def require_current_source_content(self) -> CurrentSource:
        if self.kind == "inline":
            if not self.content or self.workspace_id is not None or self.relative_path is not None:
                raise ValueError("inline sources require non-empty content only")
        elif self.content is not None or self.workspace_id is None or self.relative_path is None:
            raise ValueError("workspace-file sources require workspace_id and relative_path only")
        return self


class WikiPage(BaseModel):
    """A parsed content page, excluding special generated Wiki files."""

    model_config = ConfigDict(extra="forbid")

    logical_path: str
    title: str = Field(min_length=1, max_length=500)
    created: datetime
    updated: datetime
    tags: list[str] = Field(default_factory=list)
    sources: list[SourceRef]
    revision: PositiveInt
    body: str = Field(min_length=1)

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        # The public validator lives in schema.py; importing it lazily avoids a module cycle.
        from kimi_cli.wiki.schema import validate_logical_page

        return validate_logical_page(value).as_posix()

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title cannot be blank")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        if any(not tag or tag != tag.strip() for tag in value):
            raise ValueError("tags cannot be blank or padded")
        if len(set(value)) != len(value):
            raise ValueError("tags must be unique")
        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> WikiPage:
        if self.created.tzinfo is None or self.updated.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        if self.updated < self.created:
            raise ValueError("updated timestamp cannot precede created timestamp")
        return self


class PageChange(BaseModel):
    """A revision-checked replacement for one logical Wiki content page."""

    model_config = ConfigDict(extra="forbid")

    page: WikiPage
    expected_revision: PositiveInt | None = None


class WikiCandidate(BaseModel):
    """A proposed, still-uncommitted global Wiki update."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=500)
    pages: list[PageChange] = Field(min_length=1)
    sources: list[SourceRef]
    value: Literal["high", "medium", "low"]

    @model_validator(mode="after")
    def require_unique_pages(self) -> WikiCandidate:
        paths = [change.page.logical_path for change in self.pages]
        if len(set(paths)) != len(paths):
            raise ValueError("candidate pages must have distinct logical paths")
        return self
