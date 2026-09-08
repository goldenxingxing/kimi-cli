"""Branding API endpoints."""

from __future__ import annotations

import base64
import re
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from kimi_cli.web.db.crud import (
    delete_all_branding,
    get_branding,
    upsert_branding,
)
from kimi_cli.web.db.database import db_session
from kimi_cli.web.user_auth import require_admin

# ---------------------------------------------------------------------------
# Data URL validation patterns
# ---------------------------------------------------------------------------

_LOGO_MIME_PATTERN = re.compile(
    r"^data:image/(png|svg\+xml|jpeg);base64,[A-Za-z0-9+/\n]+=*$",
    re.DOTALL,
)
_FAVICON_MIME_PATTERN = re.compile(
    r"^data:image/(x-icon|png|svg\+xml);base64,[A-Za-z0-9+/\n]+=*$",
    re.DOTALL,
)


def _check_data_url_size(data_url: str, *, max_kb: int, field: str) -> None:
    """Reject a Data URL whose payload is over the limit, or is not base64.

    The length is checked before anything is decoded: base64 is 4 characters
    per 3 bytes, so the encoded length bounds the decoded one, and a 500 MB
    request body no longer has to be materialised in memory to find out it was
    too big.

    The size error and the parse error are also kept apart properly.
    `except (IndexError, Exception)` is just `except Exception` — it caught the
    ValueError raised two lines above it — and told the two cases apart by
    looking for the word "exceeds" in the message.
    """
    limit = max_kb * 1024
    _, _, b64_part = data_url.partition(",")
    if not b64_part:
        raise ValueError(f"{field} contains invalid base64 data")

    # 4 characters per 3 bytes, minus whatever the padding stands in for: the
    # decoded length exactly for a padded payload, and a slight under-estimate
    # for an unpadded one — which is the safe direction, since the check below
    # is the authoritative one and this only decides whether to decode at all.
    # Whitespace is stripped first: the MIME pattern allows newlines inside the
    # payload, and counting those would reject a file that is within the limit.
    compact = "".join(b64_part.split())
    padding = len(compact) - len(compact.rstrip("="))
    if len(compact) // 4 * 3 - padding > limit:
        raise ValueError(f"{field} decoded size exceeds {max_kb} KB")

    try:
        raw = base64.b64decode(b64_part)
    except Exception as e:
        raise ValueError(f"{field} contains invalid base64 data") from e

    if len(raw) > limit:
        raise ValueError(f"{field} decoded size exceeds {max_kb} KB")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class BrandingResponse(BaseModel):
    """Response model shared by public and admin branding endpoints."""

    brand_name: str | None = None
    version: str | None = None
    page_title: str | None = None
    logo_url: str | None = None
    logo: str | None = None
    favicon: str | None = None


class UpdateBrandingRequest(BaseModel):
    """PUT /api/admin/branding request body.

    Each field is optional; passing ``null`` clears the setting.
    """

    brand_name: str | None = None
    version: str | None = None
    page_title: str | None = None
    logo_url: str | None = None
    logo: str | None = None
    favicon: str | None = None

    @field_validator("brand_name")
    @classmethod
    def validate_brand_name(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 30:
            raise ValueError("brand_name must be <= 30 characters")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 20:
            raise ValueError("version must be <= 20 characters")
        return v

    @field_validator("page_title")
    @classmethod
    def validate_page_title(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 60:
            raise ValueError("page_title must be <= 60 characters")
        return v

    @field_validator("logo_url")
    @classmethod
    def validate_logo_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("logo_url must start with http:// or https://")
        return v

    @field_validator("logo")
    @classmethod
    def validate_logo(cls, v: str | None) -> str | None:
        if v is not None:
            if not _LOGO_MIME_PATTERN.match(v):
                raise ValueError("logo must be a valid Data URL (PNG/SVG/JPEG)")
            _check_data_url_size(v, max_kb=512, field="logo")
        return v

    @field_validator("favicon")
    @classmethod
    def validate_favicon(cls, v: str | None) -> str | None:
        if v is not None:
            if not _FAVICON_MIME_PATTERN.match(v):
                raise ValueError("favicon must be a valid Data URL (ICO/PNG/SVG)")
            _check_data_url_size(v, max_kb=256, field="favicon")
        return v


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Public router — exempted from auth in AuthMiddleware
public_router = APIRouter(prefix="/api/branding", tags=["branding"])

# Admin router — protected by require_admin dependency
admin_router = APIRouter(prefix="/api/admin/branding", tags=["admin-branding"])


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------


@public_router.get("", summary="Get current branding settings")
async def get_public_branding() -> BrandingResponse:
    """Return the current branding configuration.

    This endpoint does not require authentication so it can be called from
    the login page and other unauthenticated contexts.
    """
    with db_session() as db:
        data = get_branding(db)
    return BrandingResponse(**data)


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------


@admin_router.get("", summary="Get branding settings (admin)")
async def get_admin_branding(
    admin: dict[str, Any] = Depends(require_admin),
) -> BrandingResponse:
    """Return branding settings for the admin form."""
    with db_session() as db:
        data = get_branding(db)
    return BrandingResponse(**data)


@admin_router.put("", summary="Update branding settings (admin)")
async def update_branding(
    body: UpdateBrandingRequest,
    admin: dict[str, Any] = Depends(require_admin),
) -> BrandingResponse:
    """Update branding settings. Only include fields that need updating;
    pass ``null`` to clear a setting.
    """
    settings = body.model_dump()
    with db_session() as db:
        upsert_branding(db, settings)
        data = get_branding(db)
    return BrandingResponse(**data)


@admin_router.delete("", summary="Reset branding to defaults (admin)", status_code=204)
async def reset_branding(
    admin: dict[str, Any] = Depends(require_admin),
) -> None:
    """Clear all custom branding settings, restoring defaults."""
    with db_session() as db:
        delete_all_branding(db)


__all__ = ["admin_router", "public_router"]
