from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SessionLocal
from .jwt_utils import decode_token
from .auth_store import get_user_by_id
from .settings import settings


async def get_db():
    async with SessionLocal() as session:
        yield session


def _get_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip()
    return None


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    token = _get_bearer_token(request) or request.cookies.get(settings.ACCESS_COOKIE_NAME, "")
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")

    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user
