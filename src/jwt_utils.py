from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from jose import jwt, JWTError

from .auth_security import new_urlsafe_token
from .settings import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: str) -> str:
    exp = _now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "type": "access", "exp": exp}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALG)


def create_refresh_token(user_id: str) -> tuple[str, str, datetime]:
    """
    返回 (refresh_jwt, jti, expires_at)
    """
    jti = new_urlsafe_token(24)
    exp = _now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": user_id, "type": "refresh", "jti": jti, "exp": exp}
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALG)
    return token, jti, exp


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALG])
    except JWTError as e:
        raise ValueError("Invalid token") from e
