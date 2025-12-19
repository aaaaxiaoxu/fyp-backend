from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .auth_security import hash_password, verify_password, new_urlsafe_token, sha256_hex
from .models import User, EmailVerificationToken, RefreshToken


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    # 统一使用 UTC aware
    return datetime.now(timezone.utc)


def _as_utc_aware(dt: datetime) -> datetime:
    """
    SQLite + SQLAlchemy 在 DateTime(timezone=True) 场景下，常见现象：
    - 写入时传入 aware datetime（带 tzinfo）
    - 读出时变成 naive datetime（tzinfo=None）
    直接和 _now() (aware) 比较会触发：
      TypeError: can't compare offset-naive and offset-aware datetimes
    因此这里做统一标准化：naive -> 视作 UTC。
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def get_user_by_email(session: AsyncSession, email: str) -> User | None:
    q = select(User).where(User.email == email.lower().strip())
    res = await session.execute(q)
    return res.scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: str) -> User | None:
    q = select(User).where(User.id == user_id)
    res = await session.execute(q)
    return res.scalar_one_or_none()


async def create_user(
    session: AsyncSession,
    email: str,
    password: str,
    nickname: str | None = None,
    avatar_url: str | None = None,
) -> User:
    nickname_val = nickname or email.split("@", 1)[0]
    avatar_val = avatar_url or "/pubilc/头像.jpeg"
    user = User(
        id=_uuid(),
        email=email.lower().strip(),
        password_hash=hash_password(password),
        is_verified=False,
        nickname=nickname_val,
        avatar_url=avatar_val,
        created_at=_now(),
    )
    session.add(user)
    await session.commit()
    return user


async def authenticate_user(session: AsyncSession, email: str, password: str) -> User | None:
    user = await get_user_by_email(session, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


async def create_email_verification(session: AsyncSession, user_id: str) -> str:
    raw = new_urlsafe_token(32)
    rec = EmailVerificationToken(
        id=_uuid(),
        user_id=user_id,
        token_hash=sha256_hex(raw),
        expires_at=_now() + timedelta(hours=24),
        used_at=None,
        created_at=_now(),
    )
    session.add(rec)
    await session.commit()
    return raw


async def verify_email_token(session: AsyncSession, raw_token: str) -> User | None:
    h = sha256_hex(raw_token)
    q = select(EmailVerificationToken).where(EmailVerificationToken.token_hash == h)
    res = await session.execute(q)
    rec = res.scalar_one_or_none()
    if not rec:
        return None

    if rec.used_at is not None:
        return None

    # ✅ 修复：naive/aware datetime compare
    if _as_utc_aware(rec.expires_at) < _now():
        return None

    await session.execute(
        update(EmailVerificationToken)
        .where(EmailVerificationToken.id == rec.id)
        .values(used_at=_now())
    )
    await session.execute(
        update(User).where(User.id == rec.user_id).values(is_verified=True)
    )
    await session.commit()

    return await get_user_by_id(session, rec.user_id)


async def store_refresh_jti(session: AsyncSession, user_id: str, jti: str, expires_at: datetime) -> None:
    rec = RefreshToken(
        id=_uuid(),
        user_id=user_id,
        jti_hash=sha256_hex(jti),
        expires_at=expires_at,
        revoked_at=None,
        created_at=_now(),
    )
    session.add(rec)
    await session.commit()


async def is_refresh_jti_valid(session: AsyncSession, jti: str) -> bool:
    h = sha256_hex(jti)
    q = select(RefreshToken).where(RefreshToken.jti_hash == h)
    res = await session.execute(q)
    rec = res.scalar_one_or_none()
    if not rec:
        return False
    if rec.revoked_at is not None:
        return False

    # ✅ 修复：naive/aware datetime compare
    if _as_utc_aware(rec.expires_at) < _now():
        return False

    return True


async def revoke_refresh_jti(session: AsyncSession, jti: str) -> None:
    h = sha256_hex(jti)
    q = select(RefreshToken).where(RefreshToken.jti_hash == h)
    res = await session.execute(q)
    rec = res.scalar_one_or_none()
    if not rec:
        return

    await session.execute(
        update(RefreshToken).where(RefreshToken.id == rec.id).values(revoked_at=_now())
    )
    await session.commit()
