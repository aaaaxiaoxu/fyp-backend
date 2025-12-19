from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SessionLocal
from .settings import settings
from .emailer import send_verification_email
from .auth_store import (
    get_user_by_email,
    create_user,
    create_email_verification,
    verify_email_token,
    authenticate_user,
    store_refresh_jti,
    is_refresh_jti_valid,
    revoke_refresh_jti,
)
from .jwt_utils import create_access_token, create_refresh_token, decode_token

router = APIRouter(prefix="/auth", tags=["Auth"])


async def get_db():
    async with SessionLocal() as session:
        yield session


def _cookie_kwargs(max_age_seconds: int) -> dict:
    return dict(
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        path="/",
        max_age=max_age_seconds,
    )


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5)
    password: str = Field(min_length=8)
    nickname: str | None = Field(default=None, max_length=200)
    avatar_url: str | None = Field(default=None, max_length=500)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5)
    password: str = Field(min_length=8)


@router.post("/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    email = req.email.lower().strip()
    if await get_user_by_email(db, email):
        raise HTTPException(status_code=400, detail="Email already registered")

    user = await create_user(
        db,
        email=email,
        password=req.password,
        nickname=req.nickname,
        avatar_url=req.avatar_url,
    )
    raw = await create_email_verification(db, user_id=user.id)

    verify_url = f"{settings.APP_BASE_URL}/auth/verify?token={raw}"
    await send_verification_email(to_email=user.email, verify_url=verify_url)

    return {
        "ok": True,
        "message": "Registered. Please verify your email.",
        "user": {
            "id": user.id,
            "email": user.email,
            "nickname": user.nickname,
            "avatar_url": user.avatar_url,
            "is_verified": user.is_verified,
        },
    }


@router.get("/verify")
async def verify(token: str = Query(...), db: AsyncSession = Depends(get_db)):
    user = await verify_email_token(db, token)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    return {"ok": True, "message": "Email verified"}


@router.post("/login")
async def login(req: LoginRequest, resp: Response, db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Email not verified")

    access = create_access_token(user.id)
    refresh, jti, refresh_exp = create_refresh_token(user.id)
    await store_refresh_jti(db, user_id=user.id, jti=jti, expires_at=refresh_exp)

    resp.set_cookie(
        settings.ACCESS_COOKIE_NAME,
        access,
        **_cookie_kwargs(settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60),
    )
    resp.set_cookie(
        settings.REFRESH_COOKIE_NAME,
        refresh,
        **_cookie_kwargs(settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600),
    )

    return {
        "ok": True,
        "access_token": access,
        "user": {
            "id": user.id,
            "email": user.email,
            "nickname": user.nickname,
            "avatar_url": user.avatar_url,
            "is_verified": user.is_verified,
        },
    }


@router.post("/refresh")
async def refresh(request: Request, resp: Response, db: AsyncSession = Depends(get_db)):
    refresh_token = request.cookies.get(settings.REFRESH_COOKIE_NAME, "")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")

    try:
        payload = decode_token(refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token type")

    user_id = payload.get("sub")
    jti = payload.get("jti")
    if not user_id or not jti:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if not await is_refresh_jti_valid(db, jti):
        raise HTTPException(status_code=401, detail="Refresh token revoked or expired")

    # 轮换：撤销旧 refresh，发新 refresh
    await revoke_refresh_jti(db, jti)

    access = create_access_token(user_id)
    new_refresh, new_jti, new_exp = create_refresh_token(user_id)
    await store_refresh_jti(db, user_id=user_id, jti=new_jti, expires_at=new_exp)

    resp.set_cookie(
        settings.ACCESS_COOKIE_NAME,
        access,
        **_cookie_kwargs(settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60),
    )
    resp.set_cookie(
        settings.REFRESH_COOKIE_NAME,
        new_refresh,
        **_cookie_kwargs(settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600),
    )

    return {"ok": True, "access_token": access}


@router.post("/logout")
async def logout(request: Request, resp: Response, db: AsyncSession = Depends(get_db)):
    refresh_token = request.cookies.get(settings.REFRESH_COOKIE_NAME, "")
    if refresh_token:
        try:
            payload = decode_token(refresh_token)
            if payload.get("type") == "refresh" and payload.get("jti"):
                await revoke_refresh_jti(db, payload["jti"])
        except Exception:
            pass

    resp.delete_cookie(settings.ACCESS_COOKIE_NAME, path="/")
    resp.delete_cookie(settings.REFRESH_COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
async def me(request: Request, db: AsyncSession = Depends(get_db)):
    access = request.cookies.get(settings.ACCESS_COOKIE_NAME, "")
    if not access:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        payload = decode_token(access)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    from .auth_store import get_user_by_id
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "id": user.id,
        "email": user.email,
        "nickname": user.nickname,
        "avatar_url": user.avatar_url,
        "is_verified": user.is_verified,
    }
