from __future__ import annotations

from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .settings import settings


class Base(DeclarativeBase):
    pass


def _sqlite_url() -> str:
    p = Path(settings.SQLITE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{p.as_posix()}"


engine = create_async_engine(_sqlite_url(), echo=settings.SQLITE_ECHO)
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(bind=engine, expire_on_commit=False)


async def init_db() -> None:
    from . import models  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
