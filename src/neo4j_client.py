from __future__ import annotations

from typing import AsyncIterator

from neo4j import AsyncDriver, AsyncGraphDatabase

from .settings import settings

_driver: AsyncDriver | None = None


async def init_neo4j_driver() -> None:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )


async def close_neo4j_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def get_neo4j_driver() -> AsyncIterator[AsyncDriver]:
    if _driver is None:
        await init_neo4j_driver()
    assert _driver is not None
    yield _driver
