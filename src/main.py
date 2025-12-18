from __future__ import annotations

from fastapi import FastAPI

from .db import init_db
from .auth_api import router as auth_router
from .conversation_api import router as conversation_router
from .graphrag_retriever import close_neo4j_driver

app = FastAPI()

app.include_router(auth_router)
app.include_router(conversation_router)

@app.on_event("startup")
async def _startup():
    await init_db()

@app.on_event("shutdown")
async def _shutdown():
    await close_neo4j_driver()
