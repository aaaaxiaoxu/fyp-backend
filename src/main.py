from fastapi import FastAPI
from .graphrag_api import router as graphrag_router
from .graphrag_retriever import close_driver
app = FastAPI()

app.include_router(graphrag_router)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.on_event("shutdown")
async def _shutdown():
    await close_driver()
    
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
