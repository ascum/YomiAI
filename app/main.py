"""
app/main.py — FastAPI application factory.

Entry point:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.lifespan import lifespan
from app.api.routes import health, search, recommend, interact, profile, llm, auth

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

app = FastAPI(
    title="NBA Multimodal Recommendation API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(search.router)
app.include_router(recommend.router)
app.include_router(interact.router)
app.include_router(profile.router)
app.include_router(llm.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
