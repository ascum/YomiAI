"""
app/api/schemas.py — Pydantic request/response models.

Moved from api.py (SearchRequest, InteractRequest, AskLLMRequest).
"""
from typing import Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str = ""
    image_base64: Optional[str] = None
    top_k: int = 20


class InteractRequest(BaseModel):
    user_id: str
    item_id: str
    action: str          # "click" | "skip" | "cart"
    session_id: Optional[str] = None
    source: Optional[str] = "web_ui"


class AskLLMRequest(BaseModel):
    item_id: str
    title: str
    author: str
    user_prompt: str = "Why should I read this book? Give me a short 2-sentence pitch."
