"""
app/services/llm.py — Gemma-4-E4B-it LLM lazy-loader and Wikipedia grounding.

Extracted from api.py (_ensure_llm_loaded, ask_llm endpoint logic).
"""
import functools
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import torch

from threading import Thread
from typing import Generator

import torch
from transformers import TextIteratorStreamer

log = logging.getLogger("nba_api")

_llm_pipeline = None

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def ensure_loaded() -> bool:
    """
    Lazy-load Qwen2.5-1.5B-Instruct on first call.
    Subsequent calls return immediately.
    Returns True if LLM is ready, False on failure.
    """
    global _llm_pipeline
    if _llm_pipeline is not None:
        return True
    try:
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        log.info(f"[Qwen] Loading {MODEL_ID}…")
        _llm_pipeline = hf_pipeline(
            "text-generation",
            model=MODEL_ID,
            device=device,
            torch_dtype=torch.float16,
        )
        log.info("[Qwen] LLM ready ✓")
        return True
    except Exception as e:
        log.warning(f"[Qwen] Failed to load: {e}")
        return False


def get_pipeline():
    return _llm_pipeline


import re

def _extract_series_name(title: str) -> str:
    """Strip volume/chapter noise, return bare series name."""
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r",?\s*Vol\.?\s*\d+", "", t, flags=re.IGNORECASE)
    t = re.sub(r",?\s*Volume\s*\d+", "", t, flags=re.IGNORECASE)
    t = re.sub(r",?\s*Chapter\s*\d+", "", t, flags=re.IGNORECASE)
    return t.strip().rstrip(",").strip()


def _extract_vol_number(title: str) -> str | None:
    """Return volume number string from title, e.g. '77'."""
    m = re.search(r"Vol\.?\s*(\d+)", title, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _build_wiki_queries(title: str, author: str) -> list[str]:
    """Return Wikipedia search queries ordered most → least specific."""
    series = _extract_series_name(title)
    vol = _extract_vol_number(title)
    queries = []
    if vol:
        queries.append(f"{series} Vol. {vol} manga")
        queries.append(f"{series} volume {vol} plot")
    queries.append(f"{series} manga arc plot summary")
    queries.append(series)
    return [q for q in queries if q.strip()]


def _is_author_page(hit_title: str, author: str) -> bool:
    """True if the Wikipedia hit looks like an author biography, not a content page."""
    ht = hit_title.lower()
    au = author.lower()
    return au in ht or ht in au


def _wiki_search(query: str) -> dict | None:
    """Run one Wikipedia search and return the top hit dict, or None."""
    query_safe = urllib.parse.quote(query)
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={query_safe}&utf8=&format=json&srlimit=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "NBA-AI-Assistant/1.0"})
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        data = json.loads(resp.read())
    hits = data.get("query", {}).get("search", [])
    return hits[0] if hits else None


def _wiki_extract(page_title: str) -> str:
    """Fetch a large plain-text extract for a Wikipedia page title."""
    page_safe = urllib.parse.quote(page_title)
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&exsentences=40&explaintext=1"
        f"&titles={page_safe}&format=json&redirects=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "NBA-AI-Assistant/1.0"})
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        data = json.loads(resp.read())
    pages = data["query"]["pages"]
    page_id = list(pages.keys())[0]
    if page_id == "-1":
        return ""
    return pages[page_id].get("extract", "")


@functools.lru_cache(maxsize=128)
def fetch_wikipedia_summary(title: str, author: str = "") -> str:
    """
    Try multiple Wikipedia queries from specific to broad, skipping author
    biography pages. Returns the best extract found, or a fallback string.
    """
    try:
        queries = _build_wiki_queries(title, author)
        for query in queries:
            hit = _wiki_search(query)
            if not hit:
                continue
            if author and _is_author_page(hit["title"], author):
                log.debug(f"[Wiki] Skipping author page '{hit['title']}', trying next query…")
                continue
            extract = _wiki_extract(hit["title"])
            if extract:
                log.debug(f"[Wiki] Hit via '{query}' → '{hit['title']}'")
                return f"Wikipedia Context ({hit['title']}): {extract}"
        return "No specific Wikipedia summary found. Use your training knowledge about this specific volume."
    except Exception as e:
        log.warning(f"Wikipedia fetch for '{title}' failed: {e}")
        return "No context available."


_BOOKS_FIELDS = "items(volumeInfo(title,subtitle,description,authors))"

# Rate-limiting state for Google Books
_books_last_call: float = 0.0          # epoch time of last successful request
_books_disabled_until: float = 0.0     # skip all requests until this epoch time
_BOOKS_MIN_INTERVAL: float = 1.0       # minimum seconds between requests
_BOOKS_COOLDOWN: float = 120.0         # seconds to back off after a 429


def _books_request(raw_query: str) -> list:
    """Run one Google Books search using proper query operators. Returns items list."""
    global _books_last_call, _books_disabled_until

    # Skip entirely if we're in a 429 cooldown window
    if time.time() < _books_disabled_until:
        remaining = int(_books_disabled_until - time.time())
        log.debug(f"[Books] Rate-limit cooldown active, skipping ({remaining}s left)")
        raise urllib.error.HTTPError(None, 429, "Cooldown active", {}, None)

    # Throttle: enforce minimum gap between requests
    gap = time.time() - _books_last_call
    if gap < _BOOKS_MIN_INTERVAL:
        time.sleep(_BOOKS_MIN_INTERVAL - gap)

    params = urllib.parse.urlencode({
        "q":           raw_query,
        "maxResults":  "5",
        "printType":   "books",
        "langRestrict":"en",
        "fields":      _BOOKS_FIELDS,
    })
    url = f"https://www.googleapis.com/books/v1/volumes?{params}"
    log.debug(f"[Books] GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "NBA-AI-Assistant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            _books_last_call = time.time()
            return json.loads(resp.read()).get("items", [])
    except urllib.error.HTTPError as e:
        if e.code == 429:
            _books_disabled_until = time.time() + _BOOKS_COOLDOWN
            log.warning(f"[Books] 429 received — pausing Google Books for {int(_BOOKS_COOLDOWN)}s")
        raise


def _fetch_google_books(title: str, author: str) -> str:
    """
    Query Google Books using intitle:/inauthor: operators for precision.
    Falls back to title-only if the author causes zero results (e.g. wrong author in DB).
    Returns a formatted string with description, or '' if nothing useful found.
    """
    series = _extract_series_name(title)
    vol    = _extract_vol_number(title)
    base   = f"{series} Vol. {vol}" if vol else series

    # Ordered from most to least specific — intitle:/inauthor: are exact-field operators
    query_candidates = [
        f'intitle:"{base}" inauthor:"{author}"',  # precise: both fields
        f'intitle:"{base}"',                       # title only — catches wrong-author entries
    ]

    items = []
    for raw_query in query_candidates:
        items = _books_request(raw_query)
        if items:
            log.debug(f"[Books] Query '{raw_query}' → {len(items)} result(s)")
            break

    if not items:
        return ""

    # Prefer the item whose title contains the volume number
    best = None
    if vol:
        for item in items:
            if vol in item.get("volumeInfo", {}).get("title", "").lower():
                best = item["volumeInfo"]
                break
    if best is None:
        best = items[0]["volumeInfo"]

    description = best.get("description", "").strip()
    subtitle    = best.get("subtitle", "").strip()
    book_title  = best.get("title", "").strip()

    if not description:
        return ""

    parts = [f"Google Books — {book_title}"]
    if subtitle:
        parts.append(f"Volume Title: {subtitle}")
    parts.append(f"Description: {description}")
    log.debug(f"[Books] Using '{book_title}' (subtitle: '{subtitle}')")
    return "\n".join(parts)


@functools.lru_cache(maxsize=128)
def fetch_book_context(title: str, author: str = "") -> str:
    """
    Primary context fetcher. Tries Google Books first (volume-specific),
    falls back to Wikipedia if Google Books returns nothing useful.
    """
    try:
        google_result = _fetch_google_books(title, author)
        if google_result:
            return google_result
        log.debug("[Books] No Google Books result, falling back to Wikipedia…")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            log.info(f"[Books] Rate-limited for '{title}', using Wikipedia fallback")
        else:
            log.warning(f"[Books] Google Books HTTP {e.code} for '{title}': {e}")
    except Exception as e:
        log.warning(f"[Books] Google Books fetch failed for '{title}': {e}")

    return fetch_wikipedia_summary(title, author)


import numpy as np
from app.core import models as models_core

def rerank_context(query: str, raw_text: str, text_encoder, top_k: int = 3) -> str:
    """
    Split raw_text into sentences, embed them, and return the top_k most 
    semantically similar to the query using the existing BGE-M3 encoder.
    """
    if not raw_text or text_encoder is None:
        return raw_text

    # Basic sentence splitting
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if len(s.strip()) > 20]
    if len(sentences) <= top_k:
        return raw_text

    try:
        # 1. Embed the query
        query_vec = models_core.encode_text(query, text_encoder)
        if query_vec is None: return raw_text

        # 2. Embed all sentences
        sentence_vecs = []
        for s in sentences:
            v = models_core.encode_text(s, text_encoder)
            sentence_vecs.append(v if v is not None else np.zeros((1, 1024)))
        
        # 3. Compute cosine similarities
        sims = []
        for v in sentence_vecs:
            # query_vec is (1, 1024), v is (1, 1024)
            score = np.dot(query_vec, v.T).item()
            sims.append(score)

        # 4. Pick top_k
        top_indices = np.argsort(sims)[::-1][:top_k]
        # Keep original order for flow, or sort by score? Let's sort by score for relevance.
        results = [sentences[i] for i in top_indices]
        return " ".join(results)

    except Exception as e:
        log.warning(f"Reranking failed: {e}")
        return raw_text


def _normalize_genre(genre: str) -> str:
    """Extract English-only genre text, stripping non-ASCII characters."""
    # Prefer English text already wrapped in parentheses e.g. "(Action, Adventure)"
    m = re.search(r'\(([A-Za-z,&\s]+)\)', genre)
    if m:
        return m.group(1).strip()
    # Fall back: strip all non-ASCII then clean up leftover punctuation
    ascii_only = re.sub(r'[^\x00-\x7F]+', '', genre)
    ascii_only = re.sub(r'[,\s]{2,}', ', ', ascii_only).strip().strip(',').strip()
    return ascii_only or genre


def generate_stream(title: str, author: str, user_prompt: str, local_description: str = "", text_encoder=None, genre: str = "") -> Generator[str, None, None]:
    """Yields token chunks as they are generated by the LLM."""
    if not ensure_loaded():
        yield "AI model failed to load."
        return

    llm = _llm_pipeline
    ground_truth = fetch_book_context(title, author)
    
    # Semantic Reranking: pull plot/arc sentences, not author bios or sales figures.
    if ground_truth and text_encoder:
        _series = _extract_series_name(title)
        _vol = _extract_vol_number(title) or ""
        _rerank_q = f"plot summary arc story characters events {_series} volume {_vol}".strip()
        ground_truth = rerank_context(_rerank_q, ground_truth, text_encoder, top_k=5)

    # Prefer the most specific source — Google Books over generic local desc
    _gt_valid = ground_truth and not ground_truth.startswith("No ")
    description_for_model = ground_truth if _gt_valid else local_description

    # Guard: no useful description → return static message, never call the model
    _EMPTY_SIGNALS = ("No specific Wikipedia", "No context available", "No summary found", "No description")
    _is_empty = (
        not description_for_model
        or not description_for_model.strip()
        or any(description_for_model.strip().startswith(s) for s in _EMPTY_SIGNALS)
    )
    if _is_empty:
        yield (
            f"**Genre:** {_normalize_genre(genre) if genre else 'Unknown'}\n"
            "**Plot Summary:** No description is available for this book.\n"
            "**Pitch:** Check your local library or bookstore for more details."
        )
        return

    clean_genre = _normalize_genre(genre) if genre else "Unknown"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a book description assistant. "
                "Your ONLY job is to rewrite the provided book description into the output format. "
                "Do NOT use your own knowledge. Do NOT add plot points not in the description. "
                "Use ONLY what is written in the description below.\n\n"
                f"Book: '{title}' by {author}\n\n"
                f"Description:\n{description_for_model}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite the description above into this exact format:\n\n"
                f"**Genre:** {clean_genre}\n"
                "**Plot Summary:** [2-3 sentences using ONLY the description above]\n"
                "**Pitch:** [1 sentence on why to read this specific volume]\n\n"
                "Do not add anything not in the description."
            ),
        },
    ]

    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _device = next(llm.model.parameters()).device
    inputs = llm.tokenizer(text=prompt, return_tensors="pt").to(_device)

    _tok = getattr(llm.tokenizer, "tokenizer", llm.tokenizer)  # unwrap AutoProcessor → inner tokenizer
    streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=_tok.eos_token_id,
        use_cache=True,
    )

    thread = Thread(target=llm.model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        if new_text:
            yield new_text


def generate(title: str, author: str, user_prompt: str, local_description: str = "", text_encoder=None, genre: str = "") -> tuple[str, dict]:
    """
    Synchronous wrapper for generate_stream.
    Returns (full_answer, timings_dict).
    """
    import time
    timings = {"wiki_fetch_ms": 0.0, "llm_gen_ms": 0.0}
    
    start_wiki = time.perf_counter()
    fetch_book_context(title, author)  # result cached; this just warms the cache
    timings["wiki_fetch_ms"] = round((time.perf_counter() - start_wiki) * 1000, 2)

    start_gen = time.perf_counter()
    full_text = []
    for chunk in generate_stream(title, author, user_prompt, local_description, text_encoder, genre):
        full_text.append(chunk)
    
    timings["llm_gen_ms"] = round((time.perf_counter() - start_gen) * 1000, 2)
    return "".join(full_text).strip(), timings


