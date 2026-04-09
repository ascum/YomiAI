"""
app/services/llm.py — Qwen LLM lazy-loader and Wikipedia grounding.

Extracted from api.py (_ensure_llm_loaded, ask_llm endpoint logic).
"""
import json
import logging
import urllib.parse
import urllib.request

import torch

log = logging.getLogger("nba_api")

_llm_pipeline = None


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
        log.info("[Qwen] Lazy-loading Qwen2.5-1.5B-Instruct…")
        _llm_pipeline = hf_pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
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


def fetch_wikipedia_summary(title: str) -> str:
    """Return a short Wikipedia extract for the book title, or a fallback string."""
    try:
        query_safe = urllib.parse.quote(f"{title} (novel)")
        search_url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={query_safe}&utf8=&format=json&srlimit=1"
        )
        req_search = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_search, timeout=3) as resp:
            search_data = json.loads(resp.read())

        hits = search_data.get("query", {}).get("search", [])
        if not hits:
            return "Use your internal knowledge to summarize the plot."

        page_title  = urllib.parse.quote(hits[0]["title"])
        extract_url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exsentences=5&explaintext=1"
            f"&titles={page_title}&format=json"
        )
        req_extract = urllib.request.Request(extract_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_extract, timeout=3) as resp:
            extract_data = json.loads(resp.read())

        pages   = extract_data["query"]["pages"]
        page_id = list(pages.keys())[0]
        if page_id == "-1":
            return "Use your internal knowledge to summarize the plot."

        plot_text = pages[page_id].get("extract", "")
        return f"Here is factual internet summary information about the book: {plot_text}\n\nUse this information to accurately summarize the plot."

    except Exception as e:
        log.warning(f"Wikipedia fetch for '{title}' failed: {e}")
        return "Use your internal knowledge to summarize the plot."


def generate(title: str, author: str, user_prompt: str) -> str:
    """Run a grounded LLM response about a book. Assumes ensure_loaded() returned True."""
    llm            = _llm_pipeline
    ground_truth   = fetch_wikipedia_summary(title)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI book assistant. The user wants detailed information about a book. "
                "Provide a short response structured EXACTLY like this using Markdown:\n\n"
                "**Genre:** [Genre]\n**Plot Summary:** [2-3 sentences summarizing the plot]\n\n"
                f"{ground_truth}"
            ),
        },
        {"role": "user", "content": f"Tell me about the book '{title}' by {author}."},
    ]

    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.model.device)
    out_tokens = llm.model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        pad_token_id=llm.tokenizer.eos_token_id,
    )
    answer = llm.tokenizer.decode(
        out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return answer
