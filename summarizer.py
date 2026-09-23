"""Chapter splitting + summarisation.

The free path uses a small open LLM in-process so the Space never depends on a
paid key. Gemini stays available for visitors who supply their own key.
"""
import os
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# bfloat16, not float32: measured 1568 MB vs 2686 MB peak RSS for this
# model, against a ~2.7 GB ceiling once Streamlit's own footprint is added.
# It is also ~2.4x faster at decode, since generation is
# memory-bandwidth-bound. transformers 5.x renamed torch_dtype -> dtype.
LLM_MODEL = os.getenv("SUMMARY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))

SUMMARY_PROMPT = """You summarise academic and educational documents.

Write a detailed, multi-paragraph summary of the chapter below, in the SAME
language as the original. Explain key points, definitions and concepts.
Reproduce every theorem and formula exactly, preserving LaTeX notation, and
explain what each one means. Use paragraphs, no bullets, no headings, no page
numbers. Return clean Markdown only."""

_lock = threading.Lock()
_tok = None
_model = None


def _load():
    global _tok, _model
    with _lock:
        if _model is None:
            _tok = AutoTokenizer.from_pretrained(LLM_MODEL)
            _model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
            _model.eval()
    return _tok, _model


def split_markdown_into_chapters(markdown_text: str):
    parts = re.split(r"(?m)^#{1,2} (.+)$", markdown_text)
    if len(parts) > 1:
        titles = [parts[i].strip() for i in range(1, len(parts), 2)]
        bodies = [parts[i + 1].strip() for i in range(2, len(parts), 2)]
        return list(zip(titles, bodies))
    return [("Full Document", markdown_text)]


def _summarize_local(body: str) -> str:
    tok, model = _load()
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": body[:6000]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def _summarize_gemini(body: str, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return (model.generate_content(f"{SUMMARY_PROMPT}\n\n{body}").text or "").strip()


def summarize_chapter(body: str, backend: str, api_key: str | None = None) -> str:
    if not body.strip():
        return "*This chapter is empty.*"
    if backend == "Gemini":
        if not api_key:
            raise ValueError("Paste a Gemini API key in the sidebar.")
        return _summarize_gemini(body, api_key)
    return _summarize_local(body)
