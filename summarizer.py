"""Chapter splitting and summarisation.

The free path runs a small open model in-process so the hosted demo never
depends on a paid key. Claude is available for visitors who supply their own
CLAUDE_API_KEY, and is markedly better on dense technical material.
"""
import os
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# bfloat16, not float32: measured 1568 MB vs 2686 MB peak RSS for this model,
# against a ~2.7 GB ceiling once Streamlit's own footprint is added. It is also
# ~2.4x faster at decode, since generation is memory-bandwidth-bound.
# transformers 5.x renamed torch_dtype -> dtype.
LOCAL_MODEL = os.getenv("SUMMARY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "400"))

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
CLAUDE_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "4000"))

SUMMARY_PROMPT = """You summarise academic and educational documents.

Write a detailed, multi-paragraph summary of the chapter below, in the SAME
language as the original. Explain key points, definitions and concepts.
Reproduce every theorem and formula exactly, preserving LaTeX notation, and
explain what each one means. Use paragraphs, no bullets, no headings, no page
numbers. Return clean Markdown only."""

_lock = threading.Lock()
_tok = None
_model = None


def _load_local():
    global _tok, _model
    with _lock:
        if _model is None:
            _tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
            _model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
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
    tok, model = _load_local()
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
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def _summarize_claude(body: str, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        system=SUMMARY_PROMPT,
        messages=[{"role": "user", "content": body}],
    )

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        reason = getattr(detail, "explanation", None) or "declined by a safety classifier"
        return f"*[Claude declined this chapter: {reason}]*"

    parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(parts).strip()


def summarize_chapter(body: str, backend: str, api_key: str | None = None) -> str:
    if not body.strip():
        return "*This chapter is empty.*"
    if backend == "Claude":
        if not api_key:
            raise ValueError("Paste a Claude API key in the sidebar.")
        return _summarize_claude(body, api_key)
    return _summarize_local(body)
