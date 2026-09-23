"""OCR backends.

Nothing is written to disk: on a shared host the filesystem is global, and the
original code persisted every upload under static/.

Two backends:
  * Tesseract (ara+fra+eng) - free, always on, good on clean printed text.
  * Claude vision - for handwriting, mathematical notation and complex tables,
    using the visitor's own CLAUDE_API_KEY.
"""
import base64
import io
import os

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

OCR_PROMPT = """You are an OCR and document layout engine for scanned educational
content in Arabic, English and French.

1. Detect the language(s) automatically.
2. Extract text exactly as it appears. No translation, no correction.
3. Preserve formatting, diacritics, punctuation, bullets, lists and structure.
4. Output GitHub-flavored Markdown only.
5. For RTL text use <div align="right">.
6. Use borderless HTML <table> for multi-column layouts.
7. Replace unreadable text with [Unreadable].
8. Preserve maths as LaTeX using $...$ / $$...$$.
9. Bold theorems, lemmas and proofs, e.g. **Theorem 1:**.
10. Return the transcription only, with no preamble or commentary."""

TESS_LANGS = "ara+fra+eng"

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")
MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "8000"))


def pdf_to_images(pdf_bytes: bytes, start: int, end: int, dpi: int = 200):
    """Render a 1-based inclusive page range. PyMuPDF works on raw bytes, so no
    poppler subprocess and no temp file."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        end = min(end, doc.page_count)
        for n in range(start - 1, end):
            pix = doc.load_page(n).get_pixmap(dpi=dpi)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    return images


def ocr_tesseract(image: Image.Image) -> str:
    text = pytesseract.image_to_string(image.convert("RGB"), lang=TESS_LANGS)
    return text.strip() or "*[No text detected]*"


def _png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def ocr_claude(image: Image.Image, api_key: str) -> str:
    """Transcribe one page with Claude vision.

    Adaptive thinking is on: multilingual layout with RTL, LaTeX and tables is
    exactly the kind of work that benefits from it. Server-side fallbacks are
    enabled so a safety refusal on one page routes elsewhere instead of failing
    the whole extraction.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    response = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _png_b64(image),
                        },
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
    )

    # A refusal returns HTTP 200 with stop_reason "refusal" rather than raising,
    # so stop_reason must be checked before reading content.
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        reason = getattr(detail, "explanation", None) or "declined by a safety classifier"
        return f"*[Claude declined this page: {reason}]*"

    # content is a list of blocks; with thinking on, only text blocks carry the
    # transcription.
    parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(parts).strip() or "*[No text detected]*"


def run_ocr(image: Image.Image, backend: str, api_key: str | None = None) -> str:
    if backend == "Claude":
        if not api_key:
            raise ValueError("Paste a Claude API key in the sidebar, or use Tesseract.")
        return ocr_claude(image, api_key)
    return ocr_tesseract(image)
