"""OCR backends. Nothing is written to disk: on a shared Space the filesystem is
global, and the original code persisted every upload under static/."""
import io

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
10. No extra explanation."""

TESS_LANGS = "ara+fra+eng"


def pdf_to_images(pdf_bytes: bytes, start: int, end: int, dpi: int = 200):
    """Render a 1-based inclusive page range. PyMuPDF avoids the poppler
    subprocess and works on raw bytes, so no temp file is needed."""
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


def ocr_gemini(image: Image.Image, api_key: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([OCR_PROMPT, image.convert("RGB")])
    return (response.text or "").strip()


def run_ocr(image: Image.Image, backend: str, api_key: str | None = None) -> str:
    if backend == "Gemini":
        if not api_key:
            raise ValueError("Paste a Gemini API key in the sidebar, or use Tesseract.")
        return ocr_gemini(image, api_key)
    return ocr_tesseract(image)
