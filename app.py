"""Gradio app running the pipeline server-side, for local or GPU use.

The hosted demo (web/index.html) does the same steps entirely in the browser.

API (call with @gradio/client or gradio_client):
    /extract   (file, start, end, backend, api_key) -> (markdown, info)
    /summarize (text, backend, api_key)             -> summary markdown

backend is "Tesseract" (free, always on) or "Vision LLM" (needs the visitor's
own Anthropic API key, sent with each request and never stored or logged).
Uploads are processed in memory and nothing is written to disk.
"""
try:
    import spaces  # ZeroGPU: must be imported before torch / gradio
except ImportError:  # running locally
    spaces = None

import pathlib

import fitz  # PyMuPDF
import gradio as gr
from PIL import Image

from ocr_backends import pdf_to_images, run_ocr
from summarizer import summarize_chapter

BACKENDS = ["Tesseract", "Vision LLM"]
MAX_PAGES = 5               # per extraction request; each page is one OCR pass
MAX_BYTES = 20 * 1024 * 1024
MAX_SUMMARY_CHARS = 20_000


def _backend(name):
    return name if name in BACKENDS else "Tesseract"


def extract(file, start: float = 1, end: float = 1, backend: str = "Tesseract", api_key: str = ""):
    if not file:
        raise gr.Error("Upload a PDF or an image.")
    path = pathlib.Path(file)
    if path.stat().st_size > MAX_BYTES:
        raise gr.Error("That file is over 20 MB. Try a smaller one.")
    backend, api_key = _backend(backend), (api_key or "").strip() or None
    data = path.read_bytes()

    try:
        if data[:5] == b"%PDF-":
            with fitz.open(stream=data, filetype="pdf") as doc:
                total = doc.page_count
            start = max(1, min(int(start or 1), total))
            end = max(start, min(int(end or start), total, start + MAX_PAGES - 1))
            images = pdf_to_images(data, start, end)
            info = {"type": "pdf", "total_pages": total, "start": start, "end": end}
        else:
            try:
                image = Image.open(path)
                image.load()
            except Exception:
                raise gr.Error("Use a PDF, PNG, JPG or WebP file.")
            images = [image]
            info = {"type": "image", "total_pages": 1, "start": 1, "end": 1}

        chunks = []
        for i, img in enumerate(images):
            text = run_ocr(img, backend, api_key)
            header = f"## Page {info['start'] + i}\n\n" if len(images) > 1 else ""
            chunks.append(header + text)
    except ValueError as e:  # e.g. Vision LLM chosen without a key
        raise gr.Error(str(e))
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Extraction failed: {e}")

    info["backend"] = backend
    return "\n\n".join(chunks), info


def summarize(text: str, backend: str = "Tesseract", api_key: str = ""):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Nothing to summarize.")
    backend = "Vision LLM" if _backend(backend) == "Vision LLM" else "Tesseract"
    try:
        return summarize_chapter(text[:MAX_SUMMARY_CHARS], backend, (api_key or "").strip() or None)
    except ValueError as e:
        raise gr.Error(str(e))


with gr.Blocks(title="OCR → Summarizer") as demo:
    gr.Markdown(
        "# OCR → Summarizer\n"
        "Multilingual OCR (Arabic / French / English) to Markdown, then chapter summaries. "
        "[Source](https://github.com/Majd1029/OCR-Summarizer_Agents)"
    )
    with gr.Row():
        with gr.Column():
            # No file_types filter: API callers may upload without an extension,
            # so the type is checked from the content in extract().
            file = gr.File(label="PDF or image", type="filepath")
            with gr.Row():
                start = gr.Number(1, precision=0, label="Start page")
                end = gr.Number(1, precision=0, label=f"End page (max {MAX_PAGES} pages)")
            backend = gr.Radio(BACKENDS, value="Tesseract", label="Engine")
            api_key = gr.Textbox(label="Anthropic API key (Vision LLM only)", type="password")
            run = gr.Button("Extract text", variant="primary")
        with gr.Column():
            markdown = gr.Textbox(label="Extracted Markdown", lines=18)
            info = gr.JSON(label="Info")
            summarize_btn = gr.Button("Summarize this text")
            summary = gr.Textbox(label="Summary", lines=10)

    run.click(extract, [file, start, end, backend, api_key], [markdown, info], api_name="extract")
    summarize_btn.click(summarize, [markdown, backend, api_key], summary, api_name="summarize")

demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
