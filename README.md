# OCR-Summarizer-Agents

**[Try the live demo](https://ocr-summarizeragents-web.vercel.app)** — runs in your browser; the free Tesseract path needs no key, and three printed sample documents are bundled.

This project provides a modular OCR (Optical Character Recognition) and summarization pipeline supporting multiple OCR engines (Gemini, OpenAI, Tesseract, EasyOCR, PaddleOCR) and LLM-based summarization. It extracts text from scanned documents, images, and PDFs, preserves mathematical formulas and theorems, and can summarize Markdown chapters with high fidelity.

## Hosted demo

A trimmed build of this pipeline runs **entirely in the browser** as a static
page on Vercel (`web/index.html`). There is no server: uploaded files never leave
the visitor's device.

| Step | Runs on | Library |
|---|---|---|
| PDF pages → images (200 dpi) | browser | pdf.js |
| OCR, free path (`ara+fra+eng`) | browser (WebAssembly) | Tesseract.js |
| OCR, vision LLM path | Anthropic API, with the visitor's own key | Anthropic JS SDK |
| Summaries, free path | browser (WebGPU, or WebAssembly) | transformers.js + Qwen2.5-0.5B-Instruct |
| Summaries, vision LLM path | Anthropic API, with the visitor's own key | Anthropic JS SDK |

Two OCR backends, chosen by what you are reading:

| Backend | Good at | Cost |
|---|---|---|
| **Tesseract** (`ara+fra+eng`) | Clean printed French / English | Free, a few MB of language data on first use |
| **Vision LLM** (Anthropic API) | Handwriting, mathematical notation, complex tables | Your own API key |

Tesseract is the default because it needs no key, and the page says plainly that
the hard cases — scanned maths, handwriting — are where the vision LLM pulls ahead.
That is what this project was built for; the free path is a floor, not the
point. Three synthetic printed documents are bundled under `web/samples/` so
you can try it without uploading anything.

Arabic is configured in Tesseract, but its output is less reliable than on Latin
script, especially on real scans. The bundled Arabic sample is kept deliberately:
running it through both backends shows the gap directly.

The free summarizer downloads the model once (a few hundred MB) and the browser
caches it. The API key, when used, goes from the browser straight to Anthropic
and is never stored. Up to 5 PDF pages are read per run.

To deploy: on vercel.com, **Add New → Project**, import this repo, set **Root
Directory** to `web`, Framework **Other**, no build command.

### Running the pipeline in Python

`app.py` is a Gradio app with the same two steps server-side (`ocr_backends.py`,
`summarizer.py`), for running locally or on a GPU machine:

```bash
sudo apt install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fra   # see packages.txt
pip install gradio -r requirements.txt
python app.py
```

Dependencies for `app.py` are in `requirements.txt`. The full local stack
for `OCR_Extractor.py`, `chapter_summarizer.py` and the `EXP/` backends
(EasyOCR, PaddleOCR) is in `requirements-full.txt`.

---

## 📁 Project Structure

    OCR-Summarizer-Agents/
    ├── OCR_Extractor.py            # Streamlit app for advanced OCR extraction and Markdown saving
    ├── chapter_summarizer.py       # Streamlit app for summarizing extracted Markdown chapters
    ├── EXP/
    │   ├── main_tesseract.py       # Tesseract OCR implementation
    │   ├── main_paddle.py          # PaddleOCR implementation
    │   └── main_easy.py            # EasyOCR implementation
    ├── utils/
    │   ├── pdf_utils.py            # PDF/image/table handling and Markdown post-processing (formulas/theorems)
    │   └── chapter_utils.py        # Chapter summarization utilities (Gemini/OpenAI prompt logic)
    ├── requirements.txt            # Python dependencies
    └── .gitignore

> `static/` and `outputs/` are created at runtime by the app and are not part of
> this repository. `Samples/` was removed — add your own documents if you want
> sample inputs.

## 🚀 Features

- Extract text from images and PDFs using:
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - Gemini and OpenAI LLM APIs (with advanced Markdown and math/theorem preservation)
- Mathematical formulas and theorems are accurately detected, preserved, and highlighted in both extraction and summarization
- Modular utility functions in the `utils` folder
- Streamlit UI for easy interaction

## 🛠️ Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Majd1029/OCR-Summarizer-Agents
    cd OCR-Summarizer-Agents
    ```

2. **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install OCR Engine Requirements** (if needed):
    ```bash
    pip install paddleocr
    pip install easyocr
    ```

## 📄 Usage

- Run the OCR extractor:
    ```bash
    streamlit run OCR_Extractor.py
    ```
- Run the chapter summarizer:
    ```bash
    streamlit run chapter_summarizer.py
    ```
- Run engine-specific scripts (from the `EXP` folder):
    ```bash
    streamlit run EXP/main_tesseract.py
    streamlit run EXP/main_easy.py
    streamlit run EXP/main_paddle.py
    ```

## 📦 Dependencies

See `requirements.txt` for a full list. Major libraries include:

- `pytesseract`
- `easyocr`
- `paddleocr`
- `opencv-python`
- `pdf2image`
- `Pillow`
- `streamlit`
- `openai`
- `google-generativeai`
- `langdetect`

## 📌 Notes

- Ensure **Tesseract** is installed and added to your system path if using Tesseract.
- **PDF support** is enabled via `pdf2image` and related libraries.
- LLM features require valid Gemini and OpenAI API keys.
- Mathematical formulas and theorems are preserved and highlighted throughout the pipeline.

## 📃 License

This project is for academic or personal use.  
Please check individual OCR engine licenses for their specific terms.

