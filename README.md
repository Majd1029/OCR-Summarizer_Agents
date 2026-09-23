# OCR-Summarizer-Agents

**[Try the live demo](https://majd-ocr-summarizer.streamlit.app)** — free Tesseract path needs no key; three printed sample documents are bundled.

This project provides a modular OCR (Optical Character Recognition) and summarization pipeline supporting multiple OCR engines (Gemini, OpenAI, Tesseract, EasyOCR, PaddleOCR) and LLM-based summarization. It extracts text from scanned documents, images, and PDFs, preserves mathematical formulas and theorems, and can summarize Markdown chapters with high fidelity.

## Hosted demo

`app.py` is a trimmed Streamlit build of this pipeline, deployed on Streamlit
Community Cloud. Two OCR backends, chosen by what you are reading:

| Backend | Good at | Cost |
|---|---|---|
| **Tesseract** (`ara+fra+eng`) | Clean printed French / English | Free, always on |
| **Claude vision** (`claude-opus-5`) | Handwriting, mathematical notation, complex tables | Your own `CLAUDE_API_KEY` |

Tesseract is the default because it needs no key, and the app says plainly that
the hard cases — scanned maths, handwriting — are where Claude pulls ahead.
That is what this project was built for; the free path is a floor, not the
point. Three synthetic printed documents are bundled under `demo_samples/` so
you can try it without uploading anything.

Arabic is configured in Tesseract but its output is unreliable even on clean printed pages, so the bundled samples are Latin-script only and the app points Arabic users at Claude. That gap is the reason this project used a vision model in the first place.

Summaries run on a small open model in-process, so the free path is
end-to-end free. Uploads are processed in memory and discarded with the session.

Dependencies for the hosted demo are in `requirements.txt`. The full local stack
for `OCR_Extractor.py`, `chapter_summarizer.py` and the `EXP/` backends
(EasyOCR, PaddleOCR) is in `requirements-full.txt` — EasyOCR and OpenCV alone
exceed the hosted memory ceiling.

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

