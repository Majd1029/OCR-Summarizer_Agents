# OCR Pipeline

This project provides a modular OCR (Optical Character Recognition) pipeline supporting multiple OCR engines (Gemini, OpenAI, Tesseract, EasyOCR, PaddleOCR) and LLM-based summarization. It extracts text from scanned documents, images, and PDFs, and can summarize Markdown chapters.

## 📁 Project Structure

    OCR_pipe/
    ├── OCR_Extractor.py        # Advanced OCR extraction and Markdown saving
    ├── chapter_summarizer.py   # Summarize extracted Markdown chapters
    ├── EXP/
    │   ├── main_tesseract.py       # Tesseract OCR implementation
    │   ├── main_paddle.py          # PaddleOCR implementation
    │   └── main_easy.py            # EasyOCR implementation
    ├── utils/
    │   ├── pdf_utils.py            # General utility functions (PDF/image/table handling)
    │   └── chapter_utils.py    # Chapter summarization utilities
    ├── outputs/
    │   ├── saved_markdown/     # (Legacy) Extracted Markdown files
    │   └── summaries/          # Summarized Markdown chapters (current)
    ├── static/                 # Temporary images and files
    ├── requirements.txt        # Python dependencies
    ├── Samples
    │   ├── Sample_book.pdf         # Sample PDF for OCR
    │   └── sample_qcm (1).jpg      # Sample image for OCR
    └── .gitignore

## 🚀 Features

- Extract text from images and PDFs using:
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - Gemini and OpenAI LLM APIs
- Save extracted Markdown files to `outputs/saved_markdown`
- Summarize Markdown chapters and save to `outputs/summaries`
- Modular utility functions in the `utils` folder
- Streamlit UI for easy interaction

## 🛠️ Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Majd1029/ocr_pipeline
    cd OCR_pipe
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

4. **Install OCR Engine Requirements**:
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
- Run engine-specific scripts:
    ```bash
    streamlit run main_tesseract.py
    streamlit run main_easy.py
    streamlit run main_paddle.py
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
- Extracted Markdown files are saved in `outputs/saved_markdown`.
- Summaries are saved in `outputs/summaries`.

## 📃 License

This project is for academic or personal use.  
Please check individual OCR engine licenses for their specific terms.

