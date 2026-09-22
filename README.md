# OCR-Summarizer-Agents

This project provides a modular OCR (Optical Character Recognition) and summarization pipeline supporting multiple OCR engines (Gemini, OpenAI, Tesseract, EasyOCR, PaddleOCR) and LLM-based summarization. It extracts text from scanned documents, images, and PDFs, preserves mathematical formulas and theorems, and can summarize Markdown chapters with high fidelity.

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
    ├── outputs/
    │   ├── saved_markdown/         # Extracted Markdown files
    │   └── summaries/              # Summarized Markdown chapters
    ├── static/                     # Temporary images and files
    ├── requirements.txt            # Python dependencies
    │   ├── Sample_book.pdf         # Sample PDF for OCR
    │   └── sample_qcm (1).jpg      # Sample image for OCR
    └── .gitignore

## 🚀 Features

- Extract text from images and PDFs using:
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - Gemini and OpenAI LLM APIs (with advanced Markdown and math/theorem preservation)
- Save extracted Markdown files to `outputs/saved_markdown`
- Summarize Markdown chapters and save to `outputs/summaries`
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
- Extracted Markdown files are saved in `outputs/saved_markdown`.
- Summaries are saved in `outputs/summaries`.
- Mathematical formulas and theorems are preserved and highlighted throughout the pipeline.

## 📃 License

This project is for academic or personal use.  
Please check individual OCR engine licenses for their specific terms.

