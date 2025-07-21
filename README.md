# OCR Pipeline

This project provides a modular OCR (Optical Character Recognition) pipeline that supports multiple OCR engines, including Gemini and openAI APIs. Also including Tesseract, EasyOCR, and PaddleOCR. It can be used to extract text from scanned documents, images, and PDFs with options to switch between engines based on use-case requirements.

## 📁 Project Structure 
    ocr_pipeline-main/
    ├── app.py # Main app interface
    ├── app_gemini.py # Alternative app version (possibly using Gemini API or LLM integration)
    ├── main_tesseract.py # Tesseract OCR implementation
    ├── main_easy.py # EasyOCR implementation
    ├── main_paddle.py # PaddleOCR implementation
    ├── Sample_book.pdf # Sample PDF for OCR
    ├── sample_qcm (1).jpg # Sample image for OCR
    ├── requirements.txt # Python dependencies
    ├── Rapport of the Part 1.txt # Possibly documentation or a report
    └── .gitignore

## 🚀 Features

- Extract text from images and PDFs using:
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- Easy-to-run scripts for different engines
- Modular structure to add more OCR engines or features

## 🛠️ Installation

1. **Clone the repository** (if not already):
   ```bash
   git clone <repo-url>
   cd ocr_pipeline-main
2. **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

4. **Install OCR Engine Requirements**:

Tesseract: Install from here.

PaddleOCR:
    ```bash
    pip install paddleocr

EasyOCR:
    ```bash
    pip install easyocr


## 📄 Usage

- Each OCR engine has its own script. You can run them like:

    ```bash
    streamlit run main_tesseract.py
    streamlit run main_easy.py
    streamlit run main_paddle.py

- Run the app (if GUI/API is provided):
    ```bash
    streamlit run app.py
## 📦 Dependencies

See `requirements.txt` for a full list. Major libraries include:

- `pytesseract`
- `easyocr`
- `paddleocr`
- `opencv-python`
- `pdf2image`
- `Pillow`

## 📌 Notes

- Ensure **Tesseract** is installed and added to your system path if using Tesseract.
- **PDF support** is enabled via `pdf2image` and related libraries.
- `app_gemini.py` may require access to an **LLM API** — check the script for API keys or tokens.

## 📃 License

This project is for academic or personal use.  
Please check individual OCR engine licenses for their specific terms.

