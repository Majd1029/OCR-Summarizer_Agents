import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from typing import List, Dict


# ---------- OCR Preprocessing ---------- #
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Apply preprocessing for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize for better OCR (scaling factor)
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(resized, h=30)

    # Contrast adjustment (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    return thresh


def run_ocr(image: np.ndarray, lang: str = "eng") -> Dict:
    """Run Tesseract OCR and return structured data."""
    config = f"--oem 3 --psm 6 -l {lang}"  # OEM=3 (LSTM), PSM=6 (block of text)
    return pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config=config
    )


def group_lines(data: Dict) -> List[str]:
    """Group OCR words into lines of text."""
    lines = {}
    n = len(data['level'])
    for i in range(n):
        key = (data['page_num'][i], data['block_num'][i],
               data['par_num'][i], data['line_num'][i])
        txt = data['text'][i].strip()
        if txt:
            lines.setdefault(key, []).append(txt)
    return [' '.join(words) for words in lines.values()]


def format_markdown(lines: List[str]) -> str:
    """Apply simple Markdown formatting heuristics to OCR lines."""
    md = []
    for line in lines:
        if line.startswith(("1.", "•", "- ")):
            md.append(f"- {line[2:].strip()}")
        elif line.endswith(":"):
            md.append(f"## {line}")
        else:
            md.append(line)
    return "\n\n".join(md)


def handle_file(file) -> List[np.ndarray]:
    """Convert uploaded file (PDF or image) to list of images."""
    if file.name.lower().endswith(".pdf"):
        images = convert_from_bytes(file.read())
        return [np.array(img) for img in images]
    else:
        img = Image.open(file).convert("RGB")
        return [np.array(img)]


# ---------- Streamlit UI ---------- #

st.set_page_config(page_title="🧠 OCR Extractor", layout="wide")
st.title("🧠 OCR to Markdown with Tesseract")
st.markdown(
    "Upload a **PDF** or **image** and extract structured text using **Tesseract OCR**."
)

uploaded = st.file_uploader(
    "📂 Upload a scanned document (PDF/Image)",
    type=["pdf", "jpg", "png", "jpeg"]
)

if uploaded:
    pages = handle_file(uploaded)
    all_output = []

    # Allow multiple languages
    lang = st.selectbox(
        "🌍 OCR Language(s)",
        ["eng", "eng+fra", "eng+fra+ara", "ara"],
        help="You can combine languages (e.g., 'eng+ara')."
    )

    with st.spinner("🔍 Running OCR..."):
        for i, img in enumerate(pages, start=1):
            st.subheader(f"📄 Page {i}")

            # Show original and preprocessed side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Page", use_column_width=True)
            with col2:
                processed = preprocess_image(img)
                st.image(processed, caption="Preprocessed for OCR", use_column_width=True, channels="GRAY")

            # OCR processing
            ocr_data = run_ocr(processed, lang=lang)
            lines = group_lines(ocr_data)
            markdown = format_markdown(lines)
            all_output.append(markdown)

            # Tabs for outputs
            tab1, tab2 = st.tabs(["📝 Markdown", "📊 OCR Data"])
            with tab1:
                st.code(markdown, language="markdown")
            with tab2:
                st.dataframe({k: v for k, v in ocr_data.items()})

    # Download button for all pages
    full_md = "\n\n---\n\n".join(all_output)
    st.download_button(
        "💾 Download Extracted Markdown",
        full_md,
        file_name="output.md",
        mime="text/markdown"
    )

else:
    st.warning("Please upload a PDF or image to start OCR.")