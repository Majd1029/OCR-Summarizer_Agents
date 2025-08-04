import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes

# Optional (Windows): specify Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize for better OCR (optional, adjust as needed)
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(resized, h=30)
    # Contrast adjustment
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    return thresh


def run_ocr(image, lang="eng"):
    # Use --oem 3 (default LSTM), --psm 6 (assume a block of text)
    config = f"--oem 3 --psm 6 -l {lang}"
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    return data


def group_lines(data):
    lines = {}
    n = len(data['level'])
    for i in range(n):
        key = (data['page_num'][i], data['block_num'][i], data['par_num'][i], data['line_num'][i])
        txt = data['text'][i].strip()
        if txt:
            lines.setdefault(key, []).append(txt)
    return [' '.join(words) for words in lines.values()]


def format_markdown(lines):
    md = []
    for line in lines:
        if line.startswith("1.") or line.startswith("•") or line.startswith("- "):
            md.append(f"- {line[2:].strip()}")
        elif line.endswith(":"):
            md.append(f"## {line}")
        else:
            md.append(line)
    return "\n\n".join(md)


def handle_file(file):
    if file.name.lower().endswith(".pdf"):
        images = convert_from_bytes(file.read())
        return [np.array(img) for img in images]
    else:
        img = Image.open(file).convert("RGB")
        return [np.array(img)]


# ---------- Streamlit UI ---------- #

st.set_page_config("OCR Extractor", layout="wide")
st.title("🧠 OCR to Markdown with Tesseract")
st.markdown("Upload a **PDF** or **image**. The app will extract structured text using Tesseract OCR.")

uploaded = st.file_uploader("Upload a scanned document (PDF/Image)", type=["pdf", "jpg", "png", "jpeg"])

if uploaded:
    pages = handle_file(uploaded)
    all_output = []

    lang = st.selectbox("OCR Language(s)", ["eng", "eng+fra", "eng+fra+ara", "ara"])

    with st.spinner("🔍 Processing..."):
        for i, img in enumerate(pages):
            st.subheader(f"📄 Page {i + 1}")
            st.image(img, caption="Original Page", use_column_width=True)

            processed = preprocess_image(img)
            ocr_data = run_ocr(processed, lang=lang)
            lines = group_lines(ocr_data)
            markdown = format_markdown(lines)
            all_output.append(markdown)

            st.markdown("---")
            st.markdown("#### 📝 Extracted Text (Markdown)")
            st.code(markdown, language="markdown")

    full_md = "\n\n---\n\n".join(all_output)
    st.download_button("💾 Download Markdown", full_md, file_name="output.md", mime="text/markdown")
