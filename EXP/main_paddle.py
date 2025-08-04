import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
from paddleocr import PaddleOCR
from PIL import Image
import io
import json

# Initialize PaddleOCR once (Arabic or English as needed)
ocr = PaddleOCR(use_angle_cls=True, lang='ar')  # change to 'en' for English

# --- Helper Functions ---

def extract_images_from_pdf(pdf_file):
    images = []
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)
    return images

def ocr_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        result = ocr.ocr(tmp.name, cls=True)
        os.unlink(tmp.name)
    return result[0]

def structure_text_from_ocr_result(ocr_result):
    plain_text = []
    markdown_blocks = []
    json_sections = {"blocks": []}

    for line in ocr_result:
        text = line[1][0].strip()
        box = line[0]
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        min_y = min(y_coords)
        max_y = max(y_coords)
        height = max_y - min_y

        plain_text.append(text)

        # Markdown heuristics
        if ":" in text or "؟" in text:
            markdown_blocks.append(f"### {text}")
        elif " - " in text or text.startswith("- "):
            markdown_blocks.append(f"- {text}")
        elif text.count("|") >= 2:
            markdown_blocks.append(text)  # crude table row
        else:
            markdown_blocks.append(text)

        json_sections["blocks"].append({
            "text": text,
            "bbox": box,
            "line_height": height
        })

    return "\n".join(plain_text), "\n".join(markdown_blocks), json_sections

# --- Streamlit UI ---

st.set_page_config(page_title="📄 PaddleOCR Layout Extractor", layout="wide")
st.title("📄 PaddleOCR Layout & Markdown Extractor")

uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.info("Running PaddleOCR… please wait ⏳")

    if uploaded_file.type == "application/pdf":
        images = extract_images_from_pdf(uploaded_file)
    else:
        images = [uploaded_file.read()]

    for i, img in enumerate(images):
        ocr_result = ocr_image_bytes(img)
        plain_text, markdown_text, json_output = structure_text_from_ocr_result(ocr_result)

        st.success(f"✅ Page {i+1} processed")

        with st.expander(f"📜 Page {i+1} - Plain Text"):
            st.text_area("Plain Text", plain_text, height=300)

        with st.expander(f"🧾 Page {i+1} - Markdown Format"):
            st.code(markdown_text, language="markdown")

        with st.expander(f"📦 Page {i+1} - JSON Structure"):
            st.json(json_output)

else:
    st.warning("Please upload a document to extract.")