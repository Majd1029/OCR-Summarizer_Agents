import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
from paddleocr import PaddleOCR
from PIL import Image
import io
import json
from typing import List, Tuple, Dict, Any

# --- Initialize OCR (cached so it only loads once) ---
@st.cache_resource
def load_ocr(lang: str = "ar"):
    """Load PaddleOCR model only once per session."""
    return PaddleOCR(use_angle_cls=True, lang=lang)

ocr = load_ocr("ar")  # change 'ar' to 'en' if needed


# --- Helper Functions ---
def extract_images_from_pdf(pdf_bytes: bytes) -> List[bytes]:
    """Extracts all pages from a PDF as images in PNG byte format."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # higher resolution
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)
    return images


def ocr_image_bytes(image_bytes: bytes) -> Any:
    """Runs OCR on an image given as bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        result = ocr.ocr(tmp.name, cls=True)
    os.unlink(tmp.name)  # clean up temp file
    return result[0] if result else []


def structure_text_from_ocr_result(ocr_result: Any) -> Tuple[str, str, Dict]:
    """Converts OCR result into plain text, Markdown heuristics, and JSON."""
    plain_text = []
    markdown_blocks = []
    json_sections = {"blocks": []}

    for line in ocr_result:
        text = line[1][0].strip()
        box = line[0]
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        min_y, max_y = min(y_coords), max(y_coords)
        height = max_y - min_y

        if text:
            plain_text.append(text)

            # --- Markdown heuristics ---
            if ":" in text or "؟" in text:
                markdown_blocks.append(f"### {text}")
            elif text.startswith("- ") or " - " in text:
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
    st.info("⚙️ Running PaddleOCR… please wait ⏳")

    # Handle PDF vs Image
    if uploaded_file.type == "application/pdf":
        images = extract_images_from_pdf(uploaded_file.read())
    else:
        images = [uploaded_file.read()]

    # Process each page
    for i, img in enumerate(images, start=1):
        ocr_result = ocr_image_bytes(img)
        plain_text, markdown_text, json_output = structure_text_from_ocr_result(ocr_result)

        st.success(f"✅ Page {i} processed")

        # --- Preview Tabs instead of expanders ---
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📜 Plain Text", "🧾 Markdown", "📦 JSON", "🖼️ Image"]
        )

        with tab1:
            st.text_area("Plain Text", plain_text, height=300)

        with tab2:
            st.code(markdown_text, language="markdown")

        with tab3:
            st.json(json_output)

        with tab4:
            st.image(img, caption=f"Page {i}", use_column_width=True)

    # Option to download all extracted text
    all_text = "\n\n".join([structure_text_from_ocr_result(ocr_image_bytes(img))[0] for img in images])
    st.download_button(
        label="📥 Download Extracted Text",
        data=all_text,
        file_name="ocr_extracted_text.txt",
        mime="text/plain"
    )

else:
    st.warning("📂 Please upload a document to extract.")