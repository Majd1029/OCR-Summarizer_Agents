import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
import json
import re
from typing import List, Dict

# ----------------- Streamlit Config ----------------- #
st.set_page_config(page_title="📄 EasyOCR Extractor", layout="wide")
st.title("📄 OCR Pipeline with EasyOCR → Markdown & JSON")

# Sidebar options
lang_selection = st.sidebar.multiselect(
    "🌍 Select OCR languages", ["ar", "fr", "en"], default=["en"]
)
output_format = st.sidebar.selectbox("📤 Output format", ["Markdown", "JSON"])
gpu_enabled = st.sidebar.checkbox("Use GPU (if available)", value=False)

# Preprocessing settings
st.sidebar.markdown("⚙️ **Preprocessing Settings**")
block_size = st.sidebar.slider("Adaptive Threshold Block Size", 3, 51, 11, step=2)
denoise_strength = st.sidebar.slider("Denoising Strength (h)", 5, 50, 30, step=5)

uploaded_file = st.file_uploader(
    "📂 Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"]
)

# ----------------- Helpers ----------------- #
def pdf_to_images(file) -> List[Image.Image]:
    return convert_from_bytes(file.read())

def preprocess_image(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
    )
    denoised = cv2.fastNlMeansDenoising(thresh, h=denoise_strength)
    return denoised

def fix_common_ocr_errors(text: str) -> str:
    text = re.sub(r'https?I+', 'https://', text, flags=re.IGNORECASE)
    text = re.sub(r'wwW|wWw|WwW|WWW', 'www', text)
    text = re.sub(r'[\s_]+com[\s_]+', '.com.', text)
    text = re.sub(r'[\s_]+com', '.com', text)
    text = re.sub(r'[\s_]+', '.', text)
    text = re.sub(r'(?<=\w)[,;](?=\w)', '.', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r'\s*@\s*', '@', text)
    text = re.sub(r'\s*\.\s*', '.', text)
    return text

# ----------------- Layout Analysis ----------------- #
def analyze_layout(ocr_results: List) -> Dict:
    def group_by_lines(boxes, y_thresh=24):
        lines, current_line, prev_y = [], [], None
        for box in boxes:
            y = (box['bbox'][1] + box['bbox'][3]) / 2
            if prev_y is None or abs(y - prev_y) < y_thresh:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
            prev_y = y
        if current_line:
            lines.append(current_line)
        return lines

    blocks = []
    for bbox, text, conf in ocr_results:
        if not text.strip():
            continue
        text = fix_common_ocr_errors(text)
        xs, ys = [pt[0] for pt in bbox], [pt[1] for pt in bbox]
        blocks.append({
            "text": text.strip(),
            "conf": conf,
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            "center_y": (min(ys) + max(ys)) / 2,
            "center_x": (min(xs) + max(xs)) / 2
        })

    blocks = sorted(blocks, key=lambda x: x["center_y"])
    lines = group_by_lines(blocks)

    structured = {"title": "", "sections": []}
    title_set = False

    for line in lines:
        line = sorted(line, key=lambda x: x["bbox"][0])
        texts = [b["text"] for b in line]

        if not title_set and len(" ".join(texts)) > 5:
            structured["title"] = " ".join(texts)
            title_set = True
        elif len(line) >= 2:
            if structured["sections"] and "table" in structured["sections"][-1]:
                structured["sections"][-1]["table"].append(texts)
            else:
                structured["sections"].append({"table": [texts]})
        else:
            structured["sections"].append({"paragraph": texts[0]})
    return structured

# ----------------- Output Formatters ----------------- #
def to_markdown(structured: Dict) -> str:
    lines = [f"# {structured['title']}"]
    for sec in structured['sections']:
        if "paragraph" in sec:
            lines.append(f"\n{sec['paragraph']}\n")
        elif "table" in sec:
            table = sec['table']
            headers = "| " + " | ".join(table[0]) + " |"
            sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
            lines.extend([headers, sep])
            for row in table[1:]:
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def to_json(structured: Dict) -> str:
    return json.dumps(structured, indent=2, ensure_ascii=False)

# ----------------- Main Processing ----------------- #
def process_image(image: Image.Image, reader) -> tuple:
    preprocessed = preprocess_image(image)
    results = reader.readtext(preprocessed, detail=1, contrast_ths=0.05)
    structured = analyze_layout(results)
    annotated_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    for bbox, text, conf in results:
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return structured, annotated_img, results

# ----------------- App Logic ----------------- #
if uploaded_file:
    reader = easyocr.Reader(lang_selection, gpu=gpu_enabled)

    if uploaded_file.type == "application/pdf":
        images = pdf_to_images(uploaded_file)
    else:
        images = [Image.open(uploaded_file).convert("RGB")]

    full_structured = []

    for i, img in enumerate(images, start=1):
        st.subheader(f"📄 Page {i}")

        structured, annotated_img, raw_results = process_image(img, reader)
        full_structured.append(structured)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_column_width=True)
        with col2:
            st.image(annotated_img, caption="OCR Annotated", use_column_width=True)

        tab1, tab2, tab3 = st.tabs(["📝 Markdown/JSON", "📊 Structured JSON", "🔍 Raw OCR"])
        with tab1:
            if output_format == "Markdown":
                st.code(to_markdown(structured), language="markdown")
            else:
                st.code(to_json(structured), language="json")
        with tab2:
            st.json(structured)
        with tab3:
            st.write(raw_results)

    # Download final doc
    if output_format == "Markdown":
        full_output = "\n\n---\n\n".join(to_markdown(s) for s in full_structured)
        mime, ext = "text/markdown", "md"
    else:
        full_output = to_json({"pages": full_structured})
        mime, ext = "application/json", "json"

    st.download_button("💾 Download Full Output", data=full_output.encode("utf-8"),
                       file_name=f"ocr_output.{ext}", mime=mime)