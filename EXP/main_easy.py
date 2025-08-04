import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
import json
import io
import re

st.set_page_config(layout="wide")
st.title("📄 OCR Pipeline with EasyOCR: PDF/Image → Markdown & JSON")

# Sidebar options
lang_selection = st.sidebar.multiselect("Select OCR languages", ["ar", "fr", "en"], default=["en"])
output_format = st.sidebar.selectbox("Output format", ["Markdown", "JSON"])

# Upload image or PDF
uploaded_file = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"])

# Function to convert PDF to list of images
def pdf_to_images(file):
    return convert_from_bytes(file.read())

# Heuristic: Simple layout analysis
def analyze_layout(ocr_results):
    import numpy as np

    def group_by_lines(boxes, y_thresh=24):
        lines = []
        current_line = []
        prev_y = None
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

    # Step 1: convert raw OCR output into blocks
    blocks = []
    for bbox, text, conf in ocr_results:
        if not text.strip():
            continue
        text = fix_common_ocr_errors(text)
        # bbox is a list of 4 points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        blocks.append({
            "text": text.strip(),
            "conf": conf,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "center_y": (y1 + y2) / 2,
            "center_x": (x1 + x2) / 2
        })

    # Step 2: sort by top Y
    blocks = sorted(blocks, key=lambda x: x["center_y"])

    # Step 3: group into lines
    lines = group_by_lines(blocks)

    structured = {"title": "", "sections": []}
    title_set = False

    for line in lines:
        line = sorted(line, key=lambda x: x["bbox"][0])  # sort left-to-right
        texts = [b["text"] for b in line]

        # If first line and long, it's likely a title
        if not title_set and len(" ".join(texts)) > 5:
            structured["title"] = " ".join(texts)
            title_set = True
        elif len(line) >= 2:
            # Treat as table row
            if structured["sections"] and "table" in structured["sections"][-1]:
                structured["sections"][-1]["table"].append(texts)
            else:
                structured["sections"].append({"table": [texts]})
        else:
            structured["sections"].append({"paragraph": texts[0]})

    return structured

# Format structured output
def to_markdown(structured):
    lines = [f"# {structured['title']}"]
    for sec in structured['sections']:
        if "paragraph" in sec:
            lines.append(f"\n{sec['paragraph']}\n")
        elif "table" in sec:
            table = sec['table']
            if len(table[0]) > 1:
                headers = "| " + " | ".join(table[0]) + " |"
                sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
                lines.extend([headers, sep])
            else:
                lines.append(table[0][0])
    return "\n".join(lines)

def to_json(structured):
    return json.dumps(structured, indent=2, ensure_ascii=False)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)
    return denoised

# Process image
def process_image(image, reader):
    preprocessed = preprocess_image(image)
    results = reader.readtext(preprocessed, detail=1, contrast_ths=0.05)
    structured = analyze_layout(results)
    annotated_img = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    return structured, annotated_img

def fix_common_ocr_errors(text):
    # Replace common OCR mistakes in URLs
    text = re.sub(r'https?I+', 'https://', text, flags=re.IGNORECASE)
    text = re.sub(r'wwW|wWw|WwW|WWW', 'www', text)
    text = re.sub(r'[\s_]+com[\s_]+', '.com.', text)
    text = re.sub(r'[\s_]+com', '.com', text)
    text = re.sub(r'[\s_]+', '.', text)  # Replace spaces/underscores with dots in URLs
    text = re.sub(r'(?<=\w)[,;](?=\w)', '.', text)  # Replace comma/semicolon between words with dot
    # Remove double dots
    text = re.sub(r'\.\.+', '.', text)
    # Fix common email mistakes
    text = re.sub(r'\s*@\s*', '@', text)
    text = re.sub(r'\s*\.\s*', '.', text)
    return text

# Main logic
if uploaded_file:
    reader = easyocr.Reader(lang_selection, gpu=True)  # Set to True if you have a CUDA GPU

    images = []
    if uploaded_file.type == "application/pdf":
        pages = pdf_to_images(uploaded_file)
        images = pages
    else:
        images = [Image.open(uploaded_file).convert("RGB")]

    full_structured = []
    for i, img in enumerate(images):
        st.subheader(f"📄 Page {i+1}")
        structured, annotated_img = process_image(img, reader)
        full_structured.append(structured)
        st.image(annotated_img, use_column_width=True)

    st.subheader("📤 Extracted Structured Output")
    if output_format == "Markdown":
        output_text = "\n\n---\n\n".join(to_markdown(s) for s in full_structured)
        st.code(output_text, language="markdown")
    else:
        output_text = to_json({"pages": full_structured})
        st.code(output_text, language="json")

    st.download_button("💾 Download Result", data=output_text.encode("utf-8"),
                       file_name="ocr_output.md" if output_format == "Markdown" else "ocr_output.json")
