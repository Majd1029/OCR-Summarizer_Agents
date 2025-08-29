import os
import uuid
import base64
from io import StringIO

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

import google.generativeai as genai
import openai

from utils.pdf_utils import convert_pdf_to_images, display_markdown_with_tables


# =========================
# 🔧 Setup & Config
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Storage paths
STATIC_FOLDER = "static"
OUTPUT_FOLDER = os.path.join("outputs", "saved_markdown")

os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Session state initialization
if "markdown_results" not in st.session_state:
    st.session_state.markdown_results = ""

# =========================
# 📌 Constants
# =========================
OCR_PROMPT = """
### SYSTEM PROMPT ###
You are a cutting-edge OCR and Document Layout Analysis Engine designed to process scanned educational content in multiple languages, including Arabic, English, and French.

### CORE OBJECTIVE ###
Automatically detect the primary language(s) in the image and extract all text with perfect fidelity to its original visual and linguistic format.
Preserve text direction (Arabic and other RTL languages must remain RTL; English and other LTR languages remain LTR).

### DETAILED INSTRUCTIONS ###
1. Detect and recognize the language(s) automatically.
2. Extract text exactly as it appears (no translation, no correction).
3. Preserve formatting, diacritics, punctuation, bullets, lists, and structure.
4. Output only in Markdown format (GitHub-flavored).
5. For RTL text, use `<div align="right">`.
6. Use borderless HTML `<table>` for multi-column layouts.
7. Replace unreadable text with `[Unreadable]`.
8. Preserve math formulas as-is using LaTeX and `$...$` / `$$...$$`.
9. Mark theorems, lemmas, proofs in bold (e.g., `**Theorem 1:**`).
10. No extra explanations or translations.
"""


# =========================
# 🔍 AI Model Functions
# =========================
def upload_to_gemini(image_path):
    """Upload an image to Gemini for OCR."""
    return genai.upload_file(path=image_path, display_name="Diagram")


def extract_text_gemini(image_file):
    """Extract text from image using Gemini."""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content([image_file, OCR_PROMPT])
        return response.text if response else None
    except Exception as e:
        return f"❌ Gemini OCR failed: {e}"


def extract_text_openai(image_path):
    """Extract text from image using OpenAI GPT-4o Vision."""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert OCR assistant specialized in Arabic, French, and English educational documents. Extract clean Markdown text only."},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": OCR_PROMPT},
                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                 ]}
            ],
            max_tokens=4096,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI OCR failed: {e}"


def extract_text(image_path, model_choice):
    """Dispatch OCR to selected model."""
    if model_choice == "Gemini":
        gemini_file = upload_to_gemini(image_path)
        return extract_text_gemini(gemini_file)
    return extract_text_openai(image_path)


# =========================
# 🎨 Streamlit UI
# =========================
st.title("📖 Advanced OCR Extractor")

# Sidebar: Model selection
st.sidebar.subheader("⚙️ AI Model Selection")
selected_model = st.sidebar.radio("Choose AI model:", ["Gemini", "OpenAI"])

# File upload
uploaded_file = st.file_uploader("📂 Upload an image or PDF", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_base = os.path.splitext(uploaded_file.name)[0]

    # =========================
    # 📑 PDF Handling
    # =========================
    if file_ext == "pdf":
        st.subheader("📑 Page Range Selection")
        start_page = st.number_input("Start Page", min_value=1, step=1, value=1)
        end_page = st.number_input("End Page", min_value=start_page, step=1, value=start_page)

        if st.button("🚀 Convert & Extract"):
            pdf_path = os.path.join(STATIC_FOLDER, f"{uuid.uuid4().hex}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Converting PDF to images..."):
                images = convert_pdf_to_images(pdf_path, start_page, end_page)

            st.subheader("📸 Converted Pages")
            st.image(images, caption=[f"Page {i+start_page}" for i in range(len(images))], use_container_width=True)

            # Extract text
            st.session_state.markdown_results = ""
            for img_path in images:
                with st.spinner(f"Extracting text from {os.path.basename(img_path)}..."):
                    text = extract_text(img_path, selected_model)
                    st.session_state.markdown_results += f"### Page: {os.path.basename(img_path)}\n{text}\n\n"

    # =========================
    # 🖼 Image Handling
    # =========================
    else:
        img_path = os.path.join(STATIC_FOLDER, f"{uuid.uuid4().hex}_{uploaded_file.name}")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("🖼 Uploaded Image")
        st.image(img_path, caption="Uploaded Image", use_container_width=True)

        if st.button("🚀 Extract Text"):
            with st.spinner("Processing image..."):
                text = extract_text(img_path, selected_model)
                st.session_state.markdown_results = text if text else "*No text found.*"


# =========================
# 💾 Results & Download
# =========================
if st.session_state.markdown_results:
    st.subheader("📄 Extracted Content")
    display_markdown_with_tables(st.session_state.markdown_results)

    # File naming
    suffix = f"_({start_page}-{end_page})" if uploaded_file and file_ext == "pdf" else ""
    default_name = f"Extracted_{file_base}{suffix}.md"

    st.markdown("### 📝 Save Extracted Text")
    custom_name = st.text_input("File name (.md):", value=default_name)
    if not custom_name.endswith(".md"):
        custom_name += ".md"

    # Save file
    save_path = os.path.join(OUTPUT_FOLDER, custom_name)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(st.session_state.markdown_results)

    st.success(f"✅ Saved to `{save_path}`")

    # Download button
    st.download_button(
        label="📥 Download Markdown",
        data=st.session_state.markdown_results,
        file_name=custom_name,
        mime="text/markdown"
    )

# Reset
if st.button("🧹 Clear Extracted Content"):
    st.session_state.markdown_results = ""