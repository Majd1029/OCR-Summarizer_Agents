import os
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
import google.generativeai as genai
import uuid
import re
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
API_KEY = os.getenv('GEMINI_AI_API_KEY')
genai.configure(api_key=API_KEY)

STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

if "markdown_results" not in st.session_state:
    st.session_state.markdown_results = ""

def upload_to_gemini(image_path):
    sample_file = genai.upload_file(path=image_path, display_name="Diagram")
    return sample_file


def extract_text(image_file):
    prompt = (
        "### SYSTEM PROMPT ###\n"
        "You are a cutting-edge OCR and Document Layout Analysis Engine designed to process scanned educational content in multiple languages, including Arabic, English, and French.\n\n"
        
        "### CORE OBJECTIVE ###\n"
        "Automatically detect the primary language(s) in the image and extract all text with perfect fidelity to its original visual and linguistic format.\n"
        "Preserve text direction (Arabic and other RTL languages must remain RTL; English and other LTR languages remain LTR).\n\n"
        
        "### DETAILED INSTRUCTIONS ###\n"
        "1. Detect and recognize the language(s) in the image automatically; no manual language selection needed.\n"
        "2. Extract all text exactly as it appears—do not correct spelling or translate.\n"
        "3. Preserve all original formatting, including diacritics, punctuation, bullets, line breaks, paragraphs, and numbered lists.\n"
        "4. Structure the output entirely in Markdown format:\n"
        "   - Use `##` headings for question or section numbers.\n"
        "   - Convert multiple-choice options or tabular data into GitHub-flavored Markdown tables.\n"
        "   - For Arabic or other RTL languages, maintain right-to-left order and align text accordingly (use `<div align=\"right\">` if needed).\n"
        "5. For multi-column layouts or side-by-side text blocks, reconstruct layout using borderless HTML `<table>` tags.\n"
        "6. Ignore all non-textual elements such as bindings, stains, or background marks.\n"
        "7. For any unreadable text, use the placeholder `[Unreadable]`.\n"
        "8. Do NOT add any explanations, inferred answers, or translations.\n"
        "9. Output **only Markdown**, no HTML outside the specified use cases, and no additional commentary.\n\n"
        
        "### EXAMPLES ###\n"
        "- Arabic question with multiple-choice answers should appear as a `##` heading followed by a Markdown table, preserving RTL formatting.\n"
        "- English paragraphs and lists should maintain LTR formatting with clear Markdown headings and tables.\n"
        "- French text follows the same rules, preserving accents and formatting.\n\n"
        
        "### FINAL NOTE ###\n"
        "Your output must be a single clean Markdown document that faithfully replicates the text and layout of the scanned page, respecting the detected language(s) and document structure."
    )

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([image_file, prompt])
    return response.text if response else None


def convert_pdf_to_images(pdf_path, start, end):
    images = convert_from_path(pdf_path)
    saved_images = []
    for i in range(start-1, end):
        image_path = os.path.join(STATIC_FOLDER, f"page_{i+1}_{uuid.uuid4().hex}.png")
        images[i].save(image_path, "PNG")
        saved_images.append(image_path)
    return saved_images

def display_markdown_with_tables(md_text):
    table_blocks = re.findall(r"((?:\|.+\|\n)+)", md_text)
    parts = re.split(r"(?:\|.+\|\n)+", md_text)
    output_parts = []

    for i, part in enumerate(parts):
        if part.strip():
            st.markdown(part.strip())

        if i < len(table_blocks):
            table_md = table_blocks[i].strip()
            try:
                df = pd.read_csv(StringIO(table_md), sep="|", engine='python')
                df = df.dropna(axis=1, how='all')  # Remove empty columns
                df.columns = df.columns.str.strip()
                df = df.drop(index=0) if all(df.iloc[0].str.contains('---')) else df  # drop separator row
                st.table(df)
            except Exception as e:
                st.markdown("⚠️ Failed to render table properly.")
                st.code(table_md)

# Streamlit UI
st.title("📖 Advanced OCR Extractor")

# Add language selection
st.sidebar.subheader("Language Settings")
doc_language = st.sidebar.selectbox("Select document language", ["Arabic", "English", "French"])

uploaded_file = st.file_uploader("Upload an image or PDF", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext == "pdf":
        st.subheader("Page Range Selection")
        start_page = st.number_input("Start Page", min_value=1, step=1)
        end_page = st.number_input("End Page", min_value=start_page, step=1)

        if st.button("Convert & Extract"):
            with st.spinner("Converting PDF to images..."):
                pdf_path = os.path.join(STATIC_FOLDER, f"{uuid.uuid4().hex}.pdf")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                images = convert_pdf_to_images(pdf_path, start_page, end_page)

            st.session_state.markdown_results = ""
            for img_path in images:
                with st.spinner(f"Processing {img_path}..."):
                    gemini_file = upload_to_gemini(img_path)
                    text = extract_text(gemini_file)
                    page_result = text.strip() if text and text.strip() else "*No text found on this page.*"
                    st.session_state.markdown_results += f"### Page: {img_path.split('/')[-1]}\n{page_result}\n\n"

    else:
        img_path = os.path.join(STATIC_FOLDER, f"{uuid.uuid4().hex}_{uploaded_file.name}")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Extract Text"):
            with st.spinner("Uploading to Gemini..."):
                gemini_file = upload_to_gemini(img_path)
                text = extract_text(gemini_file)
                if text and text.strip():
                    st.session_state.markdown_results = text
                else:
                    st.session_state.markdown_results = "*No text found in the image.*"

# Display and Download
if st.session_state.markdown_results:
    st.subheader("📄 Extracted Content")
    display_markdown_with_tables(st.session_state.markdown_results)

    filename = uploaded_file.name.rsplit('.', 1)[0]
    suffix = f"_({start_page}-{end_page})" if file_ext == "pdf" else ""
    download_name = f"Extracted_{filename}{suffix}.md"

    st.download_button(
        label="📥 Download Extracted Markdown",
        data=st.session_state.markdown_results,
        file_name=download_name,
        mime="text/markdown"
    )


if st.button("🧹 Clear Extracted Content"):
    st.session_state.markdown_results = ""