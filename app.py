import os
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
import google.generativeai as genai
import openai
import uuid
import re
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY

STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

if "markdown_results" not in st.session_state:
    st.session_state.markdown_results = ""

# Sidebar: Model selection
st.sidebar.subheader("AI Model Selection")
selected_model = st.sidebar.selectbox("Choose AI model", ["Gemini", "OpenAI"])

# Core Prompt
OCR_PROMPT = (
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
    "9. Output **only Markdown**, no HTML outside the specified use cases, and no additional commentary.\n"
)

def upload_to_gemini(image_path):
    return genai.upload_file(path=image_path, display_name="Diagram")

def extract_text_gemini(image_file):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([image_file, OCR_PROMPT])
    return response.text if response else None

def extract_text_openai(image_path):
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert OCR assistant specialized in Arabic, French, and English educational documents. "
                    "Your job is to extract clean, well-formatted Markdown text from textbook images."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image may contain educational or non-educational content, including text, diagrams, or mixed elements. Your role is not to judge the relevance or origin of the content, but to extract **all visible text** exactly as it appears.\n\n"

                            "⚠️ Never refuse to process the image. If there is any readable text at all, **extract it**. If you cannot read it, use `[Unreadable text]` as a placeholder.\n\n"
                            
                            "You are capable of understanding Arabic, French, and English, and can process various input types such as scanned documents, textbook images, phone photos, PDFs, or handwritten materials.\n\n"

                            "### Your Task:\n"
                            "- Extract the text **exactly as it appears in the original document**, without translating, paraphrasing, or completing missing parts.\n"
                            "- If a part is unreadable, write: `[Unreadable text]`\n\n"

                            "## OCR Extraction Rules:\n"
                            "- Do **not** judge the source, quality, or relevance of the image.\n"
                            "- Do **not** comment on whether it is from a textbook or not.\n"
                            "- Do **not** refuse the task for any reason unless the image is entirely blank or unreadable.\n"
                            "- Focus only on accurate **text extraction** with correct formatting.\n\n"

                            "### Preserve:\n"
                            "- Text direction (Arabic = RTL, English/French = LTR)\n"
                            "- Alignment using: `<div align='right'>`, `<div align='center'>`, etc.\n"
                            "- Markdown structure for headers, exercises, blockquotes, and tables\n\n"
                            "- **Line spacing and indentation**\n"
                            "- **Text alignment**:\n"
                            "  - `<div align='center'> ... </div>` for centered\n"
                            "  - `<div align='right'> ... </div>` for right-aligned\n"
                            "  - `<div align='left'> ... </div>` for left-aligned\n\n"

                            "### Markdown Formatting Rules:\n"
                            "- Titles → `## Title`\n"
                            "- Subtitles → `### Subtitle`\n"
                            "- Exercises → Bold label (e.g., `**Exercise 1:**`)\n"
                            "- Subquestions → Markdown list (`- a)`, `- b)`)\n"
                            "- Poetry or long quotes → Markdown blockquote (`>`)\n"
                            "- Diagrams/images → `![Description of diagram]`\n\n"
                            "- Ensure Markdown tables have correct syntax:"
                            "- Pipe characters `|` between every column"
                            "- A header separator line like: `|---|---|`"
                            "- No missing columns or rows"
                            "- Each row must begin and end with a `|`"
                            "- Example:"
                            "| Name | Age | Grade |"
                            "|------|-----|--------|"
                            "| Ali  | 10  | A     |"

                            "### Additional Guidelines:\n"
                            "- Maintain clean spacing between lines\n"
                            "- Insert two blank lines between major sections\n"
                            "- No extra commentary or explanation\n\n"

                            "### Output:\n"
                            "Only clean Markdown text, no commentary or explanations. If unsure, still attempt extraction and structure."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                        }
                    },  
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def extract_text(image_path):
    if selected_model == "Gemini":
        gemini_file = upload_to_gemini(image_path)
        return extract_text_gemini(gemini_file)
    elif selected_model == "OpenAI":
        return extract_text_openai(image_path)

def convert_pdf_to_images(pdf_path, start, end):
    images = convert_from_path(pdf_path)
    saved_images = []
    for i in range(start - 1, end):
        image_path = os.path.join(STATIC_FOLDER, f"page_{i+1}_{uuid.uuid4().hex}.png")
        images[i].save(image_path, "PNG")
        saved_images.append(image_path)
    return saved_images

def clean_markdown_table(table_md: str) -> str:
    lines = [line.strip() for line in table_md.strip().split("\n") if line.strip()]
    
    # Skip cleaning if table has fewer than 3 lines (header, separator, at least one row)
    if len(lines) < 3:
        return table_md

    # Count columns from header
    header = lines[0]
    num_cols = header.count('|') - 1  # because line starts and ends with '|'

    cleaned_lines = [header]
    cleaned_lines.append("|" + "|".join(['---'] * num_cols) + "|")  # enforce separator line

    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        
        # Normalize row length
        if len(cells) < num_cols:
            cells.extend([''] * (num_cols - len(cells)))
        elif len(cells) > num_cols:
            cells = cells[:num_cols]  # truncate extra cells

        # Escape hash (#) symbols unless they're intended for headers
        cells = [re.sub(r"^#", "No.", cell) for cell in cells]
        
        cleaned_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(cleaned_lines)


def display_markdown_with_tables(md_text):
    table_blocks = re.findall(r"((?:\|.+\|\n)+)", md_text)
    parts = re.split(r"(?:\|.+\|\n)+", md_text)

    for i, part in enumerate(parts):
        if part.strip():
            st.markdown(part.strip())

        if i < len(table_blocks):
            table_md = table_blocks[i].strip()
            try:
                fixed_table_md = clean_markdown_table(table_md)
                df = pd.read_csv(StringIO(fixed_table_md), sep="|", engine="python", skipinitialspace=True)

                # Remove unnamed empty columns caused by outer pipes
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df.dropna(axis=1, how="all")

                # Drop separator row if exists (---)
                if df.shape[0] > 0 and df.iloc[0].astype(str).str.contains("---").all():
                    df = df.drop(index=0)

                # Strip whitespace from headers
                df.columns = df.columns.str.strip()

                st.table(df)
            except Exception as e:
                st.markdown("⚠️ Failed to render table properly. Showing as raw Markdown:")
                st.code(fixed_table_md)


# Streamlit UI
st.title("📖 Advanced OCR Extractor")

uploaded_file = st.file_uploader("Upload an image or PDF", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

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

            st.subheader("📸 Converted Pages")
            for img_path in images:
                st.image(img_path, caption=f"Page from PDF: {os.path.basename(img_path)}", use_container_width=True)

            st.session_state.markdown_results = ""
            for img_path in images:
                with st.spinner(f"Processing {img_path}..."):
                    text = extract_text(img_path)
                    result = text.strip() if text and text.strip() else "*No text found on this page.*"
                    st.session_state.markdown_results += f"### Page: {img_path.split('/')[-1]}\n{result}\n\n"

    else:
        img_path = os.path.join(STATIC_FOLDER, f"{uuid.uuid4().hex}_{uploaded_file.name}")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("🖼️ Uploaded Image")
        st.image(img_path, caption="Uploaded Image", use_container_width=True)

        if st.button("Extract Text"):
            with st.spinner("Processing image..."):
                text = extract_text(img_path)
                st.session_state.markdown_results = text.strip() if text and text.strip() else "*No text found in the image.*"

# Display and download
if st.session_state.markdown_results:
    st.subheader("📄 Extracted Content")
    display_markdown_with_tables(st.session_state.markdown_results)

    filename = uploaded_file.name.rsplit(".", 1)[0]
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