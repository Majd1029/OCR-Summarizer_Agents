import os
import uuid
import re
import pandas as pd
from pdf2image import convert_from_path
from io import StringIO
import streamlit as st

STATIC_FOLDER = "static"

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
    if len(lines) < 3:
        return table_md
    header = lines[0]
    num_cols = header.count('|') - 1
    cleaned_lines = [header]
    cleaned_lines.append("|" + "|".join(['---'] * num_cols) + "|")
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) < num_cols:
            cells.extend([''] * (num_cols - len(cells)))
        elif len(cells) > num_cols:
            cells = cells[:num_cols]
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
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                df = df.dropna(axis=1, how="all")
                if df.shape[0] > 0 and df.iloc[0].astype(str).str.contains("---").all():
                    df = df.drop(index=0)
                df.columns = df.columns.str.strip()
                st.table(df)
            except Exception as e:
                st.markdown("⚠️ Failed to render table properly. Showing as raw Markdown:")
                st.code(fixed_table_md)

def highlight_formulas_and_theorems(md_text: str) -> str:
    """
    Ensures that mathematical formulas and theorem-like blocks are preserved in Markdown.
    - Wraps detected formulas in `$...$` or `$$...$$` if not already.
    - Highlights theorem-like blocks.
    """
    # Preserve inline math: e.g., x^2 + y^2 = z^2
    md_text = re.sub(
        r'(?<!\$)(\b[a-zA-Z0-9_]+\s*=\s*[^.,;\n]+)(?!\$)',
        r'$\1$',
        md_text
    )

    # Preserve display math: lines starting and ending with math symbols
    md_text = re.sub(
        r'(?<!\$)\n([ \t]*[\\\(\[].+[\\\)\]])\n(?!\$)',
        r'\n$$\1$$\n',
        md_text
    )

    # Highlight theorem-like blocks
    theorem_patterns = [
        r'(Theorem\s*\d*\.?.*?:)', r'(Lemma\s*\d*\.?.*?:)', r'(Corollary\s*\d*\.?.*?:)', r'(Proof\s*:)', r'(Definition\s*:)', r'(Proposition\s*:)', r'(Remark\s*:)'
    ]
    for pattern in theorem_patterns:
        md_text = re.sub(pattern, r'**\1**', md_text, flags=re.IGNORECASE)

    return md_text