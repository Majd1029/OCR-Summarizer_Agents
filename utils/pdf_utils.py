import os
import re
import uuid
from io import StringIO

import pandas as pd
import streamlit as st
from pdf2image import convert_from_path


# =========================
# 🔧 Constants
# =========================
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)


# =========================
# 📑 PDF Utilities
# =========================
def convert_pdf_to_images(pdf_path: str, start: int, end: int) -> list[str]:
    """
    Convert a PDF into images for a given page range.

    Args:
        pdf_path (str): Path to the input PDF file.
        start (int): First page number (1-based).
        end (int): Last page number (inclusive).

    Returns:
        list[str]: List of saved image file paths.
    """
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        st.error(f"❌ Failed to convert PDF: {e}")
        return []

    saved_images = []
    total_pages = len(images)

    # Clamp end to max pages
    end = min(end, total_pages)

    for i in range(start - 1, end):
        image_name = f"page_{i+1}_{uuid.uuid4().hex}.png"
        image_path = os.path.join(STATIC_FOLDER, image_name)
        images[i].save(image_path, "PNG")
        saved_images.append(image_path)

    return saved_images


# =========================
# 🧹 Markdown Table Cleaning
# =========================
def clean_markdown_table(table_md: str) -> str:
    """
    Clean and normalize a Markdown table string.

    - Ensures consistent column counts.
    - Adds a proper header separator line.
    - Replaces `#` at the start of cells with `No.`.
    - Pads or truncates cells as needed.

    Args:
        table_md (str): Raw Markdown table.

    Returns:
        str: Cleaned Markdown table.
    """
    lines = [line.strip() for line in table_md.strip().split("\n") if line.strip()]
    if len(lines) < 2:  # not enough rows for a table
        return table_md

    header = lines[0]
    num_cols = header.count("|") - 1

    cleaned_lines = [header]
    cleaned_lines.append("|" + "|".join(["---"] * num_cols) + "|")

    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]

        # Adjust cell count
        if len(cells) < num_cols:
            cells.extend([""] * (num_cols - len(cells)))
        elif len(cells) > num_cols:
            cells = cells[:num_cols]

        # Normalize cell content
        cells = [re.sub(r"^#", "No.", cell) for cell in cells]
        cleaned_lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(cleaned_lines)


# =========================
# 📊 Table Display
# =========================
def display_markdown_with_tables(md_text: str) -> None:
    """
    Display Markdown text in Streamlit with improved table rendering.

    - Extracts Markdown table blocks.
    - Cleans and converts them to pandas DataFrames.
    - Displays them using `st.table`.
    - Falls back to raw Markdown if parsing fails.

    Args:
        md_text (str): Full Markdown text (possibly with tables).
    """
    # Split text into non-table parts and table blocks
    table_blocks = re.findall(r"((?:\|.+\|\n)+)", md_text)
    parts = re.split(r"(?:\|.+\|\n)+", md_text)

    for i, part in enumerate(parts):
        if part.strip():
            st.markdown(part.strip())

        if i < len(table_blocks):
            raw_table = table_blocks[i].strip()
            fixed_table = clean_markdown_table(raw_table)

            try:
                df = pd.read_csv(
                    StringIO(fixed_table),
                    sep="|",
                    engine="python",
                    skipinitialspace=True
                )

                # Drop auto-generated / empty columns
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                df = df.dropna(axis=1, how="all")

                # Remove markdown separator row if still present
                if df.shape[0] > 0 and df.iloc[0].astype(str).str.contains("---").all():
                    df = df.drop(index=0)

                df.columns = df.columns.str.strip()
                st.table(df)

            except Exception:
                st.markdown("⚠️ Failed to render table, showing raw Markdown:")
                st.code(fixed_table)


# =========================
# ✨ Math & Theorem Highlighting
# =========================
def highlight_formulas_and_theorems(md_text: str) -> str:
    """
    Enhance Markdown text with math and theorem formatting.

    - Wraps inline equations in `$...$`.
    - Wraps display equations in `$$...$$`.
    - Highlights theorem-like statements in bold.

    Args:
        md_text (str): Input Markdown text.

    Returns:
        str: Enhanced Markdown text.
    """
    # Inline math: simple equalities not already wrapped in $
    md_text = re.sub(
        r"(?<!\$)(\b[a-zA-Z0-9_]+\s*=\s*[^.,;\n]+)(?!\$)",
        r"$\1$",
        md_text
    )

    # Display math: lines enclosed in \(...\) or \[...\]
    md_text = re.sub(
        r"(?<!\$)\n([ \t]*[\\\(\[].+[\\\)\]])\n(?!\$)",
        r"\n$$\1$$\n",
        md_text
    )

    # Bold theorem-like keywords
    theorem_patterns = [
        r"(Theorem\s*\d*\.?.*?:)",
        r"(Lemma\s*\d*\.?.*?:)",
        r"(Corollary\s*\d*\.?.*?:)",
        r"(Proof\s*:)",
        r"(Definition\s*:)",
        r"(Proposition\s*:)",
        r"(Remark\s*:)"
    ]
    for pattern in theorem_patterns:
        md_text = re.sub(pattern, r"**\1**", md_text, flags=re.IGNORECASE)

    return md_text