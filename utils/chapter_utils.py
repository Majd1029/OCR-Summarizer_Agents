import os
import re
import langdetect
import google.generativeai as genai
from openai import OpenAI


# =========================
# 📌 Prompt Template
# =========================
SUMMARY_PROMPT_TEMPLATE = """
You are an expert AI trained in summarizing academic and educational documents.

Your task is to read and summarize the provided Markdown chapter content into a **detailed and comprehensive summary**, keeping the same language as the original content.

### Guidelines:
- Use the **same language** as the original text.
- Create a **longer summary with multiple paragraphs**.
- Explain key points, important definitions, and concepts.
- **Identify, accurately reproduce, and clearly explain every theorem and mathematical formula. 
  Do not omit or paraphrase formulas or theorems; preserve their original notation (including LaTeX if present).**
- For each theorem or formula, provide a brief explanation of its meaning and significance.
- Maintain logical flow and structure.
- Do **not** include headings or page numbers.
- Do **not** bullet or number anything; use paragraph form.
- Make the result easy to study from and faithful to the original content.

Only return clean, properly spaced, multi-paragraph **Markdown-formatted** text.
"""


# =========================
# 📑 Markdown Splitting
# =========================
def split_markdown_into_chapters(markdown_text: str) -> list[tuple[str, str]]:
    """
    Splits markdown text into chapters based on H1/H2 headings.

    Args:
        markdown_text (str): Raw markdown document.

    Returns:
        list[tuple[str, str]]: List of (chapter_title, chapter_body).
                               If no headings are found, returns one "Full Document" chapter.
    """
    # Split into alternating [pre-text, heading, body, heading, body...]
    chapters = re.split(r'(?m)^#{1,2} (.+)$', markdown_text)

    if len(chapters) > 1:
        titles = [chapters[i].strip() for i in range(1, len(chapters), 2)]
        bodies = [chapters[i + 1].strip() for i in range(1, len(chapters), 2)]
    else:
        titles = ["Full Document"]
        bodies = [markdown_text]

    return list(zip(titles, bodies))


# =========================
# ✨ Summarization
# =========================
def summarize_chapter(
    markdown_text: str,
    output_filename: str = "summary",
    model_choice: str = "Gemini"
) -> tuple[str, str | None]:
    """
    Summarizes a single markdown chapter.

    Args:
        markdown_text (str): The content of the chapter in Markdown.
        output_filename (str): Suggested filename for saving the summary.
        model_choice (str): Either "Gemini" or "OpenAI".

    Returns:
        tuple[str, str|None]: (summary_text, saved_file_path)  
                              If error → ("⚠️ Error ...", None).
    """
    try:
        # Detect language (default to Arabic on failure)
        try:
            lang = langdetect.detect(markdown_text)
        except Exception:
            lang = "ar"

        # Build summarization prompt
        prompt = f"{SUMMARY_PROMPT_TEMPLATE.strip()}\n\nLanguage: {lang}\n\nChapter Content:\n\n{markdown_text}"

        # Generate summary
        if model_choice == "Gemini":
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            summary_text = (response.text or "").strip()
        else:  # OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SUMMARY_PROMPT_TEMPLATE.strip()},
                    {"role": "user", "content": f"Language: {lang}\n\nChapter Content:\n\n{markdown_text}"}
                ],
                max_tokens=4000
            )
            summary_text = response.choices[0].message.content.strip()

        if not summary_text:
            summary_text = "*No summary generated.*"

        # Ensure output directory exists
        output_folder = os.path.join("outputs", "summaries")
        os.makedirs(output_folder, exist_ok=True)

        # Sanitize filename
        safe_filename = "".join(c for c in output_filename if c.isalnum() or c in (" ", "_", "-")).rstrip()
        if not safe_filename:
            safe_filename = "summary"

        # Avoid overwriting existing files
        output_path = os.path.join(output_folder, f"{safe_filename}.md")
        count = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_folder, f"{safe_filename}_{count}.md")
            count += 1

        # Save summary
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        return summary_text, output_path

    except Exception as e:
        return f"⚠️ Error during summarization: {e}", None


# =========================
# 🧹 Utility
# =========================
def is_effectively_empty_chapter(text: str) -> bool:
    """
    Determine if a chapter is effectively empty.

    Args:
        text (str): Chapter text.

    Returns:
        bool: True if empty or only contains page markers.
    """
    stripped = text.strip()
    if not stripped:
        return True

    # Detect page markers like: "### Page: ...\n```markdown\n```"
    page_marker_pattern = r"^### Page: .+\n```markdown\n*\s*```$"
    return bool(re.match(page_marker_pattern, stripped, re.MULTILINE))