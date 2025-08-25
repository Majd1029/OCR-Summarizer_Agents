import os
import langdetect
import google.generativeai as genai
from openai import OpenAI
import re

SUMMARY_PROMPT_TEMPLATE = """
You are an expert AI trained in summarizing academic and educational documents.

Your task is to read and summarize the provided Markdown chapter content into a **detailed and comprehensive summary**, keeping the same language as the original content.

### Guidelines:
- Use the **same language** as the original text.
- Create a **longer summary with multiple paragraphs**.
- Explain key points, important definitions, and concepts.
- **Identify, accurately reproduce, and clearly explain every theorem and mathematical formula. Do not omit or paraphrase formulas or theorems; preserve their original notation (including LaTeX if present).**
- For each theorem or formula, provide a brief explanation of its meaning and significance.
- Maintain logical flow and structure.
- Do **not** include headings or page numbers.
- Do **not** bullet or number anything; use paragraph form.
- Make the result easy to study from and faithful to the original content.

Only return clean, properly spaced, multi-paragraph **Markdown-formatted** text.
"""

def split_markdown_into_chapters(markdown_text: str):
    """
    Splits markdown text into chapters based on headings (## or #).
    Returns a list of (chapter_title, chapter_body) tuples.
    Duplicate chapter titles are allowed and preserved by index.
    """
    chapters = re.split(r'(?m)^#{1,2} (.+)$', markdown_text)
    chapter_titles = []
    chapter_bodies = []
    if len(chapters) > 1:
        for i in range(1, len(chapters), 2):
            chapter_title = chapters[i].strip()
            chapter_body = chapters[i+1].strip()
            chapter_titles.append(chapter_title)
            chapter_bodies.append(chapter_body)
    else:
        # No headings found, treat as single chapter
        chapter_titles = ["Full Document"]
        chapter_bodies = [markdown_text]
    # Return with index to preserve duplicates
    return [(f"{title}", body) for title, body in zip(chapter_titles, chapter_bodies)]

def summarize_chapter(
    markdown_text: str,
    output_filename: str = "summary",
    model_choice: str = "Gemini"
) -> tuple[str, str]:
    """
    Summarizes the given markdown_text and saves it to the specified output_filename.
    Returns (summary_text, output_path).
    """
    try:
        try:
            lang = langdetect.detect(markdown_text)
        except:
            lang = "ar"  # Default to Arabic if detection fails

        prompt = f"{SUMMARY_PROMPT_TEMPLATE.strip()}\n\nLanguage: {lang}\n\nChapter Content:\n\n{markdown_text}"

        if model_choice == "Gemini":
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            summary_text = response.text.strip() if response and response.text else "*No summary generated.*"
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

        # Ensure output_filename is unique if needed (for duplicate chapter names)
        output_folder = os.path.join("outputs", "summaries")
        os.makedirs(output_folder, exist_ok=True)
        safe_filename = "".join(c for c in output_filename if c.isalnum() or c in (' ', '_', '-')).rstrip()
        base_path = os.path.join(output_folder, f"{safe_filename}.md")
        output_path = base_path
        count = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_folder, f"{safe_filename}_{count}.md")
            count += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        return summary_text, output_path

    except Exception as e:
        return f"⚠️ Error during summarization: {e}", None