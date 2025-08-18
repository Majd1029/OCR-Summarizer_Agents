import os
import langdetect
import google.generativeai as genai
from openai import OpenAI

SUMMARY_PROMPT_TEMPLATE = """
You are an expert AI trained in summarizing academic and educational documents.

Your task is to read and summarize the provided Markdown chapter content into a **detailed and comprehensive summary**, keeping the same language as the original content.

### Guidelines:
- Use the **same language** as the original text.
- Create a **longer summary with multiple paragraphs**.
- Explain key points, important definitions, and concepts.
- Maintain logical flow and structure.
- Do **not** include headings or page numbers.
- Do **not** bullet or number anything; use paragraph form.
- Make the result easy to study from.

Only return clean, properly spaced, multi-paragraph **Markdown-formatted** text.
"""

def summarize_chapter(markdown_text: str, chapter_name: str = "summary", model_choice: str = "Gemini") -> tuple[str, str]:
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
                    {"role": "system", "content": "You are an expert AI trained in summarizing academic and educational documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000
            )
            summary_text = response.choices[0].message.content.strip()

        # Save to .md file
        safe_filename = "".join(c for c in chapter_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        output_folder = os.path.join("outputs", "summaries")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{safe_filename}_summary.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        return summary_text, output_path

    except Exception as e:
        return f"⚠️ Error during summarization: {e}", None