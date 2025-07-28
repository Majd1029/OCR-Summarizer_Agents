import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import langdetect

# Load Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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


def summarize_chapter(markdown_text: str, chapter_name: str = "summary") -> tuple[str, str]:
    try:
        import langdetect
        try:
            lang = langdetect.detect(markdown_text)
        except:
            lang = "en"

        prompt = f"{SUMMARY_PROMPT_TEMPLATE.strip()}\n\nLanguage: {lang}\n\nChapter Content:\n\n{markdown_text}"

        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)

        if not response or not response.text.strip():
            return "*No summary generated.*", None

        summary_text = response.text.strip()

        # Save to .md file
        safe_filename = "".join(c for c in chapter_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        output_path = f"summaries/{safe_filename}_summary.md"
        os.makedirs("summaries", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        return summary_text, output_path

    except Exception as e:
        return f"⚠️ Error during summarization: {e}", None



# Streamlit app for summarizing Markdown chapters
st.header("📂 Summarize a Markdown File")

uploaded_md = st.file_uploader("Upload a previously extracted Markdown (.md) file", type=["md"])

if uploaded_md:
    md_content = uploaded_md.read().decode("utf-8")
    st.subheader("📄 Markdown File Preview")
    st.code(md_content, language="markdown")

    chapter_name = uploaded_md.name.rsplit(".", 1)[0]  # Remove .md extension

    if st.button("🧠 Summarize This File"):
        with st.spinner("Summarizing..."):
            summary, summary_path = summarize_chapter(md_content, chapter_name)
            st.subheader("📘 Chapter Summary")
            st.markdown(summary)

            if summary_path and os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 Download Summary",
                        data=f.read(),
                        file_name=os.path.basename(summary_path),
                        mime="text/markdown"
                    )
            else:
                st.error("Summary file could not be generated or found.")



