import os
import google.generativeai as genai
import openai
from dotenv import load_dotenv
import streamlit as st
import langdetect
from utils.chapter_utils import summarize_chapter, split_markdown_into_chapters


# Load API keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_AI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY


# Streamlit app for summarizing Markdown chapters
st.header("📂 Summarize a Markdown File")

model_choice = st.radio("Choose summarization model:", ["Gemini", "OpenAI"])

uploaded_md = st.file_uploader("Upload a previously extracted Markdown (.md) file", type=["md"])

if uploaded_md:
    md_content = uploaded_md.read().decode("utf-8")
    st.subheader("📄 Markdown File Preview")
    st.code(md_content, language="markdown")

    # --- Summarization mode selection ---
    st.markdown("### 🛠️ Summarization Mode")
    summarize_mode = st.radio(
        "Would you like to summarize by chapters or summarize the entire file as one?",
        ["By Chapters", "All Together"]
    )

    if summarize_mode == "By Chapters":
        st.markdown("### 📑 Chapter Selection and Output Naming")
        chapters = split_markdown_into_chapters(md_content)
        chapter_titles = [title for title, _ in chapters]
        chapter_bodies = [body for _, body in chapters]

        if len(chapter_titles) > 1:
            selected_chapters = st.multiselect("Select chapters to summarize:", chapter_titles, default=chapter_titles)
        else:
            selected_chapters = chapter_titles

        st.markdown("### 📝 Choose Output File Names for Selected Chapters")
        output_names = {}
        for idx, chapter in enumerate(selected_chapters):
            default_name = chapter.replace(" ", "_").replace("/", "_")
            # Use idx in both the key and the dictionary to guarantee uniqueness
            output_names[(chapter, idx)] = st.text_input(
                f"Output file name for '{chapter}' (instance {idx+1}):",
                value=default_name,
                key=f"outname_{default_name}_{idx}"
            )

        if st.button("🧠 Summarize Selected Chapters"):
            with st.spinner("Summarizing..."):
                for idx, chapter in enumerate(chapter_titles):
                    if chapter in selected_chapters:
                        chapter_body = chapter_bodies[idx]
                        output_name = output_names[(chapter, idx)]
                        summary, summary_path = summarize_chapter(chapter_body, output_name, model_choice)
                        st.subheader(f"📘 Summary for: {chapter}")
                        st.markdown(summary)
                        if summary_path and os.path.exists(summary_path):
                            with open(summary_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    label=f"📥 Download Summary for {chapter}",
                                    data=f.read(),
                                    file_name=os.path.basename(summary_path),
                                    mime="text/markdown",
                                    key=f"dl_{chapter}_{idx}"
                                )
                        else:
                            st.error(f"Summary file for '{chapter}' could not be generated or found.")
    else:  # All Together
        st.markdown("### 📝 Output File Name")
        default_name = uploaded_md.name.rsplit(".", 1)[0] + "_summary"
        output_name = st.text_input("Output file name for the summary:", value=default_name)
        if st.button("🧠 Summarize Entire File"):
            with st.spinner("Summarizing..."):
                summary, summary_path = summarize_chapter(md_content, output_name, model_choice)
                st.subheader("📘 Summary for Entire File")
                st.markdown(summary)
                if summary_path and os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="📥 Download Summary",
                            data=f.read(),
                            file_name=os.path.basename(summary_path),
                            mime="text/markdown",
                            key="dl_all"
                        )
                else:
                    st.error("Summary file could not be generated or found.")



