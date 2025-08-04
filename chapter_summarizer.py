import os
import google.generativeai as genai
import openai
from dotenv import load_dotenv
import streamlit as st
import langdetect
from utils.chapter_utils import summarize_chapter


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

    chapter_name = uploaded_md.name.rsplit(".", 1)[0]  # Remove .md extension

    if st.button("🧠 Summarize This File"):
        with st.spinner("Summarizing..."):
            summary, summary_path = summarize_chapter(md_content, chapter_name, model_choice)
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



