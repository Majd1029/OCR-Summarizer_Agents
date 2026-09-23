import pathlib

import streamlit as st

from ocr_backends import pdf_to_images, run_ocr
from summarizer import split_markdown_into_chapters, summarize_chapter

st.set_page_config(page_title="OCR → Summarizer", page_icon="📖", layout="wide")
st.title("📖 OCR → Summarizer")
st.caption(
    "Multilingual OCR (Arabic / French / English) to Markdown, then chapter summaries. "
    "[Source](https://github.com/Majd1029/OCR-Summarizer_Agents)"
)

st.session_state.setdefault("markdown", "")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("⚙️ Backend")
    backend = st.radio(
        "Engine",
        ["Tesseract — printed text", "Claude — handwriting, maths, tables"],
        captions=[
            "Free, always on. Reads clean printed French and English well. "
            "Arabic comes out unreliable even on printed text, so use Claude "
            "for that. Also struggles with handwriting, mathematical notation "
            "and complex table layouts.",
            "Far better on exactly those hard cases, and what this project was "
            "built around. Needs your own Anthropic API key.",
        ],
    )
    backend = "Claude" if backend.startswith("Claude") else "Tesseract"

    api_key = None
    if backend == "Tesseract":
        st.info(
            "Tesseract is the free path, so it is the default. It reads clean "
            "printed Latin-script documents well. For Arabic, handwriting or "
            "mathematical notation — what this project was originally built "
            "for — switch to Claude. That is where the difference shows.",
            icon=":material/info:",
        )

    if backend == "Claude":
        api_key = st.text_input("Claude API key", type="password")
        st.caption(
            "Held in this browser session only — never stored, logged or sent "
            "anywhere except Anthropic. Create one at console.anthropic.com."
        )
        if not api_key:
            st.warning("No key entered — switch to Tesseract or paste a key.")

    st.divider()
    if st.button("🧹 Clear results"):
        st.session_state.markdown = ""
        st.rerun()

# ------------------------------------------------------------- extraction
tab_extract, tab_summarize = st.tabs(["1 · Extract", "2 · Summarize"])

with tab_extract:
    SAMPLES = pathlib.Path("demo_samples")
    sample_files = (
        sorted(f for f in SAMPLES.iterdir()
               if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"))
        if SAMPLES.is_dir() else []
    )

    uploaded = st.file_uploader("Upload a PDF or image",
                                type=["pdf", "png", "jpg", "jpeg"])

    if sample_files and not uploaded:
        st.caption("No document to hand? Try one of these — clean printed "
                   "Latin-script text, which is what the free Tesseract path "
                   "handles well.")
        cols = st.columns(len(sample_files))
        for col, f in zip(cols, sample_files):
            if f.suffix.lower() != ".pdf":
                col.image(str(f), caption=f.stem, use_container_width=True)
            if col.button(f.stem, key=f"sample_{f.stem}",
                          width="stretch"):
                st.session_state.sample_pick = str(f)

    picked = st.session_state.get("sample_pick")
    if picked and not uploaded:
        class _Local:
            """Adapter so a bundled sample behaves like a Streamlit upload."""

            def __init__(self, path):
                self.name = pathlib.Path(path).name
                self._b = pathlib.Path(path).read_bytes()

            def getvalue(self):
                return self._b

            def read(self):
                return self._b

        uploaded = _Local(picked)
        st.caption(f"Using bundled sample: **{uploaded.name}**")

    if uploaded:
        is_pdf = uploaded.name.lower().endswith(".pdf")

        if is_pdf:
            c1, c2 = st.columns(2)
            start = c1.number_input("Start page", min_value=1, step=1, value=1)
            end = c2.number_input("End page", min_value=int(start), step=1,
                                  value=int(start))
            st.caption("Keep the range small — each page is a separate OCR pass.")

        if st.button("🚀 Extract text", type="primary",
                     disabled=(backend == "Claude" and not api_key)):
            try:
                if is_pdf:
                    with st.spinner("Rendering pages..."):
                        images = pdf_to_images(uploaded.getvalue(), int(start), int(end))
                else:
                    from PIL import Image
                    images = [Image.open(uploaded)]

                st.image(images, width=220,
                         caption=[f"Page {i + 1}" for i in range(len(images))])

                chunks, bar = [], st.progress(0.0, "Running OCR...")
                for i, img in enumerate(images):
                    bar.progress(i / len(images), f"OCR page {i + 1}/{len(images)}")
                    text = run_ocr(img, backend, api_key)
                    header = f"## Page {i + 1}\n\n" if len(images) > 1 else ""
                    chunks.append(header + text)
                bar.empty()

                st.session_state.markdown = "\n\n".join(chunks)
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    if st.session_state.markdown:
        st.subheader("📄 Extracted Markdown")
        st.markdown(st.session_state.markdown, unsafe_allow_html=True)
        with st.expander("Raw Markdown"):
            st.code(st.session_state.markdown, language="markdown")
        st.download_button(
            "📥 Download .md", st.session_state.markdown,
            file_name=f"Extracted_{uploaded.name.rsplit('.', 1)[0]}.md"
            if uploaded else "extracted.md",
            mime="text/markdown",
        )

# ------------------------------------------------------------ summarizing
with tab_summarize:
    source = st.radio("Summarize", ["The text I just extracted", "An uploaded .md file"],
                      horizontal=True)

    content = st.session_state.markdown
    if source.endswith(".md file"):
        md_file = st.file_uploader("Upload a Markdown file", type=["md"], key="mdup")
        content = md_file.read().decode("utf-8") if md_file else ""

    if not content:
        st.info("Nothing to summarize yet — extract something in tab 1 first.")
    else:
        chapters = split_markdown_into_chapters(content)
        titles = [t for t, _ in chapters]
        selected = st.multiselect("Chapters", titles, default=titles[:3])

        if st.button("🧠 Summarize", type="primary",
                     disabled=(backend == "Claude" and not api_key)):
            for title, body in chapters:
                if title not in selected:
                    continue
                st.subheader(f"📘 {title}")
                with st.spinner(f"Summarizing {title}..."):
                    try:
                        summary = summarize_chapter(body, backend, api_key)
                    except ValueError as e:
                        st.error(str(e))
                        break
                st.markdown(summary)
                st.download_button(
                    f"📥 Download summary — {title}", summary,
                    file_name=f"{title.replace(' ', '_')[:60]}_summary.md",
                    mime="text/markdown", key=f"dl_{title}",
                )
