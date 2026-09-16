"""
app.py

Streamlit UI for the Retrieval-Augmented AI Assistant for Technical Documents.

Run with:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import VectorStore, generate_answer

load_dotenv()

st.set_page_config(page_title="Technical Docs RAG Assistant", page_icon="📚")

st.title("📚 Retrieval-Augmented AI Assistant for Technical Documents")
st.caption("Upload technical documents, then ask questions grounded in their content.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning(
        "No ANTHROPIC_API_KEY found in your environment. "
        "Create a `.env` file (see `.env.example`) before asking questions."
    )

# --- Persistent state -------------------------------------------------------

if "store" not in st.session_state:
    st.session_state.store = VectorStore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, sources}

# --- Sidebar: document upload & ingestion -----------------------------------

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Add .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest into knowledge base"):
        with st.spinner("Chunking and embedding documents..."):
            total_chunks = 0
            for uploaded in uploaded_files:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                n = st.session_state.store.add_document(tmp_path)
                total_chunks += n
                os.unlink(tmp_path)

            st.success(f"Ingested {len(uploaded_files)} file(s), {total_chunks} chunks total.")

    st.divider()
    if st.button("Clear knowledge base"):
        st.session_state.store.clear()
        st.session_state.chat_history = []
        st.success("Knowledge base cleared.")

# --- Main: question answering ------------------------------------------------

question = st.chat_input("Ask a question about your uploaded documents...")

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander("Sources used"):
                for i, src in enumerate(turn["sources"], start=1):
                    st.markdown(f"**Excerpt {i}** — `{src['source']}`")
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context and generating an answer..."):
            hits = st.session_state.store.query(question)
            answer = generate_answer(question, hits)
            st.write(answer)
            if hits:
                with st.expander("Sources used"):
                    for i, src in enumerate(hits, start=1):
                        st.markdown(f"**Excerpt {i}** — `{src['source']}`")
                        st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))

    st.session_state.chat_history.append({"question": question, "answer": answer, "sources": hits})
