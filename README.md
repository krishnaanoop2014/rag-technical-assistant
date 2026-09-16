# Retrieval-Augmented AI Assistant for Technical Documents

A RAG (Retrieval-Augmented Generation) system that ingests technical documents
(PDF, Markdown, text), stores them in a local vector database, and answers
questions grounded in that content using Claude — with a Streamlit chat UI.

## How it works

```
Documents (.pdf/.md/.txt)
        │
        ▼
   Chunking (rag_pipeline.chunk_text)
        │
        ▼
Embeddings (sentence-transformers, local, free)
        │
        ▼
  Chroma vector store (persisted to disk)
        │
        ▼
Query → retrieve top-k relevant chunks
        │
        ▼
Claude (Anthropic API) generates an answer
using ONLY the retrieved chunks as context
```

## Project structure

```
rag-assistant/
├── app.py              # Streamlit chat UI
├── ingest.py           # CLI to bulk-load documents into the vector store
├── rag_pipeline.py      # Core RAG logic (chunking, embedding, retrieval, generation)
├── requirements.txt
├── .env.example
├── .gitignore
└── sample_docs/
    └── sample_networking_notes.md
```

## Setup

1. **Clone / create the repo** (see "Push to GitHub" below if starting fresh).

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set your Anthropic API key:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your key:
   # ANTHROPIC_API_KEY=sk-ant-...
   ```
   Get a key from https://console.anthropic.com if you don't have one.

## Usage

### Option A — Command line ingestion + quick test

```bash
# Ingest the included sample doc (or point at your own file/folder)
python ingest.py sample_docs/

# Then test retrieval + generation from Python:
python3 -c "
from rag_pipeline import ask
result = ask('When should I use UDP instead of TCP?')
print(result['answer'])
"
```

### Option B — Streamlit chat UI (recommended)

```bash
streamlit run app.py
```

Then in the browser:
1. Upload one or more `.txt`, `.md`, or `.pdf` files in the sidebar.
2. Click **"Ingest into knowledge base"**.
3. Ask questions in the chat box at the bottom — answers show which document
   excerpts they were grounded in.

## Why this counts as a real project (not a toy)

- **Persistent vector store** — Chroma data is saved to `./chroma_store/`, so
  your knowledge base survives restarts.
- **Grounded answers, not hallucinated ones** — the system prompt forces
  Claude to answer only from retrieved context and admit when it doesn't know.
- **Source attribution** — every answer shows which chunks/documents it used,
  so you can verify the answer yourself.
- **Configurable** — chunk size, embedding model, and the Claude model are all
  environment-variable driven (see `.env.example`), so it's easy to extend or
  swap components (e.g., a different embedding model, or Pinecone/Weaviate
  instead of Chroma) later.

## Extending it (good "v2" ideas for your resume)

- Add re-ranking of retrieved chunks with a cross-encoder before generation.
- Support `.docx` and `.html` ingestion.
- Add conversation memory so follow-up questions can reference earlier turns.
- Deploy the Streamlit app (e.g., Streamlit Community Cloud or a small VPS)
  and link the live demo from your resume/GitHub README.
- Swap the local Chroma store for a hosted vector DB (Pinecone, Qdrant Cloud)
  to demonstrate production-style architecture.

## Push to GitHub

If this is a brand-new project:

```bash
cd rag-assistant
git init
git add .
git commit -m "Initial commit: RAG assistant for technical documents"
git branch -M main
git remote add origin https://github.com/<your-username>/rag-technical-assistant.git
git push -u origin main
```

Create the empty repo on GitHub first (github.com/new) before running the
`push` command, and make sure `.env` is **not** committed (it's already in
`.gitignore`).
