"""
rag_pipeline.py

Core Retrieval-Augmented Generation (RAG) logic for the Technical Document Assistant.

This module handles:
    1. Loading documents (.txt, .md, .pdf) from disk
    2. Splitting them into overlapping chunks
    3. Embedding chunks and storing them in a local Chroma vector database
    4. Retrieving the most relevant chunks for a user query
    5. Calling Claude to generate a grounded answer using only the retrieved context

Everything runs locally except the final answer-generation call to Groq's
free API, so you only need a GROQ_API_KEY to get answers - embeddings and
storage are free and run on your machine.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "technical_docs")
EMBEDDING_MODEL_NAME = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Groq's free tier hosts open-weight models (Llama, etc.) at very low latency.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 120    # characters of overlap between consecutive chunks
TOP_K = 4              # number of chunks to retrieve per query


# ---------------------------------------------------------------------------
# 1. Loading documents
# ---------------------------------------------------------------------------

def load_document_text(file_path: str) -> str:
    """Read a .txt, .md, or .pdf file and return its raw text content."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .md, or .pdf")


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-based chunks.

    A simple sliding-window chunker. Good enough for most technical docs;
    swap in a token-aware or sentence-aware splitter later if you need more
    precision on long, structured documents.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# 3. Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_document(self, file_path: str) -> int:
        """Chunk a document and add it to the vector store. Returns #chunks added."""
        text = load_document_text(file_path)
        chunks = chunk_text(text)
        if not chunks:
            return 0

        source_name = Path(file_path).name
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> List[Dict]:
        """Return the top_k most relevant chunks for a question."""
        results = self.collection.query(query_texts=[question], n_results=top_k)

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "source": meta.get("source"), "distance": dist})
        return hits

    def clear(self):
        """Delete and recreate the collection (useful during development)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embedding_fn
        )


# ---------------------------------------------------------------------------
# 4. Answer generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer the user's question using ONLY the provided context excerpts.
If the context does not contain enough information to answer, say so plainly
instead of guessing. Always mention which source(s) you drew from when relevant.
Keep answers concise and technically precise."""


def build_context_block(hits: List[Dict]) -> str:
    """Format retrieved chunks into a single context string for the prompt."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[Excerpt {i} - source: {hit['source']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: List[Dict], client: Groq = None) -> str:
    """Send the retrieved context + question to a Groq-hosted LLM and return the answer."""
    if not hits:
        return "I couldn't find anything relevant in the ingested documents to answer that."

    client = client or Groq()  # reads GROQ_API_KEY from env
    context = build_context_block(hits)

    user_message = (
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 5. High-level convenience function
# ---------------------------------------------------------------------------

def ask(question: str, store: VectorStore = None, top_k: int = TOP_K) -> Dict:
    """End-to-end: retrieve relevant chunks, generate an answer, return both."""
    store = store or VectorStore()
    hits = store.query(question, top_k=top_k)
    answer = generate_answer(question, hits)
    return {"answer": answer, "sources": hits}
