"""
ingest.py

Command-line tool to load documents into the vector store.

Usage:
    python ingest.py path/to/file.pdf
    python ingest.py path/to/docs_folder/
    python ingest.py path/to/docs_folder/ --clear   # wipe the store first
"""

import argparse
from pathlib import Path

from rag_pipeline import VectorStore

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def collect_files(target: str) -> list:
    path = Path(target)
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if path.is_dir():
        return [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    raise FileNotFoundError(f"No such file or directory: {target}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("target", help="Path to a file or a folder of documents")
    parser.add_argument("--clear", action="store_true", help="Clear the existing store first")
    args = parser.parse_args()

    store = VectorStore()
    if args.clear:
        print("Clearing existing vector store...")
        store.clear()

    files = collect_files(args.target)
    if not files:
        print(f"No supported files (.txt, .md, .pdf) found at: {args.target}")
        return

    total_chunks = 0
    for file_path in files:
        try:
            n = store.add_document(str(file_path))
            total_chunks += n
            print(f"  + {file_path.name}: {n} chunks")
        except Exception as e:
            print(f"  ! Failed on {file_path.name}: {e}")

    print(f"\nDone. Ingested {len(files)} file(s), {total_chunks} chunks total.")


if __name__ == "__main__":
    main()
