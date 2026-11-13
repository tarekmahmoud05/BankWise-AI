"""Shortcut:  python ingest.py [--reset] [--embedder hash] [--data DIR] [--index DIR]

Same as `python -m bankwise_rag.cli ingest ...`. The defaults (bge-m3, data/raw -> data/sample,
chroma_db/) live in bankwise_rag/config.py. Add --reset when you change the embedder.
"""
import sys

from bankwise_rag.cli import main

if __name__ == "__main__":
    sys.exit(main(["ingest", *sys.argv[1:]]))
