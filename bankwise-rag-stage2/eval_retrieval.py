"""Shortcut:  python eval_retrieval.py [questions.json] [--embedder hash] [--language all] [--k 5]

Same as `python -m bankwise_rag.cli eval ...`. Defaults: eval/eval_questions_bankwise.json,
bge-m3, language ar (see bankwise_rag/config.py).
"""
import sys

from bankwise_rag.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
