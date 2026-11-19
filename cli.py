"""Terminal chat.   Run from version2/:   python -m chatbot.cli [--provider gemini] [--debug]"""
import argparse
import logging
import uuid

from . import config
from .agent import BankWiseAgent
from .llm_clients import make_llm
from .retriever import Retriever


def build_agent(provider=None, model=None) -> BankWiseAgent:
    llm = make_llm(provider, model)          # first: fails fast if the API key is missing
    retriever = Retriever()
    print("loading embedding model (first run downloads BAAI/bge-m3, ~2 GB)...", flush=True)
    retriever.warmup()
    return BankWiseAgent(llm, retriever)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="anthropic | openai | gemini (default: LLM_PROVIDER in .env)")
    ap.add_argument("--model")
    ap.add_argument("--debug", action="store_true", help="show tools used, retrieval score, latency, tokens")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    agent = build_agent(args.provider, args.model)
    sid = uuid.uuid4().hex
    print(f"BankWise AI ready (provider={args.provider or config.LLM_PROVIDER}). "
          "/new = new chat, empty line = quit\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            break
        if msg == "/new":
            agent.reset(sid)
            print("(new conversation)\n")
            continue
        r = agent.chat(sid, msg)
        print(f"\nbot> {r.text}\n")
        if args.debug:
            print(f"     [tools={r.tools_used} abstained={r.abstained} reason={r.reason or '-'} "
                  f"top_score={r.top_score:.2f} {r.latency_ms}ms tokens={r.input_tokens}/{r.output_tokens}]\n")


if __name__ == "__main__":
    main()
