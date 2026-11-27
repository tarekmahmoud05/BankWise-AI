"""The orchestrator: conversation memory + LLM tool-use loop + rules enforced in CODE.

One user message goes through:
    1. input hygiene (empty / too long)
    2. loop (max MAX_TOOL_STEPS): ask the LLM; if it requests tools, run them and give the results back
    3. guards: abstain, citations -> sources footer, eligibility disclaimer, error fallback
    4. save the clean turn into the session history
"""
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .agent_tools import ToolRunner, all_tool_specs
from .llm_clients import LLMClient
from .prompts import SYSTEM_PROMPT
from .retriever import Retriever

log = logging.getLogger("bankwise.agent")

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_QUESTION = re.compile(r"السؤال:\s*(.+)")
_CITE = re.compile(r"\[\s*(S\d+(?:\s*[,،]\s*S\d+)*)\s*\]")
FALLBACK_EN = ("Sorry, I don't have confirmed information to answer this. "
               "Please call Banque Misr on 19888 or visit your nearest branch.")
ERROR_EN = "A temporary error occurred. Please try again shortly, or call Banque Misr on 19888."
EMPTY_EN = "Please type your question and I will help."


def _is_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text or ""))


@dataclass
class AgentReply:
    text: str                                   # what to show the customer (body + sources + disclaimer)
    answer: str = ""                            # body only
    sources: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    abstained: bool = False
    reason: str = ""                            # "", no_knowledge, empty_input, llm_error, loop_limit
    top_score: float = 0.0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    suggestions: list = field(default_factory=list)

class BankWiseAgent:
    def __init__(self, llm: LLMClient, retriever: Optional[Retriever] = None):
        self.llm = llm
        self.retriever = retriever or Retriever()
        self.tools = all_tool_specs()
        self._history = {}                      # session_id -> list of neutral messages
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def reset(self, session_id: str):
        with self._lock:
            self._history.pop(session_id, None)

    def chat(self, session_id: str, user_message: str) -> AgentReply:
        t0 = time.time()
        user_message = (user_message or "").strip()[: config.MAX_USER_CHARS]
        ar = _is_arabic(user_message) or not user_message
        if not user_message:
            return self._done(AgentReply(config.EMPTY_AR if ar else EMPTY_EN, abstained=True,
                                         reason="empty_input"), t0)

        with self._lock:
            history = list(self._history.get(session_id, []))
        messages = history + [{"role": "user", "content": user_message}]
        runner = ToolRunner(self.retriever)
        in_tok = out_tok = 0
        if ar:
            lang_rule = "\n\n# Language for this reply\nThe customer wrote in Arabic. Reply in Arabic."
        else:
            lang_rule = ("\n\n# Language for this reply\nThe customer wrote in English. Write your ENTIRE reply "
                         "in English, even if the sources or tool results are in Arabic. Translate what you need.")
        system_prompt = SYSTEM_PROMPT + lang_rule
        try:
            final_text = None
            for _ in range(config.MAX_TOOL_STEPS):
                resp = self.llm.chat(system_prompt, messages, self.tools)
                in_tok += resp.input_tokens
                out_tok += resp.output_tokens
                if not resp.tool_calls:
                    final_text = resp.text
                    break
                messages.append({"role": "assistant", "content": resp.text,
                                 "tool_calls": [{"id": c.id, "name": c.name, "args": c.args,
                                                 **({"extra": c.extra} if c.extra else {})}
                                                for c in resp.tool_calls]})
                for c in resp.tool_calls:
                    messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                     "content": runner.run(c.name, c.args)})
            if final_text is None:
                reply = self._abstain(ar, "loop_limit")
            else:
                reply = self._finalize(final_text, runner, ar)
        except Exception as exc:                  # never show a stack trace to a customer
            log.error("agent error: %r", exc)
            reply = AgentReply(config.ERROR_AR if ar else ERROR_EN, abstained=False, reason="llm_error")

        reply.tools_used = list(runner.tools_used)
        reply.top_score = runner.top_score
        reply.input_tokens, reply.output_tokens = in_tok, out_tok

        # remember the clean turn only (no tool plumbing, no footers)
        with self._lock:
            h = self._history.setdefault(session_id, [])
            h += [{"role": "user", "content": user_message},
                  {"role": "assistant", "content": reply.answer or reply.text}]
            del h[:-config.MAX_HISTORY_MESSAGES]
        return self._done(reply, t0)

    # ------------------------------------------------------------------ guards
    def _abstain(self, ar, reason):
        text = config.FALLBACK_AR if ar else FALLBACK_EN
        return AgentReply(text, answer=text, abstained=True, reason=reason)

    def _finalize(self, text: str, runner: ToolRunner, ar: bool) -> AgentReply:
        text = (text or "").strip()
        used_rule_engine = runner.eligibility_calls > 0

        # Guard 1 - evidence: it searched, found nothing, and used no calculator -> fixed fallback,
        # whatever the LLM wrote. (The LLM is told to do this itself; here we make sure.)
        if not text or (runner.searches and not runner.searches_with_hits and not used_rule_engine) \
                or config.FALLBACK_AR in text:
            return self._abstain(ar, "no_knowledge")

        # Guard 2 - sources are built from metadata of the chunks the LLM CITED, never typed by the LLM.
        cited = []
        for group in _CITE.findall(text):
            for ref in re.findall(r"S\d+", group):
                if ref in runner.hits_by_ref and ref not in cited:
                    cited.append(ref)
        body = re.sub(r"[ \t]+([.,،؛:!؟?])", r"\1", _CITE.sub("", text)).strip()
        body = re.sub(r"[ \t]{2,}", " ", body)
        sources = [self._source(ref, runner.hits_by_ref[ref]) for ref in cited]

        # Guard 3 - eligibility results always carry the bank's own disclaimer, verbatim.
        parts = [body]
        if sources:
            title = "المصادر:" if ar else "Sources:"
            parts.append(title + "\n" + "\n".join(f"- {s['label']}" for s in sources))
        if runner.disclaimers:
            parts.append("\n".join(runner.disclaimers))
        return AgentReply("\n\n".join(parts), answer=body, sources=sources,
                          suggestions=self._suggestions(runner, cited))
    @staticmethod
    def _suggestions(runner: ToolRunner, cited: list, limit: int = 3) -> list:
        """أسئلة الـ FAQ اللي اتجابت في البحث بس ماتستخدمتش في الإجابة."""
        used = {runner.hits_by_ref[r].id for r in cited}
        out = []
        for hit in runner.hits_by_ref.values():              # الأقرب الأول
            if hit.id in used or hit.metadata.get("needs_review"):
                continue
            m = _QUESTION.search(hit.text)
            q = m.group(1).strip() if m else ""
            if q and q not in out:
                out.append(q)
                used.add(hit.id)
            if len(out) == limit:
                break
        return out

    @staticmethod
    def _source(ref, hit):
        m = hit.metadata
        doc = m.get("source_document") or m.get("source_json") or "Banque Misr FAQ"
        date = m.get("accessed")
        label = f"{hit.id} — {doc}" + (f" ({date}{'~' if m.get('accessed_estimated') else ''})" if date else "")
        return {"ref": ref, "id": hit.id, "source_document": doc, "date": date, "label": label}

    @staticmethod
    def _done(reply: AgentReply, t0: float) -> AgentReply:
        reply.latency_ms = int((time.time() - t0) * 1000)
        log.info("turn: tools=%s abstained=%s reason=%s top_score=%.2f latency_ms=%d tokens=%d/%d",
                 reply.tools_used, reply.abstained, reply.reason, reply.top_score,
                 reply.latency_ms, reply.input_tokens, reply.output_tokens)
        return reply
