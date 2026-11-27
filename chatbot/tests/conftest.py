import sys
from pathlib import Path

VERSION2 = Path(__file__).resolve().parents[2]
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))
from chatbot import config  # noqa: E402,F401  (adds eligibility/ and embedding/ to sys.path)

import pytest  # noqa: E402

from chatbot.llm_clients import LLMResponse, ToolCall  # noqa: E402


class ScriptedLLM:
    """A fake LLM that replays a script, so the whole agent runs offline and deterministically.
    Each script step is an LLMResponse, an Exception, or a callable(messages) -> LLMResponse."""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), []

    def chat(self, system, messages, tools):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if callable(step) else step


def tool_call(name, **args):
    return LLMResponse(tool_calls=[ToolCall(f"call_{name}", name, args)], input_tokens=10, output_tokens=5)


def say(text):
    return LLMResponse(text=text, input_tokens=10, output_tokens=5)


def make_hit(cid="CAR-003", score=0.82, text=None, **meta):
    base = {"id": cid, "product": "car_loans", "source_document": "faq_car_loans.json",
            "accessed": "2026-10-22", "accessed_estimated": True, "needs_review": False}
    base.update(meta)
    return {"chunk_id": f"{cid}:ar", "id": cid, "score": score,
            "context_text": text or ("[قروض السيارات]\nالسؤال: إيه المستندات المطلوبة؟\n"
                             "الإجابة: بطاقة الرقم القومي وإثبات الدخل."),
            "metadata": base}


@pytest.fixture
def scripted():
    return ScriptedLLM
