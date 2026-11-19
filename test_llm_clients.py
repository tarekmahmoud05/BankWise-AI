"""Adapters tested with the REAL provider SDKs talking to a mock HTTP server (no API key, no network)."""
import json

import httpx
import pytest

try:                      # recent Anthropic SDKs are built on httpx2, older ones on httpx
    import httpx2 as _anthropic_http
except ImportError:
    _anthropic_http = httpx

from chatbot import config
from chatbot.agent import BankWiseAgent
from chatbot.agent_tools import all_tool_specs
from chatbot.llm_clients import (AnthropicLLM, OpenAICompatLLM, make_llm,
                                 to_anthropic_messages, to_openai_messages)
from chatbot.retriever import Retriever
from .conftest import make_hit

NEUTRAL = [
    {"role": "user", "content": "ايه مستندات قرض العربية؟"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "name": "search_faq", "args": {"query": "مستندات"}},
        {"id": "t2", "name": "search_faq", "args": {"query": "شروط"}}]},
    {"role": "tool", "tool_call_id": "t1", "name": "search_faq", "content": "result one"},
    {"role": "tool", "tool_call_id": "t2", "name": "search_faq", "content": "result two"},
]


def test_anthropic_groups_parallel_tool_results_in_one_user_message():
    out = to_anthropic_messages(NEUTRAL)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]


def test_openai_format_has_one_tool_message_per_call_and_json_string_arguments():
    out = to_openai_messages(NEUTRAL)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool"]
    assert out[1]["content"] is None
    args = out[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str) and json.loads(args) == {"query": "مستندات"}


# ------------------------------------------------------------ Anthropic SDK over a mock server
def _anthropic_client(handler):
    import anthropic
    return anthropic.Anthropic(api_key="test", base_url="https://mock.test",
                               http_client=_anthropic_http.Client(transport=_anthropic_http.MockTransport(handler)))


def _anthropic_reply(content, stop):
    return _anthropic_http.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                                     "content": content, "stop_reason": stop, "stop_sequence": None,
                                     "usage": {"input_tokens": 11, "output_tokens": 7}})


def test_anthropic_adapter_request_and_response_shape():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _anthropic_reply([{"type": "text", "text": "تمام"},
                                 {"type": "tool_use", "id": "toolu_1", "name": "search_faq",
                                  "input": {"query": "مستندات"}}], "tool_use")

    llm = AnthropicLLM("claude-test", client=_anthropic_client(handler))
    resp = llm.chat("SYS", NEUTRAL[:1], all_tool_specs())
    body = seen["body"]
    assert body["system"] == "SYS" and body["model"] == "claude-test" and body["max_tokens"] == config.LLM_MAX_TOKENS
    assert {t["name"] for t in body["tools"]} >= {"search_faq", "check_car_loan_eligibility"}
    assert all("input_schema" in t for t in body["tools"])
    assert resp.text == "تمام" and resp.tool_calls[0].name == "search_faq"
    assert resp.tool_calls[0].args == {"query": "مستندات"} and (resp.input_tokens, resp.output_tokens) == (11, 7)


def test_full_agent_through_the_real_anthropic_sdk():
    """User question -> Claude asks for search_faq -> our RAG runs -> Claude answers with [S1]."""
    requests = []

    def handler(req):
        body = json.loads(req.content)
        requests.append(body)
        if len(requests) == 1:
            return _anthropic_reply([{"type": "tool_use", "id": "toolu_1", "name": "search_faq",
                                      "input": {"query": "المستندات المطلوبة لقرض السيارة"}}], "tool_use")
        last = body["messages"][-1]
        assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
        assert "إثبات الدخل" in last["content"][0]["content"]          # our chunk reached the model
        return _anthropic_reply([{"type": "text", "text": "تحتاج بطاقة الرقم القومي وإثبات الدخل [S1]."}], "end_turn")

    agent = BankWiseAgent(AnthropicLLM("claude-test", client=_anthropic_client(handler)),
                          Retriever(search_fn=lambda q, k, p: [make_hit()], min_score=0.5))
    r = agent.chat("s", "ايه المستندات المطلوبة لقرض العربية؟")
    assert len(requests) == 2 and "المصادر:" in r.text and r.sources[0]["id"] == "CAR-003"


# ------------------------------------------------------------ OpenAI-compatible SDK over a mock server
def _openai_client(handler):
    from openai import OpenAI
    return OpenAI(api_key="test", base_url="https://mock.test/v1",
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _openai_reply(message, finish):
    return httpx.Response(200, json={"id": "c1", "object": "chat.completion", "created": 1, "model": "m",
                                     "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}})


def test_openai_adapter_request_and_response_shape():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return _openai_reply({"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "search_faq", "arguments": json.dumps({"query": "x"})}}]}, "tool_calls")

    resp = OpenAICompatLLM("m", client=_openai_client(handler)).chat("SYS", NEUTRAL[:1], all_tool_specs())
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["tools"][0]["type"] == "function" and body["tools"][0]["function"]["name"] == "search_faq"
    assert resp.tool_calls[0].args == {"query": "x"} and (resp.input_tokens, resp.output_tokens) == (12, 6)


def test_openai_adapter_survives_malformed_tool_arguments():
    def handler(req):
        bad_call = {"id": "c", "type": "function", "function": {"name": "search_faq", "arguments": "{not json"}}
        return _openai_reply({"role": "assistant", "content": "", "tool_calls": [bad_call]}, "tool_calls")
    llm = OpenAICompatLLM("m", client=_openai_client(handler))
    resp = llm.chat("S", NEUTRAL[:1], [])
    assert resp.tool_calls[0].args == {}


# ------------------------------------------------------------ factory
def test_factory_gives_clear_errors(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for p in ("anthropic", "openai", "gemini"):
        with pytest.raises(RuntimeError, match="API_KEY"):
            make_llm(p)
    with pytest.raises(ValueError):
        make_llm("nope")


def test_factory_builds_each_provider(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(k, "k")
    assert isinstance(make_llm("anthropic"), AnthropicLLM)
    assert isinstance(make_llm("openai"), OpenAICompatLLM)
    g = make_llm("gemini")
    assert isinstance(g, OpenAICompatLLM) and "generativelanguage.googleapis.com" in str(g.client.base_url)
