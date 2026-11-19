"""Provider-neutral LLM client.

The agent talks to ONE interface:   llm.chat(system, messages, tools) -> LLMResponse
Each provider adapter translates to/from that provider's wire format, so swapping Claude / OpenAI /
Gemini is a one-line change in .env (LLM_PROVIDER) and never touches the agent.

Neutral message format (what agent.py builds):
    {"role": "user",      "content": "text"}
    {"role": "assistant", "content": "text or ''", "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "result text"}

Neutral tool spec:  {"name", "description", "parameters": <JSON schema>}
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def chat(self, system: str, messages: list, tools: list) -> LLMResponse: ...


# =============================================================== Anthropic (Claude)
def to_anthropic_messages(messages):
    out = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            prev = out[-1] if out else None
            # Anthropic wants ALL results of one assistant turn inside ONE user message.
            if prev and prev["role"] == "user" and isinstance(prev["content"], list) \
                    and prev["content"] and prev["content"][0].get("type") == "tool_result":
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicLLM:
    def __init__(self, model: str, api_key: Optional[str] = None, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=config.LLM_TIMEOUT_S)
        self.client, self.model = client, model

    def chat(self, system, messages, tools):
        kwargs = dict(model=self.model, system=system, max_tokens=config.LLM_MAX_TOKENS,
                      messages=to_anthropic_messages(messages))
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
        if config.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.LLM_TEMPERATURE
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text, calls, resp.usage.input_tokens, resp.usage.output_tokens)


# =============================================================== OpenAI-compatible (OpenAI, Gemini, ...)
def to_openai_messages(messages):
    out = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                      "function": {"name": tc["name"],
                                                   "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                                     for tc in m["tool_calls"]]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


class OpenAICompatLLM:
    """Works with OpenAI and with any OpenAI-compatible endpoint, including Google Gemini
    (base_url = config.GEMINI_OPENAI_BASE_URL)."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, client=None):
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=config.LLM_TIMEOUT_S)
        self.client, self.model = client, model

    def chat(self, system, messages, tools):
        kwargs = dict(model=self.model, max_tokens=config.LLM_MAX_TOKENS,
                      messages=[{"role": "system", "content": system}] + to_openai_messages(messages))
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if config.LLM_TEMPERATURE is not None:
            kwargs["temperature"] = config.LLM_TEMPERATURE
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {}))
        usage = getattr(resp, "usage", None)
        return LLMResponse(msg.content or "", calls,
                           getattr(usage, "prompt_tokens", 0) or 0,
                           getattr(usage, "completion_tokens", 0) or 0)


# =============================================================== factory
def make_llm(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model or config.LLM_MODEL or config.DEFAULT_MODELS.get(provider, "")
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in version2/.env)")
        return AnthropicLLM(model, api_key=key)
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (put it in version2/.env)")
        return OpenAICompatLLM(model, api_key=key)
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in version2/.env)")
        return OpenAICompatLLM(model, api_key=key, base_url=config.GEMINI_OPENAI_BASE_URL)
    raise ValueError(f"unknown LLM_PROVIDER '{provider}' (use anthropic | openai | gemini)")
