import os
from chatbot import config
from openai import OpenAI, APIStatusError

key = os.getenv("GEMINI_API_KEY")
print("key loaded:", bool(key), "| length:", len(key or ""))
print("LLM_PROVIDER:", config.LLM_PROVIDER, "| LLM_MODEL:", repr(config.LLM_MODEL))

c = OpenAI(api_key=key, base_url=config.GEMINI_OPENAI_BASE_URL, max_retries=0, timeout=30)

def show(e):
    if isinstance(e, APIStatusError):
        return f"{e.status_code} {e.response.text[:300]}"
    return repr(e)

print("\n== الموديلات المتاحة ==")
try:
    print([m.id for m in c.models.list()])
except Exception as e:
    print("LIST ERROR:", show(e))

tool = {"type": "function", "function": {"name": "ping", "description": "test",
        "parameters": {"type": "object", "properties": {}}}}

print("\n== تجربة ==")
for model in ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]:
    for with_tools in (False, True):
        label = "with tools" if with_tools else "plain"
        try:
            kw = {"tools": [tool]} if with_tools else {}
            c.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}],
                                      max_tokens=20, **kw)
            print("OK  ", model, label)
        except Exception as e:
            print("FAIL", model, label, "->", show(e))