from fastapi.testclient import TestClient

from chatbot.agent import BankWiseAgent
from chatbot.api import create_app
from chatbot.retriever import Retriever
from .conftest import ScriptedLLM, make_hit, say, tool_call


def test_chat_endpoint_round_trip():
    llm = ScriptedLLM([tool_call("search_faq", query="x"), say("جواب [S1]"), say("تمام")])
    agent = BankWiseAgent(llm, Retriever(search_fn=lambda q, k, p: [make_hit()], min_score=0.5))
    with TestClient(create_app(agent)) as c:
        assert c.get("/health").json() == {"status": "ok"}
        r = c.post("/chat", json={"session_id": "u1", "message": "سؤال"}).json()
        assert r["answer"] == "جواب" and r["sources"][0]["id"] == "CAR-003" and r["tools_used"] == ["search_faq"]
        assert c.post("/reset", json={"session_id": "u1"}).json() == {"ok": True}
        assert c.post("/chat", json={"session_id": "", "message": "x"}).status_code == 422   # validation
