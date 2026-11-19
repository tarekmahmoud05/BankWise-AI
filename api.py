"""FastAPI backend.   Run from version2/:   uvicorn chatbot.api:app --reload

POST /chat   {"session_id": "abc", "message": "..."}  ->  reply + sources + diagnostics
POST /reset  {"session_id": "abc"}
GET  /health
The agent (LLM client + embedding model + Chroma) is built ONCE at startup, not per request.
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .agent import BankWiseAgent


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(max_length=4000)


class ResetIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


class ChatOut(BaseModel):
    reply: str
    answer: str
    sources: list
    abstained: bool
    reason: str
    tools_used: list
    latency_ms: int


def create_app(agent: Optional[BankWiseAgent] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            from .cli import build_agent
            app.state.agent = build_agent()
        else:
            app.state.agent = agent
        yield

    app = FastAPI(title="BankWise AI", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # plain `def` (not async): the LLM client is blocking, FastAPI runs it in a worker thread
    @app.post("/chat", response_model=ChatOut)
    def chat(body: ChatIn):
        r = app.state.agent.chat(body.session_id, body.message)
        return ChatOut(reply=r.text, answer=r.answer, sources=r.sources, abstained=r.abstained,
                       reason=r.reason, tools_used=r.tools_used, latency_ms=r.latency_ms)

    @app.post("/reset")
    def reset(body: ResetIn):
        app.state.agent.reset(body.session_id)
        return {"ok": True}

    return app


app = create_app()
