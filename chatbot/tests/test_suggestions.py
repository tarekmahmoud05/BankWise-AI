"""Suggested questions must follow the customer's language.

The FAQ chunks (and so the suggestions) are Arabic. If an English chat showed Arabic suggestions,
clicking one would send Arabic text and the whole chat would flip to Arabic. So for English chats the
agent translates the suggestions with one extra, tool-less LLM call.
"""
import json

from fastapi.testclient import TestClient

from chatbot.agent import BankWiseAgent, _parse_string_list
from chatbot.api import create_app
from chatbot.retriever import Retriever
from .conftest import ScriptedLLM, make_hit, say, tool_call

AR_Q = ["إيه المستندات المطلوبة لقرض السيارة؟", "ما هي مدة سداد قرض السيارة؟"]
EN_Q = ["What documents are required for a car loan?", "What is the repayment period of a car loan?"]


def hits():
    """S1 is the entry the answer uses; S2 and S3 become the suggestions."""
    return [make_hit("CAR-001", 0.90),
            make_hit("CAR-002", 0.80, text=f"[قروض السيارات]\nالسؤال: {AR_Q[0]}\nالإجابة: أ"),
            make_hit("CAR-003", 0.70, text=f"[قروض السيارات]\nالسؤال: {AR_Q[1]}\nالإجابة: ب")]


def agent_with(steps, hit_list=None):
    retr = Retriever(search_fn=lambda q, k, p: hit_list if hit_list is not None else hits(), min_score=0.5)
    llm = ScriptedLLM(steps)
    return BankWiseAgent(llm, retr), llm


def english_turn(translation):
    """Steps for one English chat turn: search, answer, then the translation call."""
    return [tool_call("search_faq", query="مستندات قرض السيارة"), say("You need an ID [S1]."), translation]


# ---------------------------------------------------------------- the fix
def test_english_chat_gets_english_suggestions():
    agent, llm = agent_with(english_turn(say(json.dumps(EN_Q))))
    r = agent.chat("s", "what documents do I need for a car loan?")
    assert r.suggestions == EN_Q
    assert r.answer == "You need an ID."                       # the answer itself is untouched


def test_translation_is_a_separate_toolless_call_with_only_the_questions_in_it():
    agent, llm = agent_with(english_turn(say(json.dumps(EN_Q))))
    agent.chat("s", "what documents do I need for a car loan?")
    call = llm.calls[2]
    assert call["tools"] == []
    assert "English" in call["system"] and "BankWise AI" not in call["system"]   # not the main prompt
    assert json.loads(call["messages"][0]["content"]) == AR_Q                  # only the Arabic questions go in
    assert len(call["messages"]) == 1                                          # no chat history leaks in


def test_translation_tokens_are_counted_and_history_stays_clean():
    agent, llm = agent_with(english_turn(say(json.dumps(EN_Q))))
    r = agent.chat("s", "what documents do I need for a car loan?")
    assert (r.input_tokens, r.output_tokens) == (30, 15)                       # 3 LLM calls x (10, 5)
    assert [m["role"] for m in agent._history["s"]] == ["user", "assistant"]
    assert agent._history["s"][1]["content"] == "You need an ID."              # no suggestions in the memory


def test_arabic_chat_keeps_arabic_suggestions_and_makes_no_extra_call():
    agent, llm = agent_with([tool_call("search_faq", query="x"), say("تحتاج بطاقة [S1].")])
    r = agent.chat("s", "إيه المستندات المطلوبة؟")
    assert r.suggestions == AR_Q and len(llm.calls) == 2


def test_clicked_english_suggestion_keeps_the_chat_in_english():
    agent, llm = agent_with(english_turn(say(json.dumps(EN_Q))) +
                            [tool_call("search_faq", query="مدة سداد قرض السيارة"), say("Up to 8 years [S1].")])
    first = agent.chat("s", "what documents do I need for a car loan?")
    agent.chat("s", first.suggestions[1])                                      # what the button sends
    assert "ENTIRE reply in English" in llm.calls[3]["system"]                 # lang rule of the 2nd turn


# ---------------------------------------------------------------- failures never break the reply
def test_translation_exception_hides_suggestions_but_keeps_the_answer():
    agent, _ = agent_with(english_turn(RuntimeError("quota exceeded")))
    r = agent.chat("s", "what documents do I need for a car loan?")
    assert r.suggestions == [] and r.answer == "You need an ID." and not r.abstained and r.reason == ""


def test_unusable_translations_are_dropped_not_shown_in_arabic():
    for bad in ("sorry, I can't", "[\"only one\"]", "{\"a\": 1}", "[1, 2]", json.dumps(AR_Q)):
        agent, _ = agent_with(english_turn(say(bad)))
        r = agent.chat("s", "what documents do I need for a car loan?")
        assert r.suggestions == [], bad
        assert r.answer == "You need an ID."


def test_json_wrapped_in_a_code_fence_or_a_sentence_is_accepted():
    fenced = "```json\n" + json.dumps(EN_Q) + "\n```"
    agent, _ = agent_with(english_turn(say(fenced)))
    assert agent.chat("s", "documents for a car loan?").suggestions == EN_Q
    chatty = "Here you go: " + json.dumps(EN_Q)
    agent, _ = agent_with(english_turn(say(chatty)))
    assert agent.chat("s", "documents for a car loan?").suggestions == EN_Q


def test_one_bad_item_does_not_remove_the_good_one():
    agent, _ = agent_with(english_turn(say(json.dumps([EN_Q[0], AR_Q[1]]))))      # 2nd item left in Arabic
    assert agent.chat("s", "documents for a car loan?").suggestions == [EN_Q[0]]


# ---------------------------------------------------------------- cache
def test_same_question_is_translated_only_once():
    agent, llm = agent_with(english_turn(say(json.dumps(EN_Q))) +
                            [tool_call("search_faq", query="x"), say("Again [S1].")])   # no 2nd translation step
    agent.chat("a", "documents for a car loan?")
    r = agent.chat("b", "documents for a car loan, again?")
    translation_calls = [c for c in llm.calls if c["tools"] == []]
    assert r.suggestions == EN_Q and len(translation_calls) == 1               # 2nd turn used the cache


def test_failed_translation_is_not_cached():
    agent, llm = agent_with(english_turn(RuntimeError("down")) +
                            english_turn(say(json.dumps(EN_Q))))
    assert agent.chat("a", "documents for a car loan?").suggestions == []
    assert agent.chat("b", "documents for a car loan?").suggestions == EN_Q    # tried again and worked


# ---------------------------------------------------------------- plumbing
def test_english_chunks_are_also_recognised_as_suggestions():
    en_hits = [make_hit("CAR-001", 0.9),
               make_hit("CAR-002", 0.8, text="[Car loans]\nQuestion: What is the minimum age?\nAnswer: 21.")]
    agent, llm = agent_with([tool_call("search_faq", query="x"), say("Ok [S1].")], hit_list=en_hits)
    r = agent.chat("s", "documents for a car loan?")
    assert r.suggestions == ["What is the minimum age?"] and len(llm.calls) == 2   # already English: no call


def test_parse_string_list():
    assert _parse_string_list('["a", " b "]', 2) == ["a", "b"]
    assert _parse_string_list("", 1) is None and _parse_string_list("[]", 1) is None
    assert _parse_string_list('["a"]', 2) is None and _parse_string_list("not json [", 1) is None


def test_api_returns_the_suggestions():
    llm = ScriptedLLM(english_turn(say(json.dumps(EN_Q))))
    agent = BankWiseAgent(llm, Retriever(search_fn=lambda q, k, p: hits(), min_score=0.5))
    with TestClient(create_app(agent)) as c:
        r = c.post("/chat", json={"session_id": "u1", "message": "documents for a car loan?"}).json()
    assert r["suggestions"] == EN_Q
