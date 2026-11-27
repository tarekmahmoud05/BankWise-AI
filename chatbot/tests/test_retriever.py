from chatbot.retriever import Retriever
from .conftest import make_hit


def test_hits_above_threshold_are_returned():
    r = Retriever(search_fn=lambda q, k, p: [make_hit(score=0.8), make_hit("CAR-008", 0.3)], min_score=0.5)
    res = r.retrieve("مستندات قرض السيارة")
    assert not res.abstained and [h.id for h in res.hits] == ["CAR-003"] and res.top_score == 0.8


def test_abstains_when_everything_is_below_threshold():
    r = Retriever(search_fn=lambda q, k, p: [make_hit(score=0.31)], min_score=0.5)
    res = r.retrieve("what is the weather")
    assert res.abstained and res.reason == "below_threshold" and res.hits == []


def test_empty_query_never_reaches_the_search_function():
    def boom(*a):
        raise AssertionError("must not be called")
    res = Retriever(search_fn=boom).retrieve("   ")
    assert res.abstained and res.reason == "empty_query"


def test_empty_index_result_abstains():
    assert Retriever(search_fn=lambda q, k, p: []).retrieve("سؤال").reason == "no_results"


def test_very_long_query_is_truncated():
    seen = {}
    Retriever(search_fn=lambda q, k, p: seen.setdefault("q", q) and [make_hit()]).retrieve("ا" * 5000)
    assert len(seen["q"]) == 1000
