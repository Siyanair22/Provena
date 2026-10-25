"""Fusion, reranking fallback and decomposition. No database, no network."""

import pytest

from provena.core import config
from provena.query import decompose as dq
from provena.query.retrieval import Retriever, rrf_fuse


# -------------------------------------------------------------------- RRF --

def test_rrf_rewards_agreement_between_legs():
    fused = [d for d, _ in rrf_fuse([["a", "b", "c"], ["c", "b", "x"]], k=60)]
    # b is 2nd in both; c is 3rd and 1st; a is 1st in one list only.
    assert fused[:2] == ["b", "c"] or fused[:2] == ["c", "b"]
    assert set(fused) == {"a", "b", "c", "x"}
    assert fused.index("a") > fused.index("b")


def test_rrf_scores_are_rank_based():
    (top, score), *_ = rrf_fuse([["only"]], k=60)
    assert top == "only" and score == pytest.approx(1 / 61)


def test_rrf_ties_break_deterministically():
    once = rrf_fuse([["a", "b"], ["b", "a"]])
    again = rrf_fuse([["a", "b"], ["b", "a"]])
    assert once == again and [d for d, _ in once] == ["a", "b"]


def test_rrf_handles_an_empty_leg():
    """No query embedding means the vector leg is empty."""
    assert [d for d, _ in rrf_fuse([[], ["t1", "t2"]])] == ["t1", "t2"]


# ----------------------------------------------------------- reranking -----

class FakeReranker:
    def __init__(self, ranked):
        self.ranked = ranked

    def rerank(self, query, passages, top_n):
        return self.ranked


def _retriever(candidates, ranked):
    r = object.__new__(Retriever)
    r.search = lambda q, limit=None, document_ids=None: [dict(c, fusion={}) for c in candidates]
    r.reranker = FakeReranker(ranked)
    return r


CANDS = [{"id": f"c{i}", "content": f"text {i}", "score": 0.5, "score_kind": "cosine"} for i in range(4)]


def test_rerank_reorders_and_rescores():
    results, reranked = _retriever(CANDS, [(2, 0.9), (0, 0.4)]).retrieve("q", top_k=2)
    assert reranked
    assert [r["id"] for r in results] == ["c2", "c0"]
    assert results[0]["score"] == 0.9 and results[0]["score_kind"] == "rerank"


def test_reranker_failure_keeps_fused_order():
    results, reranked = _retriever(CANDS, None).retrieve("q", top_k=3)
    assert not reranked
    assert [r["id"] for r in results] == ["c0", "c1", "c2"]


def test_no_candidates_skips_the_reranker():
    assert _retriever([], [(0, 1.0)]).retrieve("q") == ([], False)


def test_context_carries_page_numbers():
    ctx = Retriever.build_context([{"content": "x", "page_number": 4, "heading_context": "Intro"}])
    assert "[Source 1] (page 4) Intro" in ctx


# -------------------------------------------------------- decomposition ----

@pytest.mark.parametrize("q", [
    "What was total revenue in 2023 and how does it compare with the target?",
    "Compare enterprise churn with SMB churn over the last two years",
    "What is the churn rate? What drove it last quarter?",
])
def test_compound_questions_pass_the_gate(q):
    assert dq.should_decompose(q)


@pytest.mark.parametrize("q", [
    "What is the churn rate?",
    "Show revenue and profit by segment",
    "Explain the methodology",
])
def test_simple_questions_skip_the_model_call(q):
    assert not dq.should_decompose(q)


def test_parse_accepts_a_valid_split_and_caps_it():
    raw = '{"questions": ["a one", "b two", "c three", "d four", "a one"]}'
    assert dq.parse_sub_questions(raw, "orig") == ["a one", "b two", "c three"]


@pytest.mark.parametrize("raw", ["not json", '{"questions": "x"}', '{"questions": []}', "[]", None])
def test_parse_falls_back_to_the_original(raw):
    assert dq.parse_sub_questions(raw, "orig") == ["orig"]


def test_decompose_never_calls_the_model_for_simple_questions():
    class Boom:
        def chat(self, *a, **k):
            raise AssertionError("should not be called")
    assert dq.decompose(Boom(), "What is churn?") == ["What is churn?"]


def test_decompose_survives_a_model_failure():
    class Fails:
        def chat(self, *a, **k):
            raise RuntimeError("rate limited")
    q = "What was revenue and how does it compare with the target?"
    assert dq.decompose(Fails(), q) == [q]


def test_verdicts():
    assert dq.verdict([], True) == "NOT_FOUND"
    assert dq.verdict([{"score": 0.9}], False) == "PARTIAL"
    assert dq.verdict([{"score": config.RERANK_ANSWERED_SCORE}], True) == "ANSWERED"
    assert dq.verdict([{"score": 0.01}], True) == "PARTIAL"
