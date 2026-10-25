"""
Orchestrator flows with every external service faked.

These pin the behaviour that used to fail quietly: a failed leg merged as if
it were content, a retrieval error reported as "no information", and a RAG
hit on a table stub answered from the stub's two sample rows.
"""

import pytest

from provena.memory import MemoryManager
from provena.providers.llm import AnswerError
from provena.query import router as router_mod
from provena.query.retrieval import RetrievalError
from provena.query.router import QueryOrchestrator

TABLE = "11111111-1111-1111-1111-111111111111"


class FakeLLM:
    def __init__(self, answer="prose", fail=False):
        self.answer, self.fail, self.chats = answer, fail, []

    def generate_answer(self, query, context, **_):
        if self.fail:
            raise AnswerError("The language model could not generate an answer.")
        return self.answer

    def chat(self, messages, **_):
        self.chats.append(messages)
        return "merged"

    def rewrite_followup(self, query, history):
        return query


class FakeRetriever:
    def __init__(self, results=None, error=False):
        self.results, self.error, self.calls = results or [], error, []

    def retrieve(self, query, top_k=None, document_ids=None):
        self.calls.append((query, document_ids))
        if self.error:
            raise RetrievalError("The document store could not be searched.")
        return [dict(r) for r in self.results], True

    @staticmethod
    def build_context(results):
        return "ctx"


class FakeGenerator:
    def __init__(self, sql="SELECT 1 FROM document_facts WHERE table_id = 'x'"):
        self.sql, self.calls, self.outcomes = sql, [], []

    def generate(self, query, document_ids=None, table_ids=None):
        self.calls.append(table_ids)
        if not self.sql:
            return {"sql": None, "explanation": "Could not interpret the question."}
        return {"sql": self.sql, "params": {}, "pattern": "column_series",
                "tables_used": table_ids or [TABLE], "max_rows": None}

    def record_execution(self, pattern, success, repaired=False):
        self.outcomes.append(success)


class FakeExecutor:
    def __init__(self, success=True):
        self.success = success

    def execute(self, sql, params=None, max_rows=None):
        if not self.success:
            return {"success": False, "error": "column \"x\" does not exist", "row_count": 0}
        return {"success": True, "columns": ["label", "value"], "type_oids": [25, 1700],
                "rows": [["a", 1], ["b", 2]], "row_count": 2, "sql_executed": sql}

    def format_results(self, results, query, slots=None):
        return "computed prose"


def chunk(i, table_ids=()):
    return {"id": f"c{i}", "content": f"text {i}", "score": 0.8, "score_kind": "rerank",
            "source_file": "r.pdf", "page_number": i, "metadata": {"table_ids": list(table_ids)}}


@pytest.fixture
def make(monkeypatch):
    def build(*, llm=None, retriever=None, generator=None, executor=None):
        o = object.__new__(QueryOrchestrator)
        o.llm = llm or FakeLLM()
        o.retriever = retriever or FakeRetriever([chunk(1)])
        o.sql_generator = generator or FakeGenerator()
        o.sql_executor = executor or FakeExecutor()
        o.memory = MemoryManager()
        o.router = None
        return o
    # Decomposition is exercised in its own tests; keep one question here.
    monkeypatch.setattr(router_mod, "decompose", lambda llm, q: [q])
    return build


def test_every_response_has_artifacts_and_segments(make):
    for route in ("SQL", "RAG", "HYBRID"):
        result = make().process_query("q", force_route=route)
        assert isinstance(result["artifacts"], list)
        assert isinstance(result["segments"], list)


def test_sql_answer_is_marked_computed(make):
    result = make().process_query("q", force_route="SQL")
    assert result["segments"] == [{"kind": "computed", "text": "computed prose"}]
    assert result["artifacts"][0]["type"] == "bar"


def test_rag_sources_are_locatable(make):
    result = make().process_query("q", force_route="RAG")
    source = result["sources"][0]
    assert source["page_number"] == 1 and source["source_file"] == "r.pdf"
    assert result["segments"][0]["kind"] == "retrieved"
    assert "_results" not in result


def test_retrieval_failure_is_an_error_not_no_information(make):
    result = make(retriever=FakeRetriever(error=True)).process_query("q", force_route="RAG")
    assert result["error"] == "The document store could not be searched."
    assert "couldn't find" not in result["answer"]


def test_answer_failure_is_an_error_not_an_answer(make):
    result = make(llm=FakeLLM(fail=True)).process_query("q", force_route="RAG")
    assert result["error"] and not result["answer"].startswith("Error generating answer")


def test_failed_answers_are_not_remembered(make):
    o = make(retriever=FakeRetriever(error=True))
    o.process_query("q", session_id="s", force_route="RAG")
    assert not o.memory.session_exists("s")


def test_hybrid_does_not_merge_a_failed_sql_leg(make):
    llm = FakeLLM()
    result = make(llm=llm, executor=FakeExecutor(success=False)).process_query("q", force_route="HYBRID")
    assert llm.chats == []  # no merge call
    assert result["sql_error"].startswith("The query could not be run")
    assert result["answer"] == "prose"  # the RAG leg alone
    assert result["segments"][0]["kind"] == "retrieved"
    assert "error" not in result


def test_hybrid_reports_a_failed_rag_leg(make):
    result = make(retriever=FakeRetriever(error=True)).process_query("q", force_route="HYBRID")
    assert result["rag_error"] and result["answer"] == "computed prose"
    assert result["artifacts"][0]["provenance"]["answer_is_llm_merged"] is False


def test_hybrid_with_both_legs_is_marked_unverified(make):
    result = make().process_query("q", force_route="HYBRID")
    assert result["answer"] == "merged"
    assert result["segments"][0]["kind"] == "unverified"
    assert result["artifacts"][0]["provenance"]["answer_is_llm_merged"] is True


def test_both_legs_failing_is_an_error(make):
    result = make(retriever=FakeRetriever(error=True), executor=FakeExecutor(success=False)).process_query(
        "q", force_route="HYBRID")
    assert result["error"]


def test_rag_hit_on_a_table_stub_is_upgraded_to_hybrid(make):
    generator = FakeGenerator()
    o = make(retriever=FakeRetriever([chunk(1, [TABLE]), chunk(2)]), generator=generator)
    result = o.process_query("q", force_route="RAG")
    assert result["route"] == "HYBRID" and result["bridged_from"] == "RAG"
    assert generator.calls == [[TABLE]]  # SQL scoped to the stub's table
    assert result["artifacts"]


def test_bridge_falls_back_to_rag_when_the_table_query_fails(make):
    o = make(retriever=FakeRetriever([chunk(1, [TABLE])]), executor=FakeExecutor(success=False))
    result = o.process_query("q", force_route="RAG")
    assert result["route"] == "RAG" and "bridged_from" not in result


def test_stub_outside_the_top_passages_does_not_bridge(make):
    results = [chunk(i) for i in range(router_mod.BRIDGE_TOP_N)] + [chunk(9, [TABLE])]
    result = make(retriever=FakeRetriever(results)).process_query("q", force_route="RAG")
    assert result["route"] == "RAG"


def test_sql_execution_outcome_is_recorded(make):
    generator = FakeGenerator()
    make(generator=generator, executor=FakeExecutor(success=False)).process_query("q", force_route="SQL")
    assert generator.outcomes == [False]


def test_summary_failure_keeps_the_artifact(make):
    class NoProse(FakeExecutor):
        def format_results(self, *a, **k):
            raise AnswerError("rate limited")
    result = make(executor=NoProse()).process_query("q", force_route="SQL")
    assert result["artifacts"] and "error" not in result
    assert result["prose_error"] == "rate limited"
