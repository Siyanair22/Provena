"""Heuristic routing. Pure regex classification: no LLM is constructed."""

import pytest

from provena.query.router import QueryRouter


@pytest.fixture
def router():
    return object.__new__(QueryRouter)  # skips __init__, which builds an LLM client


@pytest.mark.parametrize("q", [
    "How many rows are in the revenue table?",
    "Total count of orders",
    "Sum the values in column 2023",
    "Show all rows and columns of the table",
])
def test_sql(router, q):
    assert router.classify_heuristic(q) == "SQL"


@pytest.mark.parametrize("q", [
    "Explain the methodology and describe the approach",
    "Why does the report describe the process this way?",
])
def test_rag(router, q):
    assert router.classify_heuristic(q) == "RAG"


def test_viz_word_breaks_a_one_signal_tie(router):
    assert router.classify_heuristic("Plot the average by segment") == "SQL"


def test_printed_figure_vetoes_the_viz_tie_break(router):
    assert router.classify_heuristic("Plot the average from figure 3") is None


@pytest.mark.parametrize("q", ["Tell me about revenue", "revenue in 2023", "Compare explain"])
def test_ambiguous_goes_to_the_model(router, q):
    assert router.classify_heuristic(q) is None
