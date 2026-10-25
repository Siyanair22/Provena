"""Page attribution and table stubs. No OCR, no database, no embeddings."""

import pytest

from provena.core import config
from provena.ingest import chunker
from provena.ingest.extractor import Page, MistralOCRExtractor, page_for_offset, page_spans
from provena.ingest.pipeline import excise_tables

PAGES = [Page(1, "# Intro\n\nFirst page."), Page(2, "Second page."), Page(3, "Third page.")]


def joined():
    return MistralOCRExtractor.pages_to_markdown(PAGES)


def test_offsets_map_to_pages():
    md = joined()
    spans = page_spans(md)
    assert page_for_offset(spans, md.index("First")) == 1
    assert page_for_offset(spans, md.index("Second")) == 2
    assert page_for_offset(spans, md.index("Third")) == 3


def test_offset_zero_is_the_first_page_not_the_last():
    """Offset 0 is inside the first marker; it used to fall through to page 3."""
    assert page_for_offset(page_spans(joined()), 0) == 1


def test_offset_inside_a_later_marker_belongs_to_that_page():
    md = joined()
    assert page_for_offset(page_spans(md), md.index("<!-- PAGE 2")) == 2


@pytest.fixture
def processor(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JINA_API_KEY", "test")
    return chunker.DocumentProcessor(chunk_size=40, overlap=None, output_dir=str(tmp_path))


def test_chunks_carry_page_numbers_and_table_ids(processor):
    table_id = "0f0e0d0c-0b0a-0908-0706-050403020100"
    md = joined().replace("Second page.", f"[TABLE 1 — Revenue | page 2 | table_id={table_id}]")
    chunks = processor.process_text(md, source_file="report.pdf", save_chunks=False, store=False)

    assert chunks and all(c["source_file"] == "report.pdf" for c in chunks)
    pages = {c["page_number"] for c in chunks}
    assert pages == {1, 2, 3}
    stub = [c for c in chunks if table_id in c["content"]]
    assert stub and stub[0]["table_ids"] == [table_id]
    assert all(c["table_ids"] == [] for c in chunks if table_id not in c["content"])


class FakeTable:
    table_index = 0
    page_start = 2
    page_end = 3
    headers = ["Segment", "Revenue"]
    rows = [["A", "1"], ["B", "2"], ["C", "3"]]
    raw_fragments = ["| Segment | Revenue |\n|---|---|\n| A | 1 |", "| B | 2 |\n| C | 3 |"]


def test_excise_replaces_every_fragment_with_one_stub():
    md = "Before.\n\n" + FakeTable.raw_fragments[0] + "\n\n<!-- PAGE 3 -->\n\n" + FakeTable.raw_fragments[1] + "\n\nAfter."
    out = excise_tables(md, [FakeTable], [{"table_index": 0, "table_id": "tid", "description": "Revenue"}])
    assert out.count("table_id=tid") == 1
    assert "| B | 2 |" not in out and "| A | 1 |" not in out
    assert "pages 2-3" in out and "3 rows" in out
    assert out.startswith("Before.") and out.rstrip().endswith("After.")
    assert "<!-- PAGE 3 -->" in out  # page attribution survives excision


def test_excise_leaves_unregistered_tables_alone():
    md = FakeTable.raw_fragments[0]
    assert excise_tables(md, [FakeTable], []) == md
