"""
The ingestion status path, with OCR, the database and embeddings faked.

A NameError after the "completed" update used to flip every successful
upload to "failed". This drives process_pdf to the end and checks the last
status written.
"""

from contextlib import contextmanager

import pytest

from provena.ingest import pipeline as pl
from provena.ingest.extractor import Page


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        last = self.log[-1][0]
        if last.startswith("SELECT id, status FROM documents"):
            return None
        if last.startswith("INSERT INTO documents"):
            return ("doc-1",)
        return None

    def close(self):
        pass


class FakeConn:
    def __init__(self, log):
        self.log = log

    def cursor(self):
        return FakeCursor(self.log)

    def commit(self):
        pass


@pytest.fixture
def run(monkeypatch, tmp_path):
    log = []

    @contextmanager
    def conn(readonly=False):
        yield FakeConn(log)

    class Extractor:
        def extract_pdf_pages(self, path):
            return [Page(1, "# Title\n\nSome text.")]

        pages_to_markdown = staticmethod(pl.MistralOCRExtractor.pages_to_markdown)

    class Tables:
        def extract_tables_from_pages(self, pages, name):
            return []

    class Processor:
        def __init__(self, **kwargs):
            pass

        def process_text(self, text, source_file, save_chunks, store, document_id):
            self.source_file = source_file
            return [{"chunk_id": 0}]

    monkeypatch.setattr(pl, "get_connection", conn)
    monkeypatch.setattr(pl, "ensure_schema", lambda: None)
    monkeypatch.setattr(pl, "MistralOCRExtractor", Extractor)
    monkeypatch.setattr(pl, "MarkdownTableExtractor", Tables)
    monkeypatch.setattr(pl, "DocumentProcessor", Processor)

    pdf = tmp_path / "abc123_report.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    def go(processor=Processor):
        monkeypatch.setattr(pl, "DocumentProcessor", processor)
        result = pl.IngestionPipeline(output_dir=str(tmp_path)).process_pdf(
            str(pdf), display_name="report.pdf"
        )
        statuses = [p[0] for sql, p in log if sql.startswith("UPDATE documents SET status")]
        return result, statuses, log

    return go


def test_successful_ingest_ends_completed(run):
    result, statuses, _ = run()
    assert result["status"] == "completed"
    assert statuses[-1] == "completed"
    assert "failed" not in statuses


def test_display_name_is_what_gets_registered(run):
    _, _, log = run()
    insert = next(p for sql, p in log if sql.startswith("INSERT INTO documents"))
    assert insert[0] == "report.pdf"


def test_a_real_failure_is_recorded_with_its_message(run):
    class Broken:
        def __init__(self, **kwargs):
            pass

        def process_text(self, *a, **k):
            raise RuntimeError("Jina API request failed: 503")

    result, statuses, log = run(Broken)
    assert result["status"] == "failed" and "503" in result["error"]
    assert statuses[-1] == "failed"
