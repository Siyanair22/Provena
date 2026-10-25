"""Flask endpoints with the services faked. No database, no API keys."""

import io
import threading
import uuid

import pytest

from provena.api import app as api


class FakeMemory:
    def __init__(self):
        self.sessions = {"s1"}

    def session_exists(self, sid):
        return sid in self.sessions

    def clear(self, sid):
        self.sessions.discard(sid)

    def clear_all(self):
        n = len(self.sessions)
        self.sessions.clear()
        return n


class FakeOrchestrator:
    def __init__(self):
        self.calls = []

    def process_query(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": "ok", "route": "RAG", "artifacts": [], "segments": []}


class FakePipeline:
    def __init__(self, result):
        self.result, self.calls = result, []

    def process_pdf(self, path, force=False, display_name=None):
        self.calls.append((path, display_name))
        return self.result


@pytest.fixture
def client(monkeypatch, tmp_path):
    api.app.testing = True
    monkeypatch.setattr(api, "UPLOAD_FOLDER", tmp_path)
    monkeypatch.setitem(api.services.__dict__, "memory", FakeMemory())
    monkeypatch.setitem(api.services.__dict__, "orchestrator", FakeOrchestrator())
    return api.app.test_client()


def pdf(name="report.pdf", body=b"%PDF-1.4 hello"):
    return {"file": (io.BytesIO(body), name)}


# ----------------------------------------------------------------- query ---

@pytest.mark.parametrize("body, code", [
    (None, "bad_body"),
    ({}, "bad_query"),
    ({"query": "   "}, "bad_query"),
    ({"query": 5}, "bad_query"),
    ({"query": "x" * 3000}, "query_too_long"),
    ({"query": "q", "document_ids": "abc"}, "bad_document_ids"),
    ({"query": "q", "document_ids": ["not-a-uuid"]}, "bad_document_ids"),
    ({"query": "q", "force_route": "SQLX"}, "bad_route"),
    ({"query": "q", "session_id": ""}, "bad_session"),
])
def test_query_validation(client, body, code):
    response = client.post("/api/query", json=body) if body is not None else client.post(
        "/api/query", data="nope", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["code"] == code


def test_valid_query_reaches_the_orchestrator(client):
    doc = str(uuid.uuid4())
    response = client.post("/api/query", json={"query": " q ", "document_ids": [doc], "session_id": "s"})
    assert response.status_code == 200
    call = api.services.orchestrator.calls[-1]
    assert call == {"query": "q", "session_id": "s", "document_ids": [doc], "force_route": None}


def test_empty_document_list_means_no_filter(client):
    client.post("/api/query", json={"query": "q", "document_ids": []})
    assert api.services.orchestrator.calls[-1]["document_ids"] is None


def test_orchestrator_crash_does_not_leak_the_exception(client, monkeypatch):
    def boom(**_):
        raise RuntimeError("password=hunter2 host=db.internal")
    monkeypatch.setattr(api.services.orchestrator, "process_query", boom)
    response = client.post("/api/query", json={"query": "q"})
    assert response.status_code == 500
    assert "hunter2" not in response.get_data(as_text=True)


# ---------------------------------------------------------------- upload ---

def test_upload_success_is_201_and_keeps_the_display_name(client, monkeypatch):
    fake = FakePipeline({"status": "completed", "doc_id": "d", "tables_created": ["t"], "chunk_count": 3})
    monkeypatch.setitem(api.services.__dict__, "pipeline", fake)
    response = client.post("/api/documents/upload", data=pdf(), content_type="multipart/form-data")
    assert response.status_code == 201
    body = response.get_json()
    assert body["document_id"] == "d" and body["filename"] == "report.pdf" and body["table_count"] == 1
    saved, display = fake.calls[0]
    assert display == "report.pdf" and saved.endswith("_report.pdf") and saved != "report.pdf"


def test_same_name_uploads_do_not_collide(client, monkeypatch):
    fake = FakePipeline({"status": "completed", "doc_id": "d"})
    monkeypatch.setitem(api.services.__dict__, "pipeline", fake)
    for _ in range(2):
        client.post("/api/documents/upload", data=pdf(), content_type="multipart/form-data")
    assert fake.calls[0][0] != fake.calls[1][0]


def test_failed_ingest_is_422_with_the_reason_and_no_file_left(client, monkeypatch, tmp_path):
    fake = FakePipeline({"status": "failed", "doc_id": "d", "error": "OCR returned no pages"})
    monkeypatch.setitem(api.services.__dict__, "pipeline", fake)
    response = client.post("/api/documents/upload", data=pdf(), content_type="multipart/form-data")
    assert response.status_code == 422
    assert response.get_json()["error"] == "OCR returned no pages"
    assert list(tmp_path.iterdir()) == []


def test_duplicate_upload_is_200_skipped(client, monkeypatch):
    monkeypatch.setitem(api.services.__dict__, "pipeline", FakePipeline({"status": "skipped", "skipped": True, "doc_id": None}))
    response = client.post("/api/documents/upload", data=pdf(), content_type="multipart/form-data")
    assert response.status_code == 200 and response.get_json()["skipped"] is True


def test_non_pdf_content_is_rejected(client, monkeypatch, tmp_path):
    monkeypatch.setitem(api.services.__dict__, "pipeline", FakePipeline({}))
    response = client.post("/api/documents/upload", data=pdf(body=b"MZ\x90 not a pdf"),
                           content_type="multipart/form-data")
    assert response.status_code == 400 and response.get_json()["code"] == "not_pdf"
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------ sessions & misc ----

def test_clear_session_reports_whether_it_existed(client):
    assert client.post("/api/sessions/s1/clear").get_json()["cleared"] is True
    assert client.post("/api/sessions/s1/clear").get_json()["cleared"] is False


def test_bad_pagination_is_400(client):
    assert client.get("/api/documents?limit=abc").status_code == 400


def test_bad_table_filter_is_400(client):
    assert client.get("/api/tables?document_ids=nope").status_code == 400


def test_unknown_document_id_is_404_without_a_query(client):
    assert client.get("/api/documents/not-a-uuid").status_code == 404


# ----------------------------------------------------------------- reset ---

def test_health_does_not_count_as_activity(client, monkeypatch):
    monkeypatch.setattr(api, "_last_activity_time", 0.0)
    monkeypatch.setattr(api, "get_connection", lambda readonly=False: (_ for _ in ()).throw(RuntimeError("down")))
    response = client.get("/health")
    assert response.status_code == 503
    assert api._last_activity_time == 0.0


def test_requests_are_refused_while_resetting(client, monkeypatch):
    monkeypatch.setattr(api, "_resetting", True)
    response = client.post("/api/query", json={"query": "q"})
    assert response.status_code == 503 and response.get_json()["code"] == "resetting"


def test_reset_waits_for_in_flight_requests(monkeypatch):
    """A running request used to race the TRUNCATE."""
    order = []
    monkeypatch.setattr(api, "_inflight", 1)
    monkeypatch.setattr(api.services, "memory", FakeMemory(), raising=False)

    class Conn:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def cursor(self):
            return self
        def execute(self, sql, *a):
            if sql.startswith("TRUNCATE"):
                order.append("truncate")
        def fetchall(self):
            return []
        def commit(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(api, "get_connection", lambda readonly=False: Conn())

    t = threading.Thread(target=api.perform_reset)
    t.start()
    t.join(timeout=0.3)
    assert order == []  # still waiting

    with api._state:
        order.append("request done")
        api._inflight = 0
        api._state.notify_all()
    t.join(timeout=5)
    assert order == ["request done", "truncate"]


def test_failed_truncate_leaves_files_and_reports_failure(monkeypatch, tmp_path):
    (tmp_path / "keep.pdf").write_bytes(b"x")
    monkeypatch.setattr(api, "UPLOAD_FOLDER", tmp_path)
    monkeypatch.setattr(api, "_inflight", 0)

    def down(readonly=False):
        raise RuntimeError("db down")
    monkeypatch.setattr(api, "get_connection", down)

    summary = api.perform_reset()
    assert summary["status"] == "failed"
    assert (tmp_path / "keep.pdf").exists()
