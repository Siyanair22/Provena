"""
Provena REST API

Flask REST API for the Provena RAG + SQL system.
Provides endpoints for document upload, querying, and session management.
"""

import shutil
import logging
import threading
import time
import uuid
from functools import cached_property
from typing import Any, Optional

from flask import Flask, g, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from psycopg2 import sql as psql

from provena.core import config
from provena.core.db import get_connection
from provena.core.schema import TRUNCATE_TABLES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = config.UPLOAD_DIR
ALLOWED_EXTENSIONS = {"pdf"}
ROUTES = ("SQL", "RAG", "HYBRID")
MAX_QUERY_CHARS = 2000
MAX_DOCUMENT_IDS = 50

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

# Cheap and credential-free, so it stays at import; config.require() runs in
# main() and the services fail with their own messages if a key is missing.
config.ensure_dirs()


class _Services:
    """
    The app's long-lived components, built on first use.

    They used to be constructed at import, which meant importing this module
    needed every credential and a live database. Lazy construction keeps the
    server's behaviour identical and lets tests swap in fakes by assignment
    (cached_property stores on the instance, so `services.orchestrator = fake`
    works).
    """

    @cached_property
    def memory(self):
        from provena.memory import MemoryManager
        return MemoryManager()

    @cached_property
    def orchestrator(self):
        from provena.query.router import QueryOrchestrator
        return QueryOrchestrator(memory=self.memory)

    @cached_property
    def pipeline(self):
        # Builds the schema once per process (IngestionPipeline.__init__).
        from provena.ingest.pipeline import IngestionPipeline
        return IngestionPipeline()


services = _Services()


# =================== Reset coordination =================== #

# One condition guards three pieces of state. A reset waits for in-flight
# requests to drain before truncating: an upload or query that was already
# running used to race the TRUNCATE and write into freshly wiped tables.
_state = threading.Condition()
_resetting = False
_inflight = 0
_last_activity_time = time.time()
_timer_started = False
INACTIVITY_TIMEOUT_SECONDS = config.INACTIVITY_TIMEOUT_SECONDS
RESET_DRAIN_TIMEOUT_SECONDS = 300

# Endpoints that neither count as user activity nor hold off a reset. /health
# is the important one: any uptime poller used to keep the wipe from ever
# firing.
_PASSIVE_ENDPOINTS = {"health", "reset_database"}


# =================== Helper Functions =================== #

def error_response(message: str, status: int, code: str, **extra: Any):
    """
    A client-safe error body.

    Handlers used to return str(e), which put psycopg2 messages, hostnames and
    SQL fragments in front of the browser. The detail goes to the log instead.
    """
    return jsonify({"error": message, "code": code, **extra}), status


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
    """Parse and clamp an integer query parameter. Raises ValueError if not an int."""
    raw = request.args.get(name)
    value = default if raw is None else int(raw)
    return max(lo, min(hi, value))


def get_document_by_id(doc_id: str) -> Optional[dict]:
    """Fetch document from database by UUID."""
    with get_connection(readonly=True) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, filename, file_hash, status, page_count,
                   table_count, chunk_count, error_message,
                   uploaded_at, completed_at
            FROM documents
            WHERE id = %s::uuid
            """,
            (doc_id,)
        )
        row = cur.fetchone()
        cur.close()

    if not row:
        return None

    return {
        "id": str(row[0]),
        "filename": row[1],
        "file_hash": row[2],
        "status": row[3],
        "page_count": row[4],
        "table_count": row[5],
        "chunk_count": row[6],
        "error_message": row[7],
        "uploaded_at": row[8].isoformat() if row[8] else None,
        "completed_at": row[9].isoformat() if row[9] else None,
    }


# =================== Middleware =================== #

@app.before_request
def track_activity():
    """Track API activity, count in-flight requests, block during reset."""
    global _last_activity_time, _inflight

    if not app.testing:
        start_inactivity_timer()

    if request.endpoint in _PASSIVE_ENDPOINTS or request.method == "OPTIONS":
        return None

    with _state:
        if _resetting:
            return error_response("System is resetting. Please try again.", 503, "resetting")
        _last_activity_time = time.time()
        _inflight += 1
        g.counted = True
    return None


@app.teardown_request
def release_request(_exc):
    global _inflight
    if g.pop("counted", False):
        with _state:
            _inflight -= 1
            _state.notify_all()


# =================== Reset Function =================== #

def perform_reset() -> dict:
    """
    Core database reset. Clears all data and returns system to initial state.

    Steps:
    1. Wait for in-flight requests to finish (new ones get 503)
    2. Truncate documents, document_chunks, table_registry, document_facts
    3. Clear filesystem artifacts (uploads, output) - only if step 2 succeeded
    4. Clear in-memory session state

    Returns a summary with "status": "success", "failed" or "already_resetting".
    """
    global _resetting

    with _state:
        if _resetting:
            return {"status": "already_resetting"}
        _resetting = True
        drained = _state.wait_for(lambda: _inflight == 0, timeout=RESET_DRAIN_TIMEOUT_SECONDS)

    try:
        summary = {
            "status": "success",
            "tables_dropped": [],
            "tables_truncated": [],
            "files_removed": 0,
            "sessions_cleared": 0,
            "errors": [],
        }

        if not drained:
            summary["status"] = "failed"
            summary["errors"].append("Timed out waiting for in-flight requests")
            return summary

        # One TRUNCATE across a fixed set of tables.
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"TRUNCATE TABLE {', '.join(TRUNCATE_TABLES)} RESTART IDENTITY CASCADE"
                )
                conn.commit()
                cur.close()
            summary["tables_truncated"] = list(TRUNCATE_TABLES)
            logger.info(f"Truncated: {', '.join(TRUNCATE_TABLES)}")
        except Exception:
            # Stop here. Deleting files and sessions after a failed TRUNCATE
            # used to leave rows pointing at uploads that no longer existed,
            # and the endpoint still reported success.
            logger.exception("Database truncation failed")
            summary["status"] = "failed"
            summary["errors"].append("TRUNCATE failed")
            return summary

        # Sweep any relation left behind by the pre-fact-store design.
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    r"""SELECT tablename FROM pg_tables
                        WHERE schemaname='public' AND tablename LIKE 'tbl\_%\_extracted'"""
                )
                for (name,) in cur.fetchall():
                    cur.execute(
                        psql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(psql.Identifier(name))
                    )
                    summary["tables_dropped"].append(name)
                conn.commit()
                cur.close()
        except Exception as e:
            logger.warning(f"Legacy table sweep failed: {e}")

        # Clear filesystem artifacts
        files_removed = 0
        for dir_path in (UPLOAD_FOLDER, config.MARKDOWN_DIR, config.CHUNKS_DIR):
            if dir_path.exists():
                for item in dir_path.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                            files_removed += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            files_removed += 1
                    except Exception as e:
                        logger.error(f"Failed to remove {item}: {e}")
                        summary["errors"].append(f"Could not remove {item.name}")

        summary["files_removed"] = files_removed
        summary["sessions_cleared"] = services.memory.clear_all()

        logger.info(f"Reset complete: {summary}")
        return summary

    finally:
        with _state:
            _resetting = False
            _state.notify_all()


# =================== API Endpoints =================== #

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Does not count as activity."""
    try:
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()

        return jsonify({
            "status": "healthy",
            "service": "provena-api",
            "database": "connected"
        }), 200
    except Exception:
        logger.exception("Health check failed")
        return jsonify({"status": "unhealthy", "database": "unreachable"}), 503


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    """
    Upload a PDF document for processing.

    Expected: multipart/form-data with 'file' field
    Optional: 'force' parameter to reprocess existing files

    201 processed, 200 already processed (skipped), 422 processing failed.
    """
    if "file" not in request.files:
        return error_response("No file provided", 400, "no_file")

    file = request.files["file"]

    if file.filename == "":
        return error_response("Empty filename", 400, "empty_filename")

    if not allowed_file(file.filename):
        return error_response("Only PDF files are allowed", 400, "not_pdf")

    filename = secure_filename(file.filename) or "upload.pdf"
    # A unique prefix on disk: two different PDFs both called report.pdf used
    # to overwrite each other. The user-facing name stays unprefixed.
    filepath = UPLOAD_FOLDER / f"{uuid.uuid4().hex[:12]}_{filename}"

    try:
        file.save(str(filepath))
        with open(filepath, "rb") as fh:
            if fh.read(5) != b"%PDF-":
                filepath.unlink(missing_ok=True)
                return error_response("File is not a PDF", 400, "not_pdf")

        logger.info(f"Saved uploaded file: {filepath}")

        force = request.form.get("force", "false").lower() == "true"
        result = services.pipeline.process_pdf(
            str(filepath), force=force, display_name=filename
        )
    except Exception:
        logger.exception("Upload failed")
        filepath.unlink(missing_ok=True)
        return error_response("Upload failed", 500, "upload_failed")

    status = result.get("status")
    skipped = result.get("skipped", False)
    if status == "failed" or skipped:
        # Nothing on disk is referenced by a completed document.
        filepath.unlink(missing_ok=True)

    body = {
        "message": (
            "Document already processed" if skipped
            else "Document processing failed" if status == "failed"
            else "Document uploaded and processed"
        ),
        "document_id": result.get("doc_id"),
        "filename": filename,
        "status": status,
        "skipped": skipped,
        "table_count": len(result.get("tables_created", [])),
        "fact_count": result.get("fact_count", 0),
        "chunk_count": result.get("chunk_count", 0),
    }

    # A failed ingestion used to return 201 with the error dropped, so the
    # client reported success for a document that could never be queried.
    # The pipeline's message is the one stored in documents.error_message.
    if status == "failed":
        body["error"] = result.get("error") or "Processing failed"
        body["code"] = "ingest_failed"
        return jsonify(body), 422

    return jsonify(body), 200 if skipped else 201


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """
    List all documents with their processing status.

    Query params:
    - status: Filter by status (pending, extracting, parsing_tables, chunking, completed, failed)
    - limit: Max results (default 50, at most 200)
    - offset: Pagination offset (default 0)
    """
    try:
        limit = _int_arg("limit", 50, 1, 200)
        offset = _int_arg("offset", 0, 0, 10**9)
    except ValueError:
        return error_response("limit and offset must be integers", 400, "bad_pagination")

    status_filter = request.args.get("status")

    try:
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, filename, file_hash, status, page_count,
                       table_count, chunk_count, error_message,
                       uploaded_at, completed_at
                FROM documents
                WHERE (%(status)s::text IS NULL OR status = %(status)s)
                ORDER BY uploaded_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                {"status": status_filter, "limit": limit, "offset": offset},
            )
            rows = cur.fetchall()

            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE (%(status)s::text IS NULL OR status = %(status)s)",
                {"status": status_filter},
            )
            total = cur.fetchone()[0]
            cur.close()
    except Exception:
        logger.exception("List documents failed")
        return error_response("Could not list documents", 500, "db_error")

    documents = [
        {
            "id": str(row[0]),
            "filename": row[1],
            "file_hash": row[2],
            "status": row[3],
            "page_count": row[4],
            "table_count": row[5],
            "chunk_count": row[6],
            "error_message": row[7],
            "uploaded_at": row[8].isoformat() if row[8] else None,
            "completed_at": row[9].isoformat() if row[9] else None,
        }
        for row in rows
    ]

    return jsonify({
        "documents": documents,
        "total": total,
        "limit": limit,
        "offset": offset
    }), 200


@app.route("/api/documents/<doc_id>", methods=["GET"])
def get_document(doc_id: str):
    """Get detailed information about a specific document."""
    if not _is_uuid(doc_id):
        return error_response("Document not found", 404, "not_found")

    try:
        document = get_document_by_id(doc_id)
    except Exception:
        logger.exception(f"Error fetching document {doc_id}")
        return error_response("Could not fetch document", 500, "db_error")

    if not document:
        return error_response("Document not found", 404, "not_found")

    return jsonify(document), 200


def _validate_query_body(data: Any):
    """Return (params, None) or (None, error_response)."""
    if not isinstance(data, dict):
        return None, error_response("Request body must be a JSON object", 400, "bad_body")

    user_query = data.get("query")
    if not isinstance(user_query, str) or not user_query.strip():
        return None, error_response("'query' must be a non-empty string", 400, "bad_query")
    if len(user_query) > MAX_QUERY_CHARS:
        return None, error_response(
            f"'query' must be at most {MAX_QUERY_CHARS} characters", 400, "query_too_long"
        )

    session_id = data.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not 0 < len(session_id) <= 128):
        return None, error_response("'session_id' must be a string of 1-128 characters", 400, "bad_session")

    document_ids = data.get("document_ids")
    if document_ids is not None:
        if (
            not isinstance(document_ids, list)
            or len(document_ids) > MAX_DOCUMENT_IDS
            or not all(_is_uuid(d) for d in document_ids)
        ):
            return None, error_response(
                f"'document_ids' must be a list of at most {MAX_DOCUMENT_IDS} UUIDs",
                400, "bad_document_ids",
            )
        document_ids = document_ids or None  # [] means "no filter"

    force_route = data.get("force_route")
    if force_route is not None and force_route not in ROUTES:
        return None, error_response("Invalid force_route. Must be SQL, RAG, or HYBRID", 400, "bad_route")

    return {
        "query": user_query.strip(),
        "session_id": session_id,
        "document_ids": document_ids,
        "force_route": force_route,
    }, None


@app.route("/api/query", methods=["POST"])
def query():
    """
    Main query endpoint - processes natural language queries.

    Request body:
    {
        "query": "string (required)",
        "session_id": "string (optional)",
        "document_ids": ["uuid1", "uuid2"] (optional),
        "force_route": "SQL|RAG|HYBRID" (optional, for testing)
    }

    Response (every route):
    {
        "answer": "string",
        "route": "SQL|RAG|HYBRID",
        "original_query": "string",
        "rewritten_query": "string (if rewritten)",
        "artifacts": [...]            chart/table/scalar built from SQL rows
        "segments": [...]             answer provenance: computed|retrieved|unverified
        "error": "string"             set when the chosen route failed
    }
    SQL / HYBRID add: sql, sql_executed, sql_explanation, sql_pattern,
        tables_used, row_count, sql_error (HYBRID)
    RAG / HYBRID add: sources [{score, content, source_file, page_number,
        chunk_id, section, start_offset, end_offset, heading_context}],
        sub_questions [{question, verdict, source_ids}], rag_error (HYBRID)
    """
    params, err = _validate_query_body(request.get_json(silent=True))
    if err:
        return err

    logger.info(f"Query: {params['query']} (session={params['session_id']})")

    try:
        result = services.orchestrator.process_query(**params)
    except Exception:
        logger.exception("Query failed")
        return error_response("The query could not be processed", 500, "query_failed")

    return jsonify(result), 200


@app.route("/api/sessions/<session_id>/clear", methods=["POST"])
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    existed = services.memory.session_exists(session_id)
    services.memory.clear(session_id)
    return jsonify({"session_id": session_id, "cleared": existed}), 200


@app.route("/api/reset", methods=["POST"])
def reset_database():
    """Reset the entire database to its initial state."""
    summary = perform_reset()

    if summary.get("status") == "already_resetting":
        return jsonify({
            "message": "Reset already in progress",
            "status": "in_progress"
        }), 409

    if summary.get("status") == "failed":
        return jsonify({
            "message": "Reset failed",
            "status": "failed",
            "code": "reset_failed",
            "summary": summary,
        }), 500

    return jsonify({
        "message": "Database reset complete",
        "status": "success",
        "summary": summary
    }), 200


@app.route("/api/tables", methods=["GET"])
def list_tables():
    """
    List all tables in the registry (for debugging/transparency).

    Query params:
    - document_ids: Comma-separated UUIDs to filter by
    """
    document_ids_str = request.args.get("document_ids")
    document_ids = [d for d in document_ids_str.split(",") if d] if document_ids_str else None
    if document_ids is not None and not all(_is_uuid(d) for d in document_ids):
        return error_response("document_ids must be comma-separated UUIDs", 400, "bad_document_ids")

    # physical_table_name no longer names a relation - table rows live in
    # document_facts. It is computed here rather than stored, so the column
    # cannot drift into lying about reality, while the frontend keeps the
    # same string shape and the same React key.
    sql_text = """
        SELECT table_id,
               'tbl_' || left(document_id::text, 8) || '_t' || table_index
                   AS physical_table_name,
               document_id, semantic_description, headers, column_keys,
               row_count, fact_count, original_filename,
               page_start, page_end, created_at
        FROM table_registry
        WHERE (%(doc_ids)s::text[] IS NULL
               OR document_id::text = ANY(%(doc_ids)s::text[]))
        ORDER BY created_at DESC
    """

    try:
        with get_connection(readonly=True) as conn:
            cur = conn.cursor()
            cur.execute(sql_text, {"doc_ids": document_ids})
            rows = cur.fetchall()
            cur.close()
    except Exception:
        logger.exception("List tables failed")
        return error_response("Could not list tables", 500, "db_error")

    tables = [
        {
            "id": str(row[0]),
            "table_id": str(row[0]),
            "physical_table_name": row[1],
            "source_doc_uuid": str(row[2]),
            "semantic_description": row[3],
            # headers and row_count are hard invariants: tables-view calls
            # .map() and .toLocaleString() on them unguarded.
            "headers": row[4] or [],
            "column_keys": row[5] or [],
            "row_count": row[6] or 0,
            "fact_count": row[7] or 0,
            "original_filename": row[8],
            "page_start": row[9],
            "page_end": row[10],
            "created_at": row[11].isoformat() if row[11] else None,
        }
        for row in rows
    ]

    return jsonify({"tables": tables, "total": len(tables)}), 200


@app.route("/api/stats", methods=["GET"])
def stats():
    """
    SQL generation counters since process start.

    Exists so the SQL_STRATEGY choice (slots vs free-form) can be settled with
    data: how often each path produced SQL that validated and executed.
    """
    return jsonify({"sql": services.orchestrator.sql_generator.stats_snapshot()}), 200


# =================== Error Handlers =================== #

@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
    return error_response(f"File exceeds the {limit_mb} MB limit", 413, "too_large")


@app.errorhandler(404)
def not_found(_error):
    return error_response("Endpoint not found", 404, "not_found")


@app.errorhandler(405)
def method_not_allowed(_error):
    return error_response("Method not allowed", 405, "method_not_allowed")


@app.errorhandler(500)
def internal_error(_error):
    return error_response("Internal server error", 500, "internal")


# =================== Inactivity Timer =================== #

def _inactivity_checker():
    """Background thread that triggers reset after inactivity timeout."""
    global _last_activity_time
    while True:
        time.sleep(60)
        with _state:
            elapsed = time.time() - _last_activity_time
            busy = _inflight > 0
        if elapsed >= INACTIVITY_TIMEOUT_SECONDS and not busy:
            logger.info(f"Inactivity timeout reached ({elapsed:.0f}s). Triggering reset.")
            try:
                perform_reset()
            except Exception:
                logger.exception("Inactivity reset failed")
            with _state:
                _last_activity_time = time.time()


def start_inactivity_timer() -> None:
    """
    Start the wipe timer once per process.

    Called from the first request as well as main(): it used to start only in
    main(), so running under `flask run` or a WSGI server never wiped at all.
    """
    global _timer_started
    with _state:
        if _timer_started:
            return
        _timer_started = True
    threading.Thread(target=_inactivity_checker, daemon=True, name="inactivity-reset").start()
    logger.info(f"Inactivity timer started (timeout={INACTIVITY_TIMEOUT_SECONDS}s)")


# =================== Main =================== #


def main() -> None:
    """Entry point: `python -m provena.api.app`."""
    import os

    config.require()
    config.ensure_dirs()

    port = config.API_PORT
    debug = config.FLASK_DEBUG

    # The reloader's parent process never serves requests; only start the
    # timer and build services in the child that does.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_inactivity_timer()
        services.pipeline  # build the schema before the first upload

    logger.info(f"Starting Provena API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
