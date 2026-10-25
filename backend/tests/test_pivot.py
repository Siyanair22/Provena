"""Pivot tests. Pure functions: no database, no LLM."""

from decimal import Decimal

import pytest

from provena.query.artifacts import build_artifact
from provena.query.pivot import NUMERIC_OID, TEXT_OID, PivotRefused, is_long_format, pivot_long

COLS = ["row_index", "col_index", "column_name", "column_key", "row_label",
        "value_text", "value_num", "unit", "page_number"]


def cell(row, col, key, text, num=None, label=None):
    return [row, col, key.title(), key, label, text, num, None, 1]


def test_detects_long_format():
    assert is_long_format(COLS)
    assert not is_long_format(["label", "value"])


def test_values_are_preserved_exactly():
    rows = [
        cell(0, 0, "segment", "Enterprise", label="Enterprise"),
        cell(0, 1, "revenue", "1,204.5", Decimal("1204.5"), "Enterprise"),
        cell(1, 0, "segment", "SMB", label="SMB"),
        cell(1, 1, "revenue", "88", Decimal("88"), "SMB"),
    ]
    headers, wide, oids = pivot_long(COLS, rows)
    assert headers == ["Segment", "Revenue"]
    assert wide == [["Enterprise", Decimal("1204.5")], ["SMB", Decimal("88")]]
    assert oids == [TEXT_OID, NUMERIC_OID]


def test_rows_come_out_in_row_order_whatever_the_input_order():
    rows = [cell(1, 0, "a", "second"), cell(0, 0, "a", "first")]
    _, wide, _ = pivot_long(COLS, rows)
    assert wide == [["first"], ["second"]]


def test_duplicate_cell_is_refused_not_overwritten():
    """Two headers that normalise to one key used to keep only the last."""
    rows = [cell(0, 0, "revenue", "10", 10), cell(0, 1, "revenue", "20", 20)]
    with pytest.raises(PivotRefused):
        pivot_long(COLS, rows)


def test_mixed_column_is_text_not_numeric():
    """'12', 'n/a', '15' is not a numeric column carrying strings."""
    rows = [cell(0, 0, "x", "12", 12), cell(1, 0, "x", "n/a"), cell(2, 0, "x", "15", 15)]
    _, wide, oids = pivot_long(COLS, rows)
    assert oids == [TEXT_OID]
    assert [r[0] for r in wide] == ["12", "n/a", "15"]


def test_empty_cells_do_not_make_a_column_text():
    rows = [cell(0, 0, "x", "12", 12), cell(1, 0, "x", ""), cell(2, 0, "x", None)]
    _, _, oids = pivot_long(COLS, rows)
    assert oids == [NUMERIC_OID]


def _exec(rows):
    return {"success": True, "columns": COLS, "type_oids": [23, 21, 25, 25, 25, 25, 1700, 25, 21],
            "rows": rows, "row_count": len(rows),
            "sql_executed": "SELECT * FROM document_facts WHERE table_id = 't'"}


def test_artifact_reports_pivot_and_refusal():
    ok_rows = [cell(0, 0, "a", "x"), cell(1, 0, "a", "y")]
    art = build_artifact(_exec(ok_rows))[0]
    assert art["classifier"]["pivoted"] is True

    dup_rows = [cell(0, 0, "a", "x"), cell(0, 1, "a", "y")]
    art = build_artifact(_exec(dup_rows))[0]
    assert art["classifier"]["pivoted"] is False
    assert any("raw cells" in n for n in art["notes"])
