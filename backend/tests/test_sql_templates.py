"""
Template rendering tests. No database, no network, no API keys.

The templates are where result shape and row limits are decided, so these pin
the two ways they used to be subtly wrong: collapsing rows that share a label,
and a cell cap that cut wide tables short.
"""

import pytest

from provena.query import sql_templates
from provena.query.sql_templates import PatternError, cell_budget, render
from provena.query.sql_validator import enforce_limit, validate_sql

T = "00000000-0000-0000-0000-000000000001"

ALL = {
    "cell_lookup": {"column_key": "revenue", "row_filter": "enterprise"},
    "scalar_aggregate": {"agg": "SUM", "column_key": "revenue"},
    "row_filter": {"row_filter": "enterprise"},
    "column_series": {"column_key": "revenue"},
    "comparison": {"column_keys": ["2022", "2023"]},
    "list_table": {},
}


@pytest.mark.parametrize("pattern", sorted(ALL))
def test_every_pattern_renders_valid_scoped_sql(pattern):
    sql, params = render({"pattern": pattern, "table_id": T, **ALL[pattern]})
    ok, error = validate_sql(sql)
    assert ok, error
    assert params["table_id"] == T


def test_values_are_bound_not_interpolated():
    sql, params = render({"pattern": "cell_lookup", "table_id": T,
                          "column_key": "x'; DROP TABLE documents; --", "row_filter": None})
    assert "DROP" not in sql
    assert params["column_key"].startswith("x'")


def test_unknown_pattern_and_missing_table_are_refused():
    with pytest.raises(PatternError):
        render({"pattern": "nope", "table_id": T})
    with pytest.raises(PatternError):
        render({"pattern": "list_table"})
    with pytest.raises(PatternError):
        render({"pattern": "row_filter", "table_id": T, "row_filter": None})


def test_unsupported_aggregate_is_refused():
    with pytest.raises(PatternError):
        render({"pattern": "scalar_aggregate", "table_id": T, "agg": "DROP", "column_key": "x"})


def test_comparison_bounds_series_count():
    with pytest.raises(PatternError):
        render({"pattern": "comparison", "table_id": T, "column_keys": []})
    with pytest.raises(PatternError):
        render({"pattern": "comparison", "table_id": T, "column_keys": list("abcdef")})


def test_comparison_aliases_are_safe_and_unique():
    sql, _ = render({"pattern": "comparison", "table_id": T, "column_keys": ["a b", "a_b", "2023"]})
    assert '"a_b"' in sql and '"a_b_1"' in sql and '"y_2023"' in sql


# ---------------------------------------------------- repeated row labels --

@pytest.mark.parametrize("pattern", ["column_series", "comparison"])
def test_rows_sharing_a_label_are_not_collapsed(pattern):
    """
    Grouping on row_label used to max() two 'Other' rows into one number
    that appeared nowhere in the document.
    """
    sql, _ = render({"pattern": pattern, "table_id": T, **ALL[pattern]})
    assert "GROUP BY row_label\n" not in sql
    assert "PARTITION BY row_label" in sql
    if pattern == "column_series":
        assert "max(value_num)" not in sql
    else:
        assert "GROUP BY row_index, row_label" in sql


def test_row_filter_picks_rows_in_document_order():
    """A LIMIT over DISTINCT row_index with no ORDER BY is an arbitrary subset."""
    sql, _ = render({"pattern": "row_filter", "table_id": T, "row_filter": "x"})
    matching = sql.split("SELECT f.", 1)[0]
    assert "ORDER BY row_index" in matching


# -------------------------------------------------------------- cell caps --

def test_cell_patterns_get_a_budget_sized_to_the_table():
    """
    The executor cap counts result rows; for these patterns a result row is a
    cell. 1000 cells stopped a 12-column table at row 83.
    """
    assert cell_budget("list_table", 12) == sql_templates._MAX_ROWS * 12
    assert cell_budget("row_filter", 3) == sql_templates._MAX_ROWS * 3
    assert cell_budget("list_table", 0) == sql_templates._MAX_ROWS


@pytest.mark.parametrize("pattern", ["cell_lookup", "scalar_aggregate", "column_series", "comparison"])
def test_row_shaped_patterns_keep_the_default_cap(pattern):
    assert cell_budget(pattern, 12) is None


def test_budget_survives_enforce_limit():
    sql, _ = render({"pattern": "list_table", "table_id": T})
    limited = enforce_limit(sql, cell_budget("list_table", 12))
    assert f"LIMIT {sql_templates._MAX_ROWS * 12}" in limited.upper()
