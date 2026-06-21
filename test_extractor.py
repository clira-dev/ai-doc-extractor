"""Tests for the AI document extractor.

Run with:  pytest -q

All tests use StubClient or small test doubles — no API key or network required.
"""

from __future__ import annotations

import pytest

from extractor import (
    MalformedThenFixedClient,
    STUB_SAMPLE_OUTPUT,
    StringNumberClient,
    StubClient,
    build_extraction_prompt,
    extract_document,
    normalize_extraction,
    parse_llm_json,
    validate_extraction,
)


SAMPLE_INVOICE_PATH = "sample_invoice.txt"


@pytest.fixture
def sample_text() -> str:
    with open(SAMPLE_INVOICE_PATH, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Field extraction (StubClient)
# --------------------------------------------------------------------------- #


def test_extract_sample_invoice_fields(sample_text):
    result = extract_document(sample_text, client=StubClient())

    assert result["vendor"] == "Acme Cloud Supplies, LLC"
    assert result["date"] == "2025-03-14"
    assert result["currency"] == "USD"
    assert result["subtotal"] == pytest.approx(365.0)
    assert result["tax"] == pytest.approx(31.03)
    assert result["total"] == pytest.approx(396.03)
    assert len(result["line_items"]) == 4
    assert result["line_items"][0]["description"] == "Managed Postgres (prod)"


def test_extract_includes_confidence_and_validation(sample_text):
    result = extract_document(sample_text, client=StubClient())

    assert "confidence" in result
    assert "validation" in result
    assert result["confidence"]["level"] in ("high", "medium", "low")
    assert result["validation"]["total_adds_up"] is True
    assert result["validation"]["subtotal_matches_lines"] is True


def test_build_extraction_prompt_includes_document(sample_text):
    prompt = build_extraction_prompt(sample_text)
    assert "INV-2025-0847" in prompt
    assert "JSON" in prompt


# --------------------------------------------------------------------------- #
# Total validation logic
# --------------------------------------------------------------------------- #


def test_validation_passes_when_totals_add_up():
    data = normalize_extraction(STUB_SAMPLE_OUTPUT.copy())
    validation = validate_extraction(data)

    assert validation["subtotal_matches_lines"] is True
    assert validation["total_adds_up"] is True
    assert validation["issues"] == []


def test_validation_flags_mismatched_total():
    bad = STUB_SAMPLE_OUTPUT.copy()
    bad["total"] = 999.99
    data = normalize_extraction(bad)
    validation = validate_extraction(data)

    assert validation["total_adds_up"] is False
    assert any("subtotal+tax" in issue for issue in validation["issues"])


def test_validation_flags_line_sum_mismatch():
    bad = STUB_SAMPLE_OUTPUT.copy()
    bad["subtotal"] = 100.0
    data = normalize_extraction(bad)
    validation = validate_extraction(data)

    assert validation["subtotal_matches_lines"] is False


# --------------------------------------------------------------------------- #
# Malformed JSON repair
# --------------------------------------------------------------------------- #


def test_parse_llm_json_heuristic_repair_trailing_comma():
    client = StubClient()
    raw = '{"vendor": "Acme", "total": 10.00,}'

    parsed, repaired = parse_llm_json(raw, client, allow_repair=False)

    assert repaired is True
    assert parsed["vendor"] == "Acme"
    assert parsed["total"] == pytest.approx(10.0)


def test_parse_llm_json_retries_via_client_on_hard_failure():
    fixed = {"vendor": "Test Co", "total": 10.0, "line_items": []}
    client = MalformedThenFixedClient(fixed)

    parsed, repaired = parse_llm_json('not json at all', client)

    assert repaired is True
    assert client.call_count == 1
    assert parsed["vendor"] == "Test Co"


def test_malformed_then_fixed_client_end_to_end():
    fixed = {
        "vendor": "Retry LLC",
        "date": "2025-06-01",
        "currency": "USD",
        "line_items": [],
        "subtotal": None,
        "tax": None,
        "total": 50.0,
    }
    client = MalformedThenFixedClient(fixed)
    result = extract_document("invoice from Retry LLC", client=client)

    assert result["vendor"] == "Retry LLC"
    assert client.call_count >= 2


# --------------------------------------------------------------------------- #
# Number coercion
# --------------------------------------------------------------------------- #


def test_number_coercion_from_strings():
    result = extract_document("any text", client=StringNumberClient())

    assert result["subtotal"] == pytest.approx(31.50)
    assert result["tax"] == pytest.approx(2.52)
    assert result["total"] == pytest.approx(34.02)
    assert result["line_items"][0]["qty"] == 3
    assert result["line_items"][0]["unit_price"] == pytest.approx(10.50)
    assert result["line_items"][0]["amount"] == pytest.approx(31.50)


def test_normalize_fills_missing_with_null():
    data = normalize_extraction({"vendor": "Test Vendor"})

    assert data["vendor"] == "Test Vendor"
    assert data["date"] is None
    assert data["currency"] is None
    assert data["line_items"] == []
    assert data["subtotal"] is None


# --------------------------------------------------------------------------- #
# Defensive behavior
# --------------------------------------------------------------------------- #


def test_empty_input_never_crashes():
    result = extract_document("", client=StubClient())

    assert result["vendor"] is None
    assert result["line_items"] == []
    assert "confidence" in result


def test_whitespace_only_input_never_crashes():
    result = extract_document("   \n\t  ", client=StubClient())

    assert result["total"] is None
    assert result["validation"]["fields_present"] == 0
