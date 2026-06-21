"""Tests for the AI document extractor.

Run with:  python3 -m pytest -q

Everything runs offline (no API key, no network). Test-double LLM clients live
*in this file*, not in the production module. The tests exercise the REAL
deterministic logic — the rule-based parser, the JSON-repair ladder, number
coercion, totals reconciliation, and the offline eval harness over the gold set
— not a stub echoing its own constant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import evaluator
import pytest
from extractor import (
    LLMClient,
    ParseError,
    TransportError,
    build_extraction_prompt,
    compute_confidence,
    extract_document,
    normalize_extraction,
    parse_document_text,
    parse_llm_json,
    validate_extraction,
)

GOLD_DIR = Path(__file__).resolve().parent / "gold"


# --------------------------------------------------------------------------- #
# Test doubles (kept out of the production module)
# --------------------------------------------------------------------------- #


class EchoJSONClient(LLMClient):
    """Returns a fixed JSON payload regardless of prompt (for LLM-path tests)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps(self._payload)


class MalformedThenFixedClient(LLMClient):
    """First response is invalid JSON; the repair call returns valid JSON."""

    def __init__(self, fixed_payload: dict[str, Any]) -> None:
        self._fixed = fixed_payload
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        if "BROKEN TEXT" in prompt:  # the repair prompt
            return json.dumps(self._fixed)
        return "<<<not valid json>>>"


class StringNumberClient(LLMClient):
    """Returns numeric fields as strings (to exercise number coercion)."""

    def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "vendor": "String Num LLC",
                "date": "2025-01-01",
                "currency": "USD",
                "line_items": [
                    {
                        "description": "Widget",
                        "qty": "3",
                        "unit_price": "$10.50",
                        "amount": "31.50",
                    }
                ],
                "subtotal": "31.50",
                "tax": "2.52",
                "total": "34.02",
            }
        )


class AlwaysFailsClient(LLMClient):
    """Raises a transport error (to verify failures are not swallowed silently)."""

    def complete(self, prompt: str) -> str:
        raise TransportError("backend unreachable")


# --------------------------------------------------------------------------- #
# Deterministic rule-based parser (the real offline extraction path)
# --------------------------------------------------------------------------- #


def test_parse_document_text_extracts_real_fields():
    raw = (GOLD_DIR / "01_acme_invoice.txt").read_text(encoding="utf-8")
    parsed = parse_document_text(raw)

    assert parsed["vendor"] == "Acme Cloud Supplies, LLC"
    assert parsed["date"] == "2025-03-14"
    assert parsed["currency"] == "USD"
    assert parsed["subtotal"] == pytest.approx(365.0)
    assert parsed["tax"] == pytest.approx(31.03)
    assert parsed["total"] == pytest.approx(396.03)
    assert len(parsed["line_items"]) == 4
    assert parsed["line_items"][0]["description"] == "Managed Postgres (prod)"
    assert parsed["line_items"][0]["amount"] == pytest.approx(149.0)


def test_parse_handles_thousands_separators_and_iso_currency():
    raw = (GOLD_DIR / "06_thousands.txt").read_text(encoding="utf-8")
    result = extract_document(raw)  # full deterministic pipeline

    assert result["currency"] == "USD"
    assert result["subtotal"] == pytest.approx(12750.0)
    assert result["total"] == pytest.approx(13833.75)
    assert result["line_items"][0]["amount"] == pytest.approx(9600.0)


def test_parse_handles_eur_and_slash_dates():
    eur = extract_document((GOLD_DIR / "03_eu_vat.txt").read_text(encoding="utf-8"))
    assert eur["currency"] == "EUR"
    assert eur["date"] == "2025-01-31"

    gbp = extract_document((GOLD_DIR / "04_slash_date_gbp.txt").read_text(encoding="utf-8"))
    assert gbp["currency"] == "GBP"
    assert gbp["date"] == "2025-04-07"  # 4/7/2025 -> ISO


# --------------------------------------------------------------------------- #
# Offline eval harness — real metrics over the gold set
# --------------------------------------------------------------------------- #


def test_gold_set_has_enough_diverse_fixtures():
    fixtures = list(GOLD_DIR.glob("*.txt"))
    assert len(fixtures) >= 8
    for txt in fixtures:
        assert txt.with_suffix(".json").exists(), f"missing label for {txt.name}"


def test_eval_reports_high_field_accuracy():
    report = evaluator.evaluate_gold_set()
    assert report.docs_total >= 8
    # The deterministic parser is strong on this gold set.
    assert report.field_accuracy() >= 0.95, evaluator.format_report(report)


def test_eval_arithmetic_pass_rate_reflects_reconciliation():
    report = evaluator.evaluate_gold_set()
    # Exactly one gold doc (08) has a deliberately non-reconciling total,
    # so the pass-rate must be below 100% — proving the metric is real.
    assert report.arithmetic_checkable >= 8
    assert 0.0 < report.arithmetic_pass_rate < 1.0


def test_eval_overall_precision_and_recall_are_reported():
    report = evaluator.evaluate_gold_set()
    overall = report.overall
    assert 0.0 <= overall.precision <= 1.0
    assert 0.0 <= overall.recall <= 1.0
    assert overall.f1 > 0.9


# --------------------------------------------------------------------------- #
# Totals reconciliation (the real validation logic)
# --------------------------------------------------------------------------- #


def test_validation_passes_when_totals_add_up():
    gold = normalize_extraction(
        json.loads((GOLD_DIR / "01_acme_invoice.json").read_text(encoding="utf-8"))
    )
    validation = validate_extraction(gold)

    assert validation["subtotal_matches_lines"] is True
    assert validation["total_adds_up"] is True
    assert validation["issues"] == []


def test_validation_flags_mismatched_total():
    bad = normalize_extraction(
        json.loads((GOLD_DIR / "01_acme_invoice.json").read_text(encoding="utf-8"))
    )
    bad["total"] = 999.99
    validation = validate_extraction(bad)

    assert validation["total_adds_up"] is False
    assert any("subtotal+tax" in issue for issue in validation["issues"])


def test_validation_flags_line_sum_mismatch():
    bad = normalize_extraction(
        json.loads((GOLD_DIR / "01_acme_invoice.json").read_text(encoding="utf-8"))
    )
    bad["subtotal"] = 100.0
    validation = validate_extraction(bad)

    assert validation["subtotal_matches_lines"] is False


def test_no_tax_invoice_reconciles_subtotal_to_total():
    result = extract_document((GOLD_DIR / "05_no_tax.txt").read_text(encoding="utf-8"))
    assert result["tax"] is None
    assert result["total"] == pytest.approx(850.0)
    assert result["validation"]["subtotal_matches_lines"] is True


# --------------------------------------------------------------------------- #
# JSON-repair ladder (real logic, exercised via test doubles)
# --------------------------------------------------------------------------- #


def test_parse_llm_json_clean_payload():
    parsed, repaired = parse_llm_json('{"vendor": "Acme", "total": 10.0}')
    assert repaired is False
    assert parsed["vendor"] == "Acme"


def test_parse_llm_json_strips_markdown_fences():
    raw = '```json\n{"vendor": "Fenced Co", "total": 5.0}\n```'
    parsed, repaired = parse_llm_json(raw)
    assert parsed["vendor"] == "Fenced Co"


def test_parse_llm_json_heuristic_repair_trailing_comma():
    parsed, repaired = parse_llm_json('{"vendor": "Acme", "total": 10.00,}', allow_repair=False)
    assert repaired is True
    assert parsed["total"] == pytest.approx(10.0)


def test_parse_llm_json_extracts_object_from_prose():
    raw = 'Sure! Here is the JSON:\n{"vendor": "Prose Co", "total": 7.0}\nHope that helps.'
    parsed, repaired = parse_llm_json(raw, allow_repair=False)
    assert parsed["vendor"] == "Prose Co"


def test_parse_llm_json_retries_via_client_on_hard_failure():
    fixed = {"vendor": "Test Co", "total": 10.0, "line_items": []}
    client = MalformedThenFixedClient(fixed)

    parsed, repaired = parse_llm_json("not json at all", client)

    assert repaired is True
    assert client.call_count == 1  # one repair call
    assert parsed["vendor"] == "Test Co"


def test_parse_llm_json_raises_when_unrepairable():
    with pytest.raises(ParseError):
        parse_llm_json("not json at all", allow_repair=False)


# --------------------------------------------------------------------------- #
# Number coercion (real logic, via test double)
# --------------------------------------------------------------------------- #


def test_number_coercion_from_strings():
    result = extract_document("any text", client=StringNumberClient())

    assert result["subtotal"] == pytest.approx(31.50)
    assert result["tax"] == pytest.approx(2.52)
    assert result["total"] == pytest.approx(34.02)
    assert result["line_items"][0]["qty"] == 3
    assert result["line_items"][0]["unit_price"] == pytest.approx(10.50)


def test_normalize_fills_missing_with_null():
    data = normalize_extraction({"vendor": "Test Vendor"})
    assert data["vendor"] == "Test Vendor"
    assert data["date"] is None
    assert data["line_items"] == []
    assert data["subtotal"] is None


# --------------------------------------------------------------------------- #
# Error handling — failures are surfaced, not swallowed into null silently
# --------------------------------------------------------------------------- #


def test_transport_error_degrades_to_low_confidence_review(caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="doc_extractor"):
        result = extract_document("some invoice text", client=AlwaysFailsClient())

    # Degrades gracefully...
    assert result["vendor"] is None
    assert result["confidence"]["needs_review"] is True
    # ...but the failure is logged with its class, not silently dropped.
    assert any("transport error" in r.message.lower() for r in caplog.records)


def test_typed_errors_are_distinct():
    assert issubclass(TransportError, Exception)
    assert issubclass(ParseError, Exception)
    assert TransportError is not ParseError


# --------------------------------------------------------------------------- #
# Confidence + review routing
# --------------------------------------------------------------------------- #


def test_high_confidence_when_complete_and_reconciled():
    result = extract_document((GOLD_DIR / "01_acme_invoice.txt").read_text(encoding="utf-8"))
    assert result["confidence"]["level"] == "high"
    assert result["confidence"]["needs_review"] is False


def test_low_confidence_routes_to_review():
    # A document missing most fields should be flagged for human review.
    result = extract_document("just some prose with no invoice structure at all")
    assert result["confidence"]["needs_review"] is True


def test_compute_confidence_shape():
    data = normalize_extraction({"vendor": "X", "total": 10.0})
    validation = validate_extraction(data)
    conf = compute_confidence(data, validation)
    assert set(conf) >= {
        "overall",
        "fields_score",
        "validation_score",
        "level",
        "needs_review",
    }
    assert conf["level"] in ("high", "medium", "low")


# --------------------------------------------------------------------------- #
# Prompt + defensive behavior
# --------------------------------------------------------------------------- #


def test_build_extraction_prompt_includes_document():
    raw = (GOLD_DIR / "01_acme_invoice.txt").read_text(encoding="utf-8")
    prompt = build_extraction_prompt(raw)
    assert "INV-2025-0847" in prompt
    assert "JSON" in prompt


def test_empty_input_never_crashes():
    result = extract_document("")
    assert result["vendor"] is None
    assert result["line_items"] == []
    assert "confidence" in result


def test_whitespace_only_input_never_crashes():
    result = extract_document("   \n\t  ")
    assert result["total"] is None
    assert result["validation"]["fields_present"] == 0
