#!/usr/bin/env python3
"""AI document extractor.

Turns messy invoice, receipt, or order-email text into validated structured JSON
using an LLM, with defensive parsing and arithmetic checks.

Usage
-----
    export ANTHROPIC_API_KEY="sk-ant-..."   # optional; falls back to StubClient
    python extractor.py sample_invoice.txt

Dependencies: anthropic SDK + Python standard library.
"""

from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-opus-4-8"

EXTRACTION_SCHEMA: dict[str, Any] = {
    "vendor": None,
    "date": None,
    "currency": None,
    "line_items": [],
    "subtotal": None,
    "tax": None,
    "total": None,
}

LINE_ITEM_KEYS = ("description", "qty", "unit_price", "amount")

# Tolerance for floating-point total reconciliation (one cent).
TOTAL_TOLERANCE = 0.02


# --------------------------------------------------------------------------- #
# LLM client interface
# --------------------------------------------------------------------------- #


class LLMClient(ABC):
    """Pluggable LLM backend for structured extraction."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's raw text response to *prompt*."""


class AnthropicClient(LLMClient):
    """Production client using the official Anthropic SDK."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package is required for AnthropicClient; "
                "pip install anthropic"
            ) from exc

        key = api_key if api_key is not None else os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not key:
            raise ValueError(
                f"{ANTHROPIC_API_KEY_ENV} is not set; cannot call Anthropic API."
            )

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def complete(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)


# Canonical stub output for ``sample_invoice.txt`` — deterministic, offline-safe.
STUB_SAMPLE_OUTPUT: dict[str, Any] = {
    "vendor": "Acme Cloud Supplies, LLC",
    "date": "2025-03-14",
    "currency": "USD",
    "line_items": [
        {
            "description": "Managed Postgres (prod)",
            "qty": 1,
            "unit_price": 149.0,
            "amount": 149.0,
        },
        {
            "description": "Object storage — 500 GB tier",
            "qty": 1,
            "unit_price": 45.0,
            "amount": 45.0,
        },
        {
            "description": "Premium support (monthly)",
            "qty": 2,
            "unit_price": 25.5,
            "amount": 51.0,
        },
        {
            "description": "Setup fee (one-time)",
            "qty": 1,
            "unit_price": 120.0,
            "amount": 120.0,
        },
    ],
    "subtotal": 365.0,
    "tax": 31.03,
    "total": 396.03,
}

# Marker present in the bundled sample invoice — used to return the fixed stub.
_STUB_MARKER = "INV-2025-0847"


class StubClient(LLMClient):
    """Deterministic offline client for tests and demos without an API key."""

    def __init__(self, *, canned: dict[str, Any] | None = None) -> None:
        self._canned = canned if canned is not None else STUB_SAMPLE_OUTPUT.copy()
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if _STUB_MARKER in prompt:
            return json.dumps(self._canned)
        # Generic empty shell for arbitrary inputs in tests.
        return json.dumps(
            {
                "vendor": None,
                "date": None,
                "currency": None,
                "line_items": [],
                "subtotal": None,
                "tax": None,
                "total": None,
            }
        )


class MalformedThenFixedClient(LLMClient):
    """Test helper: first extraction response is invalid; repair returns valid JSON."""

    def __init__(self, fixed_payload: dict[str, Any]) -> None:
        self._fixed = fixed_payload
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        if "BROKEN TEXT" in prompt:
            return json.dumps(self._fixed)
        if self.call_count == 1:
            return "<<<not valid json>>>"
        return json.dumps(self._fixed)


class StringNumberClient(LLMClient):
    """Test helper: returns numeric fields as strings."""

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
                        "unit_price": "10.50",
                        "amount": "31.50",
                    }
                ],
                "subtotal": "31.50",
                "tax": "2.52",
                "total": "34.02",
            }
        )


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #


def build_extraction_prompt(text: str) -> str:
    """Build a strict JSON-only extraction prompt for *text*."""
    return f"""You are a precise document parser. Extract structured invoice/receipt data from the text below.

Return ONLY a single JSON object (no markdown fences, no commentary) with exactly these keys:
- vendor (string or null)
- date (ISO date YYYY-MM-DD or null)
- currency (3-letter code or null)
- line_items (array of objects, each with: description, qty, unit_price, amount)
- subtotal (number or null)
- tax (number or null)
- total (number or null)

Use null for any field you cannot determine. Numbers must be JSON numbers, not strings.

DOCUMENT TEXT:
---
{text.strip()}
---
"""


REPAIR_PROMPT_TEMPLATE = """The following text was supposed to be valid JSON but failed to parse.
Fix it and return ONLY the corrected JSON object (no markdown, no explanation).

BROKEN TEXT:
---
{broken}
---
"""


# --------------------------------------------------------------------------- #
# JSON parsing + repair
# --------------------------------------------------------------------------- #


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences if the model wrapped its JSON."""
    match = _FENCE_RE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Attempt to parse *raw* as a JSON object; return None on failure."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _heuristic_repair(raw: str) -> str:
    """Apply lightweight fixes for common LLM JSON mistakes."""
    text = _strip_fences(raw)
    # Drop trailing commas before } or ].
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # Extract the outermost { ... } if surrounded by prose.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


def parse_llm_json(
    raw: str,
    client: LLMClient,
    *,
    allow_repair: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Parse LLM output as JSON, optionally repairing once via the client.

    Returns (parsed_dict, was_repaired).
  """
    cleaned = _strip_fences(raw)
    parsed = _try_parse_json(cleaned)
    if parsed is not None:
        return parsed, False

    repaired_locally = _heuristic_repair(raw)
    parsed = _try_parse_json(repaired_locally)
    if parsed is not None:
        return parsed, True

    if not allow_repair:
        return {}, True

    # One LLM-assisted repair attempt.
    repair_prompt = REPAIR_PROMPT_TEMPLATE.format(broken=raw[:4000])
    repaired_raw = client.complete(repair_prompt)
    cleaned = _heuristic_repair(repaired_raw)
    parsed = _try_parse_json(cleaned)
    return (parsed if parsed is not None else {}), True


# --------------------------------------------------------------------------- #
# Normalization + validation
# --------------------------------------------------------------------------- #


def _coerce_number(value: Any) -> float | None:
    """Coerce a value to float, or None if impossible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "nan"}:
            return None
        # Strip currency symbols and thousands separators.
        text = re.sub(r"[^0-9.\-]", "", text.replace(",", ""))
        if text in {"", ".", "-"}:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    num = _coerce_number(value)
    if num is None:
        return None
    return int(round(num))


def normalize_extraction(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce types and fill missing keys with null."""
    out: dict[str, Any] = {}
    for key in EXTRACTION_SCHEMA:
        if key == "line_items":
            items = raw.get("line_items")
            if not isinstance(items, list):
                out["line_items"] = []
                continue
            normalized_items: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized_items.append(
                    {
                        "description": (
                            str(item["description"]).strip()
                            if item.get("description") not in (None, "")
                            else None
                        ),
                        "qty": _coerce_int(item.get("qty")),
                        "unit_price": _coerce_number(item.get("unit_price")),
                        "amount": _coerce_number(item.get("amount")),
                    }
                )
            out["line_items"] = normalized_items
        elif key in ("subtotal", "tax", "total"):
            out[key] = _coerce_number(raw.get(key))
        else:
            val = raw.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                out[key] = None
            else:
                out[key] = str(val).strip()
    return out


def validate_extraction(data: dict[str, Any]) -> dict[str, Any]:
    """Run arithmetic checks and return a validation summary."""
    issues: list[str] = []
    warnings: list[str] = []

    line_items = data.get("line_items") or []
    subtotal = data.get("subtotal")
    tax = data.get("tax")
    total = data.get("total")

    line_sum = 0.0
    line_sum_known = True
    for i, item in enumerate(line_items):
        qty = item.get("qty")
        unit_price = item.get("unit_price")
        amount = item.get("amount")
        if amount is None:
            line_sum_known = False
            warnings.append(f"line_items[{i}].amount is null")
            continue
        line_sum += amount
        if qty is not None and unit_price is not None:
            expected = round(qty * unit_price, 2)
            if abs(expected - amount) > TOTAL_TOLERANCE:
                warnings.append(
                    f"line_items[{i}]: qty*unit_price ({expected}) != amount ({amount})"
                )

    subtotal_ok: bool | None = None
    if subtotal is not None and line_sum_known:
        subtotal_ok = abs(line_sum - subtotal) <= TOTAL_TOLERANCE
        if not subtotal_ok:
            issues.append(
                f"line_items sum ({line_sum:.2f}) != subtotal ({subtotal:.2f})"
            )

    total_ok: bool | None = None
    if subtotal is not None and tax is not None and total is not None:
        expected_total = round(subtotal + tax, 2)
        total_ok = abs(expected_total - total) <= TOTAL_TOLERANCE
        if not total_ok:
            issues.append(
                f"subtotal+tax ({expected_total:.2f}) != total ({total:.2f})"
            )
    elif subtotal is not None and total is not None and tax is None:
        total_ok = abs(subtotal - total) <= TOTAL_TOLERANCE
        if not total_ok:
            warnings.append(
                f"subtotal ({subtotal:.2f}) != total ({total:.2f}) with no tax field"
            )

    required_present = sum(
        1
        for k in ("vendor", "date", "currency", "total")
        if data.get(k) not in (None, "")
    )

    return {
        "line_items_sum": round(line_sum, 2) if line_sum_known else None,
        "subtotal_matches_lines": subtotal_ok,
        "total_adds_up": total_ok,
        "issues": issues,
        "warnings": warnings,
        "fields_present": required_present,
        "fields_expected": 4,
    }


def compute_confidence(data: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """Derive a simple confidence score from completeness and validation."""
    fields_score = validation["fields_present"] / validation["fields_expected"]
    checks_passed = 0
    checks_total = 0
    for key in ("subtotal_matches_lines", "total_adds_up"):
        val = validation.get(key)
        if val is not None:
            checks_total += 1
            if val:
                checks_passed += 1

    validation_score = (
        checks_passed / checks_total if checks_total else 1.0
    )
    overall = round(0.6 * fields_score + 0.4 * validation_score, 3)

    return {
        "overall": overall,
        "fields_score": round(fields_score, 3),
        "validation_score": round(validation_score, 3),
        "level": (
            "high" if overall >= 0.85
            else "medium" if overall >= 0.6
            else "low"
        ),
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def default_client() -> LLMClient:
    """Return AnthropicClient when a key is set, otherwise StubClient."""
    if os.environ.get(ANTHROPIC_API_KEY_ENV):
        return AnthropicClient()
    return StubClient()


def extract_document(text: str, client: LLMClient | None = None) -> dict[str, Any]:
    """Extract structured invoice data from messy *text*.

    Builds a strict extraction prompt, calls the LLM, parses and validates the
    JSON, and returns the normalized dict plus ``confidence`` and ``validation``
    blocks. Never raises on bad input — returns best-effort null-filled output.
    """
    if not text or not str(text).strip():
        empty = normalize_extraction({})
        validation = validate_extraction(empty)
        return {
            **empty,
            "confidence": compute_confidence(empty, validation),
            "validation": validation,
        }

    llm = client if client is not None else default_client()

    try:
        prompt = build_extraction_prompt(text)
        raw_response = llm.complete(prompt)
        parsed, _repaired = parse_llm_json(raw_response, llm)
        normalized = normalize_extraction(parsed)
    except Exception:
        normalized = normalize_extraction({})

    validation = validate_extraction(normalized)
    confidence = compute_confidence(normalized, validation)

    return {
        **normalized,
        "confidence": confidence,
        "validation": validation,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "usage: python extractor.py <document.txt>\n"
            f"       set {ANTHROPIC_API_KEY_ENV} to use Claude ({DEFAULT_MODEL}); "
            "otherwise StubClient is used.",
            file=sys.stderr,
        )
        return 2 if args else 0

    path = Path(args[0])
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    result = extract_document(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
