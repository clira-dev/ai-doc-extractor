#!/usr/bin/env python3
"""AI document extractor — turn messy invoice/receipt text into validated JSON.

This module exposes a deterministic extraction pipeline that can run with zero
network access (a rule-based parser over the raw text) and an optional LLM path
(``AnthropicClient``) for documents the deterministic parser cannot handle.

The pieces that carry the accuracy of this system are all deterministic and are
measured offline against a labelled gold set (see ``gold/`` and ``evaluator.py``):

* the JSON-repair ladder (``parse_llm_json`` + heuristics),
* number coercion (``_coerce_number`` / ``_coerce_int``),
* totals reconciliation (``validate_extraction``),
* and the rule-based text parser (``parse_document_text``).

Pipeline
--------
    raw text
      -> deterministic parse  (rule-based)  OR  LLM extract + JSON-repair
      -> normalize (type coercion, fill nulls)
      -> validate (arithmetic reconciliation: line items, subtotal, tax, total)
      -> confidence score
      -> route to human review when confidence is low

Usage
-----
    # Offline, deterministic:
    python3 extractor.py sample_invoice.txt

    # With the LLM path (optional):
    export ANTHROPIC_API_KEY="..."   # never printed or logged
    python3 extractor.py --llm sample_invoice.txt

Standard library only. The ``anthropic`` SDK is imported lazily and only when
the LLM path is explicitly requested.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Logging — structured, opt-in via DOC_EXTRACTOR_LOG_LEVEL (default WARNING).
# Never logs secret values; only references the API key by env-var name.
# --------------------------------------------------------------------------- #

logger = logging.getLogger("doc_extractor")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("DOC_EXTRACTOR_LOG_LEVEL", "WARNING").upper())


# --------------------------------------------------------------------------- #
# Typed errors — distinguish transport vs parse vs validation failures so
# callers can react differently (retry transport, route parse to review, etc.).
# --------------------------------------------------------------------------- #


class ExtractionError(Exception):
    """Base class for all extraction failures."""


class TransportError(ExtractionError):
    """The LLM backend could not be reached or returned a transport-level error."""


class ParseError(ExtractionError):
    """The model/text output could not be parsed into a JSON object."""


class DataValidationError(ExtractionError):
    """Extracted data failed a structural precondition (not an arithmetic check)."""


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

# Tolerance for floating-point reconciliation (one cent + epsilon).
TOTAL_TOLERANCE = 0.02

# Confidence below this routes the extraction to human review.
REVIEW_THRESHOLD = 0.85


# --------------------------------------------------------------------------- #
# LLM client interface (production only — test doubles live in test_extractor.py)
# --------------------------------------------------------------------------- #


class LLMClient(ABC):
    """Pluggable LLM backend for structured extraction."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's raw text response to *prompt*.

        Implementations should raise :class:`TransportError` on network /
        backend failures so the pipeline can distinguish them from parse errors.
        """


class AnthropicClient(LLMClient):
    """Production client using the official Anthropic SDK (lazy-imported)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic  # noqa: F401  (presence check)
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise TransportError(
                "anthropic package is required for AnthropicClient; pip install anthropic"
            ) from exc

        key = api_key if api_key is not None else os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not key:
            raise TransportError(f"{ANTHROPIC_API_KEY_ENV} is not set; cannot call Anthropic API.")

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str) -> str:  # pragma: no cover - needs network
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                # Opus 4.8: adaptive thinking only (no budget_tokens / sampling).
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # SDK raises typed errors; wrap as transport.
            logger.error("LLM transport failure: %s", type(exc).__name__)
            raise TransportError(str(exc)) from exc

        parts: list[str] = []
        for block in message.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


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
# JSON parsing + repair ladder (deterministic; tested directly)
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
    client: LLMClient | None = None,
    *,
    allow_repair: bool = True,
) -> tuple[dict[str, Any], bool]:
    """Parse LLM output as JSON, optionally repairing once via the client.

    Repair ladder:
      1. strip fences, parse;
      2. heuristic repair (trailing commas, surrounding prose), parse;
      3. one LLM-assisted repair call (only if ``allow_repair`` and a client).

    Returns ``(parsed_dict, was_repaired)``. Raises :class:`ParseError` if every
    rung of the ladder fails (no more silent empty dict).
    """
    cleaned = _strip_fences(raw)
    parsed = _try_parse_json(cleaned)
    if parsed is not None:
        return parsed, False

    repaired_locally = _heuristic_repair(raw)
    parsed = _try_parse_json(repaired_locally)
    if parsed is not None:
        logger.info("parse_llm_json: recovered via heuristic repair")
        return parsed, True

    if not allow_repair or client is None:
        raise ParseError("LLM output is not valid JSON and repair is disabled")

    logger.info("parse_llm_json: attempting LLM-assisted repair")
    repair_prompt = REPAIR_PROMPT_TEMPLATE.format(broken=raw[:4000])
    repaired_raw = client.complete(repair_prompt)
    cleaned = _heuristic_repair(repaired_raw)
    parsed = _try_parse_json(cleaned)
    if parsed is None:
        raise ParseError("LLM-assisted repair did not produce valid JSON")
    return parsed, True


# --------------------------------------------------------------------------- #
# Deterministic rule-based text parser (the offline-measurable extraction path)
# --------------------------------------------------------------------------- #

_DATE_PATTERNS = [
    # 2025-03-14
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    # 03/14/2025 or 3/14/25
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), "mdy"),
    # March 14, 2025  /  Mar 14 2025
    (
        re.compile(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
            r"(\d{1,2}),?\s+(\d{4})\b",
            re.IGNORECASE,
        ),
        "mname",
    ),
]

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Currency symbol -> ISO code.
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

# A money token, e.g. $1,234.56 / 1234.56 / 45.00
_MONEY_RE = re.compile(r"[-+]?\$?€?£?¥?\s?\d[\d,]*\.\d{2}\b")

# Field labels we look for on their own lines.
_LABEL_RE = {
    "subtotal": re.compile(r"\bsub\s*-?\s*total\b", re.IGNORECASE),
    "tax": re.compile(r"\b(sales\s+)?tax\b|\bvat\b|\bgst\b", re.IGNORECASE),
    "total": re.compile(r"\b(grand\s+)?total\b|\bamount\s+due\b|\btotal\s+due\b", re.IGNORECASE),
}


def _first_money(line: str) -> float | None:
    """Return the last money value on a line (totals sit at line end)."""
    matches = _MONEY_RE.findall(line)
    if not matches:
        return None
    return _coerce_number(matches[-1])


def _normalize_date(raw_match: tuple[str, ...], kind: str) -> str | None:
    try:
        if kind == "iso":
            y, m, d = raw_match
        elif kind == "mdy":
            m, d, y = raw_match
            if len(y) == 2:
                y = "20" + y
        elif kind == "mname":
            mon, d, y = raw_match
            m = str(_MONTHS[mon[:3].lower()])
        else:
            return None
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except (ValueError, KeyError):
        return None


def _detect_currency(text: str) -> str | None:
    # Explicit ISO code wins (e.g. "TOTAL DUE (USD)").
    iso = re.search(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", text)
    if iso:
        return iso.group(1)
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    return None


def _looks_like_line_item(line: str) -> bool:
    """A line-item row: prose then >=2 money/qty columns ending in an amount."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("-", "=", "_")):
        return False
    # Must contain at least one money value.
    if not _MONEY_RE.search(stripped):
        return False
    # Skip label rows (subtotal/tax/total) — those are totals, not items.
    low = stripped.lower()
    if any(rx.search(low) for rx in _LABEL_RE.values()):
        return False
    return True


def _parse_line_item(line: str) -> dict[str, Any] | None:
    """Parse a single line-item row into description/qty/unit_price/amount."""
    monies = _MONEY_RE.findall(line)
    if not monies:
        return None
    amount = _coerce_number(monies[-1])
    unit_price = _coerce_number(monies[-2]) if len(monies) >= 2 else None

    # Description = text before the first money/number column.
    first_money_pos = _MONEY_RE.search(line)
    head = line[: first_money_pos.start()] if first_money_pos else line
    # qty is the trailing standalone integer in the head (e.g. "... 2").
    qty = None
    qty_match = re.search(r"(\d+)\s*$", head.strip())
    if qty_match:
        qty = int(qty_match.group(1))
        head = head[: qty_match.start()].rstrip()
    description = head.strip(" \t.-") or None

    return {
        "description": description,
        "qty": qty,
        "unit_price": unit_price,
        "amount": amount,
    }


def parse_document_text(text: str) -> dict[str, Any]:
    """Deterministically extract structured fields from raw document text.

    This is the offline, network-free extraction path. It is intentionally
    rule-based so its accuracy can be measured against a labelled gold set.
    Returns a dict in :data:`EXTRACTION_SCHEMA` shape (un-normalized).
    """
    out: dict[str, Any] = {
        k: (list(v) if isinstance(v, list) else v) for k, v in EXTRACTION_SCHEMA.items()
    }
    if not text or not text.strip():
        return out

    lines = text.splitlines()

    # Vendor: the "From:" line, else the first non-empty non-title line.
    for _i, line in enumerate(lines):
        m = re.match(r"\s*from:\s*(.+)", line, re.IGNORECASE)
        if m and m.group(1).strip():
            out["vendor"] = m.group(1).strip()
            break
    if out["vendor"] is None:
        for line in lines:
            s = line.strip()
            if not s:
                continue
            # Skip obvious headers like "INVOICE #..." or rule lines.
            if re.match(r"^(invoice|receipt|order|bill)\b", s, re.IGNORECASE):
                continue
            if set(s) <= set("=-_ "):
                continue
            out["vendor"] = s
            break

    # Date: first parseable date, preferring an "Invoice Date" line.
    date_line = next((ln for ln in lines if re.search(r"invoice\s+date", ln, re.IGNORECASE)), None)
    search_space = [date_line] if date_line else lines
    for line in search_space:
        if line is None:
            continue
        for rx, kind in _DATE_PATTERNS:
            m = rx.search(line)
            if m:
                norm = _normalize_date(m.groups(), kind)
                if norm:
                    out["date"] = norm
                    break
        if out["date"]:
            break
    if out["date"] is None:
        for line in lines:
            for rx, kind in _DATE_PATTERNS:
                m = rx.search(line)
                if m:
                    norm = _normalize_date(m.groups(), kind)
                    if norm:
                        out["date"] = norm
                        break
            if out["date"]:
                break

    out["currency"] = _detect_currency(text)

    # Totals: scan labelled lines. Check subtotal/tax before total so "subtotal"
    # is not captured by the "total" regex.
    for line in lines:
        low = line.lower()
        if out["subtotal"] is None and _LABEL_RE["subtotal"].search(low):
            out["subtotal"] = _first_money(line)
            continue
        if out["tax"] is None and _LABEL_RE["tax"].search(low):
            out["tax"] = _first_money(line)
            continue
        if (
            out["total"] is None
            and _LABEL_RE["total"].search(low)
            and not _LABEL_RE["subtotal"].search(low)
        ):
            out["total"] = _first_money(line)

    # Line items.
    items: list[dict[str, Any]] = []
    for line in lines:
        if _looks_like_line_item(line):
            item = _parse_line_item(line)
            if item and item["amount"] is not None:
                items.append(item)
    out["line_items"] = items

    return out


# --------------------------------------------------------------------------- #
# Normalization + validation (deterministic; tested directly)
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
            issues.append(f"line_items sum ({line_sum:.2f}) != subtotal ({subtotal:.2f})")

    total_ok: bool | None = None
    if subtotal is not None and tax is not None and total is not None:
        expected_total = round(subtotal + tax, 2)
        total_ok = abs(expected_total - total) <= TOTAL_TOLERANCE
        if not total_ok:
            issues.append(f"subtotal+tax ({expected_total:.2f}) != total ({total:.2f})")
    elif subtotal is not None and total is not None and tax is None:
        total_ok = abs(subtotal - total) <= TOTAL_TOLERANCE
        if not total_ok:
            warnings.append(f"subtotal ({subtotal:.2f}) != total ({total:.2f}) with no tax field")

    required_present = sum(
        1 for k in ("vendor", "date", "currency", "total") if data.get(k) not in (None, "")
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
    """Derive a confidence score from completeness and arithmetic validation."""
    fields_score = validation["fields_present"] / validation["fields_expected"]
    checks_passed = 0
    checks_total = 0
    for key in ("subtotal_matches_lines", "total_adds_up"):
        val = validation.get(key)
        if val is not None:
            checks_total += 1
            if val:
                checks_passed += 1

    validation_score = checks_passed / checks_total if checks_total else 1.0
    overall = round(0.6 * fields_score + 0.4 * validation_score, 3)

    return {
        "overall": overall,
        "fields_score": round(fields_score, 3),
        "validation_score": round(validation_score, 3),
        "level": ("high" if overall >= REVIEW_THRESHOLD else "medium" if overall >= 0.6 else "low"),
        "needs_review": overall < REVIEW_THRESHOLD,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _assemble_result(normalized: dict[str, Any]) -> dict[str, Any]:
    validation = validate_extraction(normalized)
    confidence = compute_confidence(normalized, validation)
    return {**normalized, "confidence": confidence, "validation": validation}


def extract_document(
    text: str,
    client: LLMClient | None = None,
    *,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Extract structured invoice data from messy *text*.

    By default this runs the **deterministic** rule-based parser (no network).
    Pass ``use_llm=True`` (with a ``client`` or ``ANTHROPIC_API_KEY`` set) to use
    the LLM extraction path with the JSON-repair ladder.

    Returns the normalized dict plus ``confidence`` and ``validation`` blocks.
    Transport / parse failures are logged and degrade to a low-confidence,
    null-filled result that is flagged ``needs_review`` — they are no longer
    silently swallowed (the failure class is logged at ERROR/WARNING).
    """
    if not text or not str(text).strip():
        return _assemble_result(normalize_extraction({}))

    if not use_llm and client is None:
        # Deterministic offline path.
        parsed = parse_document_text(text)
        return _assemble_result(normalize_extraction(parsed))

    # LLM path.
    llm = client if client is not None else AnthropicClient()
    try:
        prompt = build_extraction_prompt(text)
        raw_response = llm.complete(prompt)
        parsed, _repaired = parse_llm_json(raw_response, llm)
        normalized = normalize_extraction(parsed)
    except TransportError as exc:
        logger.error("extract_document: transport error: %s", exc)
        normalized = normalize_extraction({})
    except ParseError as exc:
        logger.warning("extract_document: parse error, routing to review: %s", exc)
        normalized = normalize_extraction({})

    return _assemble_result(normalized)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    use_llm = False
    if "--llm" in args:
        use_llm = True
        args.remove("--llm")

    if not args or args[0] in ("-h", "--help"):
        print(
            "usage: python3 extractor.py [--llm] <document.txt>\n"
            "  default: deterministic offline parser (no network)\n"
            f"  --llm:   use Claude ({DEFAULT_MODEL}); reads {ANTHROPIC_API_KEY_ENV}",
            file=sys.stderr,
        )
        return 2 if args else 0

    path = Path(args[0])
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    result = extract_document(text, use_llm=use_llm)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
