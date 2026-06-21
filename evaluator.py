#!/usr/bin/env python3
"""Offline evaluation harness for the document extractor.

Runs the **deterministic** extraction pipeline (``parse_document_text`` ->
``normalize_extraction`` -> ``validate_extraction``) over the labelled gold set
in ``gold/`` and reports field-level precision / recall / accuracy plus the
arithmetic-validation pass-rate. No network, no API key, no LLM stub echo —
this measures the real parsing, coercion, and reconciliation logic.

Run directly to print a report:

    python3 evaluator.py

Or import :func:`evaluate_gold_set` for programmatic use (tests do this).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extractor import (
    extract_document,
    normalize_extraction,
)

GOLD_DIR = Path(__file__).resolve().parent / "gold"

# Scalar fields compared by exact value; line_items compared element-wise.
SCALAR_FIELDS = ("vendor", "date", "currency", "subtotal", "tax", "total")
LINE_ITEM_FIELDS = ("description", "qty", "unit_price", "amount")

# Money comparison tolerance (one cent).
_TOL = 0.005


def _values_match(pred: Any, gold: Any) -> bool:
    """Compare a predicted field value to the gold value."""
    if gold is None:
        return pred is None
    if pred is None:
        return False
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(pred) - float(gold)) <= _TOL
    return str(pred).strip() == str(gold).strip()


@dataclass
class FieldCounts:
    """Confusion-style counts for one field across the whole gold set."""

    tp: int = 0  # predicted a value, and it matched gold
    fp: int = 0  # predicted a value, but it was wrong (or gold was null)
    fn: int = 0  # gold had a value, but prediction was null
    tn: int = 0  # both null (correct abstention)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def accuracy(self) -> float:
        denom = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _score_field(counts: FieldCounts, pred: Any, gold: Any) -> None:
    if gold is None and pred is None:
        counts.tn += 1
    elif gold is None and pred is not None:
        counts.fp += 1
    elif gold is not None and pred is None:
        counts.fn += 1
    elif _values_match(pred, gold):
        counts.tp += 1
    else:
        counts.fp += 1  # wrong value = a false positive on that field


@dataclass
class EvalReport:
    per_field: dict[str, FieldCounts] = field(default_factory=dict)
    docs_total: int = 0
    arithmetic_pass: int = 0  # docs whose totals reconcile
    arithmetic_checkable: int = 0  # docs where a total check could run

    @property
    def overall(self) -> FieldCounts:
        agg = FieldCounts()
        for c in self.per_field.values():
            agg.tp += c.tp
            agg.fp += c.fp
            agg.fn += c.fn
            agg.tn += c.tn
        return agg

    @property
    def arithmetic_pass_rate(self) -> float:
        return (
            self.arithmetic_pass / self.arithmetic_checkable if self.arithmetic_checkable else 1.0
        )

    def field_accuracy(self) -> float:
        return self.overall.accuracy


def _iter_gold() -> list[tuple[str, str, dict[str, Any]]]:
    """Yield (name, raw_text, gold_dict) for every gold fixture."""
    pairs: list[tuple[str, str, dict[str, Any]]] = []
    for txt_path in sorted(GOLD_DIR.glob("*.txt")):
        json_path = txt_path.with_suffix(".json")
        if not json_path.exists():
            continue
        raw = txt_path.read_text(encoding="utf-8")
        gold = normalize_extraction(json.loads(json_path.read_text(encoding="utf-8")))
        pairs.append((txt_path.name, raw, gold))
    return pairs


def evaluate_gold_set() -> EvalReport:
    """Run the deterministic pipeline over the gold set and score it."""
    report = EvalReport()
    fields = list(SCALAR_FIELDS) + [f"line_item.{f}" for f in LINE_ITEM_FIELDS]
    report.per_field = {f: FieldCounts() for f in fields}

    for _name, raw, gold in _iter_gold():
        report.docs_total += 1
        result = extract_document(raw)  # deterministic path

        # Scalar fields.
        for f in SCALAR_FIELDS:
            _score_field(report.per_field[f], result.get(f), gold.get(f))

        # Line-item fields, aligned by index (missing rows count as FN).
        pred_items = result.get("line_items") or []
        gold_items = gold.get("line_items") or []
        for i in range(max(len(pred_items), len(gold_items))):
            p_item = pred_items[i] if i < len(pred_items) else {}
            g_item = gold_items[i] if i < len(gold_items) else {}
            for f in LINE_ITEM_FIELDS:
                _score_field(
                    report.per_field[f"line_item.{f}"],
                    p_item.get(f),
                    g_item.get(f),
                )

        # Arithmetic-validation pass-rate (measured on what the pipeline parsed).
        validation = result["validation"]
        total_ok = validation.get("total_adds_up")
        if total_ok is not None:
            report.arithmetic_checkable += 1
            if total_ok:
                report.arithmetic_pass += 1

    return report


def format_report(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"Gold documents evaluated: {report.docs_total}")
    lines.append("")
    lines.append(f"{'field':<22}{'prec':>7}{'rec':>7}{'acc':>7}{'f1':>7}")
    lines.append("-" * 50)
    for name, c in report.per_field.items():
        lines.append(f"{name:<22}{c.precision:>7.2f}{c.recall:>7.2f}{c.accuracy:>7.2f}{c.f1:>7.2f}")
    lines.append("-" * 50)
    o = report.overall
    lines.append(
        f"{'OVERALL':<22}{o.precision:>7.2f}{o.recall:>7.2f}{o.accuracy:>7.2f}{o.f1:>7.2f}"
    )
    lines.append("")
    lines.append(f"Field-level accuracy: {report.field_accuracy() * 100:.1f}%")
    lines.append(
        f"Arithmetic-validation pass-rate: "
        f"{report.arithmetic_pass_rate * 100:.1f}% "
        f"({report.arithmetic_pass}/{report.arithmetic_checkable} docs reconcile)"
    )
    return "\n".join(lines)


def main() -> int:
    report = evaluate_gold_set()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
