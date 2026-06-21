# AI Document Extractor

A small, production-minded Python tool that turns messy invoice, receipt, and
order-email text into clean, validated structured JSON using an LLM — with
defensive parsing, number coercion, and arithmetic checks so bad extractions are
caught before they reach your database.

It is the reusable delivery template for the common freelance gig: *"use AI to
extract structured data from messy text."*

---

## What it does

Given raw document text (the kind you actually get from forwarded emails,
scanned receipts, and vendor PDFs pasted into Slack), the tool:

1. **Builds a strict extraction prompt** — asks the model for a fixed JSON
   schema: `vendor`, `date`, `currency`, `line_items[]`, `subtotal`, `tax`,
   `total`.
2. **Calls an LLM** — `AnthropicClient` (Claude) in production, or a
   deterministic `StubClient` when no API key is set.
3. **Parses defensively** — strips markdown fences, repairs common JSON
   mistakes, and retries once if the response is malformed.
4. **Normalizes types** — coerces string numbers (`"$1,234.56"`, `"3"`) to
   floats/ints; fills missing fields with `null`.
5. **Validates arithmetic** — checks that line items sum to subtotal and
   `subtotal + tax ≈ total` (within one cent).
6. **Returns confidence + validation** — a simple score and a list of issues so
   downstream code can route low-confidence extractions to human review.

**Business value:** turn messy invoices, receipts, and order emails into clean
structured data automatically — ready for accounting systems, ERP imports, or
analytics pipelines — without manual re-keying.

---

## How to run

```bash
pip install -r requirements.txt

# Optional: use real Claude extraction (claude-opus-4-8)
export ANTHROPIC_API_KEY="sk-ant-..."

python extractor.py sample_invoice.txt
```

Without `ANTHROPIC_API_KEY`, the CLI uses `StubClient` and prints the
deterministic sample output — useful for demos and CI.

### Run the tests (offline — no API key or network)

```bash
pip install -r requirements.txt
python -m pytest -q
```

---

## Acceptance / "done"

A run is considered correct when:

- `python -m pytest -q` passes offline using `StubClient`.
- `python extractor.py sample_invoice.txt` prints valid structured JSON.
- The validation block reports `total_adds_up: true` on the bundled sample.
- No secrets are committed; the API key is read from `ANTHROPIC_API_KEY` only.

The committed `output/sample_output.json` is the exact result of running the
extractor on `sample_invoice.txt` via `StubClient`.

---

## Project layout

```
ai-doc-extractor/
├── extractor.py           # extraction pipeline + CLI
├── requirements.txt       # anthropic + pytest
├── sample_invoice.txt     # realistic messy invoice input
├── test_extractor.py      # offline pytest suite
├── output/
│   └── sample_output.json # committed before/after demo output
└── README.md
```

---

## Architecture notes

- **`LLMClient`** — pluggable interface; swap `AnthropicClient` for OpenAI,
  a local model, or a mock in tests.
- **`StubClient`** — returns fixed JSON for the bundled sample so tests and
  portfolio demos never need network access.
- **Repair path** — heuristic fixes (trailing commas, fence stripping) first;
  one LLM repair call if still unparseable.
- **Never crashes on bad input** — empty text, garbage JSON, and API errors
  degrade to null-filled output with a low confidence score.

---

## Sample input → output

**Input** (`sample_invoice.txt`): a realistic vendor invoice with mixed
formatting, currency symbols, and a tax line.

**Output** (`output/sample_output.json`): structured JSON with four line items,
validated totals (`$365.00 + $31.03 = $396.03`), and confidence metadata.

---

Built by [clira](https://clira.dev) — AI delivery for teams that need production-quality code, not demos.
