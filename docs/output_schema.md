# Company analysis output: integration guide (v1.0)

For the FinTech/backend side. This describes the JSON returned for one
company. The machine-readable contract is
[`schemas/company_analysis.schema.json`](../schemas/company_analysis.schema.json),
and real examples are in [`docs/examples/`](examples/).

> Proposed v1.0 from the AI/ML side. Please review the field names and
> error handling. Changes are easy now and harder once both sides build
> on it.

## How to get it

| Way | Call | Result |
|---|---|---|
| Command line | `python main.py analyze RELIANCE.NS` | `outputs/<SYMBOL>_<time>/summary.json` (+ `report.md`) |
| Local HTTP | `python main.py serve`, then `GET http://127.0.0.1:8765/api/analyze?company=RELIANCE.NS` | the JSON body |
| Python | `from src.pipeline.analyze import analyze_company` | the same dict |

`company` accepts a ticker (`RELIANCE.NS`, `TCS.BO`, `AAPL`) or a company
name (`Reliance Industries`). One call takes about 3–6 seconds (it downloads
statements). The local server handles one analysis at a time.

## Versioning

- `schema_version` is `"company_analysis/1.0"`.
- **Minor** version (1.1, …): fields are only added. Ignore fields you do
  not know.
- **Major** version (2.0): fields renamed or removed, or meanings changed.
  We will tell you first.

## Status, errors and HTTP codes

`status` is `"ok"` or `"error"`. When `status` is `"error"`, `error` is
filled. Sections that were not reached are `null`.

| `error.code` | Meaning | HTTP | CLI exit |
|---|---|---|---|
| `empty_input` | Nothing was entered | 400 | 2 |
| `unknown_company` | No listed security matched the input | 404 | 2 |
| `not_a_company` | Only ETFs, funds or indices matched; they have no financial statements. `candidates` lists them. | 404 | 2 |
| `no_financial_data` | The company exists but the provider has no quarterly statements. `candidates` may list other listings. | 404 | 2 |
| `provider_error` | The data provider failed (network, rate limit). Retry later. | 502 | 3 |
| `internal_error` | Unexpected bug; details in `error.log` in the run folder | 500 | 2 |

`error.candidates` is a list of `{symbol, name, exchange, type}`. It is
useful for a "did you mean" list.

## Top-level fields

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `company_analysis/1.0` |
| `status` | `ok` / `error` | |
| `generated_at` | ISO-8601 UTC | When this result was produced |
| `request` | object | `input`, `prefer_indian`, `as_of` as received |
| `company` | object / null | Symbol, name, exchange, country, sector, industry, `reporting_currency` (statements), `trading_currency` (share price), `resolution` (how the input was matched) |
| `source` | object / null | Provider, `retrieved_at`, and per statement: available, number of periods, latest period |
| `coverage` | object / null | Usable quarters, first and latest period, fiscal quarter-end months, share of quarters with each key field |
| `metrics` | object / null | **Deterministic figures**: `latest` (values, ratios, changes), all `periods`, `ratio_definitions` |
| `risk_flags` | object / null | **Deterministic rules**: `summary` + one entry per rule |
| `ml_signal` | object / null | **Experimental model output.** See the display rules below. |
| `data_quality` | object / null | Audit counts and findings, and the ratio-formula check |
| `warnings` | list of `{code, message}` | Things the user should know (see below) |
| `error` | object / null | See above |
| `disclaimer` | string | **Always show it** |
| `files` | object | Local output files (relative paths). Ignore these in an API. |

### Units

- Money values (`metrics.*.values`) are in the **reporting currency's units**,
  not thousands. Example: `revenue: 94036000000` = 94.0 billion.
- The currency is in `metrics.currency`. RELIANCE.NS reports in INR, US
  companies in USD. Never compare amounts across companies without
  converting them.
- Ratios and changes are fractions: `0.249` = 24.9%. Debt/equity and the
  current ratio are multiples (`1.54` = 1.54x).
- `period_end` is the fiscal quarter end, not the filing date. Results are
  typically published 3–8 weeks later.
- A `null` value means the provider did not report it. It never means zero.

### `metrics.latest.changes`

- `revenue_change_qoq`: vs the previous quarter. Often seasonal, e.g.
  December → March.
- `revenue_change_yoy`: vs the same quarter a year earlier. Use this one for
  "is the company shrinking?". It is `null` when that quarter is not
  available.

## Risk flags (rule-based)

Each flag has: `id`, `category`, `severity` (high/medium/low), `title`, `rule`,
`status` (`triggered` / `not_triggered` / `not_evaluable`), `value`,
`threshold`, `period_end`, `inputs`, and `explanation`.

- Each rule uses the latest quarter where all of its inputs are reported.
  If that is older than the latest quarter, `period_end` shows it and the
  explanation says so.
- `not_evaluable` means data is missing (e.g. no quarterly cash flow for
  RELIANCE.NS) or the rule does not apply (e.g. current ratio for banks).
  **Show it as "not available", never as "OK".**

| id | Severity | Triggered when |
|---|---|---|
| `net_loss` | medium | Net income < 0 |
| `operating_loss` | medium | Operating income < 0 |
| `revenue_decline_yoy` | medium | Revenue vs same quarter last year < −5% |
| `operating_margin_drop_yoy` | low | Operating margin down > 5 percentage points vs same quarter last year |
| `low_current_ratio` | medium | Current assets / current liabilities < 1.0 |
| `negative_equity` | high | Shareholders' equity ≤ 0 |
| `high_debt_to_equity` | medium | Total debt / equity > 2.0 |
| `high_debt_to_assets` | medium | Total debt / total assets > 0.5 |
| `low_interest_coverage` | medium | EBIT / interest expense < 1.5 |
| `negative_operating_cash_flow` | medium | Operating cash flow < 0 |
| `negative_free_cash_flow` | low | Free cash flow < 0 |

These are general screening thresholds, not tuned per industry, and
`risk_flags.summary.rules_version` identifies the rule set. Present them as
"points to look at", **not as a rating or score**. Do not add them up into
one number.

## ML signal: display rules

`ml_signal.status` is either:
- `available`: has `score`, `predicted_class`, `inputs` (with
  `imputed: true` where a ratio was missing), `top_contributions`,
  `training` and `evaluation`;
- `unavailable`: has a `reason`, e.g. banks are not covered.

The model's evaluation found **no evidence that it predicts better than
chance** (`ml_signal.evaluation.verdict`, `reports/model_evaluation.md`).
Therefore:

1. Always label it **"Experimental"**, together with
   `ml_signal.evaluation.verdict`.
2. Never show `score` as a probability, a percentage chance, or a risk score.
   `score_meaning` explains why.
3. Keep it visually separate from `metrics` and `risk_flags`, which are
   deterministic.
4. If space is limited, leave it out rather than show it without the
   caveats. Hiding it by default is reasonable.

## Warning codes

| code | Meaning |
|---|---|
| `name_search_match` | The input was matched by company name; show the matched name so the user can confirm |
| `exchange_suffix_added` | A bare symbol was matched to `.NS`/`.BO` |
| `other_listings` | The same symbol has data on other exchanges |
| `missing_statement` | A statement (often cash flow) is not available |
| `profile_unavailable` | Name, currency and sector could not be fetched |
| `stale_data` | The latest quarter is old; a newer one was probably published |
| `incomplete_latest_period` | The provider had only partial data for the newest quarter |
| `quarterly_gap` | Quarters are missing in the history |
| `one_off_items` | Unusual items are large relative to net income; margins may mislead |
| `revenue_jump` | Revenue changed more than 50% from the previous quarter |
| `negative_equity`, `duplicate_period` | See `data_quality.findings` |
| `financial_company_structure` | Bank-like statements; several ratios do not apply |

## Must-show items for any screen

- Company name and symbol, exchange, and **reporting currency**
- **Latest period** (`metrics.latest_period`) and **retrieved at**
  (`source.retrieved_at`)
- `warnings`
- `disclaimer`

## Validating

```python
from src.pipeline.schema_check import validate_payload
errors = validate_payload(result)   # [] when valid
```

Any JSON Schema 2020-12 validator (e.g. the `jsonschema` package) can also
use `schemas/company_analysis.schema.json`.
