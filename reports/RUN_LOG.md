# Run log: phases 1–5 and SEC EDGAR (2026-11-06 to 2026-11-11)

Scope, from the handoff PDF:
1. Run and stabilise the existing repository.
2. One command for any company, with ticker handling.
3. Audit data quality and verify features.
4. Evaluate the ML model honestly.
5. Define the output format for the FinTech app.

After that, a larger evaluation on SEC EDGAR history was added. Still open:
the FinTech teammate's review of `docs/output_schema.md`, the integration
itself, and final polish (phase 6).

## Environment

- Windows 11, Python 3.14.5, fresh `.venv` (Git ignores it)
- `pip install -r requirements.txt` succeeded and `pip check` found no
  conflicts. Key versions: pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1,
  yfinance 1.7.0.
- `requirements.txt` is saved as UTF‑16. pip reads it, but some tools cannot.
  It has not been changed.
- `requirements.txt` covers every import in the code (pandas, numpy,
  scikit-learn, yfinance, requests). The new code and tests add no new
  packages.

## Phase 1: running the existing scripts

All 38 scripts were run in pipeline order in a separate clone, with fresh
Yahoo Finance downloads, so the committed `data/` and `reports/` were not
touched. Interactive scripts were given `TSLA`, or Enter for the defaults.
`collect_multiple_companies_backup.py` was skipped: it is identical to
`collect_multiple_companies.py` apart from a byte-order mark.

**Result: 38/38 exited with code 0.** Exit codes hide problems, though,
because most scripts print an error and return. Problems found:

| # | Problem | Status |
|---|---|---|
| 1 | The annual collector resolved **META → META.BO** (String Metaverse Ltd, BSE) instead of Meta Platforms, because it tried `.NS`/`.BO` before the symbol as typed. The committed annual data is unaffected (it predates this). | **Fixed**: as-typed first, then `.NS`, `.BO` |
| 2 | Yahoo serves only the latest ~5–7 quarters, and the oldest column comes back nearly empty (e.g. AAPL 2025‑06‑30: 5 of 34 income fields). `collect_quarterly_data.py` overwrites the raw files, so each re-collection loses history. | **Worked around** in `main.py rebuild` (field-level merge with stored history). Running the collector script directly still overwrites. |
| 3 | No single entry point; outputs must be deleted by hand before the scripts will rerun. | **Fixed**: `main.py rebuild` writes each run to its own folder |
| 4 | `prepare_ml_data.py`: pandas 3 deprecation (`select_dtypes("object")`). A future pandas version would silently drop text columns. | **Fixed** (explicit `"str"`) |
| 5 | `inspect_altman_data.py` looks for `data/raw/<TICKER>_balance_sheet.csv` for 8 companies; only TSLA has that file, so it prints "file not found" 7 times. | Open (inspection script only) |
| 6 | `data/processed/required_financial_data_clean/` has no script that creates it. | Open (cannot be regenerated) |
| 7 | RELIANCE.NS: no quarterly cash flow from Yahoo; balance sheet is half-yearly only; Sep‑2025 quarter missing. | Data limitation; reported by the audit |
| 8 | Committed AMZN 2026‑06‑30 income statement has 5 of 47 fields (collected while Yahoo had partial data). | Reported by the audit; a fresh rebuild has the full quarter |

**Fresh data vs committed data.** All 46 quarters present in both
had identical revenue, net income, assets and debt (no restatements). A
plain re-download gives 30 labelled rows (committed: 31). With the history
merge it gives 37.

**Model results move a lot when the data changes slightly** (one positive
case in a test set can swing balanced accuracy by 0.25):

| Run | Labelled rows | Train script: LogReg balanced acc. | Date-grouped: LogReg balanced acc. | Majority baseline acc. (date-grouped) |
|---|---|---|---|---|
| Committed reports | 31 | 0.500 | 0.357 | 0.875 |
| Fresh download only | 30 | 0.750 (1 positive test case) | 0.614 | 0.846 |
| Fresh + history (`rebuild`) | 37 | 0.550 | 0.614 | 0.846 |

Company-wise validation stays at or below chance for every company. The
model is still a proof of concept. Phase 4 should not quote any of these
numbers as model quality.

## Phase 2: commands added

```
python main.py analyze RELIANCE.NS            # or "Reliance Industries", TCS, META ...
python main.py rebuild                        # fresh download + history -> runs/<time>/
python main.py rebuild --from-raw data/raw/quarterly   # offline
python main.py audit                          # audit committed data -> reports/
python main.py serve                          # local test page, http://127.0.0.1:8765
```

Ticker handling (`src/data_collection/ticker_resolver.py`), tested live:

| Input | Result |
|---|---|
| `RELIANCE.NS` | RELIANCE.NS, exact |
| `Reliance Industries` | RELIANCE.NS via name search (warning tells the user to check the match) |
| `TCS` | TCS.NS (no US listing); TCS.BO reported as an alternative |
| `META` | Meta Platforms; META.BO reported as an alternative |
| `SPY` | error `not_a_company` (ETF), exit 2 / HTTP 404 |
| `ZZZQXNOTREAL` | error `unknown_company`, exit 2 / HTTP 404 |
| empty | error `empty_input`, HTTP 400 |
| network down | error `provider_error`, exit 3 / HTTP 502. yfinance only logs a *warning* and returns an empty table here; the wrapper in `provider.py` turns that into an error instead of "unknown ticker". |

Existing scripts were changed only to make their logic reusable
(`statement_to_rows`, `build_quarterly_dataset`, `add_ratio_features`). Their
standalone behaviour is unchanged. **Check:** `rebuild --from-raw
data/raw/quarterly` reproduces all 4 committed datasets and all 8 committed
report files exactly.

## Phase 3: data-quality audit

Full report: `reports/data_quality_audit.md`. For the committed data:
0 errors, 26 warnings, 16 notes.

- **Ratio formulas:** all 5 recomputed independently. 31/31 rows match.
  Definitions: net margin = NI/revenue; operating margin = OI/revenue;
  debt/assets; debt/equity; current ratio. Each is left missing when its
  denominator is ≤ 0.
- **Target:** recomputed independently, 0 disagreements. Horizon = next
  reported quarter (70–110 days later).
- **Leakage:** no model feature is a future or identifier column. Features
  use only the same quarter's statements. Caveat: `Date` is the period end,
  not the filing date, and Yahoo does not provide filing dates.
- **Main data findings:**
  - Every company has only 5 usable quarters.
  - Unusual (one-off) items above 10% of net income in 11 company-quarters.
    AMZN and GOOGL reach 50–88%, from investment gains.
  - GOOGL total debt is missing in 4 of 5 quarters.
  - JPM has a bank statement layout (no current assets or operating income).
    It is excluded from the model.
  - RELIANCE.NS reports in INR; the other companies report in USD.
  - NVDA's fiscal quarters end in Jan/Apr/Jul/Oct.
- **Handed to phase 4 (see below for what was done):**
  - 5 of the 8 committed decline labels are the seasonal Dec→Mar drop, so
    quarter-on-quarter decline mostly measures seasonality.
  - `train_revenue_decline_model.py` splits by row position, so the same
    date (2025‑12‑31) appears in both train and test.
  - Negative equity would make Debt_To_Equity missing; the model would then
    fill it with the median, hiding that risk. No current company has
    negative equity.

## Phase 4: honest model evaluation (2026-11-07)

New: `src/ml/evaluate_revenue_decline.py` (`python main.py evaluate`; also
step 14 of `rebuild`). Full report: `reports/model_evaluation.md`.

- **Baselines:** the old scripts only compared against "always predict the
  majority class". The new script adds a seasonal rule (the majority label
  for quarters ending in the same calendar month) to show whether the model
  learns anything beyond the December drop.
- **Validation:** all on the same test rows.
  - Walk-forward: train only on labels already observable at the test date.
  - Unseen company: leave one company out.
  - Both at once.

  The old `validate_company_wise.py` trained on other companies' future
  quarters, and `validate_date_grouped.py` used labels not yet observable
  (2025‑12‑31 labels need March 2026 revenue while NVDA's 2026‑01‑31 rows
  were in test). The walk-forward schemes avoid both.
- **Statistics:**
  - Pooled precision, recall, F1, accuracy, balanced accuracy and confusion
    matrix, with sample sizes and class counts.
  - A bootstrap 95% interval for balanced accuracy.
  - A 200-shuffle permutation p-value (is it better than chance?).
- **Speed:** about 18 s. Folds are built once, plain numpy is used in the
  inner loops, and the permutations run in parallel. Results are identical
  to the first, slow version.
- **Leak fixed:** `train_revenue_decline_model.py` and
  `explain_revenue_decline.py` now split on a date boundary
  (`src/ml/splits.py`). Before, three 2025‑12‑31 rows were in test while the
  other four were in training. Their committed reports were regenerated:
  - train 23 / test 8 (was 21/10);
  - logistic regression balanced accuracy 0.36 vs majority-class accuracy
    0.875;
  - explain step: 5 of 8 correct, and the only actual decline was missed.

  The other committed reports are unchanged.

**Results**

| Data | Scheme | 5-ratio model BA (95% interval) | p | Seasonal rule BA (p) |
|---|---|---|---|---|
| Committed (31 rows) | walk-forward | 0.50 (0.25–0.78) | 0.53 | 0.50 (0.97) |
| | unseen company | 0.53 (0.32–0.74) | 0.44 | **0.77 (0.02)** |
| | both | 0.69 (0.46–0.93) | 0.13 | 0.50 (0.81) |
| Live + history (37 rows) | walk-forward | 0.50 (0.28–0.74) | 0.52 | 0.50 (0.96) |
| | unseen company | 0.53 (0.33–0.72) | 0.43 | **0.74 (0.005)** |
| | both | 0.61 (0.36–0.83) | 0.22 | 0.50 (0.75) |

**Conclusion.** There is no evidence that the model predicts revenue
declines better than chance, on either dataset or under any scheme. The only
detectable signal is seasonality, and the model captures less of it than
the simple seasonal rule. In the time-aware schemes even the seasonal rule
scores 0.50, because there is no earlier December with a known outcome to
learn from: the history is too short. A seasonality-free (year-over-year)
target would have 9 labels and 0 declines (committed) or 14 and 1 (live),
so it is not feasible without longer history, e.g. SEC EDGAR for US
companies.

**Still open:**
- Negative equity is still median-imputed. No current company has negative
  equity, and this only matters if the model is kept.
- Filing dates are not available, so the observability rule uses period
  ends.

## Phase 5: output for the FinTech app (2026-11-07)

- **`company_analysis/1.0`.** `analyze` now returns a stable JSON result
  with these sections:
  - `company`, `source` (provider, retrieval time, statements available),
    `coverage`;
  - `metrics` (deterministic);
  - `risk_flags` (11 rule-based checks; each is triggered, not triggered or
    not evaluable, with its threshold, value, period and explanation);
  - `ml_signal` (experimental, separate; it carries the phase-4 verdict and
    shows which inputs were filled with the median);
  - `data_quality`, coded `warnings`, coded `error`, and `disclaimer`.

  Unexpected exceptions are returned as `internal_error` instead of
  crashing.
- **Contract and docs:**
  - [`schemas/company_analysis.schema.json`](../schemas/company_analysis.schema.json),
    plus a small standard-library validator in `src/pipeline/schema_check.py`;
  - [`docs/output_schema.md`](../docs/output_schema.md): field reference,
    units, error/HTTP/exit codes, warning codes, risk-flag rules, ML display
    rules, versioning;
  - three real examples in `docs/examples/`, all schema-valid.
- **`report.md`:** a readable report generated from the same JSON for each
  run. The local test page shows the risk flags and the ML signal panel.
- **Checked live:**
  - AAPL: 0 of 10 evaluable flags triggered; interest coverage not
    evaluable because Apple does not report interest expense.
  - JPM: bank rules not evaluable, ML signal unavailable.
  - RELIANCE.NS: cash-flow and year-over-year rules not evaluable, because
    Yahoo has no quarterly cash flow and no Sep-2025 quarter.
- **Fixed while testing:**
  - The module was renamed `ml_signal.py`. As `signal.py` it would shadow
    Python's standard `signal` module when scripts are run from `src/ml/`.
  - A stray "null" in the page's ML panel.

## SEC EDGAR history (2026-11-07)

`python main.py edgar` downloads XBRL company facts for the 7 US
companies. The SEC needs a contact email in `SEC_EDGAR_USER_AGENT`; it is
kept in the git-ignored `.env`. The command then builds point-in-time
quarters, cross-checks them with Yahoo, and evaluates both targets.
Reports are in `reports/sec_edgar/`.

- **Data:** 477 quarters (2007–2026). 440 are point-in-time (first filed
  within 120 days of the quarter end), giving 429 labelled rows: 141
  quarter-on-quarter declines and 43 year-over-year declines.
- **Cross-check with Yahoo (35 overlapping quarters):**
  - revenue, net income, assets, equity, current assets and liabilities:
    35/35;
  - operating income: 33/35 (Tesla; Yahoo shows an adjusted figure);
  - debt: 5/31, as expected, because the EDGAR figure excludes leases.
- **Problems found in the source data and handled:**
  1. Amazon's 2012 10-K tags segment figures as company totals (Q4 2012:
     17.4B instead of 21.3B). Fourth quarters are now always derived as
     annual minus nine-month year-to-date.
  2. Quarters from before a company's first XBRL filing appear only as
     restated comparatives in later filings. For example, Apple's Dec-2008
     and Mar-2009 quarters are restated while Jun-2009 is original, which
     created a fake "decline". Revenue now uses the earliest-filed value,
     and late-filed quarters are excluded.
  3. Picking the largest value across revenue tags preferred later
     restatements. It now picks the earliest filing, and uses the largest
     value only to break ties, which excludes goods-only revenue.
- **Evaluation** (balanced accuracy, permutation p, 200 shuffles):

  | Target | Scheme | 5-ratio model | Seasonal rule |
  |---|---|---|---|
  | Quarter on quarter | walk-forward | 0.53 (p 0.17) | 0.78 (p ≤ 0.005) |
  | | unseen company | 0.52 (p 0.30) | 0.76 (p ≤ 0.005) |
  | | both | 0.51 (p 0.33) | 0.76 (p ≤ 0.005) |
  | Year over year | walk-forward | 0.57 (p 0.0498, borderline) | 0.51 |
  | | unseen company | 0.44 (p 0.90) | 0.50 |
  | | both | 0.46 (p 0.89) | 0.50 |

  **Conclusion:** the phase-4 verdict holds with 14× more data. The ratios
  carry no reliable signal for next-quarter revenue declines. Only the
  calendar does, and it doesn't for the year-over-year target. The one
  year-over-year result below 0.05 is borderline, disappears for unseen
  companies, has 15% precision, and is about what 9 tests produce by
  chance.
- **Report wording fixes:**
  - A p-value of 0.0498 printed as "0.050". It now shows 4 decimals near
    0.05 and is labelled "borderline".
  - The 200-shuffle floor is shown as "≤ 0.005".
  - Every model below 0.05 is listed, together with the expected number of
    chance results.
  - Reports were refreshed from saved results with
    `refresh_report()`, without refitting.

## Multi-agent system and distress model (2026-11-12)

This responds to the team's request:
- better ML performance;
- distress prediction with better features and more data;
- SHAP explanations;
- Quantitative Risk, Market and News agents;
- a LangGraph Manager agent;
- FinTech integration and end-to-end testing.

**Distress model** (`src/ml/distress/`, `python main.py distress`)
- **Data:** SEC XBRL frames, about 740 cached requests. That gives an annual
  panel of 114,304 company-years from 14,884 US filers. Each balance sheet is
  matched to its exact fiscal year end, which worked for 96% of rows. The
  training set has 46,947 company-years from 7,101 operating companies
  (2010–2025).
- **Results on the strictest test** (later years and unseen companies):
  - all companies: gradient boosting AUC 0.913, against 0.834 for the
    persistence baseline;
  - new distress among currently healthy companies: AUC 0.790, against 0.603
    for the best baseline.

  The company-cluster bootstrap intervals for the gain all exclude zero. Full
  report: `reports/distress/`.
- **SHAP:** a global ranking in the report, and per-company drivers in the
  Quantitative Risk Agent.
- **Spot checks:**

  | Company | Score | Rank |
  |---|---|---|
  | Apple | 0.015 | Lowest group |
  | Reliance | 0.044 | Riskier than 10% |
  | Beyond Meat | 0.968 | In distress; riskier than 90% |

**Agents** (`src/agents/`)
- **Quantitative Risk:** the existing analysis plus the distress model and
  SHAP.
- **Market:** volatility, drawdown, beta, returns relative to the index,
  valuation.
- **News:** Yahoo and Google News headlines, sentiment and events; Claude if
  a key is configured, otherwise a keyword lexicon.
- **Manager:** a LangGraph graph (resolve → three agents in parallel →
  manager) that produces the attention level, cross-checks, and a summary
  (Claude or template). Output is `research_report/1.0`.
- **API:** FastAPI (`python main.py api`) with `/health`, `/api/analyze`,
  `/api/research` and `/api/model`. CORS is open.

**Bugs found and fixed while testing**
- In the provider, annual and quarterly balance sheets shared a cache key.
  The distress model received quarterly data and found only 10 of 28 inputs.
  A regression test now covers this.
- Indian companies' size was measured in INR, while the model learned USD.
  Assets are now converted at the live exchange rate.
- Building the SEC panel was slow: a sequential download plus a slow pandas
  lookup. It now downloads in parallel within the rate limit and uses a dict
  lookup, so the build takes about 90 s.
- The test page and the API now answer the browser pane's HEAD check.

**Live check:** `main.py research` returned in 6–12 s.

| Company | Attention level | Why |
|---|---|---|
| Apple | Low | All agents agree there is no stress |
| Reliance | Moderate | 27% price fall and negative news; the Manager notes the conflict with healthy financials |
| Beyond Meat | High | Statements, model and share price all show stress |

The browser test went through FastAPI and returned HTTP 200.

## Tests

`python -m unittest discover -s tests -t .` runs 82 tests, all passing, in
about 15 seconds with no network. Besides the list below, they cover the
distress features, labels, folds and metrics, the Market and News agents,
the attention and cross-check rules, the full LangGraph flow (ok, error and
failing-agent cases, schema-validated), the FastAPI endpoints, and the
provider cache. The earlier tests cover:
- ratio values against hand calculations, and edge cases;
- the leakage and target checks;
- ticker resolution with a fake provider, and provider-failure detection;
- the audit checks and the history merge;
- the splits (no shared dates, only observable labels, no company in both
  sets) and the baselines;
- fast balanced accuracy against scikit-learn;
- end-to-end evaluation on synthetic data: a real signal is detected
  (p < 0.05) and noise is not;
- risk-flag rules: triggers, threshold edges, not-evaluable cases, fallback
  periods, banks;
- ML signal: the score equals the intercept plus the contributions;
  imputation flags; unavailable cases;
- schema: examples validate, and broken payloads are rejected;
- offline end-to-end `analyze` with a fake provider: an ok run plus all six
  error codes;
- EDGAR extraction: Q4 derivation, first-filed values, revenue tag choice,
  point-in-time marking, debt; QoQ and YoY labels; settings and the
  required contact email.
