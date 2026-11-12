# Multi-Agent Investment Research and Risk Analysis — AI/ML part

Enter a listed company (ticker or name), collect its quarterly financial
statements, calculate financial ratios, and check data quality. A
revenue-decline model is included as a **proof of concept only**.

> Informational research output only; not investment advice. The model is
> not a validated risk predictor (see `reports/RUN_LOG.md`).

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Commands

| Command | What it does | Output |
|---|---|---|
| `.venv\Scripts\python main.py analyze RELIANCE.NS` | One company: resolve ticker/name → quarterly statements → ratios → data-quality audit | `outputs/<SYMBOL>_<time>/summary.json`, `quarterly_financials.csv`, `data_quality.md` |
| `.venv\Scripts\python main.py analyze "Reliance Industries"` | Same, by company name | same |
| `.venv\Scripts\python main.py rebuild` | Full quarterly pipeline: download (merged with stored history) → dataset → target → features → audit → model evaluation → explanation | `runs/<time>/` + `run_manifest.json` |
| `.venv\Scripts\python main.py rebuild --from-raw data/raw/quarterly` | Same, offline, from the stored raw statements (reproduces the committed results exactly) | `runs/<time>/` |
| `.venv\Scripts\python main.py audit` | Data-quality audit of the committed datasets | `reports/data_quality_audit.md` / `.json` |
| `.venv\Scripts\python main.py evaluate` | Model evaluation: baselines, time- and company-aware validation, permutation test (~20 s) | `reports/model_evaluation.md` / `.json` / `_predictions.csv` |
| `.venv\Scripts\python main.py edgar` | SEC EDGAR history since ~2009 for the 7 US companies, cross-check with Yahoo, evaluation on next-quarter and year-over-year targets (needs `.env`, see below) | `data/raw/sec_edgar/quarterly/`, `reports/sec_edgar/` |
| `.venv\Scripts\python main.py serve` | Local test page for `analyze` | http://127.0.0.1:8765 |
| `.venv\Scripts\python -m unittest discover -s tests -t .` | Offline unit tests | — |

For `edgar`, create a `.env` file in the project root (it is git-ignored).
The SEC requires a name and a contact email on every request:

```
SEC_EDGAR_USER_AGENT=Multi-Agent Investment Research your.email@example.com
```

`main.py` never overwrites `data/` or the committed reports: each
`analyze`/`rebuild` run writes to a new folder (`outputs/` and `runs/` are
git-ignored), and `audit` needs `--overwrite` to replace its report.

Exit codes for `analyze`: 0 ok, 2 bad input or no data (`empty_input`,
`unknown_company`, `not_a_company`, `no_financial_data`), 3 data-provider
failure (`provider_error`).

### Ticker input

- Exchange-qualified tickers are used as typed: `RELIANCE.NS` (NSE),
  `TCS.BO` (BSE), `AAPL`.
- Bare symbols are tried as typed, then `.NS`, then `.BO`: `TCS` → `TCS.NS`,
  `META` → Meta Platforms (not `META.BO`, an unrelated BSE company). Other
  listings with data are reported. `--prefer-indian` tries `.NS`/`.BO` first.
- Anything else is searched by company name; the match is reported so it
  can be checked.

## Multi-agent research system

```
                        ┌─> Quantitative Risk Agent ─┐
company ─> resolve ─────┼─> Market Agent ────────────┼─> Manager Agent ─> research_report/1.0
 (LangGraph StateGraph) └─> News Agent ──────────────┘   (combined report)
```

| Agent | What it does | Data |
|---|---|---|
| Quantitative Risk | Ratios, 11 rule-based risk flags, the financial-distress model with SHAP explanations | Yahoo statements; model trained on SEC EDGAR |
| Market | Volatility, drawdown, beta, returns vs the index, valuation | Yahoo daily prices |
| News | Recent headlines, sentiment, events (lawsuits, downgrades, layoffs, defaults) | Yahoo Finance news, Google News RSS |
| Manager | Runs the three in parallel; findings by severity; agreement/conflict checks; attention level; written summary | Agent outputs (+ Claude if configured) |

```
python main.py distress            # train/evaluate the distress model (downloads SEC frames once)
python main.py research AAPL       # combined report on the command line
python main.py api                 # HTTP API for the FinTech app: http://127.0.0.1:8000/docs
```

Optional `.env` settings (git-ignored):

```
SEC_EDGAR_USER_AGENT=Project Name your.email@example.com   # needed for SEC downloads
ANTHROPIC_API_KEY=sk-ant-...                               # optional: Claude news classification + summary
```

Without an API key, the News Agent uses a keyword lexicon and the Manager a
template summary; everything else is identical.

## Output for the FinTech app

`analyze` (CLI and `GET /api/analyze?company=...`) returns a stable JSON
result, `company_analysis/1.0`. It has these sections:
- company and source details (provider, retrieval time, available
  statements);
- coverage;
- deterministic `metrics`;
- rule-based `risk_flags`, with their thresholds and "not evaluable"
  states;
- an `ml_signal`, kept separate and labelled experimental;
- data-quality findings, coded `warnings`, coded `error` states, and a
  disclaimer.

Also produced:
- **Integration guide:** [`docs/output_schema.md`](docs/output_schema.md)
  (field reference, units, error and HTTP codes, display rules for the ML
  signal).
- **Contract:** [`schemas/company_analysis.schema.json`](schemas/company_analysis.schema.json).
- **Examples:** [`docs/examples/`](docs/examples/).
- **Readable report:** `report.md`, generated from the same JSON for each
  run.

## Pipeline steps (what `rebuild` runs)

The original scripts still work on their own (`python src/...py`) and
refuse to overwrite existing outputs.

1. `src/data_collection/collect_quarterly_data.py` — quarterly statements
2. `src/data_collection/prepare_quarterly_dataset.py` — combined dataset
3. `src/data_collection/prepare_quarterly_ml_data.py` — modelling base (JPM excluded)
4. `src/data_collection/create_revenue_decline_target.py` — next-quarter revenue decline label
5. `src/ml/prepare_revenue_decline_features.py` — ratio features
6. `src/data_quality/` — audit, ratio/target/leakage checks
7. `src/ml/train_revenue_decline_model.py`, `validate_date_grouped.py`,
   `validate_company_wise.py`, `compare_*feature_sets.py`,
   `inspect_prediction_errors.py` — evaluation
8. `src/explainability/explain_revenue_decline.py` — coefficients and predictions
9. `src/ml/evaluate_revenue_decline.py` — the evaluation to quote (below)

## Financial-distress model (main ML model)

`python main.py distress` → `reports/distress/distress_evaluation.md`,
`models/distress_model.joblib`.

- **Target:** distress in the next fiscal year. Distress means any of:
  - a net loss with negative operating cash flow;
  - negative equity;
  - operating income below interest expense.
- **Data:** SEC EDGAR "frames" for all US filers. That gives 46,947
  company-years from 7,101 companies, fiscal years 2010–2025. Banks,
  insurers and shells are excluded.
- **Features (28):** Altman Z'' and its parts, Piotroski-style profitability
  and cash-flow measures, leverage, liquidity, interest coverage, one-year
  changes, and the current distress state.
- **Models:** gradient boosting (the final model) and logistic regression.
  They are tested against four baselines: base rate, persistence ("distressed
  now → distressed next year"), Altman Z'', and "loss this year".
- **Validation:** training uses only labels known at prediction time. Tested
  on later years, on unseen companies, and on both together. 95% intervals
  come from a bootstrap that resamples whole companies.

| Strictest test (later years and unseen companies) | Gradient boosting | Best baseline | AUC gain (95% CI) |
|---|---|---|---|
| All companies (37% base rate) | **AUC 0.913**, PR AUC 0.866 | Persistence 0.834 | +0.077 to +0.082 |
| New distress, currently healthy companies (13% base rate) | **AUC 0.790** | Loss now 0.603 | +0.176 to +0.195 |

95% of the riskiest 10% of companies by score really were in distress the
next year.

The biggest drivers, by SHAP:
1. whether the company is already in distress;
2. interest coverage;
3. operating cash flow relative to assets;
4. operating margin;
5. company size;
6. retained earnings relative to assets.

**Limits:**
- US SEC filers only. Indian and other companies are scored from Yahoo
  statements, with assets converted to USD.
- The frames data holds the latest filed values, so a few figures may be
  restated.
- "Distress" here is an accounting definition, not bankruptcy.
- Scores are risk ranks, not calibrated probabilities.

## The earlier revenue-decline model: what it is and is not good for

The model is a logistic regression on 5 current-quarter ratios (net and
operating margin, debt/assets, debt/equity, current ratio). It predicts
whether **next quarter's revenue will be lower than this quarter's**.

Evaluation (`reports/model_evaluation.md`, committed data: 31 rows, 8
companies, 8 declines):

| Validation | Model balanced accuracy | Better than chance? | Seasonal rule |
|---|---|---|---|
| Later dates only (walk-forward) | 0.50 | no (p = 0.53) | 0.50 (cannot learn December yet) |
| Unseen companies | 0.53 | no (p = 0.44) | **0.77 (p = 0.02)** |
| Unseen companies and later dates | 0.69 | no (p = 0.13) | 0.50 |

- **Not good for:** risk scoring, investment decisions, or any claim of
  predictive power. No validation scheme shows it beating chance, and on
  unseen companies it does worse than a "same calendar quarter" rule.
- **What the label mostly measures:** seasonality. 5 of 7 December
  quarters are followed by a decline, versus 3 of 24 other quarters.
- **Acceptable use:** an experimental research signal, shown with its
  limitations and the disclaimer. It must not be presented as a probability
  or a recommendation.

### Larger check: SEC EDGAR history (`main.py edgar`)

Yahoo's ~7 quarters are too few, so the same evaluation was repeated on
SEC EDGAR filings for the 7 US companies:
- 429 labelled quarters, 2009–2026, using values as first filed;
- timing from actual filing dates;
- revenue and net income match Yahoo on all 35 overlapping quarters.

Reports are in `reports/sec_edgar/`.

| Target | Validation | 5-ratio model | Seasonal rule |
|---|---|---|---|
| Next quarter vs this quarter (141 declines) | Later dates only | 0.53 (p = 0.17) | **0.78 (p ≤ 0.005)** |
| | Unseen companies | 0.52 (p = 0.30) | **0.76 (p ≤ 0.005)** |
| | Both | 0.51 (p = 0.33) | **0.76 (p ≤ 0.005)** |
| Next quarter vs same quarter last year (43 declines) | Later dates only | 0.57 (p = 0.0498, borderline) | 0.51 |
| | Unseen companies | 0.44 (p = 0.90) | 0.50 |
| | Both | 0.46 (p = 0.89) | 0.50 |

Numbers are balanced accuracy (0.5 = chance) with permutation p-values.

- **Quarter-on-quarter target.** With 14× more data the model is still no
  better than chance. A rule that only looks at the calendar quarter scores
  about 0.77: 81 of 89 December quarters are followed by a decline.
- **Year-over-year target (seasonality removed).** The only result below
  0.05 is borderline and disappears for companies the model has not seen.
  When it predicts a decline, it is right only 15% of the time. With 9
  model/scheme tests, one result near 0.05 is expected by chance.
- **Conclusion:** current-quarter margins and balance-sheet ratios do not
  predict next-quarter revenue declines for these companies. To go further,
  try different inputs (e.g. revenue trends, guidance, market data) and
  more and smaller companies, not more of the same ratios.

## Known limitations

See `reports/RUN_LOG.md`, `reports/data_quality_audit.md` and
`reports/sec_edgar/collection.md`. In short:
- Yahoo Finance gives only ~5–7 recent quarters per company.
- The revenue-decline label is mostly seasonal.
- RELIANCE.NS has no quarterly cash flow.
- SEC EDGAR covers US filers only.
- In the EDGAR data, debt excludes leases, and companies without debt
  filings have their debt ratios filled with the median.

Remaining work (FinTech review of `docs/output_schema.md` and integration,
final polish) is listed in `reports/AI_ML_HANDOFF.md` and the handoff PDF.
