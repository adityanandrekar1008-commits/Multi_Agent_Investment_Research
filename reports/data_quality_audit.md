# Data-quality audit: committed quarterly dataset

- As of: 2026-11-06
- Companies: 9; records: 50
- Findings: 0 errors, 26 warnings, 16 notes

The audit only reads data; no values were changed, filled, or removed. Findings flag things to review; they are not proof of errors.

## Coverage by company

| Ticker | Quarters | Periods | Currency | Revenue | Net inc. | Op. inc. | Assets | Debt | Equity | Cur. assets | Op. CF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AAPL | 5 | 2025-06-30 to 2026-06-30 | USD* | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| AMZN | 5 | 2025-03-31 to 2026-03-31 | USD* | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| GOOGL | 5 | 2025-06-30 to 2026-06-30 | USD* | 100% | 100% | 100% | 100% | 20% | 100% | 100% | 100% |
| JPM | 5 | 2025-06-30 to 2026-06-30 | USD* | 100% | 100% | 0% | 100% | 100% | 100% | 0% | 100% |
| META | 5 | 2025-06-30 to 2026-06-30 | USD* | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MSFT | 5 | 2025-06-30 to 2026-06-30 | USD* | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| NVDA | 5 | 2025-07-31 to 2026-07-31 | USD* | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| RELIANCE.NS | 5 | 2025-03-31 to 2026-06-30 | INR* | 100% | 100% | 100% | 40% | 40% | 40% | 40% | 0% |
| TSLA | 5 | 2025-06-30 to 2026-06-30 | USD* | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

Coverage = share of usable quarters with a value. \* currency inferred from the ticker suffix, not reported by the provider.

## Findings

| Severity | Ticker | Check | Period | Detail |
|---|---|---|---|---|
| warning | all | mixed_currencies |  | Companies report in different currencies (AAPL: USD, AMZN: USD, GOOGL: USD, JPM: USD, META: USD, MSFT: USD, NVDA: USD, RELIANCE.NS: INR, TSLA: USD). Absolute amounts are not comparable across companies; ratios are. |
| warning | AMZN | incomplete_latest_period | 2026-06-30 | The most recent period has no income-statement values (the provider had only partial data when it was collected). Re-collect the data. |
| warning | AMZN | one_off_items | 2025-03-31 | Unusual items equal 16% of net income; margins for this period may not reflect normal operations. |
| warning | AMZN | one_off_items | 2025-09-30 | Unusual items equal 48% of net income; margins for this period may not reflect normal operations. |
| warning | AMZN | one_off_items | 2026-03-31 | Unusual items equal 52% of net income; margins for this period may not reflect normal operations. |
| warning | AMZN | stale_data | 2026-03-31 | Latest period 2026-03-31 is 220 days old; a newer quarter has probably been reported. Re-collect the data. |
| warning | GOOGL | missing_values |  | 'Total Debt' missing in 4 of 5 usable periods (2025-06-30, 2025-09-30, 2025-12-31, 2026-03-31). |
| warning | GOOGL | one_off_items | 2025-09-30 | Unusual items equal 31% of net income; margins for this period may not reflect normal operations. |
| warning | GOOGL | one_off_items | 2026-03-31 | Unusual items equal 59% of net income; margins for this period may not reflect normal operations. |
| warning | GOOGL | one_off_items | 2026-06-30 | Unusual items equal 88% of net income; margins for this period may not reflect normal operations. |
| warning | JPM | missing_values |  | 'Operating Income' missing in 5 of 5 usable periods (2025-06-30, 2025-09-30, 2025-12-31, 2026-03-31, 2026-06-30). |
| warning | JPM | missing_values |  | 'Current Assets' missing in 5 of 5 usable periods (2025-06-30, 2025-09-30, 2025-12-31, 2026-03-31, 2026-06-30). |
| warning | JPM | missing_values |  | 'Current Liabilities' missing in 5 of 5 usable periods (2025-06-30, 2025-09-30, 2025-12-31, 2026-03-31, 2026-06-30). |
| warning | JPM | one_off_items | 2026-06-30 | Unusual items equal 21% of net income; margins for this period may not reflect normal operations. |
| warning | NVDA | one_off_items | 2026-07-31 | Unusual items equal 13% of net income; margins for this period may not reflect normal operations. |
| warning | RELIANCE.NS | missing_values |  | 'Total Assets' missing in 3 of 5 usable periods (2025-06-30, 2025-12-31, 2026-06-30). |
| warning | RELIANCE.NS | missing_values |  | 'Total Debt' missing in 3 of 5 usable periods (2025-06-30, 2025-12-31, 2026-06-30). |
| warning | RELIANCE.NS | missing_values |  | 'Stockholders Equity' missing in 3 of 5 usable periods (2025-06-30, 2025-12-31, 2026-06-30). |
| warning | RELIANCE.NS | missing_values |  | 'Current Assets' missing in 3 of 5 usable periods (2025-06-30, 2025-12-31, 2026-06-30). |
| warning | RELIANCE.NS | missing_values |  | 'Current Liabilities' missing in 3 of 5 usable periods (2025-06-30, 2025-12-31, 2026-06-30). |
| warning | RELIANCE.NS | missing_values |  | 'Operating Cash Flow' missing in 5 of 5 usable periods (2025-03-31, 2025-06-30, 2025-12-31, 2026-03-31, 2026-06-30). |
| warning | RELIANCE.NS | missing_values |  | 'Free Cash Flow' missing in 5 of 5 usable periods (2025-03-31, 2025-06-30, 2025-12-31, 2026-03-31, 2026-06-30). |
| warning | RELIANCE.NS | one_off_items | 2025-06-30 | Unusual items equal 33% of net income; margins for this period may not reflect normal operations. |
| warning | RELIANCE.NS | quarterly_gap | 2025-12-31 | About 1 quarter(s) missing between 2025-06-30 and 2025-12-31 (184 days). |
| warning | TSLA | one_off_items | 2025-09-30 | Unusual items equal 17% of net income; margins for this period may not reflect normal operations. |
| warning | TSLA | one_off_items | 2025-12-31 | Unusual items equal 19% of net income; margins for this period may not reflect normal operations. |
| info | all | mixed_fiscal_calendars |  | Some companies' quarters end in different months (NVDA). Rows with the same Date are not the same economic quarter, and date-based train/test splits mix calendars. |
| info | AAPL | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | AMZN | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | GOOGL | empty_period | 2025-03-31 | Provider returned this older date without income-statement values; it is dropped before modelling. |
| info | GOOGL | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | JPM | empty_period | 2025-03-31 | Provider returned this older date without income-statement values; it is dropped before modelling. |
| info | JPM | financial_company_structure |  | No current assets or operating income reported; the statements look like a bank/financial company. Margin and liquidity ratios are not comparable with other companies. |
| info | JPM | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | META | empty_period | 2025-03-31 | Provider returned this older date without income-statement values; it is dropped before modelling. |
| info | META | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | MSFT | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | NVDA | fiscal_calendar |  | Fiscal quarters end in months [1, 4, 7, 10]; periods are not aligned with calendar quarters used by other companies. |
| info | NVDA | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | RELIANCE.NS | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |
| info | TSLA | empty_period | 2025-03-31 | Provider returned this older date without income-statement values; it is dropped before modelling. |
| info | TSLA | short_history |  | Only 5 usable quarters (enough for one year-over-year comparison only). |

## Ratio formula check

Each stored ratio was recomputed with plain arithmetic, independently of the pipeline code.

| Feature | Formula | Status | Matching rows | Missing |
|---|---|---|---|---|
| Net_Income_Margin | Net Income / Total Revenue (missing if Total Revenue <= 0) | ok | 31/31 | 0 |
| Operating_Margin | Operating Income / Total Revenue (missing if Total Revenue <= 0) | ok | 31/31 | 0 |
| Debt_To_Assets | Total Debt / Total Assets (missing if Total Assets <= 0) | ok | 31/31 | 5 |
| Debt_To_Equity | Total Debt / Stockholders Equity (missing if Stockholders Equity <= 0) | ok | 31/31 | 5 |
| Current_Ratio | Current Assets / Current Liabilities (missing if Current Liabilities <= 0) | ok | 31/31 | 1 |

## Feature timing (leakage) check

- No model feature is a future/outcome column or an identifier, and every feature is a documented current-period ratio.
- Note: `Date` is the fiscal period end. Statements are published weeks later (typically 25-60 days), so a prediction for the next quarter can only be made after the filing date. The provider does not give filing dates, so this is not checked per row.

## Target check

- Definition: 1 if the next reported quarter's Total Revenue is lower than the current quarter's; only transitions 70-110 days apart are labelled.
- Horizon: next reported quarter (about 3 months)
- Labels recomputed: 31; stored: 31; disagreements: 0
