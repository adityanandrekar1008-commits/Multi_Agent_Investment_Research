

\# AI/ML Project Handoff



\## Project

Multi-Agent Investment Research and Risk Analysis System



\## Completed Components

1\. Quarterly financial data collection.

2\. Financial dataset preparation and cleaning.

3\. Financial feature engineering.

4\. Revenue-decline target creation.

5\. Logistic Regression model with missing-value imputation and feature scaling.

6\. Majority-class baseline comparison.

7\. Chronological and company-wise model validation experiments.

8\. Prediction error inspection.

9\. Explainability prototype using standardized Logistic Regression coefficients.

10\. CSV exports of test predictions and feature coefficients.



\## Important Files



\### Model and explainability

\- src/ml/train\_revenue\_decline\_model.py

\- src/ml/validate\_company\_wise.py

\- src/ml/validate\_date\_grouped.py

\- src/ml/inspect\_prediction\_errors.py

\- src/explainability/explain\_revenue\_decline.py



\### Main dataset

\- data/processed/revenue\_decline\_features.csv



\### Generated outputs

\- reports/revenue\_decline\_baseline\_results.txt

\- reports/company\_wise\_validation\_results.csv

\- reports/date\_grouped\_validation\_results.txt

\- reports/revenue\_decline\_predictions.csv

\- reports/revenue\_decline\_feature\_explanations.csv



\## Latest Explainability Results

\- Updated 2026-11-07: the split is now on a date boundary (the earlier row-position split put 2025-12-31 rows in both training and test).

\- Training observations: 23

\- Test observations: 8 (1 actual decline)

\- Correct test predictions: 5 out of 8; the one actual decline was missed

\- Standardized coefficients are available in the feature-explanations CSV.

\- Full evaluation with baselines, time- and company-aware validation and a permutation test: reports/model\_evaluation.md. No validation scheme shows the model beating chance.

\- Repeated on 429 SEC EDGAR quarters (7 US companies, 2009-2026; reports/sec\_edgar/): still no better than chance on the next-quarter target. A same-calendar-quarter rule scores about 0.77. On the year-over-year target there is one borderline result (p = 0.0498), which does not hold for unseen companies.

\- Output for the FinTech app: docs/output\_schema.md and schemas/company\_analysis.schema.json (company\_analysis/1.0, with rule-based risk flags and a separate experimental ML signal).



\## Limitations

\- The dataset is small.

\- The target indicates a decline in revenue in the next reported period, not bankruptcy or investment loss.

\- The model has not been established as a reliable investment-risk predictor.

\- Model scores are not calibrated probabilities.

\- Missing financial features are imputed, not recovered as actual values.

\- Financial statements may include unusual events that affect ratios.

\- Reporting dates may differ from the dates when information became publicly available.



\## Integration Notes

The FinTech application should treat the current ML outputs as experimental research signals.

It should not present them as guaranteed predictions or personalized investment advice.

A future integration should include clear explanations, data timestamps, and model limitations.



