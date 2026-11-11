"""Honest evaluation of the revenue-decline proof of concept.

Target: Next_Period_Revenue_Decline = 1 if the next reported quarter's
revenue is lower than the current quarter's (horizon: one quarter).

Models compared (all evaluated on the same test rows):
- majority_baseline  always predicts the most common training label
- seasonal_baseline  predicts the majority training label for quarters
                     ending in the same calendar month (captures seasonality
                     only, e.g. the usual December -> March drop)
- logreg_5_ratios    the proof-of-concept model (5 ratios, median
                     imputation, scaling, class-balanced logistic regression)
- logreg_margins     the same model with the two margin ratios only

Validation schemes:
- walk_forward              time-aware. For each reporting date, train on
                            rows whose label was already observable (the
                            next quarter's period end is on or before that
                            date), test on that date's rows.
- unseen_company            company-aware. Leave one company out; train on
                            all other companies' rows (any date). Kept for
                            comparison with validate_company_wise.py; it lets
                            the model learn from other companies' future.
- walk_forward_unseen_company  both. Train on other companies' rows whose
                            label was observable at the test date.

Metrics are computed on the pooled out-of-sample predictions of each
scheme: class counts, accuracy, balanced accuracy, precision/recall/F1 for
the decline class, confusion matrix, a bootstrap 95% interval for balanced
accuracy, and a label-permutation p-value (is the model better than chance?).

Run:  python src/ml/evaluate_revenue_decline.py
      python main.py evaluate          (same, from the project root)
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "revenue_decline_features.csv"
TARGET_FILE = (
    PROJECT_ROOT / "data" / "processed" / "quarterly_revenue_decline_target.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports"

TARGET = "Next_Period_Revenue_Decline"
RATIO_FEATURES = [
    "Net_Income_Margin",
    "Operating_Margin",
    "Debt_To_Assets",
    "Debt_To_Equity",
    "Current_Ratio",
]
MARGIN_FEATURES = ["Net_Income_Margin", "Operating_Margin"]

MIN_TRAIN_ROWS = 10
SIGNIFICANCE = 0.05
OUTPUT_NAMES = {
    "markdown": "model_evaluation.md",
    "json": "model_evaluation.json",
    "predictions": "model_evaluation_predictions.csv",
}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class MajorityBaseline:
    def fit(self, X, y):
        self.positive_rate_ = float(np.mean(y))
        self.label_ = int(Counter(y).most_common(1)[0][0])
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)

    def predict_proba(self, X):
        p = np.full(len(X), self.positive_rate_)
        return np.column_stack([1 - p, p])


class SeasonalBaseline:
    """Majority training label among quarters ending in the same calendar
    month; falls back to the overall training majority for unseen months.
    X's first column is the period-end month."""

    def fit(self, X, y):
        months = np.asarray(X)[:, 0]
        y = np.asarray(y)
        self.overall_ = MajorityBaseline().fit(None, y)
        self.rates_ = {m: float(y[months == m].mean()) for m in np.unique(months)}
        return self

    def _rates(self, X):
        return np.array([
            self.rates_.get(month, self.overall_.positive_rate_)
            for month in np.asarray(X)[:, 0]
        ])

    def predict(self, X):
        rates = self._rates(X)
        # Ties (rate exactly 0.5) fall back to the overall majority label.
        return np.where(rates > 0.5, 1,
                        np.where(rates < 0.5, 0, self.overall_.label_))

    def predict_proba(self, X):
        p = self._rates(X)
        return np.column_stack([1 - p, p])


def make_logreg():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        )),
    ])


MODELS = {
    "majority_baseline": {
        "columns": ["Period_Month"],
        "factory": MajorityBaseline,
        "description": "Always predicts the most common training label.",
    },
    "seasonal_baseline": {
        "columns": ["Period_Month"],
        "factory": SeasonalBaseline,
        "description": "Majority training label for quarters ending in the "
                       "same calendar month (seasonality only).",
    },
    "logreg_5_ratios": {
        "columns": RATIO_FEATURES,
        "factory": make_logreg,
        "description": "Proof-of-concept model: 5 ratios, logistic regression.",
    },
    "logreg_margins": {
        "columns": MARGIN_FEATURES,
        "factory": make_logreg,
        "description": "Same model with the two margin ratios only.",
    },
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

TARGET_DEFINITIONS = {
    TARGET: (
        "1 if the next reported quarter's revenue is lower than the current "
        "quarter's; 0 otherwise. Only transitions 70-110 days apart are "
        "labelled.",
        "next reported quarter (about 3 months)",
    ),
    "Next_Period_Revenue_Decline_YoY": (
        "1 if the next reported quarter's revenue is lower than the same "
        "quarter one year earlier (year over year, so seasonality cancels "
        "out); 0 otherwise.",
        "next reported quarter (about 3 months)",
    ),
}


def load_dataset(features_file, target_file=None, target_column=TARGET):
    """Load the feature dataset and attach the timing columns.

    Optional columns used when present (SEC EDGAR dataset):
      Available_Date        when the row's own figures were public (filing)
      Label_Available_Date  when its label became known (next filing)
      Test_Group            rows evaluated together (calendar quarter)
    Defaults: Date, Next_Date and Date.
    Returns (df, notes, revenue_table)."""
    notes = []
    df = pd.read_csv(features_file)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if target_column != TARGET:
        df[TARGET] = df[target_column]
    df = df.dropna(subset=["Date", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    revenue_series = None
    if "Next_Date" in df.columns:
        df["Next_Date"] = pd.to_datetime(df["Next_Date"], errors="coerce")
        if {"Total Revenue", "Next_Revenue"} <= set(df.columns):
            revenue_series = df
    elif target_file is not None and Path(target_file).exists():
        target = pd.read_csv(target_file)
        target["Date"] = pd.to_datetime(target["Date"], errors="coerce")
        target["Next_Date"] = pd.to_datetime(target["Next_Date"], errors="coerce")
        df = df.merge(target[["Ticker", "Date", "Next_Date"]],
                      on=["Ticker", "Date"], how="left")
        revenue_series = target
    if "Next_Date" not in df.columns or df["Next_Date"].isna().any():
        missing = df["Next_Date"].isna() if "Next_Date" in df.columns \
            else pd.Series(True, index=df.index)
        df.loc[missing, "Next_Date"] = df.loc[missing, "Date"] + pd.Timedelta(days=92)
        notes.append(
            f"Next_Date was approximated as Date + 92 days for "
            f"{int(missing.sum())} rows (target file not available)."
        )

    for column, default in (("Available_Date", "Date"),
                            ("Label_Available_Date", "Next_Date"),
                            ("Test_Group", "Date")):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
            df[column] = df[column].fillna(df[default])
        else:
            df[column] = df[default]

    df["Period_Month"] = df["Date"].dt.month
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df, notes, revenue_series


def yoy_target_feasibility(target):
    """How many labels a seasonality-free target would have: next quarter's
    revenue vs the same quarter one year earlier."""
    if target is None:
        return None
    # Every revenue observation used by a valid transition, per company/date.
    revenue = {}
    for _, row in target.iterrows():
        revenue[(row["Ticker"], row["Date"])] = row["Total Revenue"]
        if pd.notna(row.get("Next_Date")):
            revenue[(row["Ticker"], row["Next_Date"])] = row["Next_Revenue"]
    labels = declines = 0
    for _, row in target.iterrows():
        next_date = row["Next_Date"]
        earlier = [
            value for (ticker, date), value in revenue.items()
            if ticker == row["Ticker"] and pd.notna(next_date)
            and 350 <= (next_date - date).days <= 380 and pd.notna(value)
        ]
        if earlier and pd.notna(row["Next_Revenue"]):
            labels += 1
            declines += int(row["Next_Revenue"] < earlier[0])
    return {"labels": labels, "declines": declines}


# --------------------------------------------------------------------------
# Validation schemes
# --------------------------------------------------------------------------

def walk_forward_folds(df, unseen_company=False):
    """(train_index, test_index) pairs, one per test group (reporting date,
    or calendar quarter for EDGAR data) and, for unseen_company, per company
    within it. A training row is usable only if its label was already known
    when the test rows' figures became available:
    Label_Available_Date <= earliest Available_Date of the test rows."""
    def column(name, default):
        return (df[name] if name in df.columns else df[default]).to_numpy()

    groups = column("Test_Group", "Date")
    available = column("Available_Date", "Date")
    label_available = column("Label_Available_Date", "Next_Date")
    tickers = df["Ticker"].to_numpy()
    folds = []
    for group in np.unique(groups):
        test_index = np.flatnonzero(groups == group)
        if unseen_company:
            for ticker in np.unique(tickers[test_index]):
                rows = test_index[tickers[test_index] == ticker]
                cutoff = available[rows].min()
                folds.append((np.flatnonzero((label_available <= cutoff)
                                             & (tickers != ticker)), rows))
        else:
            cutoff = available[test_index].min()
            folds.append((np.flatnonzero(label_available <= cutoff), test_index))
    return folds


def leave_company_out_folds(df):
    tickers = df["Ticker"].to_numpy()
    return [
        (np.flatnonzero(tickers != ticker), np.flatnonzero(tickers == ticker))
        for ticker in np.unique(tickers)
    ]


def _skip_reason(y_train):
    if len(y_train) < MIN_TRAIN_ROWS:
        return f"fewer than {MIN_TRAIN_ROWS} training rows"
    if len(np.unique(y_train)) < 2:
        return "training rows contain only one class"
    return None


def model_matrices(df, models):
    return {m: df[MODELS[m]["columns"]].to_numpy(dtype=float) for m in models}


def run_folds(folds, matrices, y, models):
    """Fit and predict every fold. Returns {model: (row_index, predicted,
    score)} arrays plus {row_index: skip reason} and per-row training info."""
    out = {m: ([], [], []) for m in models}
    skipped, train_info = {}, {}
    for train_index, test_index in folds:
        y_train = y[train_index]
        reason = _skip_reason(y_train)
        if reason:
            skipped.update({int(i): reason for i in test_index})
            continue
        for i in test_index:
            train_info[int(i)] = (len(train_index), int(y_train.sum()))
        for model_name in models:
            model = MODELS[model_name]["factory"]()
            X = matrices[model_name]
            model.fit(X[train_index], y_train)
            rows, predicted, score = out[model_name]
            rows.extend(test_index.tolist())
            predicted.extend(model.predict(X[test_index]).tolist())
            score.extend(model.predict_proba(X[test_index])[:, 1].tolist())
    return out, skipped, train_info


def fast_balanced_accuracy(y_true, y_pred):
    """Binary balanced accuracy, (recall of 1 + recall of 0) / 2. Same value
    as sklearn's balanced_accuracy_score without its per-call overhead
    (used thousands of times by the bootstrap and permutation test).
    Returns None when only one class is present."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positives = y_true == 1
    negatives = ~positives
    if not positives.any() or not negatives.any():
        return None
    return 0.5 * ((y_pred[positives] == 1).mean() + (y_pred[negatives] == 0).mean())


def pooled_balanced_accuracy(y, rows, predicted):
    return fast_balanced_accuracy(y[np.asarray(rows, dtype=int)], predicted)


SCHEMES = {
    "walk_forward": {
        "folds": lambda df: walk_forward_folds(df),
        "description": "Time-aware: train only on labels already observable "
                       "at the test date; test on that date's rows.",
    },
    "unseen_company": {
        "folds": leave_company_out_folds,
        "description": "Company-aware: leave one company out, train on the "
                       "others at any date (can learn from the future of "
                       "other companies).",
    },
    "walk_forward_unseen_company": {
        "folds": lambda df: walk_forward_folds(df, unseen_company=True),
        "description": "Time- and company-aware (strictest): train only on "
                       "other companies' labels observable at the test date.",
    },
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    both = len(set(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "actual_declines": int(y_true.sum()),
        "actual_no_decline": int(len(y_true) - y_true.sum()),
        "predicted_declines": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred)) if both else None
        ),
        "precision_decline": float(precision[0]),
        "recall_decline": float(recall[0]),
        "f1_decline": float(f1[0]),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_interval(y_true, y_pred, n_resamples, seed):
    """95% percentile interval for balanced accuracy over resampled rows."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(set(y_true)) < 2 or n_resamples <= 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(y_true), len(y_true))
        value = fast_balanced_accuracy(y_true[index], y_pred[index])
        if value is not None:
            values.append(value)
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _permutation_scores(folds, matrices, shuffled, models):
    warnings.simplefilter("ignore")  # worker processes; real-label fits are logged
    out, _, _ = run_folds(folds, matrices, shuffled, models)
    return {
        m: pooled_balanced_accuracy(shuffled, out[m][0], out[m][1])
        for m in models
    }


def permutation_pvalues(folds, matrices, y, models, observed, n_permutations,
                        seed, n_jobs=-1):
    """Shuffle the labels, rerun the whole scheme (same folds), and count how
    often a model scores at least as well as on the real labels. Shuffles
    are drawn up front from `seed`, so results do not depend on n_jobs."""
    if n_permutations <= 0:
        return {m: None for m in models}
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    shuffles = [rng.permutation(y) for _ in range(n_permutations)]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_permutation_scores)(folds, matrices, s, models) for s in shuffles
    )
    at_least = {m: 0 for m in models}
    valid = {m: 0 for m in models}
    for score in scores:
        for model_name in models:
            value = score[model_name]
            if value is None or observed.get(model_name) is None:
                continue
            valid[model_name] += 1
            at_least[model_name] += int(value >= observed[model_name] - 1e-12)
    return {
        m: (None if not valid[m] else (1 + at_least[m]) / (1 + valid[m]))
        for m in models
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(features_file=None, target_file=None, output_dir=None,
             n_permutations=200, n_bootstrap=2000, seed=42, overwrite=False,
             n_jobs=-1, schemes=None, target_column=TARGET,
             dataset_label="Yahoo Finance quarterly data (committed)",
             echo=print):
    """Run the evaluation and write the report files. `schemes` limits the
    validation schemes (default: all of SCHEMES); `target_column` selects
    the label (see TARGET_DEFINITIONS)."""
    selected = {name: SCHEMES[name] for name in (schemes or SCHEMES)}
    features_file = Path(features_file or INPUT_FILE)
    target_file = Path(target_file or TARGET_FILE)
    output_dir = Path(output_dir or OUTPUT_DIR)
    paths = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    if not overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError(
            f"{OUTPUT_NAMES['markdown']} already exists in {output_dir}; "
            "use overwrite=True (--overwrite) to replace it."
        )

    df, notes, target_table = load_dataset(features_file, target_file,
                                           target_column)
    models = list(MODELS)
    definition, horizon = TARGET_DEFINITIONS[target_column]

    results = {
        "dataset": {
            "label": dataset_label,
            "features_file": str(features_file),
            "rows": int(len(df)),
            "companies": sorted(df["Ticker"].unique().tolist()),
            "date_range": [str(df["Date"].min().date()),
                           str(df["Date"].max().date())],
            "distinct_dates": int(df["Date"].nunique()),
            "class_counts": {
                "no_decline": int((df[TARGET] == 0).sum()),
                "decline": int((df[TARGET] == 1).sum()),
            },
            "declines_by_company": {
                t: [int(g[TARGET].sum()), int(len(g))]
                for t, g in df.groupby("Ticker")
            },
            "decline_rate_by_period_month": {
                int(m): {"rows": int(len(g)), "declines": int(g[TARGET].sum())}
                for m, g in df.groupby("Period_Month")
            },
            "yoy_target_feasibility": yoy_target_feasibility(target_table),
            "timing": (
                "filing dates" if (df["Available_Date"] != df["Date"]).any()
                else "period-end dates (filing dates not available)"
            ),
        },
        "target": {
            "name": target_column,
            "definition": definition,
            "horizon": horizon,
        },
        "models": {m: MODELS[m]["description"] for m in models},
        "schemes": {},
        "settings": {
            "min_train_rows": MIN_TRAIN_ROWS,
            "n_permutations": n_permutations,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        "notes": notes,
    }

    all_predictions = []
    y = df[TARGET].to_numpy()
    matrices = model_matrices(df, models)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for scheme_name, scheme in selected.items():
            echo(f"  {scheme_name}: fitting...")
            folds = scheme["folds"](df)
            out, skipped, train_info = run_folds(folds, matrices, y, models)
            records = []
            for model_name in models:
                for i, p, s in zip(*out[model_name]):
                    n_train, train_declines = train_info[i]
                    records.append({
                        "model": model_name, "Ticker": df.at[i, "Ticker"],
                        "Date": df.at[i, "Date"],
                        "group": df.at[i, "Test_Group"], "actual": int(y[i]),
                        "predicted": int(p), "score": float(s),
                        "n_train": n_train, "train_declines": train_declines,
                    })
            observed = {
                m: pooled_balanced_accuracy(y, out[m][0], out[m][1])
                for m in models
            }
            echo(f"  {scheme_name}: permutation test ({n_permutations} shuffles)...")
            pvalues = permutation_pvalues(
                folds, matrices, y,
                [m for m in models if m != "majority_baseline"],
                observed, n_permutations, seed, n_jobs=n_jobs,
            )
            scheme_result = {
                "description": scheme["description"],
                "evaluated_rows": len({(r["Ticker"], r["Date"]) for r in records}),
                "skipped_rows": len(skipped),
                "skip_reasons": dict(Counter(skipped.values())),
                "models": {},
            }
            for model_name in models:
                rows = [r for r in records if r["model"] == model_name]
                y_true = [r["actual"] for r in rows]
                y_pred = [r["predicted"] for r in rows]
                metrics = compute_metrics(y_true, y_pred) if rows else {"n": 0}
                if rows:
                    metrics["balanced_accuracy_95ci"] = bootstrap_interval(
                        y_true, y_pred, n_bootstrap, seed
                    )
                    metrics["permutation_p_value"] = pvalues.get(model_name)
                scheme_result["models"][model_name] = metrics
            if scheme_name == "walk_forward":
                scheme_result["per_date"] = _per_group(records, "group")
            if scheme_name == "unseen_company":
                scheme_result["per_company"] = _per_group(records, "Ticker")
            results["schemes"][scheme_name] = scheme_result
            for r in records:
                all_predictions.append({"scheme": scheme_name, **r})

    results["fit_warnings"] = dict(Counter(
        f"{w.category.__name__}: {str(w.message).splitlines()[0][:160]}"
        for w in caught
    ))
    results["conclusions"] = conclusions(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(all_predictions)
    if not predictions.empty:
        predictions["Date"] = pd.to_datetime(predictions["Date"]).dt.date
    predictions.to_csv(paths["predictions"], index=False)
    paths["json"].write_text(json.dumps(results, indent=2, default=str),
                             encoding="utf-8")
    paths["markdown"].write_text(render_markdown(results), encoding="utf-8")
    results["files"] = {k: str(v) for k, v in paths.items()}
    return results


def _per_group(records, key):
    groups = {}
    for r in records:
        value = r[key]
        label = str(pd.Timestamp(value).date()) if key in ("Date", "group") else value
        entry = groups.setdefault(label, {
            "rows": set(), "declines": 0, "n_train": [], "correct": Counter(),
        })
        row_id = (r["Ticker"], r["Date"])
        if row_id not in entry["rows"]:
            entry["rows"].add(row_id)
            entry["declines"] += r["actual"]
            entry["n_train"].append(r["n_train"])
        entry["correct"][r["model"]] += int(r["actual"] == r["predicted"])
    return {
        label: {
            "test_rows": len(e["rows"]),
            "actual_declines": e["declines"],
            "train_rows": (min(e["n_train"]), max(e["n_train"])),
            "correct": dict(e["correct"]),
        }
        for label, e in sorted(groups.items())
    }


# --------------------------------------------------------------------------
# Conclusions and report
# --------------------------------------------------------------------------

def _fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_p(p, n_permutations):
    """p-value text. Shows the floor of the permutation test as '<=' and
    uses 4 decimals near 0.05 so 0.0498 is not printed as 0.050."""
    if p is None:
        return "n/a"
    floor = 1 / (1 + n_permutations) if n_permutations else 0
    if floor and p <= floor + 1e-12:
        return f"<= {floor:.3f} (smallest possible with {n_permutations} shuffles)"
    if 0.04 <= p < 0.06:
        return f"{p:.4f}"
    return f"{p:.3f}"


def _p_text(p, n_permutations):
    text = _fmt_p(p, n_permutations)
    return text if text.startswith("<=") or text == "n/a" else f"= {text}"


def _chance_verdict(p):
    if p is None or p >= SIGNIFICANCE:
        return "not distinguishable from chance"
    if p >= 0.04:
        return "only borderline better than chance"
    return "better than chance"


def conclusions(results):
    """Plain-language findings, derived only from the numbers above."""
    lines = []
    beats_chance_anywhere = False
    status = {}
    n_perm = results["settings"]["n_permutations"]
    for scheme_name, scheme in results["schemes"].items():
        model = scheme["models"].get("logreg_5_ratios", {})
        seasonal = scheme["models"].get("seasonal_baseline", {})
        ba, p = model.get("balanced_accuracy"), model.get("permutation_p_value")
        s_ba, s_p = seasonal.get("balanced_accuracy"), seasonal.get("permutation_p_value")
        ci = model.get("balanced_accuracy_95ci")
        if ba is None:
            lines.append(f"{scheme_name}: not enough evaluable rows.")
            continue
        better_than_chance = p is not None and p < SIGNIFICANCE
        beats_chance_anywhere |= better_than_chance
        verdict = _chance_verdict(p)
        comparison = (
            "worse than" if s_ba is not None and ba < s_ba - 1e-9
            else "no better than" if s_ba is not None and ba <= s_ba + 1e-9
            else "better than"
        )
        status[scheme_name] = (better_than_chance, comparison == "better than")
        ci_text = f", 95% interval {_fmt(ci[0])}-{_fmt(ci[1])}" if ci else ""
        lines.append(
            f"{scheme_name}: the 5-ratio model has balanced accuracy "
            f"{_fmt(ba)}{ci_text} (permutation p {_p_text(p, n_perm)}), "
            f"{verdict}; it is {comparison} the seasonal rule ({_fmt(s_ba)}, "
            f"p {_p_text(s_p, n_perm)}) on {model['n']} test rows with "
            f"{model['actual_declines']} declines."
        )

    # Every (scheme, model) test, so other models are not silently skipped.
    tests = [
        (scheme_name, model_name, m.get("balanced_accuracy"),
         m.get("permutation_p_value"))
        for scheme_name, scheme in results["schemes"].items()
        for model_name, m in scheme["models"].items()
        if model_name != "majority_baseline" and m.get("permutation_p_value") is not None
    ]
    others = [
        f"{model} in {scheme} (balanced accuracy {_fmt(ba)}, p {_p_text(p, n_perm)})"
        for scheme, model, ba, p in tests
        if p < SIGNIFICANCE and model not in ("logreg_5_ratios", "seasonal_baseline")
    ]
    if others:
        lines.append("Other models below p = 0.05: " + "; ".join(others) + ".")
    if len(tests) > 1:
        expected = len(tests) * SIGNIFICANCE
        significant = sum(1 for *_, p in tests if p < SIGNIFICANCE)
        strong = sum(1 for *_, p in tests if p <= 0.01)
        lines.append(
            f"Multiple tests: {len(tests)} model/scheme combinations were "
            f"tested; at the 5% level about {expected:.1f} would fall below "
            f"0.05 by chance alone. {significant} did, {strong} of them with "
            "p <= 0.01, which chance rarely produces. A single result "
            "between 0.01 and 0.05 is weak evidence."
        )

    months = results["dataset"]["decline_rate_by_period_month"]
    december = months.get(12)
    if december and december["rows"]:
        others = [v for m, v in months.items() if m != 12]
        other_rows = sum(v["rows"] for v in others)
        other_declines = sum(v["declines"] for v in others)
        dec_rate = december["declines"] / december["rows"]
        other_rate = other_declines / other_rows if other_rows else 0
        seasonal = dec_rate > 2 * other_rate and dec_rate > 0.3
        lines.append(
            f"Seasonality: {december['declines']} of {december['rows']} "
            f"December quarters are followed by a decline, versus "
            f"{other_declines} of {other_rows} other quarters."
            + (" Much of the label is the normal December-to-March drop."
               if seasonal else "")
        )

    feasibility = results["dataset"].get("yoy_target_feasibility")
    is_yoy = results["target"]["name"].endswith("_YoY")
    if feasibility is not None and not is_yoy:
        if feasibility["labels"] < 50:
            lines.append(
                "A seasonality-free target (next quarter vs the same quarter "
                f"a year earlier) would have only {feasibility['labels']} "
                f"labels with {feasibility['declines']} declines in this "
                "data: it needs a longer history."
            )
        else:
            lines.append(
                "A seasonality-free (year-over-year) version of the target "
                f"has {feasibility['labels']} labels with "
                f"{feasibility['declines']} declines in this data; it is "
                "evaluated separately."
            )

    strict = status.get("walk_forward_unseen_company")
    if strict and strict[0] and strict[1]:
        overall = (
            "the model is better than chance and better than the seasonal "
            "rule under the strictest validation (unseen companies, later "
            "dates only). Treat it as a research signal for a few large "
            "companies: it is not calibrated, not a risk score and not "
            "investment advice."
        )
    elif beats_chance_anywhere:
        overall = (
            "the model beats chance in at least one scheme but not "
            "consistently, or not by more than the seasonal rule under the "
            "strictest validation. Do not rely on it; treat its output as an "
            "experimental research signal only."
        )
    else:
        overall = (
            "there is no evidence that the model predicts revenue declines "
            "better than chance. Treat its output as an experimental research "
            "signal only, never as a risk score or investment advice."
        )
    lines.append("Overall: " + overall)
    return lines


def render_markdown(results):
    data = results["dataset"]
    out = [
        "# Revenue-decline model: evaluation",
        "",
        "Generated by `src/ml/evaluate_revenue_decline.py`. Proof of concept "
        "only; not investment advice.",
        "",
        "## Conclusions",
        "",
    ]
    out += [f"- {line}" for line in results["conclusions"]]
    out += [
        "",
        "## Target",
        "",
        f"- `{results['target']['name']}`: {results['target']['definition']}",
        f"- Horizon: {results['target']['horizon']}.",
        "- Features use only the current quarter's statements. Training rows "
        "are limited to labels already known when the test rows' figures "
        "became available. "
        + ("Timing uses actual SEC filing dates." if data.get("timing") == "filing dates"
           else "Filing dates are not available, so period-end dates are "
                "used; statements are published weeks later, so this is an "
                "approximation."),
        "",
        "## Data",
        "",
        f"- Dataset: {data.get('label', '')}",
        "",
        f"- Rows: {data['rows']} from {len(data['companies'])} companies "
        f"({', '.join(data['companies'])}); {data['distinct_dates']} reporting "
        f"dates, {data['date_range'][0]} to {data['date_range'][1]}.",
        f"- Classes: {data['class_counts']['decline']} declines, "
        f"{data['class_counts']['no_decline']} no decline.",
        "",
        "Decline rate by the month the quarter ends:",
        "",
        "| Month | Rows | Declines | Rate |",
        "|---|---|---|---|",
    ]
    for month, v in data["decline_rate_by_period_month"].items():
        rate = v["declines"] / v["rows"] if v["rows"] else 0
        out.append(f"| {month} | {v['rows']} | {v['declines']} | {rate:.0%} |")
    out += [
        "",
        "Declines by company: " + ", ".join(
            f"{t} {d}/{n}" for t, (d, n) in data["declines_by_company"].items()
        ) + ".",
        "",
        "## Results",
        "",
        "Metrics are pooled over every out-of-sample prediction in a scheme. "
        "All models in a scheme are tested on the same rows. Balanced "
        "accuracy 0.5 = chance. The permutation p-value is the share of "
        f"{results['settings']['n_permutations']} label shuffles that scored "
        "at least as well; below 0.05 suggests real signal.",
        "",
    ]
    for scheme_name, scheme in results["schemes"].items():
        out += [
            f"### {scheme_name}",
            "",
            scheme["description"],
            f"Evaluated rows: {scheme['evaluated_rows']}; skipped: "
            f"{scheme['skipped_rows']}"
            + (f" ({', '.join(f'{k}: {v}' for k, v in scheme['skip_reasons'].items())})"
               if scheme["skip_reasons"] else "") + ".",
            "",
            "| Model | n | Declines (actual/predicted) | Accuracy | Balanced acc. "
            "| 95% interval | p-value | Precision | Recall | F1 | TN/FP/FN/TP |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for model_name, m in scheme["models"].items():
            if not m.get("n"):
                out.append(f"| {model_name} | 0 | | | | | | | | | |")
                continue
            ci = m.get("balanced_accuracy_95ci")
            cm = m["confusion_matrix"]
            out.append(
                f"| {model_name} | {m['n']} | {m['actual_declines']}/"
                f"{m['predicted_declines']} | {_fmt(m['accuracy'])} | "
                f"{_fmt(m['balanced_accuracy'])} | "
                f"{(_fmt(ci[0]) + '-' + _fmt(ci[1])) if ci else 'n/a'} | "
                f"{_fmt_p(m.get('permutation_p_value'), results['settings']['n_permutations']).split(' (')[0]} | "
                f"{_fmt(m['precision_decline'])} | {_fmt(m['recall_decline'])} | "
                f"{_fmt(m['f1_decline'])} | {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']} |"
            )
        out.append("")
        detail_key = "per_date" if "per_date" in scheme else \
            "per_company" if "per_company" in scheme else None
        if detail_key:
            label = "Test date" if detail_key == "per_date" else "Company"
            out += [
                f"{label} detail (correct predictions per model):",
                "",
                f"| {label} | Test rows | Declines | Train rows | "
                + " | ".join(results["models"]) + " |",
                "|---|---|---|---|" + "---|" * len(results["models"]),
            ]
            for key, d in scheme[detail_key].items():
                low, high = d["train_rows"]
                train = str(low) if low == high else f"{low}-{high}"
                out.append(
                    f"| {key} | {d['test_rows']} | {d['actual_declines']} | {train} | "
                    + " | ".join(str(d["correct"].get(m, 0)) for m in results["models"])
                    + " |"
                )
            out.append("")

    out += [
        "## Models",
        "",
    ]
    out += [f"- `{name}`: {desc}" for name, desc in results["models"].items()]
    out += [
        "",
        "## Notes",
        "",
        f"- Minimum training rows: {results['settings']['min_train_rows']}; "
        "a test row is skipped when its training set is smaller or has only "
        "one class (early dates cannot be evaluated).",
        "- `validate_company_wise.py` trains on other companies at any date, "
        "including dates after the test rows; `unseen_company` reproduces "
        "that setup for comparison, the stricter schemes do not.",
        "- `validate_date_grouped.py`, `train_revenue_decline_model.py` and "
        "`explain_revenue_decline.py` split on a date boundary but do not hold "
        "back labels that are not yet observable (e.g. 2025-12-31 labels use "
        "March 2026 revenue while NVDA's 2026-01-31 rows are in test). "
        "`walk_forward` does.",
        "- In the time-aware schemes the seasonal rule usually scores 0.50 "
        "because the data has no earlier December quarter whose outcome was "
        "already known at the first December test date, so it cannot learn "
        "the December pattern yet. This reflects the short history, not an "
        "absence of seasonality (see the decline-rate table).",
        "- With this little data, a single train/test split (as in those "
        "scripts) can swing by 0.25 balanced accuracy from one test case; "
        "use the pooled results above.",
    ]
    out += [f"- {note}" for note in results["notes"]]
    if results["fit_warnings"]:
        out += ["", "Warnings raised while fitting (count):", ""]
        out += [f"- {msg} ({count})" for msg, count in results["fit_warnings"].items()]
    out.append("")
    return "\n".join(out)


def refresh_report(output_dir):
    """Recompute the conclusions and rewrite model_evaluation.md/.json from
    the saved results, without refitting (e.g. after wording changes)."""
    output_dir = Path(output_dir)
    json_path = output_dir / OUTPUT_NAMES["json"]
    results = json.loads(json_path.read_text(encoding="utf-8"))
    results.pop("files", None)
    results["conclusions"] = conclusions(results)
    json_path.write_text(json.dumps(results, indent=2, default=str),
                         encoding="utf-8")
    (output_dir / OUTPUT_NAMES["markdown"]).write_text(
        render_markdown(results), encoding="utf-8")
    return results


def main():
    try:
        results = evaluate(overwrite=False)
    except FileExistsError as exc:
        print(exc)
        return
    print("\n".join(results["conclusions"]))
    print("\nReport:", results["files"]["markdown"])


if __name__ == "__main__":
    main()
