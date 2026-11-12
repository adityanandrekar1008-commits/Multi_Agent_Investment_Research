"""Financial-distress model: evaluation, final training, SHAP, scoring.

    python main.py distress            # build data, evaluate, train, explain

Models: logistic regression and gradient boosting (HistGradientBoosting,
handles missing values natively). Baselines: base rate, persistence
("distressed now -> distressed next year"), Altman Z'' (lower = riskier)
and "loss this year".

Validation (same idea as the revenue model):
  walk_forward           train only on labels known before the test year
  unseen_company         5 folds by company, any year
  walk_forward_unseen    both: later years and companies never seen

Metrics on pooled out-of-sample scores: ROC AUC, average precision (PR AUC),
precision and recall in the riskiest 10%, balanced accuracy at 0.5, with
company-cluster bootstrap 95% intervals, for all rows and for companies
not in distress now (new distress).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.distress.features import (
    FEATURE_DESCRIPTIONS, FEATURES, GROUP, TARGET, build_training_set,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_FILE = PROJECT_ROOT / "models" / "distress_model.joblib"
REPORT_DIR = PROJECT_ROOT / "reports" / "distress"
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "distress_training_set.csv"

N_FOLDS = 5
FIRST_TEST_YEAR = 2014
N_BOOTSTRAP = 200
SEED = 42


# ------------------------------------------------------------------ models

def make_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
    ])


def make_gbm():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED,
    )


MODELS = {"logistic": make_logistic, "gradient_boosting": make_gbm}
BASELINES = {
    "base_rate": lambda df: np.zeros(len(df)),
    "persistence": lambda df: df["distress_now"].fillna(0).to_numpy(),
    "altman_z2": lambda df: -df["altman_z2"].fillna(df["altman_z2"].median()).to_numpy(),
    "loss_now": lambda df: df["loss_now"].fillna(0).to_numpy(),
}


# ------------------------------------------------------------------ folds

def company_folds(df, n=N_FOLDS):
    companies = np.array(sorted(df[GROUP].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(companies)
    fold_of = {c: i % n for i, c in enumerate(companies)}
    return df[GROUP].map(fold_of).to_numpy()


def make_folds(df, scheme):
    years = df["year"].to_numpy()
    available = df["available_date"].to_numpy()
    label_available = df["label_available_date"].to_numpy()
    fold = company_folds(df)
    folds = []
    if scheme == "unseen_company":
        for k in range(N_FOLDS):
            folds.append((np.flatnonzero(fold != k), np.flatnonzero(fold == k)))
        return folds
    for year in range(FIRST_TEST_YEAR, int(years.max()) + 1):
        test_year = years == year
        if not test_year.any():
            continue
        cutoff = available[test_year].min()
        known = label_available <= cutoff
        if scheme == "walk_forward":
            folds.append((np.flatnonzero(known), np.flatnonzero(test_year)))
        else:
            for k in range(N_FOLDS):
                folds.append((np.flatnonzero(known & (fold != k)),
                              np.flatnonzero(test_year & (fold == k))))
    return folds


# ------------------------------------------------------------------ metrics

def _top_decile(y, s):
    k = max(1, int(round(0.1 * len(s))))
    top = np.argsort(-s, kind="stable")[:k]
    positives = y.sum()
    return float(y[top].mean()), (float(y[top].sum() / positives) if positives else None)


def metrics(y, s, threshold=0.5):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "base_rate": float(y.mean()) if len(y) else None}
    if len(np.unique(y)) < 2:
        return out
    out["roc_auc"] = float(roc_auc_score(y, s))
    out["pr_auc"] = float(average_precision_score(y, s))
    out["top10_precision"], out["top10_recall"] = _top_decile(y, s)
    pred = s >= threshold
    out["balanced_accuracy"] = float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))
    return out


def cluster_bootstrap(y, scores, groups, n=N_BOOTSTRAP, seed=SEED, against=None):
    """95% intervals for ROC AUC per model, resampling whole companies.
    `against` names a model whose AUC is subtracted (paired difference)."""
    y = np.asarray(y)
    unique = np.unique(groups)
    index_by_group = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in scores}
    for _ in range(n):
        chosen = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        base = roc_auc_score(y[idx], scores[against][idx]) if against else 0
        for m, s in scores.items():
            samples[m].append(roc_auc_score(y[idx], s[idx]) - base)
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for m, v in samples.items() if v}


# ------------------------------------------------------------------ evaluation

def run_scheme(df, scheme, echo=print):
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy().astype(int)
    scores = {m: np.full(len(df), np.nan) for m in MODELS}
    folds = make_folds(df, scheme)
    start = time.time()
    for i, (train, test) in enumerate(folds):
        if len(train) < 500 or len(np.unique(y[train])) < 2 or len(test) == 0:
            continue
        for name, factory in MODELS.items():
            model = factory().fit(X[train], y[train])
            scores[name][test] = model.predict_proba(X[test])[:, 1]
        if (i + 1) % 10 == 0:
            echo(f"    {scheme}: {i + 1}/{len(folds)} folds ({time.time() - start:.0f}s)")
    evaluated = ~np.isnan(scores["logistic"])
    for name, rule in BASELINES.items():
        scores[name] = rule(df).astype(float)
    return evaluated, scores


def evaluate(df, echo=print):
    y = df[TARGET].to_numpy().astype(int)
    groups = df[GROUP].to_numpy()
    healthy = (df["distress_now"] == 0).to_numpy()
    results = {}
    predictions = []
    for scheme in ["walk_forward", "unseen_company", "walk_forward_unseen_company"]:
        echo(f"  evaluating {scheme}...")
        evaluated, scores = run_scheme(df, scheme, echo)
        scheme_result = {"evaluated_rows": int(evaluated.sum())}
        for subset_name, mask in [("all", evaluated), ("healthy_now", evaluated & healthy)]:
            subset = {}
            for name, s in scores.items():
                threshold = 0.5 if name in MODELS else (
                    0.5 if name in ("persistence", "loss_now") else np.inf)
                subset[name] = metrics(y[mask], s[mask], threshold)
            best_baseline = max(
                (b for b in BASELINES if b != "base_rate" and subset[b].get("roc_auc")),
                key=lambda b: subset[b]["roc_auc"])
            ci = cluster_bootstrap(y[mask], {k: v[mask] for k, v in scores.items()}, groups[mask])
            diff = cluster_bootstrap(y[mask], {k: v[mask] for k, v in scores.items()
                                               if k in MODELS or k == best_baseline},
                                     groups[mask], against=best_baseline)
            for name in subset:
                if name in ci:
                    subset[name]["roc_auc_95ci"] = ci[name]
                if name in MODELS and name in diff:
                    subset[name]["auc_minus_best_baseline_95ci"] = diff[name]
            subset["_best_baseline"] = best_baseline
            scheme_result[subset_name] = subset
        results[scheme] = scheme_result
        frame = df.loc[evaluated, [GROUP, "company", "fiscal_year_end", TARGET, "distress_now"]].copy()
        frame["scheme"] = scheme
        for name in MODELS:
            frame[f"score_{name}"] = scores[name][evaluated]
        predictions.append(frame)
    return results, pd.concat(predictions, ignore_index=True)


def conclusions(results):
    lines = []
    for scheme, r in results.items():
        for subset in ("all", "healthy_now"):
            s = r[subset]
            best_b = s["_best_baseline"]
            gbm, lr = s["gradient_boosting"], s["logistic"]
            better = max(("gradient_boosting", "logistic"), key=lambda m: s[m].get("roc_auc", 0))
            diff = s[better].get("auc_minus_best_baseline_95ci")
            if diff and diff[0] > 0:
                verdict = f"better than the best baseline ({best_b}); the 95% interval of the AUC gain is {diff[0]:+.3f} to {diff[1]:+.3f}"
            elif diff and diff[1] < 0:
                verdict = f"worse than the best baseline ({best_b})"
            else:
                verdict = f"not clearly different from the best baseline ({best_b})"
            label = "all companies" if subset == "all" else "companies not in distress now (new distress)"
            lines.append(
                f"{scheme}, {label}: gradient boosting AUC {gbm.get('roc_auc', float('nan')):.3f}, "
                f"logistic {lr.get('roc_auc', float('nan')):.3f}, {best_b} "
                f"{s[best_b].get('roc_auc', float('nan')):.3f} (base rate {s[better]['base_rate']:.1%}); "
                f"the better model ({better}) is {verdict}."
            )
    return lines


# ------------------------------------------------------------------ final model + SHAP

def train_final(df, results):
    """Train the model that did best under the strictest scheme on all data."""
    strict = results["walk_forward_unseen_company"]["all"]
    name = max(MODELS, key=lambda m: strict[m].get("roc_auc", 0))
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy().astype(int)
    model = MODELS[name]().fit(X, y)
    scores = model.predict_proba(X)[:, 1]
    return name, model, {
        "model_name": name,
        "features": FEATURES,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "training_rows": int(len(df)),
        "training_companies": int(df[GROUP].nunique()),
        "training_years": [int(df["year"].min()), int(df["year"].max())],
        "base_rate": float(y.mean()),
        "score_percentiles": {str(q): float(np.percentile(scores, q))
                              for q in (10, 25, 50, 75, 90, 95, 99)},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation": strict[name],
    }


def shap_explainer(model, background):
    import shap
    if isinstance(model, Pipeline):
        imputer, scaler = model.named_steps["imputer"], model.named_steps["scaler"]
        transformed = scaler.transform(imputer.transform(background))
        return shap.LinearExplainer(model.named_steps["classifier"], transformed), \
            (lambda X: scaler.transform(imputer.transform(X)))
    return shap.TreeExplainer(model), (lambda X: X)


def shap_values(model, X, background=None):
    explainer, transform = shap_explainer(model, background if background is not None else X)
    values = explainer.shap_values(transform(X))
    values = np.asarray(values)
    if values.ndim == 3:          # (n, features, classes)
        values = values[..., 1]
    base = explainer.expected_value
    base = float(np.ravel(base)[-1])
    return values, base


def global_importance(model, df, sample=3000):
    X = df[FEATURES].sample(min(sample, len(df)), random_state=SEED).to_numpy(dtype=float)
    values, _ = shap_values(model, X)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": FEATURES[i], "description": FEATURE_DESCRIPTIONS[FEATURES[i]],
             "mean_abs_shap": float(mean_abs[i])} for i in order]


def save_model(model, meta, path=MODEL_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)


def load_model(path=MODEL_FILE):
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def explain_row(bundle, row_features, background=None):
    """Score one company-year and explain it with SHAP (log-odds units for
    gradient boosting; margin units for logistic)."""
    model, meta = bundle["model"], bundle["meta"]
    x = np.array([[row_features.get(f, np.nan) for f in meta["features"]]], dtype=float)
    score = float(model.predict_proba(x)[0, 1])
    values, base = shap_values(model, x, background)
    contributions = sorted(
        ({"feature": f, "description": meta["feature_descriptions"][f],
          "value": None if np.isnan(x[0, i]) else float(x[0, i]),
          "shap": float(values[0, i])}
         for i, f in enumerate(meta["features"])),
        key=lambda c: -abs(c["shap"]))
    percentile = float(np.mean([score >= v for v in meta["score_percentiles"].values()]))
    rank = next((q for q, v in sorted(meta["score_percentiles"].items(),
                                      key=lambda kv: -float(kv[0])) if score >= v), None)
    return {"score": score, "base_value": base, "contributions": contributions,
            "riskier_than_percent_of_training": rank}


# ------------------------------------------------------------------ report

def render_markdown(results, lines, importance, meta, data_summary):
    out = ["# Financial-distress model: evaluation", "",
           "Generated by `src/ml/distress/model.py`. Research use only; not investment advice.", "",
           "## Conclusions", ""]
    out += [f"- {l}" for l in lines]
    out += ["", "## Target and data", "",
            "- **Distress** in a fiscal year: (net loss and negative operating cash flow) or "
            "negative equity or operating income below interest expense (interest > 0.5% of assets).",
            "- **Target:** distress in the company's next fiscal year (within 12 months).",
            f"- **Data:** {data_summary['rows']} company-years, {data_summary['companies']} US "
            f"companies, fiscal years {data_summary['years'][0]}–{data_summary['years'][1]} "
            "(SEC EDGAR frames; operating companies with revenue, banks/insurers and shells excluded).",
            f"- **Base rate:** {data_summary['base_rate']:.1%} of company-years are followed by distress; "
            f"{data_summary['persistence']:.1%} of companies in distress stay in distress.",
            "- **Timing:** a company-year is usable 90 days after its fiscal year end (10-K "
            "deadline); training uses only labels known by then.", "",
            "## Results", "",
            "AUC 0.5 = random ranking, 1.0 = perfect. PR AUC is compared with the base rate. "
            "Top-10% = the riskiest tenth of companies by score.", ""]
    for scheme, r in results.items():
        out += [f"### {scheme} ({r['evaluated_rows']} test rows)", ""]
        for subset in ("all", "healthy_now"):
            s = r[subset]
            out += [f"**{'All companies' if subset == 'all' else 'Not in distress now (new distress)'}** "
                    f"(base rate {s['logistic']['base_rate']:.1%})", "",
                    "| Model | ROC AUC (95% CI) | PR AUC | Top-10% precision | Top-10% recall | Balanced acc. | AUC gain vs best baseline (95% CI) |",
                    "|---|---|---|---|---|---|---|"]
            for name, m in s.items():
                if name.startswith("_") or "roc_auc" not in m:
                    continue
                ci = m.get("roc_auc_95ci")
                gain = m.get("auc_minus_best_baseline_95ci")
                out.append(
                    f"| {name} | {m['roc_auc']:.3f}" + (f" ({ci[0]:.3f}–{ci[1]:.3f})" if ci else "")
                    + f" | {m['pr_auc']:.3f} | {m['top10_precision']:.1%} | "
                    + (f"{m['top10_recall']:.1%}" if m['top10_recall'] is not None else "n/a")
                    + f" | {m['balanced_accuracy']:.3f} | "
                    + (f"{gain[0]:+.3f} to {gain[1]:+.3f}" if gain else "") + " |")
            out.append("")
    out += ["## What drives the predictions (SHAP)", "",
            f"Final model: **{meta['model_name']}**, trained on all {meta['training_rows']} rows. "
            "Mean absolute SHAP value = average influence on the prediction.", "",
            "| Rank | Factor | Mean |SHAP| |", "|---|---|---|"]
    out += [f"| {i + 1} | {r['description']} | {r['mean_abs_shap']:.3f} |"
            for i, r in enumerate(importance[:15])]
    out += ["", "## Limitations", "",
            "- US SEC filers only; scoring Indian or other companies uses Yahoo statements, whose "
            "definitions differ slightly, and is outside the training population.",
            "- Frames hold the latest filed value, so a few figures may be restated.",
            "- Distress here is an accounting definition (losses with cash burn, negative equity, "
            "weak interest cover), not bankruptcy or default.",
            "- Scores are not calibrated probabilities; class weights are balanced.", ""]
    return "\n".join(out)


def run(echo=print, refresh=False):
    from src.ml.distress.edgar_frames import load_or_download

    panel = load_or_download(refresh=refresh, echo=echo)
    df = build_training_set(panel)
    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_FILE, index=False)
    now = df["distress_now"] == 1
    data_summary = {
        "rows": int(len(df)), "companies": int(df[GROUP].nunique()),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "base_rate": float(df[TARGET].mean()),
        "persistence": float(df.loc[now, TARGET].mean()) if now.any() else 0.0,
    }
    echo(f"Training set: {data_summary}")

    results, predictions = evaluate(df, echo)
    lines = conclusions(results)
    name, model, meta = train_final(df, results)
    importance = global_importance(model, df)
    meta["global_importance"] = importance[:15]
    meta["background_sample"] = df[FEATURES].sample(200, random_state=SEED).to_numpy(dtype=float)
    meta["data_summary"] = data_summary
    meta["conclusions"] = lines
    save_model(model, meta)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"data": data_summary, "results": results, "conclusions": lines,
               "final_model": {k: v for k, v in meta.items() if k != "background_sample"}}
    (REPORT_DIR / "distress_evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (REPORT_DIR / "distress_evaluation.md").write_text(
        render_markdown(results, lines, importance, meta, data_summary), encoding="utf-8")
    predictions.to_csv(REPORT_DIR / "distress_predictions.csv.gz", index=False, compression="gzip")
    return payload
