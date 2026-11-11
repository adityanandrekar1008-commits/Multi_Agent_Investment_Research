"""Experimental ML signal for one company's latest quarter.

The proof-of-concept model (logistic regression on 5 ratios) is trained on
the committed feature dataset and scores the latest quarter. The output is
deliberately labelled experimental and carries the evaluation verdict from
reports/model_evaluation.json, because that evaluation found no evidence the
model beats chance. It must not be shown as a probability, a risk score or
a recommendation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.evaluate_revenue_decline import (
    RATIO_FEATURES,
    TARGET,
    make_logreg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_FILE = PROJECT_ROOT / "data" / "processed" / "revenue_decline_features.csv"
EVALUATION_FILE = PROJECT_ROOT / "reports" / "model_evaluation.json"

MODEL_NAME = "logreg_5_ratios"
SCORE_MEANING = (
    "Uncalibrated logistic-regression score between 0 and 1; not a "
    "probability. Classes are re-weighted in training, so 0.5 does not equal "
    "the historical decline rate."
)
LIMITATIONS = [
    "Evaluation found no evidence that the model predicts revenue declines "
    "better than chance. A larger check on 429 SEC EDGAR quarters "
    "(7 US companies, 2009-2026) agreed: no better than chance, while a "
    "simple same-calendar-quarter rule did much better "
    "(reports/sec_edgar/).",
    "The label is mostly seasonal: December quarters are usually followed by "
    "a lower March quarter.",
    "Trained on a few dozen quarters from 8 large companies; not representative "
    "of other sectors, sizes or markets.",
    "Predicts a lower next-quarter revenue only; not losses, defaults or share "
    "price moves.",
    "Missing inputs are replaced with the training median, which can hide "
    "unusual values (e.g. negative equity).",
]


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def _evaluation_summary(evaluation_file):
    path = Path(evaluation_file)
    if not path.exists():
        return {"available": False,
                "note": "Run `python main.py evaluate` to produce it."}
    data = json.loads(path.read_text(encoding="utf-8"))
    schemes = {}
    for name, scheme in data.get("schemes", {}).items():
        model = scheme.get("models", {}).get(MODEL_NAME, {})
        schemes[name] = {
            "balanced_accuracy": model.get("balanced_accuracy"),
            "balanced_accuracy_95ci": model.get("balanced_accuracy_95ci"),
            "permutation_p_value": model.get("permutation_p_value"),
            "test_rows": model.get("n"),
        }
    conclusions = data.get("conclusions") or []
    verdict = conclusions[-1].removeprefix("Overall: ") if conclusions else None
    if verdict:
        verdict = verdict[0].upper() + verdict[1:]
    return {
        "available": True,
        "verdict": verdict,
        "schemes": schemes,
        "report": "reports/model_evaluation.md",
    }


def unavailable(reason, evaluation_file=EVALUATION_FILE):
    return {
        "status": "unavailable",
        "reason": reason,
        "label": "experimental",
        "model": MODEL_NAME,
        "evaluation": _evaluation_summary(evaluation_file),
        "limitations": LIMITATIONS,
    }


def build_ml_signal(latest, period_end, financial_company=False,
                    training_file=TRAINING_FILE, evaluation_file=EVALUATION_FILE):
    """Score one quarter.

    latest: mapping with the five ratio values (missing allowed).
    """
    if financial_company:
        return unavailable(
            "The model was not trained on banks or other financial companies "
            "(their statements lack the ratios it uses).", evaluation_file)

    values = {f: _num(latest.get(f)) for f in RATIO_FEATURES}
    if all(v is None for v in values.values()):
        return unavailable("None of the model's ratios are available for the "
                           "latest quarter.", evaluation_file)

    training_file = Path(training_file)
    if not training_file.exists():
        return unavailable(f"Training data not found: {training_file.name}.",
                           evaluation_file)
    training = pd.read_csv(training_file).dropna(subset=[TARGET])
    y = training[TARGET].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return unavailable("Training data has only one class.", evaluation_file)

    model = make_logreg()
    model.fit(training[RATIO_FEATURES].to_numpy(dtype=float), y)

    x = np.array([[np.nan if v is None else v for v in values.values()]])
    score = float(model.predict_proba(x)[0, 1])

    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    used = imputer.transform(x)[0]
    standardized = scaler.transform(used.reshape(1, -1))[0]
    contributions = classifier.coef_[0] * standardized

    inputs = {}
    for i, feature in enumerate(RATIO_FEATURES):
        inputs[feature] = {
            "value": values[feature],
            "imputed": values[feature] is None,
            "value_used": float(used[i]),
            "training_median": float(imputer.statistics_[i]),
            "contribution": float(contributions[i]),
        }
    ranked = sorted(inputs.items(), key=lambda kv: -abs(kv[1]["contribution"]))
    top = [
        {
            "feature": name,
            "contribution": info["contribution"],
            "direction": "towards decline" if info["contribution"] > 0
            else "towards no decline",
        }
        for name, info in ranked[:3]
    ]

    return {
        "status": "available",
        "label": "experimental",
        "recommended_use": "research only; do not display as a prediction, "
                           "probability or risk score",
        "model": MODEL_NAME,
        "target": "next reported quarter's revenue lower than this quarter's",
        "horizon": "one quarter",
        "period_end": period_end,
        "score": score,
        "score_meaning": SCORE_MEANING,
        "predicted_class": "decline" if score >= 0.5 else "no_decline",
        "intercept": float(classifier.intercept_[0]),
        "inputs": inputs,
        "top_contributions": top,
        "contribution_note": (
            "Contribution = coefficient x standardized input, in log-odds. "
            "Intercept + sum of contributions = logit(score). Associations "
            "in a small sample, not causes."
        ),
        "training": {
            "rows": int(len(training)),
            "companies": sorted(training["Ticker"].astype(str).unique().tolist()),
            "period_range": [str(training["Date"].min())[:10],
                             str(training["Date"].max())[:10]],
            "declines": int(y.sum()),
        },
        "evaluation": _evaluation_summary(evaluation_file),
        "limitations": LIMITATIONS,
    }
