
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "revenue_decline_features.csv"
)

OUTPUT_FILE = (
    ROOT / "reports"
    / "feature_set_comparison.csv"
)

TARGET = "Next_Period_Revenue_Decline"

FEATURE_SETS = {
    "Two complete margins": [
        "Net_Income_Margin",
        "Operating_Margin",
    ],
    "All five ratios": [
        "Net_Income_Margin",
        "Operating_Margin",
        "Debt_To_Assets",
        "Debt_To_Equity",
        "Current_Ratio",
    ],
}


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["Ticker", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    results = []

    for set_name, features in FEATURE_SETS.items():
        actual_all = []
        predicted_all = []
        baseline_all = []

        for company in sorted(df["Ticker"].unique()):
            train = df[df["Ticker"] != company]
            test = df[df["Ticker"] == company]

            if train[TARGET].nunique() < 2:
                continue

            X_train = train[features]
            y_train = train[TARGET]
            X_test = test[features]
            y_test = test[TARGET]

            model = make_model()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            baseline = DummyClassifier(
                strategy="most_frequent"
            )
            baseline.fit(X_train, y_train)
            baseline_predictions = baseline.predict(X_test)

            actual_all.extend(y_test.tolist())
            predicted_all.extend(predictions.tolist())
            baseline_all.extend(baseline_predictions.tolist())

        if not actual_all:
            continue

        matrix = confusion_matrix(
            actual_all, predicted_all, labels=[0, 1]
        )

        baseline_matrix = confusion_matrix(
            actual_all, baseline_all, labels=[0, 1]
        )

        results.append({
            "Feature_Set": set_name,
            "Features": ", ".join(features),
            "Observations": len(actual_all),
            "Model_Accuracy": accuracy_score(
                actual_all, predicted_all
            ),
            "Baseline_Accuracy": accuracy_score(
                actual_all, baseline_all
            ),
            "True_Negatives": int(matrix[0, 0]),
            "False_Alarms": int(matrix[0, 1]),
            "Missed_Declines": int(matrix[1, 0]),
            "Detected_Declines": int(matrix[1, 1]),
            "Baseline_False_Alarms": int(baseline_matrix[0, 1]),
            "Baseline_Missed_Declines": int(baseline_matrix[1, 0]),
        })

    results_df = pd.DataFrame(results)

    print("\n========== FEATURE SET COMPARISON ==========")
    print(results_df.to_string(index=False))

    print(
        "\nReminder: 31 observations and 8 declines are insufficient "
        "to establish reliable real-world predictive performance."
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print("\nExisting report not overwritten:", OUTPUT_FILE)
    else:
        results_df.to_csv(OUTPUT_FILE, index=False)
        print("\nComparison saved to:", OUTPUT_FILE)

    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()
