
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "revenue_decline_features.csv"
)

OUTPUT_FILE = (
    ROOT / "reports"
    / "company_feature_set_comparison.csv"
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

    for feature_set_name, features in FEATURE_SETS.items():
        for company in sorted(df["Ticker"].unique()):
            train = df[df["Ticker"] != company]
            test = df[df["Ticker"] == company]

            if train[TARGET].nunique() < 2:
                continue

            model = make_model()
            model.fit(train[features], train[TARGET])

            predictions = model.predict(test[features])
            actual = test[TARGET]

            matrix = confusion_matrix(
                actual, predictions, labels=[0, 1]
            )

            results.append({
                "Feature_Set": feature_set_name,
                "Company": company,
                "Observations": len(test),
                "Actual_Declines": int(actual.sum()),
                "Predicted_Declines": int(predictions.sum()),
                "Accuracy": accuracy_score(actual, predictions),
                "True_Negatives": int(matrix[0, 0]),
                "False_Alarms": int(matrix[0, 1]),
                "Missed_Declines": int(matrix[1, 0]),
                "Detected_Declines": int(matrix[1, 1]),
            })

    results_df = pd.DataFrame(results)

    print("\n========== COMPANY-WISE FEATURE COMPARISON ==========")
    print(results_df.to_string(index=False))

    print(
        "\nReminder: each company has only a few observations. "
        "These results are exploratory, not reliable predictions."
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print("\nExisting report not overwritten:", OUTPUT_FILE)
    else:
        results_df.to_csv(OUTPUT_FILE, index=False)
        print("\nReport saved to:", OUTPUT_FILE)

    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()
