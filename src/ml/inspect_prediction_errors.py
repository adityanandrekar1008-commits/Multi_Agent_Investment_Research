
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "revenue_decline_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "prediction_error_inspection.csv"
)

TARGET = "Next_Period_Revenue_Decline"

FEATURES = [
    "Net_Income_Margin",
    "Operating_Margin",
    "Debt_To_Assets",
    "Debt_To_Equity",
    "Current_Ratio",
]


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["Ticker", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    predictions = []

    # Repeat the leave-one-company-out experiment.
    for company in sorted(df["Ticker"].unique()):
        train = df[df["Ticker"] != company]
        test = df[df["Ticker"] == company]

        if train[TARGET].nunique() < 2:
            print(f"Skipping {company}: only one training class.")
            continue

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ])

        model.fit(train[FEATURES], train[TARGET])

        predicted = model.predict(test[FEATURES])
        probabilities = model.predict_proba(test[FEATURES])

        classes = list(model.named_steps["classifier"].classes_)
        decline_index = classes.index(1)

        for position, (_, row) in enumerate(test.iterrows()):
            actual = int(row[TARGET])
            prediction = int(predicted[position])

            if actual == 1 and prediction == 1:
                result = "Correctly detected decline"
            elif actual == 0 and prediction == 0:
                result = "Correctly detected no decline"
            elif actual == 0 and prediction == 1:
                result = "False alarm"
            else:
                result = "Missed revenue decline"

            record = {
                "Ticker": row["Ticker"],
                "Date": row["Date"],
                "Actual": actual,
                "Predicted": prediction,
                "Predicted_Decline_Probability": round(
                    float(probabilities[position][decline_index]), 4
                ),
                "Result": result,
            }

            # Include features for error analysis only.
            for feature in FEATURES:
                record[feature] = row[feature]

            predictions.append(record)

    result_df = pd.DataFrame(predictions)

    print("\n========== PREDICTION ERROR INSPECTION ==========")
    print("Observations:", len(result_df))

    print("\nResult counts:")
    print(result_df["Result"].value_counts().to_string())

    print("\nIncorrect predictions:")
    errors = result_df[
        result_df["Actual"] != result_df["Predicted"]
    ]

    if errors.empty:
        print("No incorrect predictions.")
    else:
        print(
            errors[
                [
                    "Ticker",
                    "Date",
                    "Actual",
                    "Predicted",
                    "Predicted_Decline_Probability",
                    "Result",
                ]
            ].to_string(index=False)
        )

    print("\nResults by company:")
    print(
        pd.crosstab(
            result_df["Ticker"],
            result_df["Result"],
        ).to_string()
    )

    # Protect any previously generated report.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print("\nReport already exists; it was not overwritten:")
        print(OUTPUT_FILE)
    else:
        result_df.to_csv(OUTPUT_FILE, index=False)
        print("\nDetailed results saved to:", OUTPUT_FILE)

    print("\nOriginal datasets were not modified.")
    print(
        "Probabilities are model scores, not calibrated real-world "
        "investment-risk probabilities."
    )


if __name__ == "__main__":
    main()
