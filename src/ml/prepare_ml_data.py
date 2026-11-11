
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "future_growth_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_ready_dataset.csv"
)

TARGET = "Revenue_Decline_Next_Year"

# Load the prepared dataset
df = pd.read_csv(INPUT_FILE)

# Keep only rows with a valid target
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

# Exclude target columns and identifiers from model features
excluded_columns = [
    TARGET,
    "Target_Next_Revenue_Growth",
    "Date",
]

X = df.drop(
    columns=[col for col in excluded_columns if col in df.columns]
)

y = df[TARGET]

# Separate numerical and categorical features
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
# "str" is listed explicitly: pandas 3 stores text as the str dtype and will
# stop matching it with "object" in a future version.
categorical_columns = X.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

# Fill missing numerical values with each column's median
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Fill missing categorical values and encode them
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]
)

# Fit preprocessing to this dataset for a preparation check only
X_processed = preprocessor.fit_transform(X)

# Save a clean, imputed dataset for inspection.
# This file is for inspection, not the final model-training workflow.
X_ready = pd.DataFrame(
    X_processed.toarray()
    if hasattr(X_processed, "toarray")
    else X_processed,
    columns=preprocessor.get_feature_names_out(),
    index=df.index,
)

X_ready[TARGET] = y

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
X_ready.to_csv(OUTPUT_FILE, index=False)

print("Preprocessing check completed.")
print("Original rows:", len(df))
print("Original input features:", X.shape[1])
print("Processed features:", X_ready.shape[1] - 1)
print("Missing values after preprocessing:", X_ready.isna().sum().sum())
print("\nTarget distribution:")
print(y.value_counts().sort_index())
print("\nSaved inspection dataset to:")
print(OUTPUT_FILE)
