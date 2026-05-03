# This file trains separate machine-learning models for the datasets inside the data folder.
# Simple idea: each disease dataset trains its own model and saves one .joblib file.

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# These are the exact dataset files you currently have inside your data folder.
# Each block tells Python: file name, possible target column names, columns to ignore, and model save location.
TRAINING_CONFIG = {
    "Diabetes / Blood Sugar": {
        "file": "data/diabetes.csv",
        "target_candidates": ["Outcome"],
        "preferred_features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age",
        ],
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "drop_columns": [],
        "output_model": "models/diabetes_model.joblib",
    },
    "Heart Disease": {
        "file": "data/heart.csv",
        "target_candidates": ["output", "target", "num", "heart_disease"],
        "preferred_features": [
            "age", "sex", "cp", "trtbps", "trestbps", "chol", "fbs",
            "thalachh", "thalach", "oldpeak", "exng", "exang", "slope", "slp", "ca", "caa", "thal", "thall",
        ],
        "zero_as_missing": [],
        "drop_columns": [],
        "output_model": "models/heart_model.joblib",
    },
    "Liver Health": {
        "file": "data/indian_liver_patient.csv",
        "target_candidates": ["Dataset", "dataset", "Selector", "target", "class"],
        "preferred_features": [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Protiens", "Albumin",
            "Albumin_and_Globulin_Ratio",
        ],
        "zero_as_missing": [],
        "drop_columns": [],
        "target_map": {1: 1, 2: 0},  # In the common liver dataset, 1 = liver disease, 2 = no liver disease.
        "output_model": "models/liver_model.joblib",
    },
    "Chronic Kidney Disease": {
        "file": "data/kidney_disease.csv",
        "target_candidates": ["classification", "class", "target", "ckd"],
        "preferred_features": [],  # Empty means use all useful columns except target/id.
        "zero_as_missing": [],
        "drop_columns": ["id"],
        "output_model": "models/kidney_model.joblib",
    },
    "Thyroid Screening": {
        "file": "data/cleaned_dataset_Thyroid1.csv",
        "target_candidates": [
            "target", "Target", "binaryClass", "BinaryClass", "class", "Class",
            "diagnosis", "Diagnosis", "thyroid", "Thyroid", "status", "Status",
            "Recurred", "recurred", "Condition", "condition",
        ],
        "preferred_features": [],
        "zero_as_missing": [],
        "drop_columns": ["id", "patient_id"],
        "output_model": "models/thyroid_model.joblib",
    },
    "Stroke Risk": {
        "file": "data/healthcare-dataset-stroke-data.csv",
        "target_candidates": ["stroke", "Stroke", "target", "Target"],
        "preferred_features": [],
        "zero_as_missing": [],
        "drop_columns": ["id"],
        "output_model": "models/stroke_model.joblib",
    },
    "Cardiovascular / BP Risk": {
        "file": "data/cardio_data_processed.csv",
        "target_candidates": ["cardio", "Cardio", "target", "Target", "Disease", "disease"],
        "preferred_features": [],
        "zero_as_missing": [],
        "drop_columns": ["id"],
        "output_model": "models/cardio_model.joblib",
    },
}


TEXT_TARGET_MAP = {
    "yes": 1, "no": 0,
    "positive": 1, "negative": 0,
    "ckd": 1, "notckd": 0,
    "ckd\t": 1, "notckd\t": 0,
    "disease": 1, "no disease": 0,
    "present": 1, "absent": 0,
    "true": 1, "false": 0,
}


def normalize_name(name):
    # This makes column matching easier by ignoring spaces, hyphens, and capital letters.
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df, candidates):
    # This finds the real column name even if capitalization is different.
    normalized_columns = {normalize_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized_columns:
            return normalized_columns[key]
    return None


def load_dataset(path):
    # This reads a CSV file and treats common missing symbols as actual missing values.
    return pd.read_csv(path, na_values=["?", "NA", "N/A", "na", "null", "None", "", " "])


def clean_target_values(series, config):
    # This converts the answer column into clean numeric labels.
    if "target_map" in config:
        return series.map(config["target_map"]).fillna(series)

    if series.dtype == "object":
        lowered = series.astype(str).str.strip().str.lower()
        if set(lowered.dropna().unique()).issubset(set(TEXT_TARGET_MAP.keys())):
            return lowered.map(TEXT_TARGET_MAP)

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(lowered)
        return pd.Series(encoded, index=series.index)

    return pd.to_numeric(series, errors="coerce")


def choose_features(df, target_col, config):
    # This chooses the columns used as model inputs.
    drop_columns = [normalize_name(c) for c in config.get("drop_columns", [])]
    target_key = normalize_name(target_col)

    preferred = config.get("preferred_features", [])
    if preferred:
        selected = []
        for feature in preferred:
            real_col = find_column(df, [feature])
            if real_col is not None and normalize_name(real_col) != target_key:
                selected.append(real_col)
        return selected

    # If no preferred feature list is given, use all columns except target and ignored columns.
    selected = []
    for col in df.columns:
        key = normalize_name(col)
        if key == target_key:
            continue
        if key in drop_columns:
            continue
        selected.append(col)
    return selected


def train_one_model(disease_name, config):
    # This trains one model for one disease module.
    dataset_path = config["file"]
    model_path = config["output_model"]

    if not os.path.exists(dataset_path):
        print(f"[SKIPPED] {disease_name}: dataset not found at {dataset_path}")
        return

    df = load_dataset(dataset_path)

    if df.empty:
        print(f"[SKIPPED] {disease_name}: dataset is empty")
        return

    target_col = find_column(df, config["target_candidates"])
    if target_col is None:
        print(f"[SKIPPED] {disease_name}: target column not found")
        print(f"          Available columns: {list(df.columns)}")
        print(f"          Add the correct target name inside target_candidates.")
        return

    feature_cols = choose_features(df, target_col, config)
    if not feature_cols:
        print(f"[SKIPPED] {disease_name}: no feature columns found")
        return

    # Clean target column.
    df[target_col] = clean_target_values(df[target_col], config)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Replace impossible zero values with missing values where needed.
    for col in config.get("zero_as_missing", []):
        real_col = find_column(df, [col])
        if real_col is not None:
            df[real_col] = df[real_col].replace(0, np.nan)

    X = df[feature_cols]
    y = df[target_col].astype(int)

    if y.nunique() < 2:
        print(f"[SKIPPED] {disease_name}: target column has only one class")
        return

    # Split safely. Stratify only if every class has at least 2 rows.
    stratify_value = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify_value
    )

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Numeric values are filled with median.
    # Categorical values are filled with the most common value and one-hot encoded.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)
            if len(model.named_steps["classifier"].classes_) == 2:
                roc_auc = roc_auc_score(y_test, probabilities[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr")
        except Exception:
            roc_auc = None

    os.makedirs("models", exist_ok=True)

    package = {
        "disease": disease_name,
        "model": model,
        "features": feature_cols,
        "target": target_col,
        "classes": model.named_steps["classifier"].classes_.tolist(),
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }

    joblib.dump(package, model_path)

    print(f"[DONE] {disease_name}")
    print(f"       Dataset : {dataset_path}")
    print(f"       Target  : {target_col}")
    print(f"       Features: {len(feature_cols)} columns")
    print(f"       Accuracy: {accuracy:.3f}")
    if roc_auc is not None:
        print(f"       ROC AUC : {roc_auc:.3f}")
    print(f"       Saved to: {model_path}")


def main():
    # This trains every available dataset model one by one.
    for disease_name, config in TRAINING_CONFIG.items():
        train_one_model(disease_name, config)


if __name__ == "__main__":
    main()
