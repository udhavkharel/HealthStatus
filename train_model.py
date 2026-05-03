# This file trains machine learning models for the diseases
# where public datasets are available and suitable.

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# This dictionary tells the program which datasets to use,
# which columns are features, and where the trained model should be saved.
TRAINING_CONFIG = {
    "Diabetes": {
        "file": "data/diabetes.csv",
        "target": "Outcome",
        "features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "output_model": "models/diabetes_model.joblib",
    },
    "Heart Disease": {
        "file": "data/heart.csv",
        "target": "output",
        "features": [
            "age",
            "sex",
            "cp",
            "trtbps",
            "chol",
            "fbs",
            "thalachh",
            "oldpeak",
        ],
        "zero_as_missing": [],
        "output_model": "models/heart_model.joblib",
    },
    "Liver Health": {
        "file": "data/liver.csv",
        "target": "Dataset",
        "features": [
            "Age",
            "Total_Bilirubin",
            "Direct_Bilirubin",
            "Alkaline_Phosphotase",
            "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase",
            "Total_Protiens",
            "Albumin",
            "Albumin_and_Globulin_Ratio",
        ],
        "zero_as_missing": [],
        "output_model": "models/liver_model.joblib",
    },
}


def clean_target_column(df, target_name):
    # This function makes sure the target column becomes numeric.
    # Some datasets may store yes/no or string-based outcomes.
    if df[target_name].dtype == "object":
        lowered = df[target_name].astype(str).str.strip().str.lower()

        if set(lowered.unique()) <= {"yes", "no"}:
            df[target_name] = lowered.map({"no": 0, "yes": 1})

        elif set(lowered.unique()) <= {"ckd", "notckd"}:
            df[target_name] = lowered.map({"notckd": 0, "ckd": 1})

        elif set(lowered.unique()) <= {"negative", "positive"}:
            df[target_name] = lowered.map({"negative": 0, "positive": 1})

    return df


def train_one_model(disease_name, config):
    # This trains one disease model from one dataset.

    dataset_path = config["file"]
    model_path = config["output_model"]

    if not os.path.exists(dataset_path):
        print(f"[SKIPPED] {disease_name}: dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    if df.empty:
        print(f"[SKIPPED] {disease_name}: dataset is empty")
        return

    missing_columns = [col for col in config["features"] + [config["target"]] if col not in df.columns]
    if missing_columns:
        print(f"[SKIPPED] {disease_name}: missing columns -> {missing_columns}")
        return

    df = clean_target_column(df, config["target"])

    # Convert all feature columns to numeric where possible.
    for col in config["features"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[config["target"]] = pd.to_numeric(df[config["target"]], errors="coerce")

    # Replace unrealistic zero values with missing values for some medical fields.
    for col in config["zero_as_missing"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Drop rows where the target is missing.
    df = df.dropna(subset=[config["target"]])

    X = df[config["features"]]
    y = df[config["target"]].astype(int)

    if len(y.unique()) < 2:
        print(f"[SKIPPED] {disease_name}: target does not contain two classes")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    os.makedirs("models", exist_ok=True)

    package = {
        "disease": disease_name,
        "model": model,
        "features": config["features"],
        "target": config["target"],
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }

    joblib.dump(package, model_path)

    print(f"[DONE] {disease_name}")
    print(f"       Accuracy: {accuracy:.3f}")
    print(f"       ROC AUC : {roc_auc:.3f}")
    print(f"       Saved to: {model_path}")


def main():
    # This loops through all configured disease datasets and trains the available ones.
    for disease_name, config in TRAINING_CONFIG.items():
        train_one_model(disease_name, config)


if __name__ == "__main__":
    main()