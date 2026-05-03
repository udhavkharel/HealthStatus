# This file trains the machine learning model for the health AI agent.

import os  # This lets Python work with folders and file paths.
import joblib  # This saves the trained model so the app can use it later.
import pandas as pd  # This reads and manages the dataset in table format.
import numpy as np  # This helps with numerical values and missing values.

from sklearn.model_selection import train_test_split  # This splits data into training and testing parts.
from sklearn.pipeline import Pipeline  # This keeps preprocessing and the model together.
from sklearn.impute import SimpleImputer  # This fills missing values with a safe statistical value.
from sklearn.ensemble import RandomForestClassifier  # This is the ML model used for classification.
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score  # These check model performance.

DATA_PATH = "data/diabetes.csv"  # This is where the training dataset should be stored.
MODEL_PATH = "models/health_risk_model.joblib"  # This is where the trained model will be saved.

FEATURES = [  # These are the columns the model will learn from.
    "Pregnancies",  # Number of pregnancies; use 0 if not applicable.
    "Glucose",  # Blood glucose value from the dataset.
    "BloodPressure",  # Blood pressure value from the dataset.
    "SkinThickness",  # Skin thickness measurement from the dataset.
    "Insulin",  # Insulin measurement from the dataset.
    "BMI",  # Body Mass Index.
    "DiabetesPedigreeFunction",  # Diabetes family-history style score in the dataset.
    "Age",  # Age of the person.
]

TARGET = "Outcome"  # This is the label column: 0 means no diabetes, 1 means diabetes in the dataset.

ZERO_AS_MISSING = [  # These columns should not normally be zero, so zero is treated as missing.
    "Glucose",  # A glucose value of 0 is not realistic for a living patient.
    "BloodPressure",  # A blood pressure value of 0 is not realistic.
    "SkinThickness",  # A zero here usually means it was not measured.
    "Insulin",  # A zero here usually means it was not measured.
    "BMI",  # BMI of 0 is not realistic.
]


def train_model():  # This function trains and saves the model.
    if not os.path.exists(DATA_PATH):  # This checks whether the dataset file exists.
        raise FileNotFoundError("Please place your Kaggle diabetes.csv file inside the data folder.")  # This explains what is missing.

    df = pd.read_csv(DATA_PATH)  # This loads the CSV dataset into a table.

    missing_columns = [col for col in FEATURES + [TARGET] if col not in df.columns]  # This finds columns that are missing.
    if missing_columns:  # This checks if any required columns are absent.
        raise ValueError(f"Your dataset is missing these columns: {missing_columns}")  # This stops training with a clear message.

    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)  # This changes unrealistic zero values into missing values.

    X = df[FEATURES]  # This keeps only the input columns for training.
    y = df[TARGET]  # This keeps only the answer column for training.

    X_train, X_test, y_train, y_test = train_test_split(  # This splits data into train and test sets.
        X,  # These are the input features.
        y,  # These are the correct answers.
        test_size=0.20,  # This keeps 20% of the data for testing.
        random_state=42,  # This keeps results repeatable.
        stratify=y,  # This keeps the diabetes/no-diabetes ratio similar in both sets.
    )

    model = Pipeline([  # This creates one combined training pipeline.
        ("imputer", SimpleImputer(strategy="median")),  # This fills missing values using the median.
        ("classifier", RandomForestClassifier(  # This creates the main prediction model.
            n_estimators=300,  # This uses 300 decision trees for stable prediction.
            random_state=42,  # This keeps model training repeatable.
            class_weight="balanced",  # This helps when one class is smaller than the other.
        )),
    ])

    model.fit(X_train, y_train)  # This trains the model using the training data.

    predictions = model.predict(X_test)  # This asks the model to predict the test data.
    probabilities = model.predict_proba(X_test)[:, 1]  # This gets diabetes-risk probabilities.

    print("Model Performance")  # This prints a heading for the results.
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))  # This shows total correct predictions.
    print("Recall:", round(recall_score(y_test, predictions), 3))  # This shows how many positive cases were found.
    print("Precision:", round(precision_score(y_test, predictions), 3))  # This shows how many positive predictions were correct.
    print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 3))  # This shows ranking performance.

    os.makedirs("models", exist_ok=True)  # This creates the models folder if it does not exist.

    package = {  # This creates a saved package for the app.
        "model": model,  # This stores the trained model.
        "features": FEATURES,  # This stores the feature order needed during prediction.
    }

    joblib.dump(package, MODEL_PATH)  # This saves the package to the models folder.

    print(f"Model saved to: {MODEL_PATH}")  # This confirms where the model was saved.


if __name__ == "__main__":  # This runs only when the file is executed directly.
    train_model()  # This starts training.