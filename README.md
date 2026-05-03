# HealthStatus
It is a Ai agent that helps to classify medical risks and provide related suggestions for its medication procedures.
# HealthStatus: Medical Screening AI Agent

HealthStatus is a simple medical screening AI agent built with Python and Streamlit. It allows users to enter health-checkup values such as blood sugar, blood pressure, kidney values, liver values, thyroid values, stroke-related values, and heart-related values. Based on the selected disease area, the app gives a risk category, short summary, visual risk score, and basic recommendation.

This project is designed for learning and demonstration purposes. It is not a real medical diagnosis system.

---

## Important Disclaimer

This app does not replace a doctor, hospital, or medical professional.  
It only gives a basic risk indication based on user-entered values and trained machine learning models.

If you have serious symptoms such as chest pain, breathing difficulty, fainting, confusion, severe weakness, very high blood pressure, or very abnormal test results, please seek urgent medical help.

---

## What This Project Does

The user first selects a health area such as diabetes, heart disease, kidney disease, liver disease, thyroid disease, stroke risk, or cardiovascular risk.

Then the user enters values from their medical report or checkup.

The app then:

1. Collects the entered health values.
2. Sends the values to the correct disease module.
3. Uses a trained machine learning model when available.
4. Uses rule-based screening where needed.
5. Gives a final risk category.
6. Shows a color-based risk chart.
7. Gives a simple explanation.
8. Saves the result locally in a CSV file.

---

## Main Features

- Simple medical dashboard
- Disease selection option
- Separate models for different diseases
- User-friendly form inputs
- Color-based risk categories
- Visual risk chart
- Basic medical Q&A assistant
- Local record saving
- Easy to run on a laptop

---

## Supported Disease Areas

The app is designed to support these health areas:

- Diabetes / Blood Sugar
- Heart Disease
- Liver Health
- Kidney Disease
- Thyroid Disease
- Stroke Risk
- Cardiovascular Risk
- Blood Pressure Screening
- Uric Acid / Gout
- Cholesterol Screening
- Anemia Screening
- Urine Test Screening
- Obesity Risk
- Metabolic Syndrome
- Dehydration Screening

Some disease areas use trained machine learning models. Others may use rule-based health screening.

---

## Project Folder Structure

Your project should look like this:

```text
HealthStatus/
│
├── app.py
├── train_model.py
├── risk_engine.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   ├── indian_liver_patient.csv
│   ├── kidney_disease.csv
│   ├── cleaned_dataset_Thyroid1.csv
│   ├── healthcare-dataset-stroke-data.csv
│   ├── cardio_data_processed.csv
│   └── patient_logs.csv
│
└── models/
    ├── diabetes_model.joblib
    ├── heart_model.joblib
    ├── liver_model.joblib
    ├── kidney_model.joblib
    ├── thyroid_model.joblib
    ├── stroke_model.joblib
    └── cardio_model.joblib