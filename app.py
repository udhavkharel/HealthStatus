# This is the main Streamlit app file.
# It shows disease options, collects user values, runs ML + rule screening, and displays visual results.

import os
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from risk_engine import (
    DISEASE_OPTIONS,
    DISEASE_HELP,
    run_analysis,
    build_ml_input,
    medical_chatbot,
)

MODEL_FILES = {
    "Diabetes / Blood Sugar": "models/diabetes_model.joblib",
    "Heart Disease": "models/heart_model.joblib",
    "Cardiovascular / BP Risk": "models/cardio_model.joblib",
    "Stroke Risk": "models/stroke_model.joblib",
    "Chronic Kidney Disease": "models/kidney_model.joblib",
    "Liver Health": "models/liver_model.joblib",
    "Thyroid Screening": "models/thyroid_model.joblib",
}

LOG_PATH = "data/patient_logs.csv"

st.set_page_config(page_title="MedScreen AI Agent", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f6fbff 0%, #eef7fb 100%);}
    .hero-box {padding: 22px; border-radius: 20px; background: linear-gradient(135deg, #0f4c81 0%, #1d7db6 100%); color: white; margin-bottom: 18px;}
    .card-box {padding: 18px; border-radius: 18px; background: white; border: 1px solid #d9e6f2; box-shadow: 0 4px 16px rgba(20,40,60,0.08);}
    .result-pill {padding: 12px 18px; border-radius: 999px; color: white; font-weight: 800; text-align: center; margin-bottom: 12px; font-size: 20px;}
    .metric-small {font-size: 14px; color: #4d6275;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    # This loads all .joblib models that exist inside the models folder.
    loaded = {}
    for disease_name, model_path in MODEL_FILES.items():
        if os.path.exists(model_path):
            loaded[disease_name] = joblib.load(model_path)
    return loaded


def save_log(record):
    # This saves the user's screening result locally.
    os.makedirs("data", exist_ok=True)
    new_row = pd.DataFrame([record])
    if os.path.exists(LOG_PATH):
        old_rows = pd.read_csv(LOG_PATH)
        new_row = pd.concat([old_rows, new_row], ignore_index=True)
    new_row.to_csv(LOG_PATH, index=False)


def render_gauge(score, color, label):
    # This creates a medical-style gauge chart from 0 to 100.
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100"},
        title={"text": f"Risk Score • {label}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 20], "color": "#d4efdf"},
                {"range": [20, 40], "color": "#fcf3cf"},
                {"range": [40, 60], "color": "#fdebd0"},
                {"range": [60, 80], "color": "#fadbd8"},
                {"range": [80, 100], "color": "#f5b7b1"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def num(label, min_value=0.0, max_value=1000.0, value=0.0, step=1.0):
    # This is a shorter helper for number inputs.
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step)


def yes_no(label):
    # This returns yes/no values in a format many datasets understand.
    return st.selectbox(label, ["no", "yes"])


def render_disease_form(disease_name):
    # This shows only the fields needed for the selected disease area.
    data = {}

    if disease_name == "Diabetes / Blood Sugar":
        data["age"] = num("Age", 1, 120, 35, 1)
        data["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"])
        data["pregnancies"] = num("Pregnancies (0 if not applicable)", 0, 25, 0, 1)
        data["glucose_type"] = st.selectbox("Blood sugar test type", ["Fasting", "Random", "Post-meal"])
        data["glucose"] = num("Blood sugar / glucose (mg/dL)", 20, 700, 100, 1)
        data["a1c"] = st.text_input("HbA1c (%) if available", "")
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)
        data["insulin"] = num("Insulin if available", 0.0, 900.0, 0.0, 1.0)
        data["skin_thickness"] = num("Skin thickness if available", 0.0, 120.0, 0.0, 1.0)
        data["pedigree"] = num("Diabetes pedigree / family-history score", 0.0, 3.0, 0.5, 0.01)
        data["diastolic"] = num("Diastolic BP", 30.0, 160.0, 80.0, 1.0)
        data["urine_glucose"] = st.selectbox("Urine glucose", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["ketones"] = st.selectbox("Urine ketones", ["Negative", "Trace", "Small", "Moderate", "Large"])

    elif disease_name == "Hypertension / Blood Pressure":
        data["systolic"] = num("Systolic BP / upper number", 50.0, 260.0, 120.0, 1.0)
        data["diastolic"] = num("Diastolic BP / lower number", 30.0, 160.0, 80.0, 1.0)

    elif disease_name in ["Heart Disease", "Cardiovascular / BP Risk", "Stroke Risk", "Metabolic Syndrome"]:
        data["age"] = num("Age", 1, 120, 45, 1)
        data["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"])
        data["sex"] = 1 if data["gender"] == "Male" else 0
        data["systolic"] = num("Systolic BP", 50.0, 260.0, 120.0, 1.0)
        data["diastolic"] = num("Diastolic BP", 30.0, 160.0, 80.0, 1.0)
        data["resting_bp"] = data["systolic"]
        data["cholesterol"] = num("Total cholesterol", 50.0, 700.0, 180.0, 1.0)
        data["glucose"] = num("Glucose / sugar", 20.0, 700.0, 100.0, 1.0)
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)
        data["max_hr"] = num("Maximum heart rate if available", 40.0, 250.0, 150.0, 1.0)
        data["oldpeak"] = num("Oldpeak / ST depression if available", 0.0, 10.0, 0.0, 0.1)
        data["cp"] = st.selectbox("Chest pain type code", [0, 1, 2, 3])
        data["fbs"] = st.selectbox("Fasting sugar above 120?", [0, 1])
        data["chest_pain"] = st.checkbox("Chest pain / discomfort now")
        data["hypertension"] = st.selectbox("Known hypertension?", [0, 1])
        data["heart_disease"] = st.selectbox("Known heart disease?", [0, 1])
        data["ever_married"] = st.selectbox("Ever married?", ["Yes", "No"])
        data["work_type"] = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        data["residence_type"] = st.selectbox("Residence type", ["Urban", "Rural"])
        data["smoking_status"] = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        data["smoke"] = 1 if data["smoking_status"] == "smokes" else 0
        data["alco"] = st.selectbox("Alcohol use?", [0, 1])
        data["active"] = st.selectbox("Physically active?", [1, 0])
        data["height"] = num("Height in cm", 80.0, 230.0, 170.0, 1.0)
        data["weight"] = num("Weight in kg", 20.0, 250.0, 70.0, 1.0)
        data["triglycerides"] = num("Triglycerides if available", 20.0, 1000.0, 150.0, 1.0)
        data["hdl"] = num("HDL if available", 10.0, 120.0, 50.0, 1.0)
        data["waist"] = num("Waist circumference if available", 40.0, 200.0, 90.0, 1.0)

    elif disease_name == "Chronic Kidney Disease":
        data["age"] = num("Age", 1, 120, 45, 1)
        data["bp"] = num("Blood pressure", 40.0, 250.0, 80.0, 1.0)
        data["sg"] = num("Specific gravity", 1.000, 1.030, 1.010, 0.001)
        data["al"] = num("Albumin", 0.0, 5.0, 0.0, 1.0)
        data["su"] = num("Sugar in urine", 0.0, 5.0, 0.0, 1.0)
        data["bgr"] = num("Blood glucose random", 20.0, 700.0, 120.0, 1.0)
        data["bu"] = num("Blood urea", 1.0, 400.0, 40.0, 1.0)
        data["creatinine"] = num("Serum creatinine", 0.1, 20.0, 1.0, 0.1)
        data["sodium"] = num("Sodium", 100.0, 180.0, 140.0, 1.0)
        data["potassium"] = num("Potassium", 1.0, 10.0, 4.0, 0.1)
        data["hemoglobin"] = num("Hemoglobin", 1.0, 25.0, 13.0, 0.1)
        data["pcv"] = num("Packed cell volume", 10.0, 70.0, 40.0, 1.0)
        data["wc"] = num("White blood cell count", 1000.0, 30000.0, 8000.0, 100.0)
        data["rc"] = num("Red blood cell count", 1.0, 10.0, 5.0, 0.1)
        data["htn"] = yes_no("Hypertension?")
        data["dm"] = yes_no("Diabetes mellitus?")
        data["cad"] = yes_no("Coronary artery disease?")
        data["appet"] = st.selectbox("Appetite", ["good", "poor"])
        data["pe"] = yes_no("Pedal edema?")
        data["ane"] = yes_no("Anemia?")
        data["urine_protein"] = st.selectbox("Urine protein", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["egfr"] = num("eGFR if available", 1.0, 150.0, 90.0, 1.0)

    elif disease_name == "Liver Health":
        data["age"] = num("Age", 1, 120, 40, 1)
        data["gender"] = st.selectbox("Gender", ["Male", "Female"])
        data["bilirubin"] = num("Total bilirubin", 0.1, 20.0, 0.8, 0.1)
        data["direct_bilirubin"] = num("Direct bilirubin", 0.0, 10.0, 0.2, 0.1)
        data["alp"] = num("Alkaline phosphotase / ALP", 10.0, 1500.0, 100.0, 1.0)
        data["alt"] = num("ALT", 1.0, 1500.0, 30.0, 1.0)
        data["ast"] = num("AST", 1.0, 1500.0, 30.0, 1.0)
        data["total_proteins"] = num("Total proteins", 1.0, 12.0, 7.0, 0.1)
        data["albumin"] = num("Albumin", 0.5, 7.0, 4.0, 0.1)
        data["ag_ratio"] = num("Albumin / Globulin ratio", 0.1, 5.0, 1.2, 0.1)

    elif disease_name == "Thyroid Screening":
        data["age"] = num("Age", 1, 120, 35, 1)
        data["gender"] = st.selectbox("Gender", ["Male", "Female"])
        data["sex"] = data["gender"]
        data["tsh"] = num("TSH", 0.0, 100.0, 2.0, 0.1)
        data["t3"] = num("T3", 0.0, 500.0, 120.0, 1.0)
        data["tt4"] = num("TT4", 0.0, 400.0, 100.0, 1.0)
        data["t4u"] = num("T4U", 0.0, 3.0, 1.0, 0.1)
        data["fti"] = num("FTI", 0.0, 400.0, 100.0, 1.0)

    elif disease_name == "Uric Acid / Gout":
        data["uric_acid"] = num("Uric acid", 1.0, 20.0, 5.5, 0.1)
        data["joint_pain"] = st.checkbox("Joint pain / swelling")

    elif disease_name == "Cholesterol / Dyslipidemia":
        data["cholesterol"] = num("Total cholesterol", 50.0, 700.0, 180.0, 1.0)
        data["ldl"] = num("LDL", 10.0, 400.0, 100.0, 1.0)
        data["hdl"] = num("HDL", 10.0, 120.0, 50.0, 1.0)
        data["triglycerides"] = num("Triglycerides", 20.0, 1000.0, 150.0, 1.0)

    elif disease_name == "Anemia Screening":
        data["hemoglobin"] = num("Hemoglobin", 1.0, 25.0, 13.0, 0.1)
        data["fatigue"] = st.checkbox("Fatigue / weakness")

    elif disease_name in ["UTI / Urine Infection", "General Urine Analysis"]:
        data["urine_protein"] = st.selectbox("Urine protein", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["urine_glucose"] = st.selectbox("Urine glucose", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["ketones"] = st.selectbox("Urine ketones", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["blood_urine"] = st.selectbox("Blood in urine", ["Negative", "Positive"])
        data["leukocytes"] = st.selectbox("Leukocytes", ["Negative", "Positive"])
        data["nitrite"] = st.selectbox("Nitrite", ["Negative", "Positive"])
        data["burning"] = st.checkbox("Burning while urinating")

    elif disease_name == "Obesity Risk":
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)
        data["waist"] = num("Waist circumference", 40.0, 200.0, 90.0, 1.0)

    elif disease_name == "Prediabetes / Insulin Resistance":
        data["glucose_type"] = "Fasting"
        data["glucose"] = num("Fasting glucose", 20.0, 700.0, 95.0, 1.0)
        data["a1c"] = st.text_input("HbA1c", "5.4")
        data["insulin"] = num("Fasting insulin", 0.0, 300.0, 8.0, 0.1)
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)

    elif disease_name == "Dehydration Screening":
        data["sodium"] = num("Sodium", 100.0, 180.0, 140.0, 1.0)
        data["dry_mouth"] = st.checkbox("Dry mouth")
        data["dizziness"] = st.checkbox("Dizziness")
        data["dark_urine"] = st.checkbox("Dark urine")

    return data


models = load_models()

st.markdown(
    """
    <div class="hero-box">
        <h2 style="margin-bottom:6px;">🩺 MedScreen AI Agent</h2>
        <div>Select a disease area, enter values from a medical report, and receive a simple color-coded screening result.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning("This is an educational screening prototype, not a diagnosis. Serious symptoms or abnormal results should be reviewed by a qualified medical professional.")

tab1, tab2 = st.tabs(["Health Screening Agent", "Medical Q&A Assistant"])

with tab1:
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        disease_name = st.selectbox("Select disease / health area", DISEASE_OPTIONS)
        st.caption(DISEASE_HELP[disease_name])

        with st.form("screening_form"):
            input_data = render_disease_form(disease_name)
            analyze_button = st.form_submit_button("Analyze Risk")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.subheader("Optional report image")
        uploaded_file = st.file_uploader("Upload lab report / medical photo", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        st.info("Image is shown for reference only. Please manually enter values from the report.")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_button:
        ml_probability = None
        ml_message = "No trained model found for this module; rule-based screening was used."

        if disease_name in models:
            package = models[disease_name]
            model_input = build_ml_input(disease_name, input_data, package["features"])
            probabilities = package["model"].predict_proba(model_input)
            classes = package.get("classes", [])

            if probabilities.shape[1] == 2:
                ml_probability = float(probabilities[0][1])
            else:
                ml_probability = float(probabilities[0].max())

            ml_message = f"Model loaded: {os.path.basename(MODEL_FILES[disease_name])}"

        result = run_analysis(disease_name, input_data, ml_probability)

        result_col, summary_col = st.columns([1, 1])

        with result_col:
            st.markdown(
                f'<div class="result-pill" style="background:{result["color"]};">Final Result: {result["label"]}</div>',
                unsafe_allow_html=True,
            )
            render_gauge(result["score"], result["color"], result["label"])
            st.caption(ml_message)

        with summary_col:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.subheader("Short Summary")
            st.write(result["summary"])
            st.subheader("Main Findings")
            for note in result["notes"]:
                st.write("• " + note)
            st.subheader("Suggested Next Step")
            st.write(result["recommendation"])
            st.markdown('</div>', unsafe_allow_html=True)

        log_record = dict(input_data)
        log_record["selected_disease"] = disease_name
        log_record["risk_label"] = result["label"]
        log_record["risk_score"] = result["score"]
        log_record["ml_probability"] = ml_probability
        save_log(log_record)
        st.success("Saved locally to data/patient_logs.csv")

with tab2:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("Ask a simple health screening question")
    question = st.text_input("Example: What is normal blood pressure?")
    if st.button("Get Answer"):
        st.write(medical_chatbot(question))
    st.caption("This is a simple FAQ assistant, not a medical diagnosis chatbot.")
    st.markdown('</div>', unsafe_allow_html=True)
