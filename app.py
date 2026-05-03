import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
    "Liver Health": "models/liver_model.joblib",
}

LOG_PATH = "data/patient_logs.csv"


st.set_page_config(
    page_title="MedScreen AI Agent",
    page_icon="🩺",
    layout="wide",
)


st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #f6fbff 0%, #eef7fb 100%);
}
.hero-box {
    padding: 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f4c81 0%, #1d7db6 100%);
    color: white;
    margin-bottom: 16px;
}
.card-box {
    padding: 16px;
    border-radius: 16px;
    background: white;
    border: 1px solid #d9e6f2;
    box-shadow: 0 4px 16px rgba(20,40,60,0.06);
}
.result-pill {
    padding: 10px 14px;
    border-radius: 999px;
    color: white;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}
.small-note {
    color: #4d6275;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    loaded = {}
    for disease_name, model_path in MODEL_FILES.items():
        if os.path.exists(model_path):
            loaded[disease_name] = joblib.load(model_path)
    return loaded


def save_log(record):
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([record])

    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(LOG_PATH, index=False)


def render_gauge(score, color, label):
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
        }
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def num(label, min_value=0.0, max_value=1000.0, value=0.0, step=1.0):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=step)


def render_disease_form(disease_name):
    data = {}

    if disease_name == "Diabetes / Blood Sugar":
        data["age"] = num("Age", 1, 120, 35, 1)
        data["pregnancies"] = num("Pregnancies (0 if not applicable)", 0, 20, 0, 1)
        data["glucose_type"] = st.selectbox("Blood sugar test type", ["Fasting", "Random", "Post-meal"])
        data["glucose"] = num("Blood sugar / glucose (mg/dL)", 20, 700, 100, 1)
        data["a1c"] = st.text_input("HbA1c (%) if available", "")
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)
        data["insulin"] = num("Insulin if available", 0.0, 900.0, 0.0, 1.0)
        data["skin_thickness"] = num("Skin thickness if available", 0.0, 120.0, 0.0, 1.0)
        data["pedigree"] = num("Diabetes pedigree / family-history score", 0.0, 3.0, 0.5, 0.01)
        data["diastolic"] = num("Diastolic blood pressure", 30.0, 160.0, 80.0, 1.0)
        data["urine_glucose"] = st.selectbox("Urine glucose", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["ketones"] = st.selectbox("Urine ketones", ["Negative", "Trace", "Small", "Moderate", "Large"])

    elif disease_name == "Hypertension / Blood Pressure":
        data["systolic"] = num("Systolic blood pressure", 50.0, 260.0, 120.0, 1.0)
        data["diastolic"] = num("Diastolic blood pressure", 30.0, 160.0, 80.0, 1.0)

    elif disease_name == "Heart Disease":
        data["age"] = num("Age", 1, 120, 45, 1)
        data["sex"] = st.selectbox("Sex code", [0, 1], help="0 = female, 1 = male for most public heart datasets")
        data["cp"] = st.selectbox("Chest pain type code", [0, 1, 2, 3], help="Dataset-style heart pain category")
        data["resting_bp"] = num("Resting blood pressure", 50.0, 260.0, 120.0, 1.0)
        data["cholesterol"] = num("Total cholesterol", 50.0, 700.0, 180.0, 1.0)
        data["fbs"] = st.selectbox("Fasting blood sugar > 120 mg/dL?", [0, 1])
        data["max_hr"] = num("Maximum heart rate achieved", 40.0, 250.0, 150.0, 1.0)
        data["oldpeak"] = num("Oldpeak / ST depression if available", 0.0, 10.0, 0.0, 0.1)
        data["chest_pain"] = st.checkbox("Current chest pain / discomfort")

    elif disease_name == "Chronic Kidney Disease":
        data["creatinine"] = num("Creatinine (mg/dL)", 0.1, 20.0, 1.0, 0.1)
        data["egfr"] = num("eGFR", 1.0, 150.0, 90.0, 1.0)
        data["urine_protein"] = st.selectbox("Urine protein", ["Negative", "Trace", "Small", "Moderate", "Large"])

    elif disease_name == "Liver Health":
        data["age"] = num("Age", 1, 120, 40, 1)
        data["bilirubin"] = num("Total bilirubin", 0.1, 20.0, 0.8, 0.1)
        data["direct_bilirubin"] = num("Direct bilirubin", 0.0, 10.0, 0.2, 0.1)
        data["alp"] = num("ALP", 10.0, 1000.0, 100.0, 1.0)
        data["alt"] = num("ALT", 1.0, 1000.0, 30.0, 1.0)
        data["ast"] = num("AST", 1.0, 1000.0, 30.0, 1.0)
        data["total_proteins"] = num("Total proteins", 1.0, 12.0, 7.0, 0.1)
        data["albumin"] = num("Albumin", 0.5, 7.0, 4.0, 0.1)
        data["ag_ratio"] = num("Albumin / Globulin ratio", 0.1, 5.0, 1.2, 0.1)

    elif disease_name == "Uric Acid / Gout":
        data["uric_acid"] = num("Uric acid (mg/dL)", 1.0, 20.0, 5.5, 0.1)
        data["joint_pain"] = st.checkbox("Joint pain / swelling present")

    elif disease_name == "Cholesterol / Dyslipidemia":
        data["total_cholesterol"] = num("Total cholesterol", 50.0, 700.0, 180.0, 1.0)
        data["ldl"] = num("LDL", 10.0, 400.0, 100.0, 1.0)
        data["hdl"] = num("HDL", 10.0, 120.0, 50.0, 1.0)
        data["triglycerides"] = num("Triglycerides", 20.0, 1000.0, 150.0, 1.0)

    elif disease_name == "Thyroid Screening":
        data["tsh"] = num("TSH", 0.0, 50.0, 2.0, 0.1)
        data["t3"] = num("T3", 0.0, 500.0, 120.0, 1.0)
        data["t4"] = num("T4", 0.0, 30.0, 8.0, 0.1)

    elif disease_name == "Anemia Screening":
        data["hemoglobin"] = num("Hemoglobin", 1.0, 25.0, 13.0, 0.1)
        data["ferritin"] = num("Ferritin if available", 0.0, 1000.0, 50.0, 1.0)
        data["fatigue"] = st.checkbox("Fatigue / weakness present")

    elif disease_name == "UTI / Urine Infection":
        data["leukocytes"] = st.selectbox("Urine leukocytes", ["Negative", "Positive"])
        data["nitrite"] = st.selectbox("Urine nitrite", ["Negative", "Positive"])
        data["blood_urine"] = st.selectbox("Blood in urine", ["Negative", "Positive"])
        data["burning"] = st.checkbox("Burning while urinating")
        data["frequency"] = st.checkbox("Frequent urination")

    elif disease_name == "General Urine Analysis":
        data["protein"] = st.selectbox("Urine protein", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["glucose_urine"] = st.selectbox("Urine glucose", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["ketones"] = st.selectbox("Urine ketones", ["Negative", "Trace", "Small", "Moderate", "Large"])
        data["blood_urine"] = st.selectbox("Blood in urine", ["Negative", "Positive"])
        data["leukocytes"] = st.selectbox("Urine leukocytes", ["Negative", "Positive"])

    elif disease_name == "Obesity Risk":
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)
        data["waist"] = num("Waist circumference (cm)", 40.0, 200.0, 90.0, 1.0)

    elif disease_name == "Metabolic Syndrome":
        data["waist"] = num("Waist circumference (cm)", 40.0, 200.0, 90.0, 1.0)
        data["triglycerides"] = num("Triglycerides", 20.0, 1000.0, 150.0, 1.0)
        data["hdl"] = num("HDL", 10.0, 120.0, 50.0, 1.0)
        data["fasting_glucose"] = num("Fasting glucose", 20.0, 700.0, 95.0, 1.0)
        data["systolic"] = num("Systolic blood pressure", 50.0, 260.0, 120.0, 1.0)
        data["diastolic"] = num("Diastolic blood pressure", 30.0, 160.0, 80.0, 1.0)

    elif disease_name == "Prediabetes / Insulin Resistance":
        data["fasting_glucose"] = num("Fasting glucose", 20.0, 700.0, 95.0, 1.0)
        data["fasting_insulin"] = num("Fasting insulin", 0.0, 300.0, 8.0, 0.1)
        data["a1c"] = num("HbA1c", 3.0, 15.0, 5.4, 0.1)
        data["bmi"] = num("BMI", 10.0, 80.0, 25.0, 0.1)

    elif disease_name == "Dehydration Screening":
        data["sodium"] = num("Sodium", 100.0, 180.0, 140.0, 1.0)
        data["dry_mouth"] = st.checkbox("Dry mouth")
        data["dizziness"] = st.checkbox("Dizziness")
        data["dark_urine"] = st.checkbox("Dark urine")

    return data


models = load_models()

st.markdown("""
<div class="hero-box">
    <h2 style="margin-bottom:6px;">🩺 MedScreen AI Agent</h2>
    <div>This upgraded version uses disease selection, smoother scoring, better visual dashboards, and a simple medical Q&A assistant.</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Health Screening Agent", "Medical Q&A Assistant"])

with tab1:
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)

        disease_name = st.selectbox("Select a disease / health area", DISEASE_OPTIONS)

        st.caption(DISEASE_HELP[disease_name])

        with st.form("screening_form"):
            input_data = render_disease_form(disease_name)
            analyze_button = st.form_submit_button("Analyze Risk")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.subheader("Optional document image")
        uploaded_file = st.file_uploader(
            "Upload lab report / medical photo (reference only)",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        st.info("Image is shown for reference only. The user must enter values manually from the medical report.")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_button:
        ml_probability = None

        if disease_name in models:
            model_package = models[disease_name]
            model_frame = build_ml_input(disease_name, input_data, model_package["features"])

            if model_frame is not None:
                ml_probability = model_package["model"].predict_proba(model_frame)[0][1]

        result = run_analysis(disease_name, input_data, ml_probability)

        result_col1, result_col2 = st.columns([1, 1])

        with result_col1:
            st.markdown(
                f'<div class="result-pill" style="background:{result["color"]};">Final Result: {result["label"]}</div>',
                unsafe_allow_html=True
            )
            render_gauge(result["score"], result["color"], result["label"])

        with result_col2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.subheader("Short Summary")
            st.write(result["summary"])

            st.subheader("Main Findings")
            if result["notes"]:
                for note in result["notes"]:
                    st.write("• " + note)
            else:
                st.write("• No major abnormal note detected from the entered values.")

            st.subheader("Recommendation")
            st.write(result["recommendation"])
            st.markdown('</div>', unsafe_allow_html=True)

        # Save the result.
        log_record = dict(input_data)
        log_record["selected_disease"] = disease_name
        log_record["risk_label"] = result["label"]
        log_record["risk_score"] = result["score"]
        log_record["ml_probability"] = ml_probability
        save_log(log_record)

        st.success("This screening result has been saved to data/patient_logs.csv")

with tab2:
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.subheader("Ask a medical screening question")
    question = st.text_input("Example: What is normal blood pressure?")

    if st.button("Get Answer"):
        answer = medical_chatbot(question)
        st.write(answer)

    st.caption("This is a simple knowledge assistant, not a diagnosis chatbot.")
    st.markdown('</div>', unsafe_allow_html=True)