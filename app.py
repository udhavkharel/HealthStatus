# This file creates the Streamlit web app for the health AI agent.

import os  # This lets the app check files and folders.
import joblib  # This loads the saved machine learning model.
import pandas as pd  # This stores submitted health records in table format.
import matplotlib.pyplot as plt  # This draws the colored risk gauge.
import streamlit as st  # This creates the web application.
from PIL import Image  # This lets Streamlit preview uploaded images.

from risk_engine import (  # This imports the risk-checking functions from the helper file.
    blood_pressure_risk,  # This checks blood pressure risk.
    glucose_risk,  # This checks blood sugar risk.
    urine_risk,  # This checks urine-test risk.
    ml_risk,  # This converts ML probability into risk.
    combine_all_risks,  # This combines all risks into one final category.
    make_model_row,  # This creates the correct model input row.
    build_summary,  # This creates a short explanation summary.
)

MODEL_PATH = "models/health_risk_model.joblib"  # This is where the trained model is stored.
LOG_PATH = "data/health_logs.csv"  # This is where submitted user records are saved.

st.set_page_config(  # This sets page information.
    page_title="Health Risk AI Agent",  # This sets the browser tab title.
    page_icon="🩺",  # This sets a medical-style icon.
    layout="wide",  # This gives the app more horizontal space.
)

st.markdown(  # This adds custom CSS styling.
    """
    <style>
    .main {background-color: #f7fbff;}
    .risk-card {padding: 22px; border-radius: 18px; color: white; font-size: 24px; font-weight: 700; text-align: center;}
    .info-box {padding: 16px; border-radius: 14px; background: #ffffff; border: 1px solid #d9e6f2;}
    </style>
    """,
    unsafe_allow_html=True,  # This allows HTML styling inside Streamlit.
)


@st.cache_resource  # This keeps the model loaded so the app runs faster.
def load_model():  # This function loads the trained model.
    if not os.path.exists(MODEL_PATH):  # This checks whether the model file exists.
        return None  # This returns nothing if the model has not been trained yet.
    return joblib.load(MODEL_PATH)  # This loads the saved model package.


def draw_risk_gauge(score, color):  # This creates a simple visual risk gauge.
    fig, ax = plt.subplots(figsize=(7, 1.1))  # This creates a wide small chart.
    ax.barh([0], [100], color="#e8eef5")  # This draws the full gray background bar.
    ax.barh([0], [score], color=color)  # This draws the colored risk amount.
    ax.set_xlim(0, 100)  # This sets the gauge range from 0 to 100.
    ax.set_yticks([])  # This hides the y-axis label.
    ax.set_xlabel("Risk Score")  # This labels the gauge.
    ax.set_title("Visual Risk Representation")  # This titles the gauge.
    ax.grid(axis="x", alpha=0.25)  # This adds light grid lines.
    st.pyplot(fig)  # This shows the chart in Streamlit.


def save_log(record):  # This saves the submitted health record.
    os.makedirs("data", exist_ok=True)  # This creates the data folder if missing.
    df = pd.DataFrame([record])  # This converts the record into a one-row table.
    if os.path.exists(LOG_PATH):  # This checks if a previous log file exists.
        old = pd.read_csv(LOG_PATH)  # This reads old records.
        df = pd.concat([old, df], ignore_index=True)  # This adds the new record to the old records.
    df.to_csv(LOG_PATH, index=False)  # This saves the updated table.


package = load_model()  # This loads the trained model package.

st.title("🩺 Health Risk AI Agent")  # This shows the main title.
st.write("This app accepts health updates, estimates risk category, and gives a short explanation.")  # This explains the app.

st.warning(  # This shows an important medical safety warning.
    "This tool is only a student prototype and not a medical diagnosis. If you have chest pain, severe weakness, breathing trouble, confusion, fainting, or very abnormal readings, seek urgent medical help."
)

left, right = st.columns([1.2, 1])  # This creates two page columns.

with left:  # This starts the left input column.
    st.subheader("1. Enter health updates")  # This creates the input section heading.

    with st.form("health_form"):  # This groups inputs into one submit form.
        age = st.number_input("Age", min_value=1, max_value=120, value=30)  # This asks for age.
        pregnancies = st.number_input("Pregnancies (enter 0 if not applicable)", min_value=0, max_value=25, value=0)  # This asks for pregnancy count.
        systolic = st.number_input("Systolic blood pressure / upper number", min_value=50, max_value=260, value=120)  # This asks for systolic BP.
        diastolic = st.number_input("Diastolic blood pressure / lower number", min_value=30, max_value=160, value=80)  # This asks for diastolic BP.
        glucose_type = st.selectbox("Blood sugar test type", ["Fasting", "Random", "Post-meal / 2-hour"])  # This asks what kind of glucose test it is.
        glucose = st.number_input("Blood sugar / glucose level (mg/dL)", min_value=20, max_value=700, value=100)  # This asks for blood sugar.
        a1c_text = st.text_input("A1C value if available, otherwise leave blank", value="")  # This asks for optional A1C.
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0)  # This asks for BMI.
        insulin = st.number_input("Insulin value if available, otherwise keep 0", min_value=0.0, max_value=900.0, value=0.0)  # This asks for insulin.
        skin_thickness = st.number_input("Skin thickness if available, otherwise keep 0", min_value=0.0, max_value=120.0, value=0.0)  # This asks for skin thickness.
        pedigree = st.number_input("Diabetes pedigree / family-history score if available", min_value=0.0, max_value=3.0, value=0.5)  # This asks for pedigree score.

        st.subheader("2. Urine test indicators")  # This creates urine-test section heading.
        protein = st.selectbox("Urine protein", ["Negative", "Trace", "Small", "Moderate", "Large"])  # This asks urine protein.
        glucose_urine = st.selectbox("Urine glucose", ["Negative", "Trace", "Small", "Moderate", "Large"])  # This asks urine glucose.
        ketones = st.selectbox("Urine ketones", ["Negative", "Trace", "Small", "Moderate", "Large"])  # This asks urine ketones.
        blood = st.selectbox("Blood in urine", ["Negative", "Positive"])  # This asks blood in urine.
        leukocytes = st.selectbox("Leukocytes in urine", ["Negative", "Positive"])  # This asks leukocytes.

        st.subheader("3. Serious symptoms")  # This creates symptoms section.
        emergency_symptoms = st.checkbox("Chest pain, shortness of breath, weakness, vision change, difficulty speaking, confusion, or fainting")  # This asks emergency symptoms.

        submitted = st.form_submit_button("Analyze Risk")  # This creates the submit button.

with right:  # This starts the right image/upload column.
    st.subheader("Optional report image")  # This creates the image section heading.
    uploaded_file = st.file_uploader("Upload lab report or urine strip photo for preview only", type=["png", "jpg", "jpeg"])  # This lets user upload an image.
    if uploaded_file is not None:  # This checks if a file was uploaded.
        image = Image.open(uploaded_file)  # This opens the uploaded image.
        st.image(image, caption="Uploaded image preview only. Please enter values manually.", use_column_width=True)  # This displays image.
    st.info("Image is shown for reference only. This prototype does not diagnose from images.")  # This explains image limitation.

if submitted:  # This runs analysis after the user presses submit.
    a1c = float(a1c_text) if a1c_text.strip() else None  # This converts A1C text into a number if entered.

    user_data = {  # This stores all user inputs in one dictionary.
        "age": age,  # This stores age.
        "pregnancies": pregnancies,  # This stores pregnancies.
        "systolic": systolic,  # This stores systolic pressure.
        "diastolic": diastolic,  # This stores diastolic pressure.
        "glucose_type": glucose_type,  # This stores test type.
        "glucose": glucose,  # This stores glucose level.
        "a1c": a1c,  # This stores A1C.
        "bmi": bmi,  # This stores BMI.
        "insulin": insulin,  # This stores insulin.
        "skin_thickness": skin_thickness,  # This stores skin thickness.
        "pedigree": pedigree,  # This stores pedigree score.
        "protein": protein,  # This stores urine protein.
        "glucose_urine": glucose_urine,  # This stores urine glucose.
        "ketones": ketones,  # This stores urine ketones.
        "blood": blood,  # This stores blood in urine.
        "leukocytes": leukocytes,  # This stores leukocytes.
        "emergency_symptoms": emergency_symptoms,  # This stores symptom flag.
    }

    probability = None  # This creates an empty ML probability.
    if package is not None:  # This checks if the trained model exists.
        model = package["model"]  # This gets the trained model.
        features = package["features"]  # This gets the expected feature order.
        model_row = make_model_row(user_data, features)  # This builds the input row for prediction.
        probability = model.predict_proba(model_row)[0][1]  # This predicts risk probability.

    bp_level, bp_notes = blood_pressure_risk(systolic, diastolic, emergency_symptoms)  # This checks BP risk.
    sugar_level, sugar_notes = glucose_risk(glucose, glucose_type, a1c)  # This checks glucose risk.
    urine_level, urine_notes = urine_risk(protein, glucose_urine, ketones, blood, leukocytes)  # This checks urine risk.
    model_level, model_notes = ml_risk(probability)  # This checks ML model risk.

    all_notes = bp_notes + sugar_notes + urine_notes + model_notes  # This combines all explanation notes.
    category, score, color = combine_all_risks(bp_level, sugar_level, urine_level, model_level)  # This creates final risk result.
    summary = build_summary(category, all_notes)  # This creates a short summary.

    st.markdown(f"<div class='risk-card' style='background:{color};'>Final Risk Category: {category}</div>", unsafe_allow_html=True)  # This shows risk category.
    draw_risk_gauge(score, color)  # This shows the visual risk gauge.

    st.subheader("Short Summary")  # This creates summary heading.
    st.write(summary)  # This shows the summary.

    st.subheader("Main Findings")  # This creates finding heading.
    for note in all_notes:  # This loops through all notes.
        st.write("• " + note)  # This shows each note.

    st.subheader("Suggested Next Step")  # This creates next-step heading.
    if category == "Critical":  # This checks critical category.
        st.error("Seek urgent medical help now, especially if serious symptoms are present.")  # This gives urgent instruction.
    elif category == "High Risk":  # This checks high-risk category.
        st.warning("Please arrange medical follow-up soon and recheck abnormal readings.")  # This gives high-risk guidance.
    elif category == "Moderate Risk":  # This checks moderate category.
        st.info("Monitor values, repeat tests if needed, and discuss results with a clinician.")  # This gives moderate guidance.
    else:  # This handles lower categories.
        st.success("Continue normal monitoring and healthy habits.")  # This gives lower-risk guidance.

    log_record = user_data.copy()  # This copies the input data for saving.
    log_record["ml_probability"] = probability  # This adds the ML probability to the saved record.
    log_record["risk_category"] = category  # This adds the final category to the saved record.
    log_record["risk_score"] = score  # This adds the risk score to the saved record.
    save_log(log_record)  # This saves the record to a CSV database.

    st.caption("Record saved locally in data/health_logs.csv")  # This confirms saving.