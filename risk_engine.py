# This file contains the simple health screening logic used by the app.
# The machine-learning model gives one signal, and these rules provide understandable health context.

import numpy as np
import pandas as pd

DISEASE_OPTIONS = [
    "Diabetes / Blood Sugar",
    "Hypertension / Blood Pressure",
    "Heart Disease",
    "Cardiovascular / BP Risk",
    "Stroke Risk",
    "Chronic Kidney Disease",
    "Liver Health",
    "Thyroid Screening",
    "Uric Acid / Gout",
    "Cholesterol / Dyslipidemia",
    "Anemia Screening",
    "UTI / Urine Infection",
    "General Urine Analysis",
    "Obesity Risk",
    "Metabolic Syndrome",
    "Prediabetes / Insulin Resistance",
    "Dehydration Screening",
]

DISEASE_HELP = {
    "Diabetes / Blood Sugar": "Enter glucose, HbA1c, BMI, insulin, and urine sugar/ketone information.",
    "Hypertension / Blood Pressure": "Enter systolic and diastolic blood pressure values.",
    "Heart Disease": "Enter heart-related values such as BP, cholesterol, chest pain type, and max heart rate.",
    "Cardiovascular / BP Risk": "Enter cardiovascular indicators such as blood pressure, cholesterol, glucose, activity, and smoking.",
    "Stroke Risk": "Enter stroke-related risk factors such as age, BP history, glucose, BMI, smoking, and heart disease.",
    "Chronic Kidney Disease": "Enter kidney and urine-related values such as creatinine, eGFR, protein, and CKD dataset-style fields.",
    "Liver Health": "Enter bilirubin, liver enzymes, protein, and albumin-related values.",
    "Thyroid Screening": "Enter thyroid lab values such as TSH, T3, TT4, T4U, and FTI.",
    "Uric Acid / Gout": "Enter uric acid value and joint pain/swelling symptoms.",
    "Cholesterol / Dyslipidemia": "Enter total cholesterol, LDL, HDL, and triglycerides.",
    "Anemia Screening": "Enter hemoglobin and ferritin information.",
    "UTI / Urine Infection": "Enter urine infection indicators and symptoms.",
    "General Urine Analysis": "Enter urine protein, glucose, ketone, blood, and leukocyte results.",
    "Obesity Risk": "Enter BMI and waist circumference.",
    "Metabolic Syndrome": "Enter waist, triglycerides, HDL, fasting glucose, and BP.",
    "Prediabetes / Insulin Resistance": "Enter fasting glucose, insulin, HbA1c, and BMI.",
    "Dehydration Screening": "Enter sodium and dehydration-type symptoms.",
}


def clamp(value, low=0, high=100):
    return max(low, min(float(value), high))


def interpolate(value, low_x, high_x, low_score, high_score):
    # This creates a smooth score, so the chart changes gradually when input changes.
    value = float(value)
    if value <= low_x:
        return low_score
    if value >= high_x:
        return high_score
    return low_score + ((value - low_x) * (high_score - low_score) / (high_x - low_x))


def score_to_label(score):
    score = clamp(score)
    if score < 20:
        return "Safe", "#2ECC71"
    if score < 40:
        return "Mild Risk", "#F1C40F"
    if score < 60:
        return "Moderate Risk", "#F39C12"
    if score < 80:
        return "High Risk", "#E74C3C"
    return "Critical", "#8B0000"


def average_score(scores):
    clean = [clamp(s) for s in scores if s is not None and not pd.isna(s)]
    return clamp(sum(clean) / len(clean)) if clean else 0


def add_ml_score(scores, notes, ml_probability, disease_name):
    # This adds model probability as another risk signal if a trained model is available.
    if ml_probability is not None:
        scores.append(ml_probability * 100)
        notes.append(f"The trained {disease_name} model estimated risk probability around {ml_probability:.2f}.")


def result(score, notes, recommendation):
    label, color = score_to_label(score)
    summary = " ".join(notes[:3]) if notes else "No strong warning was detected from the entered values."
    return {
        "score": round(clamp(score), 1),
        "label": label,
        "color": color,
        "summary": summary,
        "notes": notes,
        "recommendation": recommendation,
    }


def analyze_diabetes(data, ml_probability=None):
    scores, notes = [], []
    glucose = float(data.get("glucose", 0) or 0)
    a1c = data.get("a1c", "")
    bmi = float(data.get("bmi", 0) or 0)
    test_type = data.get("glucose_type", "Fasting")

    if test_type == "Fasting":
        if glucose < 70:
            scores.append(80); notes.append("Fasting glucose is low.")
        elif glucose < 100:
            scores.append(interpolate(glucose, 70, 99, 5, 18)); notes.append("Fasting glucose looks safer.")
        elif glucose < 126:
            scores.append(interpolate(glucose, 100, 125, 35, 58)); notes.append("Fasting glucose is above normal.")
        else:
            scores.append(interpolate(glucose, 126, 260, 65, 95)); notes.append("Fasting glucose is high.")
    else:
        if glucose < 140:
            scores.append(interpolate(glucose, 70, 139, 5, 25)); notes.append("Glucose is not strongly elevated.")
        elif glucose < 200:
            scores.append(interpolate(glucose, 140, 199, 40, 65)); notes.append("Glucose is elevated.")
        else:
            scores.append(interpolate(glucose, 200, 350, 75, 95)); notes.append("Glucose is very high.")

    if str(a1c).strip():
        a1c = float(a1c)
        if a1c < 5.7:
            scores.append(10); notes.append("HbA1c is in the usual normal range.")
        elif a1c < 6.5:
            scores.append(interpolate(a1c, 5.7, 6.4, 35, 60)); notes.append("HbA1c may suggest prediabetes range.")
        else:
            scores.append(interpolate(a1c, 6.5, 10, 70, 95)); notes.append("HbA1c is high.")

    if bmi >= 25:
        scores.append(interpolate(bmi, 25, 40, 25, 60)); notes.append("BMI is elevated and may increase diabetes-related risk.")

    if data.get("urine_glucose") not in [None, "Negative"]:
        scores.append(60); notes.append("Glucose is present in urine.")
    if data.get("ketones") in ["Moderate", "Large"]:
        scores.append(80); notes.append("Moderate or large ketones can be serious.")

    add_ml_score(scores, notes, ml_probability, "diabetes")
    return result(average_score(scores), notes, "Monitor values and discuss repeated abnormal sugar readings with a clinician.")


def analyze_bp(data):
    scores, notes = [], []
    systolic = float(data.get("systolic", 0) or 0)
    diastolic = float(data.get("diastolic", 0) or 0)
    score = max(interpolate(systolic, 110, 180, 5, 100), interpolate(diastolic, 70, 120, 5, 100))
    scores.append(score)

    if systolic < 120 and diastolic < 80:
        notes.append("Blood pressure is in a safer range.")
    elif systolic < 130 and diastolic < 80:
        notes.append("Blood pressure is slightly elevated.")
    elif systolic < 140 or diastolic < 90:
        notes.append("Blood pressure is above normal.")
    elif systolic < 180 and diastolic < 120:
        notes.append("Blood pressure is high.")
    else:
        notes.append("Blood pressure is very high and may need urgent attention.")

    return result(average_score(scores), notes, "Recheck BP properly and seek medical advice for repeated high readings.")


def analyze_heart(data, ml_probability=None):
    scores, notes = [], []
    age = float(data.get("age", 0) or 0)
    bp = float(data.get("resting_bp", data.get("systolic", 0)) or 0)
    cholesterol = float(data.get("cholesterol", 0) or 0)
    chest_pain = data.get("chest_pain", False)

    scores.extend([
        interpolate(age, 25, 80, 10, 70),
        interpolate(bp, 110, 190, 10, 90),
        interpolate(cholesterol, 150, 320, 10, 85),
    ])
    if chest_pain:
        scores.append(80); notes.append("Chest pain or discomfort was reported.")
    if bp >= 140:
        notes.append("Resting blood pressure is high.")
    if cholesterol >= 240:
        notes.append("Total cholesterol is high.")

    add_ml_score(scores, notes, ml_probability, "heart disease")
    return result(average_score(scores), notes, "Chest pain, severe breathlessness, or weakness should be treated urgently.")


def analyze_kidney(data, ml_probability=None):
    scores, notes = [], []
    creatinine = float(data.get("creatinine", 0) or 0)
    egfr = float(data.get("egfr", 0) or 0)
    protein = data.get("urine_protein", "Negative")

    if creatinine > 0:
        scores.append(interpolate(creatinine, 0.6, 5.0, 5, 95))
        if creatinine > 1.3: notes.append("Creatinine is elevated.")
    if egfr > 0:
        scores.append(100 - interpolate(egfr, 15, 120, 95, 5))
        if egfr < 60: notes.append("eGFR is low.")
    if protein != "Negative":
        scores.append(60); notes.append("Protein is present in urine.")

    add_ml_score(scores, notes, ml_probability, "kidney disease")
    return result(average_score(scores), notes, "Kidney-related abnormal values should be reviewed with a clinician.")


def analyze_liver(data, ml_probability=None):
    scores, notes = [], []
    alt = float(data.get("alt", 0) or 0)
    ast = float(data.get("ast", 0) or 0)
    bilirubin = float(data.get("bilirubin", 0) or 0)
    if alt > 0:
        scores.append(interpolate(alt, 20, 200, 5, 95))
        if alt > 40: notes.append("ALT is above the common normal range.")
    if ast > 0:
        scores.append(interpolate(ast, 20, 200, 5, 95))
        if ast > 40: notes.append("AST is above the common normal range.")
    if bilirubin > 0:
        scores.append(interpolate(bilirubin, 0.2, 4.0, 5, 95))
        if bilirubin > 1.2: notes.append("Bilirubin is elevated.")
    add_ml_score(scores, notes, ml_probability, "liver")
    return result(average_score(scores), notes, "Liver markers should be interpreted with symptoms and medical history.")


def analyze_thyroid(data, ml_probability=None):
    scores, notes = [], []
    tsh = float(data.get("tsh", 0) or 0)
    if tsh > 0:
        if 0.4 <= tsh <= 4.0:
            scores.append(12); notes.append("TSH is within a common reference range.")
        elif tsh < 0.4:
            scores.append(interpolate(tsh, 0.0, 0.39, 80, 40)); notes.append("TSH is low.")
        else:
            scores.append(interpolate(tsh, 4.1, 15, 40, 95)); notes.append("TSH is high.")
    add_ml_score(scores, notes, ml_probability, "thyroid")
    return result(average_score(scores), notes, "Thyroid values should be reviewed together, not one number alone.")


def analyze_uric(data):
    scores, notes = [], []
    uric = float(data.get("uric_acid", 0) or 0)
    scores.append(interpolate(uric, 3, 10, 5, 95))
    notes.append("Uric acid is elevated." if uric > 7 else "Uric acid is not strongly elevated.")
    if data.get("joint_pain", False):
        scores.append(70); notes.append("Joint pain or swelling was reported.")
    return result(average_score(scores), notes, "High uric acid with joint symptoms may need gout-related review.")


def analyze_cholesterol(data):
    scores, notes = [], []
    total = float(data.get("cholesterol", data.get("total_cholesterol", 0)) or 0)
    ldl = float(data.get("ldl", 0) or 0)
    hdl = float(data.get("hdl", 0) or 0)
    trig = float(data.get("triglycerides", 0) or 0)
    if total: scores.append(interpolate(total, 150, 320, 5, 95))
    if ldl: scores.append(interpolate(ldl, 70, 220, 5, 95))
    if hdl: scores.append(100 - interpolate(hdl, 30, 70, 90, 5))
    if trig: scores.append(interpolate(trig, 80, 400, 5, 95))
    if total >= 240: notes.append("Total cholesterol is high.")
    elif total >= 200: notes.append("Total cholesterol is borderline high.")
    if ldl >= 160: notes.append("LDL is high.")
    if hdl and hdl < 40: notes.append("HDL is low.")
    if trig >= 200: notes.append("Triglycerides are high.")
    return result(average_score(scores), notes, "Review cholesterol values with overall heart-risk profile.")


def analyze_anemia(data):
    scores, notes = [], []
    hb = float(data.get("hemoglobin", 0) or 0)
    if hb:
        if hb >= 13: scores.append(10); notes.append("Hemoglobin looks acceptable in simple screening.")
        elif hb >= 11: scores.append(40); notes.append("Hemoglobin is mildly low.")
        elif hb >= 8: scores.append(70); notes.append("Hemoglobin is low.")
        else: scores.append(95); notes.append("Hemoglobin is very low.")
    if data.get("fatigue", False):
        scores.append(40); notes.append("Fatigue or weakness was reported.")
    return result(average_score(scores), notes, "Possible anemia should be reviewed with a complete blood count.")


def analyze_urine(data, uti=False):
    scores, notes = [], []
    fields = ["urine_protein", "urine_glucose", "ketones", "blood_urine", "leukocytes", "nitrite"]
    for field in fields:
        value = data.get(field, "Negative")
        if value not in ["Negative", None, ""]:
            scores.append(60); notes.append(f"{field.replace('_', ' ').title()} is positive/present.")
    if uti and data.get("burning", False):
        scores.append(65); notes.append("Burning during urination was reported.")
    if not notes:
        notes.append("No major urine warning was detected from entered values.")
    return result(average_score(scores), notes, "Urine findings should be reviewed with symptoms and medical history.")


def analyze_obesity(data):
    bmi = float(data.get("bmi", 0) or 0)
    waist = float(data.get("waist", 0) or 0)
    scores, notes = [], []
    if bmi: scores.append(interpolate(bmi, 18.5, 40, 5, 95))
    if waist: scores.append(interpolate(waist, 70, 130, 5, 80))
    if bmi >= 30: notes.append("BMI is in the obesity range.")
    elif bmi >= 25: notes.append("BMI is above the normal range.")
    return result(average_score(scores), notes, "Weight risk is best reviewed together with BP, sugar, and cholesterol.")


def analyze_dehydration(data):
    scores, notes = [], []
    sodium = float(data.get("sodium", 0) or 0)
    if sodium > 145:
        scores.append(interpolate(sodium, 146, 160, 50, 90)); notes.append("Sodium is elevated.")
    elif sodium and sodium < 135:
        scores.append(60); notes.append("Sodium is low.")
    else:
        scores.append(10)
    for key in ["dry_mouth", "dizziness", "dark_urine"]:
        if data.get(key, False):
            scores.append(50); notes.append(key.replace("_", " ").title() + " was reported.")
    return result(average_score(scores), notes, "Persistent dehydration symptoms or severe weakness should be medically reviewed.")


def run_analysis(disease_name, data, ml_probability=None):
    if disease_name == "Diabetes / Blood Sugar": return analyze_diabetes(data, ml_probability)
    if disease_name == "Hypertension / Blood Pressure": return analyze_bp(data)
    if disease_name == "Heart Disease": return analyze_heart(data, ml_probability)
    if disease_name == "Cardiovascular / BP Risk": return analyze_heart(data, ml_probability)
    if disease_name == "Stroke Risk": return analyze_heart(data, ml_probability)
    if disease_name == "Chronic Kidney Disease": return analyze_kidney(data, ml_probability)
    if disease_name == "Liver Health": return analyze_liver(data, ml_probability)
    if disease_name == "Thyroid Screening": return analyze_thyroid(data, ml_probability)
    if disease_name == "Uric Acid / Gout": return analyze_uric(data)
    if disease_name == "Cholesterol / Dyslipidemia": return analyze_cholesterol(data)
    if disease_name == "Anemia Screening": return analyze_anemia(data)
    if disease_name == "UTI / Urine Infection": return analyze_urine(data, uti=True)
    if disease_name == "General Urine Analysis": return analyze_urine(data, uti=False)
    if disease_name == "Obesity Risk": return analyze_obesity(data)
    if disease_name == "Metabolic Syndrome": return analyze_heart(data, ml_probability)
    if disease_name == "Prediabetes / Insulin Resistance": return analyze_diabetes(data, ml_probability)
    if disease_name == "Dehydration Screening": return analyze_dehydration(data)
    return result(0, ["No analyzer found."], "No recommendation available.")


def normalize_feature(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def build_ml_input(disease_name, data, feature_order):
    # This maps app inputs to the columns that the trained dataset model expects.
    row = {}
    for col in feature_order:
        key = normalize_feature(col)
        value = np.nan

        mapping = {
            "age": data.get("age"),
            "gender": data.get("gender"),
            "sex": data.get("sex"),
            "pregnancies": data.get("pregnancies"),
            "glucose": data.get("glucose", data.get("fasting_glucose")),
            "bloodpressure": data.get("diastolic", data.get("bp")),
            "bp": data.get("bp", data.get("diastolic")),
            "systolic": data.get("systolic"),
            "diastolic": data.get("diastolic"),
            "skinthickness": data.get("skin_thickness"),
            "insulin": data.get("insulin", data.get("fasting_insulin")),
            "bmi": data.get("bmi"),
            "diabetespedigreefunction": data.get("pedigree"),
            "trtbps": data.get("resting_bp", data.get("systolic")),
            "trestbps": data.get("resting_bp", data.get("systolic")),
            "chol": data.get("cholesterol"),
            "cholesterol": data.get("cholesterol"),
            "fbs": data.get("fbs"),
            "thalachh": data.get("max_hr"),
            "thalach": data.get("max_hr"),
            "oldpeak": data.get("oldpeak"),
            "cp": data.get("cp"),
            "ap_hi": data.get("systolic"),
            "ap_lo": data.get("diastolic"),
            "height": data.get("height"),
            "weight": data.get("weight"),
            "smoke": data.get("smoke"),
            "alco": data.get("alco"),
            "active": data.get("active"),
            "hypertension": data.get("hypertension"),
            "heart_disease": data.get("heart_disease"),
            "ever_married": data.get("ever_married"),
            "work_type": data.get("work_type"),
            "residence_type": data.get("residence_type"),
            "avg_glucose_level": data.get("glucose"),
            "smoking_status": data.get("smoking_status"),
            "total_bilirubin": data.get("bilirubin"),
            "direct_bilirubin": data.get("direct_bilirubin"),
            "alkaline_phosphotase": data.get("alp"),
            "alamine_aminotransferase": data.get("alt"),
            "aspartate_aminotransferase": data.get("ast"),
            "total_protiens": data.get("total_proteins"),
            "total_proteins": data.get("total_proteins"),
            "albumin": data.get("albumin"),
            "albumin_and_globulin_ratio": data.get("ag_ratio"),
            "sg": data.get("sg"),
            "al": data.get("al"),
            "su": data.get("su"),
            "bgr": data.get("bgr", data.get("glucose")),
            "bu": data.get("bu"),
            "sc": data.get("creatinine"),
            "sod": data.get("sodium"),
            "pot": data.get("potassium"),
            "hemo": data.get("hemoglobin"),
            "pcv": data.get("pcv"),
            "wc": data.get("wc"),
            "rc": data.get("rc"),
            "htn": data.get("htn"),
            "dm": data.get("dm"),
            "cad": data.get("cad"),
            "appet": data.get("appet"),
            "pe": data.get("pe"),
            "ane": data.get("ane"),
            "tsh": data.get("tsh"),
            "t3": data.get("t3"),
            "tt4": data.get("tt4"),
            "t4u": data.get("t4u"),
            "fti": data.get("fti"),
        }

        if key in mapping:
            value = mapping[key]
        elif "age" in key:
            value = data.get("age")
        elif "glucose" in key or "sugar" in key:
            value = data.get("glucose", data.get("fasting_glucose"))
        elif "chol" in key:
            value = data.get("cholesterol")
        elif "bmi" in key:
            value = data.get("bmi")
        elif "pressure" in key or key in ["bp", "trtbps", "ap_hi"]:
            value = data.get("systolic", data.get("resting_bp"))

        row[col] = value

    return pd.DataFrame([row], columns=feature_order)


FAQ = {
    "blood pressure": "A common normal blood pressure is below 120/80 mmHg. Very high values like 180/120 or more can be urgent, especially with symptoms.",
    "sugar": "Fasting blood sugar below 100 mg/dL is commonly normal. 100-125 may suggest prediabetes, and 126 or more can be concerning.",
    "glucose": "Glucose means blood sugar. Normal and dangerous levels depend on whether the test was fasting, random, or after meals.",
    "hba1c": "HbA1c below 5.7% is usually normal, 5.7-6.4% may suggest prediabetes, and 6.5% or more may suggest diabetes.",
    "insulin": "Fasting insulin ranges differ by lab, but roughly 2-25 uIU/mL is often used as a broad reference range.",
    "uric acid": "Uric acid above around 7 mg/dL in men or 6 mg/dL in women may be considered high, but lab ranges vary.",
    "creatinine": "Creatinine is a kidney marker. High creatinine may suggest reduced kidney filtering, but it depends on age, sex, and muscle mass.",
    "egfr": "eGFR estimates kidney filtering. Persistent eGFR below 60 may suggest reduced kidney function.",
    "hemoglobin": "Low hemoglobin can suggest anemia. Reference ranges differ by sex and lab.",
    "tsh": "TSH is a thyroid test. A common reference range is about 0.4 to 4.0, but labs vary.",
    "cholesterol": "Total cholesterol below 200 mg/dL is often preferred. LDL is usually better lower, while HDL is generally better higher.",
    "ketones": "Urine ketones can occur with fasting, dehydration, or diabetes. Moderate or large ketones should not be ignored.",
}


def medical_chatbot(question):
    q = question.lower().strip()
    if not q:
        return "Please type a health question."
    for key, answer in FAQ.items():
        if key in q:
            return answer
    return "I can answer simple questions about blood pressure, sugar, glucose, HbA1c, insulin, uric acid, creatinine, eGFR, hemoglobin, TSH, cholesterol, and ketones."
