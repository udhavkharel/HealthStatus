# This file contains:
# 1. disease list
# 2. disease information
# 3. medical FAQ chatbot answers
# 4. risk calculation logic
# 5. ML input row builders

import pandas as pd


# These are the selectable medical screening modules shown in the app.
DISEASE_OPTIONS = [
    "Diabetes / Blood Sugar",
    "Hypertension / Blood Pressure",
    "Heart Disease",
    "Chronic Kidney Disease",
    "Liver Health",
    "Uric Acid / Gout",
    "Cholesterol / Dyslipidemia",
    "Thyroid Screening",
    "Anemia Screening",
    "UTI / Urine Infection",
    "General Urine Analysis",
    "Obesity Risk",
    "Metabolic Syndrome",
    "Prediabetes / Insulin Resistance",
    "Dehydration Screening",
]


# These are short help texts shown in the UI.
DISEASE_HELP = {
    "Diabetes / Blood Sugar": "Use glucose, HbA1c, BMI, and urine findings to estimate sugar-related risk.",
    "Hypertension / Blood Pressure": "Use systolic and diastolic blood pressure values to estimate pressure risk.",
    "Heart Disease": "Uses a mix of heart-check values and optional ML model support if a dataset/model is available.",
    "Chronic Kidney Disease": "Checks creatinine, eGFR, and urine protein to screen kidney health.",
    "Liver Health": "Checks bilirubin and liver enzymes to estimate liver-related concern.",
    "Uric Acid / Gout": "Checks uric acid level and symptoms such as joint pain or swelling.",
    "Cholesterol / Dyslipidemia": "Uses total cholesterol, LDL, HDL, and triglycerides.",
    "Thyroid Screening": "Uses TSH, T3, and T4 values to give a thyroid screening output.",
    "Anemia Screening": "Uses hemoglobin and optional ferritin information for anemia screening.",
    "UTI / Urine Infection": "Uses urine findings such as leukocytes and nitrites plus symptoms.",
    "General Urine Analysis": "Uses urine protein, glucose, ketones, blood, and leukocytes.",
    "Obesity Risk": "Uses BMI and waist circumference for obesity-related risk.",
    "Metabolic Syndrome": "Uses waist, BP, glucose, triglycerides, and HDL.",
    "Prediabetes / Insulin Resistance": "Uses fasting sugar, insulin, HbA1c, and BMI.",
    "Dehydration Screening": "Uses sodium, urine color, and dehydration-type symptoms.",
}


# This controls label and color for the final result.
def score_to_band(score):
    if score < 20:
        return "Safe", "#2ECC71"
    elif score < 40:
        return "Mild Risk", "#F1C40F"
    elif score < 60:
        return "Moderate Risk", "#F39C12"
    elif score < 80:
        return "High Risk", "#E74C3C"
    return "Critical", "#8B0000"


def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(value, maximum))


def combine_scores(score_list):
    # This combines different partial scores and keeps them in 0-100 range.
    if not score_list:
        return 0
    return clamp(sum(score_list) / len(score_list))


def build_result(score, notes, recommendation):
    label, color = score_to_band(score)
    summary = " ".join(notes[:3]) if notes else "No strong abnormal finding was detected from the entered values."

    return {
        "score": round(score, 1),
        "label": label,
        "color": color,
        "summary": summary,
        "notes": notes,
        "recommendation": recommendation,
    }


def interpolate(value, low_x, high_x, low_score, high_score):
    # This makes the score move smoothly, not in a jumpy way.
    if value <= low_x:
        return low_score
    if value >= high_x:
        return high_score
    return low_score + (value - low_x) * (high_score - low_score) / (high_x - low_x)


# -------------------------
# Disease-specific analysis
# -------------------------

def analyze_diabetes(data, ml_probability=None):
    notes = []
    scores = []

    glucose = float(data.get("glucose", 0))
    a1c = data.get("a1c", None)
    bmi = float(data.get("bmi", 0))
    test_type = data.get("glucose_type", "Fasting")
    urine_glucose = data.get("urine_glucose", "Negative")
    ketones = data.get("ketones", "Negative")

    # Smooth glucose-based risk score.
    if test_type == "Fasting":
        if glucose < 70:
            scores.append(80)
            notes.append("Fasting glucose is low and may be dangerous if symptoms are present.")
        elif glucose <= 99:
            scores.append(interpolate(glucose, 70, 99, 5, 15))
            notes.append("Fasting glucose is in the safer range.")
        elif glucose <= 125:
            scores.append(interpolate(glucose, 100, 125, 30, 55))
            notes.append("Fasting glucose is above normal and may suggest prediabetes.")
        else:
            scores.append(interpolate(glucose, 126, 220, 60, 95))
            notes.append("Fasting glucose is in a high range.")
    elif test_type == "Random":
        if glucose < 140:
            scores.append(interpolate(glucose, 70, 139, 5, 25))
            notes.append("Random glucose is not strongly elevated.")
        elif glucose < 200:
            scores.append(interpolate(glucose, 140, 199, 35, 65))
            notes.append("Random glucose is elevated.")
        else:
            scores.append(interpolate(glucose, 200, 350, 75, 95))
            notes.append("Random glucose is very high.")
    else:
        if glucose < 140:
            scores.append(interpolate(glucose, 80, 139, 5, 20))
            notes.append("Post-meal glucose is not strongly elevated.")
        elif glucose < 200:
            scores.append(interpolate(glucose, 140, 199, 35, 65))
            notes.append("Post-meal glucose is elevated.")
        else:
            scores.append(interpolate(glucose, 200, 350, 75, 95))
            notes.append("Post-meal glucose is very high.")

    if a1c not in [None, ""]:
        a1c = float(a1c)
        if a1c < 5.7:
            scores.append(interpolate(a1c, 4.0, 5.6, 5, 15))
            notes.append("HbA1c is within the usual normal range.")
        elif a1c < 6.5:
            scores.append(interpolate(a1c, 5.7, 6.4, 35, 60))
            notes.append("HbA1c is above normal and may suggest prediabetes.")
        else:
            scores.append(interpolate(a1c, 6.5, 10.0, 70, 95))
            notes.append("HbA1c is in a diabetic-range zone.")

    if bmi >= 25:
        scores.append(interpolate(bmi, 25, 40, 20, 55))
        notes.append("BMI is elevated and can increase diabetes-related risk.")

    if urine_glucose != "Negative":
        scores.append(60)
        notes.append("Glucose is present in urine.")

    if ketones in ["Trace", "Small"]:
        scores.append(45)
        notes.append("Ketones are present in urine.")
    elif ketones in ["Moderate", "Large"]:
        scores.append(80)
        notes.append("Moderate or large ketones need urgent attention, especially with diabetes.")

    if ml_probability is not None:
        ml_score = ml_probability * 100
        scores.append(ml_score)
        notes.append(f"ML model estimated diabetes-related probability: {ml_probability:.2f}")

    final_score = combine_scores(scores)
    recommendation = "Please monitor sugar values, repeat abnormal tests if needed, and discuss results with a qualified medical professional."

    return build_result(final_score, notes, recommendation)


def analyze_bp(data):
    notes = []
    scores = []

    systolic = float(data.get("systolic", 0))
    diastolic = float(data.get("diastolic", 0))

    # Use the worse of the two trends.
    sys_score = interpolate(systolic, 110, 180, 5, 100)
    dia_score = interpolate(diastolic, 70, 120, 5, 100)
    scores.append(max(sys_score, dia_score))

    if systolic < 120 and diastolic < 80:
        notes.append("Blood pressure is in the safer range.")
    elif systolic < 130 and diastolic < 80:
        notes.append("Blood pressure is slightly elevated.")
    elif systolic < 140 or diastolic < 90:
        notes.append("Blood pressure is above the normal range.")
    elif systolic < 180 and diastolic < 120:
        notes.append("Blood pressure is high and needs follow-up.")
    else:
        notes.append("Blood pressure is very high and may need urgent attention.")

    recommendation = "Maintain regular monitoring and discuss repeated high readings with a clinician."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_heart(data, ml_probability=None):
    notes = []
    scores = []

    age = float(data.get("age", 0))
    resting_bp = float(data.get("resting_bp", 0))
    cholesterol = float(data.get("cholesterol", 0))
    max_hr = float(data.get("max_hr", 0))
    chest_pain = data.get("chest_pain", False)

    scores.append(interpolate(age, 25, 80, 10, 70))
    scores.append(interpolate(resting_bp, 110, 190, 10, 90))
    scores.append(interpolate(cholesterol, 150, 320, 10, 85))

    if max_hr > 0:
        # Lower achieved max HR during exertion can be a concerning sign in simple screening.
        hr_score = 100 - interpolate(max_hr, 90, 190, 10, 80)
        scores.append(hr_score)

    if chest_pain:
        scores.append(75)
        notes.append("Chest pain was reported and should be taken seriously.")

    if cholesterol >= 240:
        notes.append("Total cholesterol is high.")
    elif cholesterol >= 200:
        notes.append("Total cholesterol is borderline high.")

    if resting_bp >= 140:
        notes.append("Resting blood pressure is high.")

    if ml_probability is not None:
        scores.append(ml_probability * 100)
        notes.append(f"ML model estimated heart-disease probability: {ml_probability:.2f}")

    recommendation = "If chest pain, breathing difficulty, or weakness is present, seek medical attention urgently."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_kidney(data):
    notes = []
    scores = []

    creatinine = float(data.get("creatinine", 0))
    egfr = float(data.get("egfr", 0))
    urine_protein = data.get("urine_protein", "Negative")

    if creatinine > 0:
        scores.append(interpolate(creatinine, 0.6, 5.0, 5, 95))
        if creatinine > 1.3:
            notes.append("Creatinine is elevated.")

    if egfr > 0:
        # Lower eGFR means higher risk.
        egfr_score = 100 - interpolate(egfr, 15, 120, 95, 5)
        scores.append(egfr_score)
        if egfr < 60:
            notes.append("eGFR is lower than usual and may suggest reduced kidney function.")

    if urine_protein != "Negative":
        scores.append(60)
        notes.append("Protein is present in urine.")

    recommendation = "Repeated abnormal kidney-related values should be reviewed by a clinician."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_liver(data, ml_probability=None):
    notes = []
    scores = []

    alt = float(data.get("alt", 0))
    ast = float(data.get("ast", 0))
    bilirubin = float(data.get("bilirubin", 0))

    if alt > 0:
        scores.append(interpolate(alt, 20, 200, 5, 95))
        if alt > 40:
            notes.append("ALT is above the normal range.")

    if ast > 0:
        scores.append(interpolate(ast, 20, 200, 5, 95))
        if ast > 40:
            notes.append("AST is above the normal range.")

    if bilirubin > 0:
        scores.append(interpolate(bilirubin, 0.2, 4.0, 5, 95))
        if bilirubin > 1.2:
            notes.append("Bilirubin is elevated.")

    if ml_probability is not None:
        scores.append(ml_probability * 100)
        notes.append(f"ML model estimated liver-disease probability: {ml_probability:.2f}")

    recommendation = "Abnormal liver markers should be reviewed along with symptoms and medical history."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_uric_acid(data):
    notes = []
    scores = []

    uric_acid = float(data.get("uric_acid", 0))
    joint_pain = data.get("joint_pain", False)

    scores.append(interpolate(uric_acid, 3.0, 10.0, 5, 95))

    if uric_acid > 7.0:
        notes.append("Uric acid is elevated.")
    else:
        notes.append("Uric acid is not strongly elevated.")

    if joint_pain:
        scores.append(70)
        notes.append("Joint pain or swelling was reported.")

    recommendation = "High uric acid with joint symptoms may need follow-up for gout-related evaluation."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_cholesterol(data):
    notes = []
    scores = []

    total = float(data.get("total_cholesterol", 0))
    ldl = float(data.get("ldl", 0))
    hdl = float(data.get("hdl", 0))
    triglycerides = float(data.get("triglycerides", 0))

    if total > 0:
        scores.append(interpolate(total, 150, 320, 5, 95))
    if ldl > 0:
        scores.append(interpolate(ldl, 70, 220, 5, 95))
    if hdl > 0:
        hdl_score = 100 - interpolate(hdl, 30, 70, 90, 5)
        scores.append(hdl_score)
    if triglycerides > 0:
        scores.append(interpolate(triglycerides, 80, 400, 5, 95))

    if total >= 240:
        notes.append("Total cholesterol is high.")
    elif total >= 200:
        notes.append("Total cholesterol is borderline high.")

    if ldl >= 160:
        notes.append("LDL is high.")
    elif ldl >= 130:
        notes.append("LDL is above ideal.")

    if hdl < 40:
        notes.append("HDL is low.")

    if triglycerides >= 200:
        notes.append("Triglycerides are high.")

    recommendation = "Lipid abnormalities should be reviewed with overall cardiovascular risk."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_thyroid(data):
    notes = []
    scores = []

    tsh = float(data.get("tsh", 0))
    t3 = float(data.get("t3", 0))
    t4 = float(data.get("t4", 0))

    if tsh > 0:
        if 0.4 <= tsh <= 4.0:
            scores.append(interpolate(tsh, 0.4, 4.0, 5, 20))
            notes.append("TSH is within the usual normal range.")
        elif tsh < 0.4:
            scores.append(interpolate(tsh, 0.0, 0.39, 70, 35))
            notes.append("TSH is lower than usual.")
        else:
            scores.append(interpolate(tsh, 4.1, 15.0, 40, 95))
            notes.append("TSH is higher than usual.")

    if t3 > 0:
        if not (80 <= t3 <= 200):
            scores.append(55)
            notes.append("T3 appears outside a typical reference range.")

    if t4 > 0:
        if not (5.0 <= t4 <= 12.0):
            scores.append(55)
            notes.append("T4 appears outside a typical reference range.")

    recommendation = "Thyroid results should be interpreted together, not from one value alone."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_anemia(data):
    notes = []
    scores = []

    hemoglobin = float(data.get("hemoglobin", 0))
    ferritin = float(data.get("ferritin", 0))
    fatigue = data.get("fatigue", False)

    if hemoglobin > 0:
        if hemoglobin >= 13:
            scores.append(interpolate(hemoglobin, 13, 18, 10, 5))
            notes.append("Hemoglobin looks acceptable for simple screening.")
        elif hemoglobin >= 11:
            scores.append(interpolate(hemoglobin, 11, 12.9, 55, 30))
            notes.append("Hemoglobin is mildly low.")
        elif hemoglobin >= 8:
            scores.append(interpolate(hemoglobin, 8, 10.9, 85, 60))
            notes.append("Hemoglobin is low and may suggest anemia.")
        else:
            scores.append(95)
            notes.append("Hemoglobin is very low and may need urgent review.")

    if ferritin > 0 and ferritin < 30:
        scores.append(65)
        notes.append("Ferritin is low and may suggest iron deficiency.")

    if fatigue:
        scores.append(40)
        notes.append("Fatigue was reported.")

    recommendation = "Possible anemia should be reviewed using full blood count and clinician advice."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_uti(data):
    notes = []
    scores = []

    leukocytes = data.get("leukocytes", "Negative")
    nitrite = data.get("nitrite", "Negative")
    blood = data.get("blood_urine", "Negative")
    burning = data.get("burning", False)
    frequency = data.get("frequency", False)

    if leukocytes == "Positive":
        scores.append(70)
        notes.append("Leukocytes are present in urine.")
    if nitrite == "Positive":
        scores.append(80)
        notes.append("Nitrite is positive and may suggest bacterial infection.")
    if blood == "Positive":
        scores.append(55)
        notes.append("Blood is present in urine.")
    if burning:
        scores.append(65)
        notes.append("Burning during urination was reported.")
    if frequency:
        scores.append(45)
        notes.append("Frequent urination was reported.")

    if not notes:
        notes.append("No strong UTI-style flag was detected from the entered values.")

    recommendation = "UTI-type symptoms with abnormal urine findings may need urine culture or clinical review."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_general_urine(data):
    notes = []
    scores = []

    for name, field_label, weight in [
        ("protein", "Protein", 60),
        ("glucose_urine", "Glucose", 60),
        ("ketones", "Ketones", 65),
        ("blood_urine", "Blood", 55),
        ("leukocytes", "Leukocytes", 55),
    ]:
        value = data.get(name, "Negative")
        if value != "Negative":
            scores.append(weight)
            notes.append(f"{field_label} is present in urine.")

    if not notes:
        notes.append("General urine strip findings do not show major warnings from the values entered.")

    recommendation = "Urine findings should be interpreted together with symptoms and medical history."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_obesity(data):
    notes = []
    scores = []

    bmi = float(data.get("bmi", 0))
    waist = float(data.get("waist", 0))

    if bmi > 0:
        scores.append(interpolate(bmi, 18.5, 40.0, 5, 95))
        if bmi >= 30:
            notes.append("BMI is in the obesity range.")
        elif bmi >= 25:
            notes.append("BMI is above the normal range.")

    if waist > 0:
        scores.append(interpolate(waist, 70, 130, 5, 80))
        notes.append("Waist circumference contributes to metabolic risk.")

    recommendation = "Weight-related risk improves when reviewed together with BP, sugar, and lipids."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_metabolic(data):
    notes = []
    scores = []

    waist = float(data.get("waist", 0))
    triglycerides = float(data.get("triglycerides", 0))
    hdl = float(data.get("hdl", 0))
    fasting_glucose = float(data.get("fasting_glucose", 0))
    systolic = float(data.get("systolic", 0))
    diastolic = float(data.get("diastolic", 0))

    if waist > 0:
        scores.append(interpolate(waist, 75, 130, 5, 85))
    if triglycerides > 0:
        scores.append(interpolate(triglycerides, 100, 400, 5, 95))
    if hdl > 0:
        hdl_score = 100 - interpolate(hdl, 30, 70, 90, 5)
        scores.append(hdl_score)
    if fasting_glucose > 0:
        scores.append(interpolate(fasting_glucose, 80, 180, 5, 95))

    bp_score = max(interpolate(systolic, 110, 180, 5, 95), interpolate(diastolic, 70, 120, 5, 95))
    scores.append(bp_score)

    notes.append("Metabolic syndrome risk uses waist, glucose, blood pressure, triglycerides, and HDL.")

    recommendation = "Multiple abnormal metabolic markers should be reviewed together."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_prediabetes(data):
    notes = []
    scores = []

    fasting_glucose = float(data.get("fasting_glucose", 0))
    fasting_insulin = float(data.get("fasting_insulin", 0))
    a1c = float(data.get("a1c", 0))
    bmi = float(data.get("bmi", 0))

    if fasting_glucose > 0:
        if fasting_glucose < 100:
            scores.append(interpolate(fasting_glucose, 70, 99, 5, 15))
            notes.append("Fasting glucose is within the safer range.")
        elif fasting_glucose < 126:
            scores.append(interpolate(fasting_glucose, 100, 125, 35, 60))
            notes.append("Fasting glucose may suggest prediabetes.")
        else:
            scores.append(interpolate(fasting_glucose, 126, 200, 70, 95))
            notes.append("Fasting glucose is in a high range.")

    if fasting_insulin > 0:
        scores.append(interpolate(fasting_insulin, 2, 35, 10, 85))
        notes.append("Fasting insulin contributes to insulin-resistance screening.")

    if a1c > 0:
        if a1c < 5.7:
            scores.append(10)
        elif a1c < 6.5:
            scores.append(interpolate(a1c, 5.7, 6.4, 35, 60))
            notes.append("HbA1c may suggest prediabetes.")
        else:
            scores.append(interpolate(a1c, 6.5, 10.0, 70, 95))
            notes.append("HbA1c is high.")

    if bmi >= 25:
        scores.append(interpolate(bmi, 25, 40, 20, 55))

    recommendation = "Insulin resistance and prediabetes should be checked with a clinician, especially if repeated values are abnormal."
    return build_result(combine_scores(scores), notes, recommendation)


def analyze_dehydration(data):
    notes = []
    scores = []

    sodium = float(data.get("sodium", 0))
    dry_mouth = data.get("dry_mouth", False)
    dizziness = data.get("dizziness", False)
    dark_urine = data.get("dark_urine", False)

    if sodium > 0:
        if sodium > 145:
            scores.append(interpolate(sodium, 146, 160, 50, 90))
            notes.append("Sodium is elevated and may fit dehydration.")
        elif sodium < 135:
            scores.append(interpolate(sodium, 120, 134, 80, 45))
            notes.append("Sodium is low and may need review.")
        else:
            scores.append(10)

    if dry_mouth:
        scores.append(40)
        notes.append("Dry mouth was reported.")
    if dizziness:
        scores.append(50)
        notes.append("Dizziness was reported.")
    if dark_urine:
        scores.append(60)
        notes.append("Dark urine was reported.")

    recommendation = "Persistent dehydration symptoms or severe weakness should be medically reviewed."
    return build_result(combine_scores(scores), notes, recommendation)


def run_analysis(disease_name, data, ml_probability=None):
    # This sends the data to the correct screening function.
    if disease_name == "Diabetes / Blood Sugar":
        return analyze_diabetes(data, ml_probability)
    elif disease_name == "Hypertension / Blood Pressure":
        return analyze_bp(data)
    elif disease_name == "Heart Disease":
        return analyze_heart(data, ml_probability)
    elif disease_name == "Chronic Kidney Disease":
        return analyze_kidney(data)
    elif disease_name == "Liver Health":
        return analyze_liver(data, ml_probability)
    elif disease_name == "Uric Acid / Gout":
        return analyze_uric_acid(data)
    elif disease_name == "Cholesterol / Dyslipidemia":
        return analyze_cholesterol(data)
    elif disease_name == "Thyroid Screening":
        return analyze_thyroid(data)
    elif disease_name == "Anemia Screening":
        return analyze_anemia(data)
    elif disease_name == "UTI / Urine Infection":
        return analyze_uti(data)
    elif disease_name == "General Urine Analysis":
        return analyze_general_urine(data)
    elif disease_name == "Obesity Risk":
        return analyze_obesity(data)
    elif disease_name == "Metabolic Syndrome":
        return analyze_metabolic(data)
    elif disease_name == "Prediabetes / Insulin Resistance":
        return analyze_prediabetes(data)
    elif disease_name == "Dehydration Screening":
        return analyze_dehydration(data)

    return build_result(0, ["No analyzer available for the selected disease module."], "No recommendation available.")


# --------------------------------
# ML input row builder for app.py
# --------------------------------

def build_ml_input(disease_name, data, feature_order):
    # This creates the correct input row for the trained ML models.

    row = None

    if disease_name == "Diabetes / Blood Sugar":
        row = {
            "Pregnancies": data.get("pregnancies", 0),
            "Glucose": data.get("glucose", 0),
            "BloodPressure": data.get("diastolic", 0),
            "SkinThickness": data.get("skin_thickness", 0),
            "Insulin": data.get("insulin", 0),
            "BMI": data.get("bmi", 0),
            "DiabetesPedigreeFunction": data.get("pedigree", 0.5),
            "Age": data.get("age", 30),
        }

    elif disease_name == "Heart Disease":
        row = {
            "age": data.get("age", 45),
            "sex": data.get("sex", 1),
            "cp": data.get("cp", 0),
            "trtbps": data.get("resting_bp", 120),
            "chol": data.get("cholesterol", 180),
            "fbs": data.get("fbs", 0),
            "thalachh": data.get("max_hr", 150),
            "oldpeak": data.get("oldpeak", 0.0),
        }

    elif disease_name == "Liver Health":
        row = {
            "Age": data.get("age", 40),
            "Total_Bilirubin": data.get("bilirubin", 0.8),
            "Direct_Bilirubin": data.get("direct_bilirubin", 0.2),
            "Alkaline_Phosphotase": data.get("alp", 100),
            "Alamine_Aminotransferase": data.get("alt", 30),
            "Aspartate_Aminotransferase": data.get("ast", 30),
            "Total_Protiens": data.get("total_proteins", 7.0),
            "Albumin": data.get("albumin", 4.0),
            "Albumin_and_Globulin_Ratio": data.get("ag_ratio", 1.2),
        }

    if row is None:
        return None

    ordered_values = [row.get(col, 0) for col in feature_order]
    return pd.DataFrame([ordered_values], columns=feature_order)


# -------------------------
# Simple medical chatbot
# -------------------------

FAQ = {
    "blood pressure": "Normal blood pressure is usually below 120/80 mmHg. Elevated starts around 120-129 systolic with diastolic below 80. High blood pressure is commonly considered 130/80 or above.",
    "sugar": "A common fasting glucose normal range is below 100 mg/dL. Prediabetes may be 100-125 mg/dL, and 126 mg/dL or above can be concerning.",
    "hb1ac": "HbA1c below 5.7% is usually considered normal. 5.7-6.4% can suggest prediabetes, and 6.5% or more may suggest diabetes.",
    "hba1c": "HbA1c below 5.7% is usually considered normal. 5.7-6.4% can suggest prediabetes, and 6.5% or more may suggest diabetes.",
    "insulin": "Fasting insulin ranges differ across labs, but many references consider roughly 2-25 uIU/mL a broad usual range. Interpretation depends on the lab and the clinical situation.",
    "uric acid": "A common upper limit is around 7 mg/dL in men and around 6 mg/dL in women, but reference ranges can vary by lab.",
    "creatinine": "Creatinine helps screen kidney function. Higher values may suggest reduced kidney filtering, but age, sex, muscle mass, and the lab matter.",
    "egfr": "eGFR estimates kidney filtering. Values below 60 may suggest reduced kidney function if persistent.",
    "hemoglobin": "Low hemoglobin may suggest anemia. Reference ranges differ by sex and lab, but values near or below 12 g/dL often need review.",
    "tsh": "TSH is a common thyroid test. Roughly 0.4 to 4.0 mIU/L is often used as a general reference range, but labs vary.",
    "cholesterol": "Total cholesterol below 200 mg/dL is often preferred. LDL is often ideally below 100 mg/dL, while HDL is generally better when higher.",
    "ketones": "Urine ketones can appear in fasting, dehydration, or uncontrolled diabetes. Moderate or large ketones should not be ignored.",
}


def medical_chatbot(question):
    if not question.strip():
        return "Please type a health question."

    q = question.lower()

    for key, answer in FAQ.items():
        if key in q:
            return answer

    return (
        "I could not find an exact stored answer for that question. "
        "Try asking about blood pressure, sugar, HbA1c, insulin, uric acid, creatinine, eGFR, hemoglobin, TSH, cholesterol, or ketones."
    )