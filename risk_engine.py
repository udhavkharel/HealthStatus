# This file contains the health-risk logic used by the AI agent.

import pandas as pd  # This lets us create a row of data for the trained model.

RISK_COLORS = {  # These colors are used in the UI for each risk level.
    "Safe": "#2ECC71",  # Green means the values look safer.
    "Mild Risk": "#F1C40F",  # Yellow means slight attention is needed.
    "Moderate Risk": "#E67E22",  # Orange means follow-up is recommended.
    "High Risk": "#E74C3C",  # Red means the result should be taken seriously.
    "Critical": "#8B0000",  # Dark red means urgent medical attention may be needed.
}

RISK_SCORES = {  # These numbers help combine different risk signals.
    "Safe": 10,  # Lowest risk score.
    "Mild Risk": 30,  # Low but noticeable risk score.
    "Moderate Risk": 55,  # Medium risk score.
    "High Risk": 80,  # High risk score.
    "Critical": 100,  # Highest risk score.
}


def blood_pressure_risk(systolic, diastolic, emergency_symptoms):  # This checks blood pressure risk.
    notes = []  # This list stores simple explanation messages.

    if systolic >= 180 or diastolic >= 120:  # This checks for very high blood pressure.
        if emergency_symptoms:  # This checks if dangerous symptoms were reported.
            notes.append("Blood pressure is in a possible emergency range with serious symptoms.")  # This explains the emergency risk.
            return "Critical", notes  # This returns the highest risk level.
        notes.append("Blood pressure is very high and should be rechecked and discussed urgently with a clinician.")  # This explains severe high blood pressure.
        return "High Risk", notes  # This returns high risk.

    if systolic >= 140 or diastolic >= 90:  # This checks stage 2 style high blood pressure.
        notes.append("Blood pressure is high and may need medical follow-up.")  # This explains the finding.
        return "High Risk", notes  # This returns high risk.

    if systolic >= 130 or diastolic >= 80:  # This checks stage 1 style high blood pressure.
        notes.append("Blood pressure is above the normal range.")  # This explains the finding.
        return "Moderate Risk", notes  # This returns moderate risk.

    if 120 <= systolic <= 129 and diastolic < 80:  # This checks elevated blood pressure.
        notes.append("Blood pressure is slightly elevated.")  # This explains the finding.
        return "Mild Risk", notes  # This returns mild risk.

    notes.append("Blood pressure is in a safer range based on the entered values.")  # This explains normal-looking BP.
    return "Safe", notes  # This returns safe.


def glucose_risk(glucose, test_type, a1c):  # This checks blood-sugar risk.
    notes = []  # This stores explanation messages.
    level = "Safe"  # This starts with the lowest risk level.

    if glucose < 70:  # This checks for low blood sugar.
        notes.append("Blood sugar is low; low glucose can become serious if symptoms are present.")  # This explains low glucose.
        level = "High Risk"  # This marks high risk.

    elif test_type == "Fasting":  # This checks fasting-glucose rules.
        if glucose >= 126:  # This checks high fasting glucose.
            notes.append("Fasting blood sugar is in a high range and needs professional review.")  # This explains high fasting glucose.
            level = "High Risk"  # This marks high risk.
        elif glucose >= 100:  # This checks borderline fasting glucose.
            notes.append("Fasting blood sugar is above the usual normal range.")  # This explains borderline glucose.
            level = "Moderate Risk"  # This marks moderate risk.
        else:  # This handles safer fasting glucose.
            notes.append("Fasting blood sugar is in a safer range based on the entered value.")  # This explains safer glucose.

    elif test_type == "Random":  # This checks random-glucose rules.
        if glucose >= 200:  # This checks high random glucose.
            notes.append("Random blood sugar is very high and needs medical review, especially if symptoms are present.")  # This explains high random glucose.
            level = "High Risk"  # This marks high risk.
        elif glucose >= 140:  # This checks moderately high random glucose.
            notes.append("Random blood sugar is raised and should be followed up.")  # This explains raised random glucose.
            level = "Moderate Risk"  # This marks moderate risk.
        else:  # This handles lower random glucose.
            notes.append("Random blood sugar is not strongly elevated based on the entered value.")  # This explains lower random glucose.

    else:  # This handles post-meal glucose values.
        if glucose >= 200:  # This checks high post-meal glucose.
            notes.append("Post-meal blood sugar is high and should be reviewed with a clinician.")  # This explains high post-meal glucose.
            level = "High Risk"  # This marks high risk.
        elif glucose >= 140:  # This checks borderline post-meal glucose.
            notes.append("Post-meal blood sugar is above the usual target range.")  # This explains borderline post-meal glucose.
            level = "Moderate Risk"  # This marks moderate risk.
        else:  # This handles lower post-meal glucose.
            notes.append("Post-meal blood sugar is in a safer range based on the entered value.")  # This explains safer post-meal glucose.

    if a1c is not None:  # This checks whether the user entered A1C.
        if a1c >= 6.5:  # This checks high A1C.
            notes.append("A1C is high and should be discussed with a medical professional.")  # This explains high A1C.
            level = max_risk(level, "High Risk")  # This increases risk if needed.
        elif a1c >= 5.7:  # This checks borderline A1C.
            notes.append("A1C is above the usual normal range.")  # This explains raised A1C.
            level = max_risk(level, "Moderate Risk")  # This increases risk if needed.

    return level, notes  # This returns glucose risk and messages.


def urine_risk(protein, glucose_urine, ketones, blood, leukocytes):  # This checks urine-test risk indicators.
    notes = []  # This stores urine-test explanations.
    level = "Safe"  # This starts with safe.

    positive_values = ["Trace", "Small", "Moderate", "Large", "Positive"]  # These mean the test is not negative.
    strong_values = ["Moderate", "Large"]  # These mean stronger abnormal results.

    if protein in positive_values:  # This checks urine protein.
        notes.append("Protein was found in urine, which can be related to kidney or urinary issues.")  # This explains protein.
        level = max_risk(level, "Moderate Risk")  # This raises risk.

    if glucose_urine in positive_values:  # This checks urine glucose.
        notes.append("Glucose was found in urine, which may occur with high blood sugar or kidney-related issues.")  # This explains glucose.
        level = max_risk(level, "Moderate Risk")  # This raises risk.

    if ketones in positive_values:  # This checks urine ketones.
        notes.append("Ketones were found in urine; this can be important for people with diabetes or poor intake.")  # This explains ketones.
        level = max_risk(level, "Moderate Risk")  # This raises risk.

    if ketones in strong_values:  # This checks stronger ketone results.
        notes.append("Moderate or large ketones can be more concerning and should not be ignored.")  # This explains stronger ketones.
        level = max_risk(level, "High Risk")  # This raises risk more.

    if blood == "Positive":  # This checks blood in urine.
        notes.append("Blood was reported in urine and should be reviewed by a clinician.")  # This explains blood in urine.
        level = max_risk(level, "Moderate Risk")  # This raises risk.

    if leukocytes == "Positive":  # This checks leukocytes in urine.
        notes.append("Leukocytes were reported in urine, which can happen with infection or inflammation.")  # This explains leukocytes.
        level = max_risk(level, "Moderate Risk")  # This raises risk.

    if not notes:  # This checks if no urine warnings were found.
        notes.append("Urine indicators entered here did not show a major warning signal.")  # This explains safer urine result.

    return level, notes  # This returns urine risk and messages.


def ml_risk(probability):  # This converts ML probability into a risk level.
    if probability is None:  # This checks if the model was not available.
        return "Safe", ["ML model was not loaded, so only rule-based screening was used."]  # This explains missing model.

    if probability >= 0.75:  # This checks very high model probability.
        return "High Risk", [f"ML model estimated high diabetes-related risk probability: {probability:.2f}."]  # This returns high risk.

    if probability >= 0.50:  # This checks moderate-high model probability.
        return "Moderate Risk", [f"ML model estimated moderate diabetes-related risk probability: {probability:.2f}."]  # This returns moderate risk.

    if probability >= 0.30:  # This checks mild model probability.
        return "Mild Risk", [f"ML model estimated mild diabetes-related risk probability: {probability:.2f}."]  # This returns mild risk.

    return "Safe", [f"ML model estimated lower diabetes-related risk probability: {probability:.2f}."]  # This returns safe.


def max_risk(current, new):  # This keeps the higher risk between two categories.
    return new if RISK_SCORES[new] > RISK_SCORES[current] else current  # This compares numeric risk scores.


def combine_all_risks(bp_level, glucose_level, urine_level, model_level):  # This combines all separate risk levels.
    levels = [bp_level, glucose_level, urine_level, model_level]  # This stores every risk source.
    highest = max(levels, key=lambda item: RISK_SCORES[item])  # This finds the highest risk level.
    return highest, RISK_SCORES[highest], RISK_COLORS[highest]  # This returns label, score, and color.


def make_model_row(user_data, feature_order):  # This converts user input into the same feature order used in training.
    row = {  # This prepares model inputs.
        "Pregnancies": user_data["pregnancies"],  # This maps pregnancies.
        "Glucose": user_data["glucose"],  # This maps blood glucose.
        "BloodPressure": user_data["diastolic"],  # Pima data uses one BP field, so we use diastolic pressure.
        "SkinThickness": user_data["skin_thickness"],  # This maps skin thickness.
        "Insulin": user_data["insulin"],  # This maps insulin.
        "BMI": user_data["bmi"],  # This maps BMI.
        "DiabetesPedigreeFunction": user_data["pedigree"],  # This maps family-history score.
        "Age": user_data["age"],  # This maps age.
    }

    return pd.DataFrame([[row[col] for col in feature_order]], columns=feature_order)  # This creates a one-row model input table.


def build_summary(category, notes):  # This creates a short human-readable summary.
    opening = f"Overall category: {category}."  # This starts the summary with the risk category.
    main_points = " ".join(notes[:4])  # This keeps the summary short by using the first few notes.
    return opening + " " + main_points  # This returns the final summary.