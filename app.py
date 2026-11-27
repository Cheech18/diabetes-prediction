from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_predictor():
    model_path = Path(__file__).with_name("diabetes_predictor.joblib")
    return joblib.load(model_path)


predictor = load_predictor()

st.title("Diabetes prediction demo")
st.write(
    "Try a fictional patient below. Change a measurement and click Predict "
    "to see how the model responds. This is an educational demo."
)

# Each entry contains the model's column name, the label, and a starting value.
fields = [
    ("Pregnancies", "Number of pregnancies", 2),
    ("Glucose", "Glucose (mg/dL)", 130),
    ("BloodPressure", "Diastolic blood pressure (mmHg)", 72),
    ("SkinThickness", "Triceps skinfold thickness (mm)", 25),
    ("Insulin", "Insulin (µU/mL)", 85),
    ("BMI", "Body mass index (BMI)", 31.0),
    ("DiabetesPedigreeFunction", "Diabetes pedigree score", 0.40),
    ("Age", "Age (years)", 35),
]

with st.form("patient_form"):
    values = {}
    for column, label, starting_value in fields:
        values[column] = st.number_input(label, value=starting_value)
    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame([values], columns=predictor["features"])
    scaled = predictor["scaler"].transform(patient)
    score = predictor["model"].decision_function(scaled)[0]
    cutoff = predictor["cutoff"]

    prediction = "Positive" if score >= cutoff else "Negative"
    st.subheader(f"Model prediction: {prediction}")
    st.write(f"Decision score: {score:.3f} · Cutoff: {cutoff:g}")
