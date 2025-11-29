import streamlit as st
import pandas as pd
import joblib

st.title("Absenteeism Prediction App")
st.write("Fill in the employee details to predict **Absenteeism Time (hours)**")

model_choice = st.selectbox(
    "Choose Prediction Model",
    ("Linear Regression", "Random Forest", "Support Vector Regressor (SVR)")
)


if model_choice == "Linear Regression":
    model = joblib.load("models/model1.pkl")

elif model_choice == "Random Forest":
    model = joblib.load("models/model2.pkl")

elif model_choice == "Support Vector Regressor (SVR)":
    model = joblib.load("models/model3.pkl")

REASONS = {
    0: "No diagnosis / Not informed",
    1: "Certain infectious and parasitic diseases",
    2: "Neoplasms",
    3: "Blood & immune disorders",
    4: "Endocrine / metabolic diseases",
    5: "Mental and behavioural disorders",
    6: "Nervous system diseases",
    7: "Eye diseases",
    8: "Ear diseases",
    9: "Circulatory system diseases",
    10: "Respiratory system diseases",
    11: "Digestive system diseases",
    12: "Skin diseases",
    13: "Musculoskeletal diseases",
    14: "Genitourinary diseases",
    15: "Pregnancy / childbirth",
    16: "Perinatal conditions",
    17: "Congenital anomalies",
    18: "Symptoms and abnormal findings",
    19: "Injuries / poisoning",
    20: "External causes of morbidity and mortality",
    21: "Factors influencing health status and contact with health services",
    22: "Patient follow-up",
    23: "Medical consultation",
    24: "Blood donation",
    25: "Lab examination",
    26: "Unjustified absence",
    27: "Physiotherapy",
    28: "Dental consultation"
}

with st.form("input_form"):
    
    reason = st.radio(
        "Reason for absence (ICD)",
        options=list(REASONS.keys()),
        format_func=lambda x: f"{x} - {REASONS[x]}"
    )

    month = st.selectbox("Month of absence", list(range(0, 13)))
    day = st.radio("Day of the week (2=Mon ... 6=Fri)", [2, 3, 4, 5, 6])
    season = st.radio("Season (0=Winter, 1=Spring, 2=Summer, 3=Autumn)", [0, 1, 2, 3])

    transport = st.number_input("Transportation expense", 118, 388, 118)
    distance = st.number_input("Distance from Residence to Work (km)", 5, 52, 5)
    service_time = st.number_input("Service time (years)", 1, 29, 1)
    age = st.number_input("Age", 27, 58, 30)
    workload = st.number_input("Work load average/day", 0, 37, 10)
    hit_target = st.selectbox("Hit target (%)", [81, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])

    disciplinary = st.radio(
        "Disciplinary failure", 
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    education = st.radio(
        "Education",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "High school",
            2: "Postgraduate",
            3: "Master",
            4: "Doctor"
        }[x]
    )

    son = st.selectbox("Number of children", [0, 1, 2, 3, 4])
    drinker = st.radio("Social Drinker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    smoker = st.radio("Social Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    pet = st.selectbox("Number of pets", [0, 1, 2, 4, 5, 8])

    weight = st.number_input("Weight (kg)", 56, 108, 70)
    height = st.selectbox("Height (cm)", [163,165,167,168,169,170,171,172,174,175,178,182,185,196])

    bmi = st.selectbox("Body Mass Index", [19,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,38])
    bmi_cat = st.radio(
        "BMI Category",
        [0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "Overweight", 2: "Obese"}[x]
    )

    submit = st.form_submit_button("Predict Absenteeism")


# ------------------------ PREDICTION ------------------------
if submit:
    input_data = pd.DataFrame([[
        reason, month, day, season, transport, distance, service_time,
        age, workload, hit_target, disciplinary, education, son,
        drinker, smoker, pet, weight, height, bmi, bmi_cat
    ]], columns=[
        "Reason for absence", "Month of absence", "Day of the week", "Season",
        "Transportation expense", "Distance from Residence to Work", "Service time",
        "Age", "Work load Average/day", "Hit target", "Disciplinary failure",
        "Education", "Son", "Social drinker", "Social smoker", "Pet",
        "Weight", "Height", "Body mass index", "BMI_Category"
    ])

    # Align columns to model expectations
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"Predicted Absenteeism Time: **{round(prediction[0],2)} hours**")
