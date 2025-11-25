import streamlit as st
import pandas as pd
import joblib

# ================= LOAD MODEL =================
model = joblib.load("models/model1.pkl")

st.title("Absenteeism Prediction App")
st.write("Fill in the employee details to predict **Absenteeism Time (hours)**")

# ================= INPUTS =================
with st.form("input_form"):

    # 1. Reason for absence
    reason = st.selectbox(
        "Reason for absence (ICD Code)",
        list(range(0, 29))
    )

    # 2. Month of absence
    month = st.selectbox(
        "Month of absence",
        list(range(0, 13))
    )

    # 3. Day of the week
    day = st.radio(
        "Day of the week (0=Mon ... 4=Fri)",
        [0, 1, 2, 3, 4]
    )

    # 4. Season
    season = st.radio(
        "Season (0=Winter, 1=Spring, 2=Summer, 3=Autumn)",
        [0, 1, 2, 3]
    )

    # 5. Transportation expense
    transport = st.number_input(
        "Transportation expense",
        min_value=118,
        max_value=388,
        value=118
    )

    # 6. Distance from Residence to Work
    distance = st.number_input(
        "Distance from Residence to Work (km)",
        min_value=5,
        max_value=52,
        value=5
    )

    # 7. Service time
    service_time = st.number_input(
        "Service time (years)",
        min_value=1,
        max_value=29,
        value=1
    )

    # 8. Age
    age = st.number_input(
        "Age",
        min_value=27,
        max_value=58,
        value=30
    )

    # 9. Work load Average/day
    workload = st.number_input(
        "Work load average/day",
        min_value=0,
        max_value=37,
        value=10
    )

    # 10. Hit target
    hit_target = st.selectbox(
        "Hit target (%)",
        [81, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    )

    # 11. Disciplinary failure
    disciplinary = st.radio(
        "Disciplinary failure",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    # 12. Education
    education = st.radio(
        "Education",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Unknown",
            1: "High School",
            2: "Graduate",
            3: "Postgraduate"
        }[x]
    )

    # 13. Son (children)
    son = st.selectbox(
        "Number of children",
        [0, 1, 2, 3, 4]
    )

    # 14. Social drinker
    drinker = st.radio(
        "Social Drinker",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    # 15. Social smoker
    smoker = st.radio(
        "Social Smoker",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    # 16. Pet
    pet = st.selectbox(
        "Number of pets",
        [0, 1, 2, 4, 5, 8]
    )

    # 17. Weight
    weight = st.number_input(
        "Weight (kg)",
        min_value=56,
        max_value=108,
        value=70
    )

    # 18. Height
    height = st.selectbox(
        "Height (cm)",
        [163,165,167,168,169,170,171,172,174,175,178,182,185,196]
    )

    # 19. Body Mass Index
    bmi = st.selectbox(
        "Body Mass Index",
        [19,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36,38]
    )

    # 20. BMI Category
    bmi_cat = st.radio(
        "BMI Category",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "Overweight",
            2: "Obese"
        }[x]
    )

    button = st.form_submit_button("Predict Absenteeism")


# ================= PREDICTION =================
if button:

    input_data = pd.DataFrame([[

        reason,
        month,
        day,
        season,
        transport,
        distance,
        service_time,
        age,
        workload,
        hit_target,
        disciplinary,
        education,
        son,
        drinker,
        smoker,
        pet,
        weight,
        height,
        bmi,
        bmi_cat

    ]], columns=[
        "Reason for absence",
        "Month of absence",
        "Day of the week",
        "Season",
        "Transportation expense",
        "Distance from Residence to Work",
        "Service time",
        "Age",
        "Work load Average/day",
        "Hit target",
        "Disciplinary failure",
        "Education",
        "Son",
        "Social drinker",
        "Social smoker",
        "Pet",
        "Weight",
        "Height",
        "Body mass index",
        "BMI_Category"
    ])

    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_data)

    prediction = model.predict(input_data)

    st.success(f" Predicted Absenteeism Time: **{round(prediction[0],2)} hours**")
