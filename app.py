import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

st.set_page_config(page_title="Absenteeism Project", layout="wide")

TARGET = "Absenteeism time in hours"


DAY_MAPPING = {
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
}

SEASON_MAPPING = {
    1: "Spring",
    2: "Summer",
    3: "Autumn",
    4: "Winter"
}

EDUCATION_MAPPING = {
    1: "High school",
    2: "Graduate",
    3: "Postgraduate",
    4: "Master or Doctor"
}

BOOL_MAPPING = {0: "No", 1: "Yes"}

REASON_MAPPING = {
    1: "Certain infectious and parasitic diseases",
    2: "Neoplasms",
    3: "Diseases of the blood and immune mechanism",
    4: "Endocrine, nutritional and metabolic diseases",
    5: "Mental and behavioural disorders",
    6: "Diseases of the nervous system",
    7: "Diseases of the eye and adnexa",
    8: "Diseases of the ear and mastoid process",
    9: "Diseases of the circulatory system",
    10: "Diseases of the respiratory system",
    11: "Diseases of the digestive system",
    12: "Diseases of the skin and subcutaneous tissue",
    13: "Diseases of the musculoskeletal system",
    14: "Diseases of the genitourinary system",
    15: "Pregnancy, childbirth and puerperium",
    16: "Certain conditions originating in the perinatal period",
    17: "Congenital malformations and chromosomal abnormalities",
    18: "Symptoms and abnormal clinical findings",
    19: "Injury, poisoning and consequences",
    20: "External causes of morbidity",
    21: "Factors influencing health services",
    22: "Patient follow-up",
    23: "Medical consultation",
    24: "Blood donation",
    25: "Laboratory examination",
    26: "Unjustified absence",
    27: "Physiotherapy",
    28: "Dental consultation"
}



if "df" not in st.session_state:
    st.session_state.df = None

if "models" not in st.session_state:
    st.session_state.models = {}


st.sidebar.title("Absenteeism Navigation")

page = st.sidebar.radio(
    "Go to",
    [" Upload", " EDA", " Preprocessing",
     " Model Training", " Evaluation", " Prediction", " Export"]
)

if page == " Upload":

    st.title(" Upload Absenteeism Dataset")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)

        st.session_state.df = df
        st.success(" File uploaded successfully")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head())
        st.write("Shape:", st.session_state.df.shape)

elif page == " EDA":

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.df

        st.title(" Exploratory Data Analysis")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        st.subheader("Absenteeism Distribution")
        plt.figure(figsize=(8, 5))
        sns.histplot(df[TARGET], kde=True)
        st.pyplot(plt)

        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap="coolwarm")
        st.pyplot(plt)

elif page == " Preprocessing":

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        df = st.session_state.df.copy()

        X = df.drop(TARGET, axis=1)
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.columns = X.columns.tolist()

        st.success(" Preprocessing complete")


# ------------------ PAGE 4 ------------------
elif page == "4) Model Training":

    if "X_train" not in st.session_state:
        st.warning("Run preprocessing first")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        if st.button("Train all models"):

            lr = LinearRegression()
            rf = RandomForestRegressor()
            gb = GradientBoostingRegressor()

            lr.fit(X_train, y_train)
            rf.fit(X_train, y_train)
            gb.fit(X_train, y_train)

            st.session_state.models = {
                "Linear Regression": lr,
                "Random Forest": rf,
                "Gradient Boosting": gb
            }

            st.success(" Models trained successfully")

elif page == "5) Evaluation":

    if not st.session_state.models:
        st.warning("Train models first")
    else:
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        for name, model in st.session_state.models.items():
            y_pred = model.predict(X_test)

            st.subheader(name)
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
            st.write("R²:", r2_score(y_test, y_pred))


elif page == "6) Prediction":

    if not st.session_state.models:
        st.warning("Train a model first")
    else:
        st.title(" Absenteeism Predictor")

        sample = {}

        day = st.selectbox("Day of Week", DAY_MAPPING.values())
        sample["Day of the week"] = list(DAY_MAPPING.keys())[
            list(DAY_MAPPING.values()).index(day)
        ]

        reason = st.selectbox("Reason for Absence", REASON_MAPPING.values())
        sample["Reason for absence"] = list(REASON_MAPPING.keys())[
            list(REASON_MAPPING.values()).index(reason)
        ]

        season = st.selectbox("Season", SEASON_MAPPING.values())
        sample["Seasons"] = list(SEASON_MAPPING.keys())[
            list(SEASON_MAPPING.values()).index(season)
        ]

        sample["Month of absence"] = st.slider("Month", 1, 12, 6)
        sample["Transportation expense"] = st.number_input("Transportation expense", 0, 500, 200)
        sample["Distance from Residence to Work"] = st.number_input("Distance (km)", 0, 50, 10)
        sample["Service time"] = st.number_input("Service time (years)", 1, 40, 10)
        sample["Age"] = st.number_input("Age", 18, 65, 30)
        sample["Work load Average/day "] = st.number_input("Workload", 100, 400, 250)
        sample["Hit target"] = st.number_input("Hit Target", 50, 100, 90)

        discipline = st.radio("Disciplinary Failure", ["Yes", "No"])
        sample["Disciplinary failure"] = 1 if discipline == "Yes" else 0

        edu = st.selectbox("Education", EDUCATION_MAPPING.values())
        sample["Education"] = list(EDUCATION_MAPPING.keys())[list(EDUCATION_MAPPING.values()).index(edu)]

        sample["Son"] = st.number_input("Children", 0, 5, 1)

        drink = st.radio("Social Drinker", ["Yes", "No"])
        smoke = st.radio("Social Smoker", ["Yes", "No"])

        sample["Social drinker"] = 1 if drink == "Yes" else 0
        sample["Social smoker"] = 1 if smoke == "Yes" else 0

        sample["Pet"] = st.number_input("Number of pets", 0, 5, 0)
        sample["Weight"] = st.number_input("Weight (kg)", 40, 150, 70)
        sample["Height"] = st.number_input("Height (cm)", 140, 200, 170)
        sample["Body mass index"] = sample["Weight"] / ((sample["Height"] / 100) ** 2)

        model_name = st.selectbox("Choose model", list(st.session_state.models.keys()))

        if st.button("Predict Absenteeism"):
            df_sample = pd.DataFrame([sample])[st.session_state.columns]
            df_sample = st.session_state.scaler.transform(df_sample)

            prediction = st.session_state.models[model_name].predict(df_sample)[0]
            st.success(f"Predicted Absenteeism Hours: {prediction:.2f} hours")



elif page == " Export":

    if not st.session_state.models:
        st.warning("Train model first")
    else:

        model_name = st.selectbox("Select model", list(st.session_state.models.keys()))

        if st.button("Download model"):
            joblib.dump(st.session_state.models[model_name], model_name + ".joblib")
            st.success(" Model saved")
