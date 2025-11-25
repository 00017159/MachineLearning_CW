import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Page config ----------
st.set_page_config(page_title="Absenteeism Prediction App", layout="wide")

st.title(" Absenteeism at Work - Machine Learning App")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Updated_dataset - Copy.csv")
    return df

df = load_data()

menu = ["Data Exploration", "Preprocessing", "Model Training", "Evaluation"]
choice = st.sidebar.selectbox("Select a Section", menu)


# ---------- DATA EXPLORATION ----------
if choice == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)


# ---------- PREPROCESSING ----------
if choice == "Preprocessing":
    st.subheader("Cleaning Data")

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    st.success("Missing values handled!")

    st.write(df.head())


# ---------- MODEL TRAINING ----------
if choice == "Model Training":
    st.subheader("Train Models")

    # Target and features
    target = "Absenteeism time in hours"

    if target not in df.columns:
        st.error("Target column not found in dataset!")
    else:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical data
        X = pd.get_dummies(X, drop_first=True)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Models
        model_choice = st.selectbox(
            "Choose Model",
            [
                "Linear Regression",
                "Random Forest",
                "Support Vector Regressor"
            ]
        )

        if st.button("Train Model"):

            if model_choice == "Linear Regression":
                model = LinearRegression()

            elif model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)

            elif model_choice == "Support Vector Regressor":
                model = SVR(kernel="rbf")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"{model_choice} trained successfully!")

            # Save in session
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred
            st.session_state["model_name"] = model_choice


# ---------- EVALUATION ----------
if choice == "Evaluation":

    if "y_test" in st.session_state:

        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        model_name = st.session_state["model_name"]

        st.subheader(f"Evaluation for {model_name}")

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Absolute Error:", mae)
        st.write("Mean Squared Error:", mse)
        st.write("R2 Score:", r2)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Page config ----------
st.set_page_config(page_title="Absenteeism Prediction App", layout="wide")

st.title(" Absenteeism at Work - Machine Learning App")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Updated_dataset - Copy.csv")
    return df

df = load_data()

menu = ["Data Exploration", "Preprocessing", "Model Training", "Evaluation"]
choice = st.sidebar.selectbox("Select a Section", menu)


# ---------- DATA EXPLORATION ----------
if choice == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)


# ---------- PREPROCESSING ----------
if choice == "Preprocessing":
    st.subheader("Cleaning Data")

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    st.success("Missing values handled!")

    st.write(df.head())


# ---------- MODEL TRAINING ----------
if choice == "Model Training":
    st.subheader("Train Models")

    # Target and features
    target = "Absenteeism time in hours"

    if target not in df.columns:
        st.error("Target column not found in dataset!")
    else:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical data
        X = pd.get_dummies(X, drop_first=True)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Models
        model_choice = st.selectbox(
            "Choose Model",
            [
                "Linear Regression",
                "Random Forest",
                "Support Vector Regressor"
            ]
        )

        if st.button("Train Model"):

            if model_choice == "Linear Regression":
                model = LinearRegression()

            elif model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)

            elif model_choice == "Support Vector Regressor":
                model = SVR(kernel="rbf")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"{model_choice} trained successfully!")

            # Save in session
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred
            st.session_state["model_name"] = model_choice


# ---------- EVALUATION ----------
if choice == "Evaluation":

    if "y_test" in st.session_state:

        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        model_name = st.session_state["model_name"]

        st.subheader(f"Evaluation for {model_name}")

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("Mean Absolute Error:", mae)
        st.write("Mean Squared Error:", mse)
        st.write("R2 Score:", r2)
