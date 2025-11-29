Absenteeism at work place Prediction model
Project Overview
This project provides a Machine Learning–driven Absenteeism Prediction System.
Given 20 employee-related input features such as: Reason for absence (ICD categories),Workload,Transportation expense,BMI & BMI category,Distance to work,Service time,Age,Education Pets, children, etc.The system predicts absenteeism time (in hours) using one of three ML models:
Linear Regression (model1.pkl)
Random Forest Regressor (model2.pkl)
Support Vector Regressor – SVR (model3.pkl)
A Streamlit user interface allows the user to enter all required features and then choose which model to use for prediction.
Prerequisities
Python 3.11+
pip (Python package manager)
VS Code recommended
Running the Program Locally
Clone the Repository
git clone https://github.com/00017159/MachineLearning_CW
cd MLDA_CW1_17159
Create Virtual Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Streamlit app link
The model is already trained and saved in the corresponding folders.
https://machinelearningcw-00017159.streamlit.app/
Folder tree structure 
MLDA_CW1_17159/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── models/
│   ├── model1.pkl
│   ├── model2.pkl
│   └── model3.pkl
│
├── paper/
│   └── Report.docx            # Final written report
│
├── ui/
│   └── app.py   # Streamlit / main app script
├── absenteeismm.ipynb                    # Notebook for EDA + modelling
├── Absenteeism_at_work.csv               # Original dataset
├── Updated_dataset - Copy.csv            # Cleaned or modified dataset
├── Attribute Information.docx            # Dataset documentation
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── .gitignore                            # Git ignore rules
Model evaluation metrics
MAE
MSE
RMSE
R square
Hyperparameter tuning
GridSearchCV is used to find the best combination of hyperparameters for the Random Forest model by testing every possible combination inside rf_params.
Faster parameter search for SVR with large search space
Models for training algorithms 
Linear Regression
Random Forest
Support Vector Regressor


