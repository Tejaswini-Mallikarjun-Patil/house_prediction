import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load(open("boston_best_model.pkl", "rb"))

st.title("Boston House Price Prediction")

# Collect 13 inputs
CRIM = st.number_input("Per capita crime rate (CRIM)", value=0.1)
ZN = st.number_input("Proportion of residential land zoned (ZN)", value=0.0)
INDUS = st.number_input("Proportion of non-retail business acres (INDUS)", value=7.0)
CHAS = st.selectbox("Charles River dummy variable (CHAS)", [0, 1])
NOX = st.number_input("Nitric oxides concentration (NOX)", value=0.5)
RM = st.number_input("Average number of rooms (RM)", value=6.0)
AGE = st.number_input("Age of property (AGE)", value=65.0)
DIS = st.number_input("Weighted distances (DIS)", value=4.0)
RAD = st.number_input("Accessibility to highways (RAD)", value=1)
TAX = st.number_input("Property-tax rate (TAX)", value=300)
PTRATIO = st.number_input("Pupil–teacher ratio (PTRATIO)", value=15.0)
B = st.number_input("Proportion of Black residents (B)", value=396.0)
LSTAT = st.number_input("Lower status population % (LSTAT)", value=12.0)

# Prediction
if st.button("Predict"):
    # Build dataframe with correct column names
    features_df = pd.DataFrame([{
        "CRIM": CRIM,
        "ZN": ZN,
        "INDUS": INDUS,
        "CHAS": CHAS,
        "NOX": NOX,
        "RM": RM,
        "AGE": AGE,
        "DIS": DIS,
        "RAD": RAD,
        "TAX": TAX,
        "PTRATIO": PTRATIO,
        "B": B,
        "LSTAT": LSTAT
    }])

    # Predict
    prediction = model.predict(features_df)
    st.success(f"Predicted House Price: {prediction[0]:.2f}")


