import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load the model and feature columns
model = joblib.load("xgb_model.pkl")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# App title
st.title("CPU Benchmark Score Predictor")
st.write("Enter a CPU s hardware specifications to get an estimated PassMark multi-thread benchmark score.")

# Input fields
st.header("CPU Specifications")

cores = st.number_input("Number of Cores", min_value=1, max_value=256, value=8)
threadMark = st.number_input("Single Thread Mark", min_value=0, max_value=5000, value=2000)
price = st.number_input("Price (USD)", min_value=0.0, max_value=10000.0, value=300.0)
TDP = st.number_input("TDP (Watts)", min_value=1, max_value=400, value=65)
testDate = st.number_input("Release Year", min_value=2000, max_value=2024, value=2022)
category = st.selectbox("Category", ["Desktop", "Laptop", "Server", "Other"])

# Predict button
if st.button("Predict cpuMark"):

    # Compute engineered features
    price_per_core = price / cores
    tdp_per_core = TDP / cores
    thread_to_core_ratio = threadMark / cores

    # Build input row with all zeros matching training columns
    input_dict = {col: 0 for col in feature_columns}

    # Fill in known values
    input_dict["cores"] = cores
    input_dict["threadMark"] = threadMark
    input_dict["price"] = price
    input_dict["TDP"] = TDP
    input_dict["testDate"] = testDate
    input_dict["price_per_core"] = price_per_core
    input_dict["tdp_per_core"] = tdp_per_core
    input_dict["thread_to_core_ratio"] = thread_to_core_ratio

    # Set the correct category column
    if category == "Server":
        input_dict["category_Server"] = 1
    elif category == "Laptop":
        input_dict["category_Laptop"] = 1
    elif category == "Other":
        input_dict["category_Other"] = 1
    # Desktop is the base category — all zeros

    # Convert to dataframe and predict
    input_df = pd.DataFrame([input_dict]).astype(float)
    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    # Display result
    st.success(f"Estimated cpuMark Score: {int(prediction):,}")
    st.caption("This is an estimate based on the XGBoost model trained on PassMark benchmark data.")
