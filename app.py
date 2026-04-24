import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load the model and feature columns
model = joblib.load("xgb_model.pkl")

with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Helper functions
def get_performance_label(score):
    if score < 2000:
        return "Budget", "#6c757d"
    elif score < 10000:
        return "Mid-Range", "#0d6efd"
    elif score < 30000:
        return "High-End", "#198754"
    else:
        return "Workstation", "#dc3545"

def get_progress_pct(score, max_score=108822):
    return min(int((score / max_score) * 100), 100)

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

    # Build input row
    input_dict = {col: 0 for col in feature_columns}
    input_dict["cores"] = cores
    input_dict["threadMark"] = threadMark
    input_dict["price"] = price
    input_dict["TDP"] = TDP
    input_dict["testDate"] = testDate
    input_dict["price_per_core"] = price_per_core
    input_dict["tdp_per_core"] = tdp_per_core
    input_dict["thread_to_core_ratio"] = thread_to_core_ratio

    if category == "Server":
        input_dict["category_Server"] = 1
    elif category == "Laptop":
        input_dict["category_Laptop"] = 1
    elif category == "Other":
        input_dict["category_Other"] = 1

    # Predict
    input_df = pd.DataFrame([input_dict]).astype(float)
    prediction_log = model.predict(input_df)[0]
    prediction = int(np.expm1(prediction_log))

    # Get label and color
    label, color = get_performance_label(prediction)
    pct = get_progress_pct(prediction)

    # Display results
    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Estimated cpuMark", value=f"{prediction:,}")
    with col2:
        st.markdown(f"### Performance Tier")
        st.markdown(
            f'<span style="background-color:{color}; color:white; padding:6px 14px; '
            f'border-radius:8px; font-weight:bold;">{label}</span>',
            unsafe_allow_html=True
        )

    # Progress bar
    st.markdown("**Score on PassMark Scale (0 — 108,822)**")
    st.progress(pct)
    st.caption(f"This CPU scores higher than approximately {pct}% of CPUs in our dataset.")
