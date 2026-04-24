import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model and features
model = joblib.load("xgb_model.pkl")
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Page config
st.set_page_config(
    page_title="CPU Benchmark Predictor",
    page_icon="🖥️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f0f0f;
        color: #f0f0f0;
    }

    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }

    .hero h1 {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
    }

    .card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 1rem;
    }

    .score-display {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
        border-radius: 16px;
        border: 1px solid #2a2a2a;
        margin-bottom: 1rem;
    }

    .score-number {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }

    .score-label {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.5rem;
        letter-spacing: 0.05em;
    }

    .tier-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
        margin-top: 1rem;
    }

    .bar-container {
        background: #2a2a2a;
        border-radius: 999px;
        height: 12px;
        width: 100%;
        margin: 1rem 0 0.5rem 0;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        transition: width 0.5s ease;
    }

    .bar-caption {
        font-size: 0.78rem;
        color: #555;
        text-align: right;
    }

    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem;
        cursor: pointer;
        margin-top: 1rem;
    }

    div[data-testid="stButton"] button:hover {
        opacity: 0.9;
    }

    label, .stNumberInput label, .stSelectbox label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
    <div class="hero">
        <h1>CPU Benchmark Predictor</h1>
        <p>Enter your CPU specs and get an estimated PassMark score instantly.</p>
    </div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">CPU Specifications</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    cores = st.number_input("Number of Cores", min_value=1, max_value=256, value=8)
    price = st.number_input("Price (USD)", min_value=0.0, max_value=10000.0, value=300.0)
    TDP = st.number_input("TDP (Watts)", min_value=1, max_value=400, value=65)
with col2:
    threadMark = st.number_input("Single Thread Mark", min_value=0, max_value=5000, value=2000)
    testDate = st.number_input("Release Year", min_value=2000, max_value=2024, value=2022)
    category = st.selectbox("Category", ["Desktop", "Laptop", "Server", "Other"])

st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("Predict Benchmark Score")

# Prediction
if predict:
    price_per_core = price / cores
    tdp_per_core = TDP / cores
    thread_to_core_ratio = threadMark / cores

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

    input_df = pd.DataFrame([input_dict]).astype(float)
    prediction_log = model.predict(input_df)[0]
    prediction = int(np.expm1(prediction_log))

    # Tier logic
    if prediction < 2000:
        tier, tier_color, tier_bg = "Budget", "#fff", "#3a3a3a"
    elif prediction < 10000:
        tier, tier_color, tier_bg = "Mid-Range", "#fff", "#0072ff"
    elif prediction < 30000:
        tier, tier_color, tier_bg = "High-End", "#fff", "#00a86b"
    else:
        tier, tier_color, tier_bg = "Workstation", "#fff", "#cc0000"

    pct = min(int((prediction / 108822) * 100), 100)

    # Score display
    st.markdown(f"""
        <div class="score-display">
            <div class="score-number">{prediction:,}</div>
            <div class="score-label">ESTIMATED PASSMARK SCORE</div>
            <div>
                <span class="tier-badge" style="background:{tier_bg}; color:{tier_color};">
                    {tier}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
        <div class="card">
            <div class="section-label">Score on PassMark Scale</div>
            <div class="bar-container">
                <div class="bar-fill" style="width:{pct}%;"></div>
            </div>
            <div class="bar-caption">Scores higher than ~{pct}% of CPUs in the dataset</div>
        </div>
    """, unsafe_allow_html=True)
