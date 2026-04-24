import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from rapidfuzz import process, fuzz

# Load model and features
model = joblib.load("xgb_model.pkl")
with open("feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Load dataset for search and similar CPUs
df = pd.read_csv("CPU_benchmark_cleaned.csv")

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

    .result-row {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
    }

    .result-cpu-name {
        font-size: 1rem;
        font-weight: 700;
        color: #f0f0f0;
        margin-bottom: 0.4rem;
    }

    .result-meta {
        font-size: 0.8rem;
        color: #666;
    }

    .result-score {
        font-size: 1.4rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
    <div class="hero">
        <h1>CPU Benchmark Predictor</h1>
        <p>Predict benchmark scores from specs or search any CPU by name.</p>
    </div>
""", unsafe_allow_html=True)

# Mode selector
mode = st.radio(
    "Choose a mode",
    ["Predict by Specs", "Search by CPU Name", "Best Value Finder"],
    horizontal=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Helper functions
def get_tier(score):
    if score < 2000:
        return "Budget", "#3a3a3a"
    elif score < 10000:
        return "Mid-Range", "#0072ff"
    elif score < 30000:
        return "High-End", "#00a86b"
    else:
        return "Workstation", "#cc0000"

def get_progress_pct(score, max_score=108822):
    return min(int((score / max_score) * 100), 100)

# ── MODE 1 — PREDICT BY SPECS ──────────────────────────────────────────
if mode == "Predict by Specs":

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

    if st.button("Predict Benchmark Score"):

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

        tier, tier_bg = get_tier(prediction)
        pct = get_progress_pct(prediction)

        # Score display
        st.markdown(f"""
            <div class="score-display">
                <div class="score-number">{prediction:,}</div>
                <div class="score-label">ESTIMATED PASSMARK SCORE</div>
                <div>
                    <span class="tier-badge" style="background:{tier_bg}; color:#fff;">
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

        # Similar CPUs from the same category
        same_cat = df[df['category'] == category].copy()
        same_cat['diff'] = (same_cat['cpuMark'] - prediction).abs()
        similar = same_cat.sort_values('diff').head(5)

        if not similar.empty:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Similar CPUs in the Same Category</div>', unsafe_allow_html=True)

            for _, row in similar.iterrows():
                cpu_score = int(row['cpuMark'])
                tier_s, tier_bg_s = get_tier(cpu_score)
                pct_similar = get_progress_pct(cpu_score)

                st.markdown(f"""
                    <div class="result-row">
                        <div class="result-cpu-name">{row['cpuName']}</div>
                        <div class="result-meta">
                            {int(row['cores'])} Cores &nbsp;·&nbsp;
                            ${row['price']:.0f} &nbsp;·&nbsp;
                            <span style="background:{tier_bg_s}; color:#fff;
                            padding: 2px 10px; border-radius:999px;
                            font-size:0.75rem; font-weight:700;">{tier_s}</span>
                        </div>
                        <div class="bar-container" style="margin-top:0.75rem;">
                            <div class="bar-fill" style="width:{pct_similar}%;"></div>
                        </div>
                        <div class="result-score">{cpu_score:,}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
# ── MODE 3 — BEST VALUE FINDER ──────────────────────────────────────────
elif mode == "Best Value Finder":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Find Best Value CPUs</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.selectbox("Category", ["Any", "Desktop", "Laptop", "Server", "Other"])
    with col2:
        min_budget = st.number_input("Min Budget (USD)", min_value=0, max_value=10000, value=100)
    with col3:
        max_budget = st.number_input("Max Budget (USD)", min_value=0, max_value=10000, value=500)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Find Best Value CPUs"):

        if min_budget >= max_budget:
            st.warning("Max budget must be greater than min budget.")
        else:
            # Filter by price range
            filtered = df[
                (df['price'] >= min_budget) &
                (df['price'] <= max_budget) &
                (df['price'] > 0)
            ].copy()

            # Filter by category
            if category_filter != "Any":
                filtered = filtered[filtered['category'] == category_filter]

            if filtered.empty:
                st.markdown('<p style="color:#555;">No CPUs found in this range. Try adjusting your filters.</p>', unsafe_allow_html=True)
            else:
                # Calculate value score and get top 5
                filtered['value_score'] = (filtered['cpuMark'] / filtered['price']).round(2)
                top5 = filtered.nlargest(5, 'value_score').reset_index(drop=True)

                st.markdown('<div class="section-label">Top 5 Best Value CPUs</div>', unsafe_allow_html=True)

                for _, row in top5.iterrows():
                    cpu_score = int(row['cpuMark'])
                    tier, tier_bg = get_tier(cpu_score)
                    pct = get_progress_pct(cpu_score)

                    with st.expander(f"{row['cpuName']}  —  {cpu_score:,} cpuMark  ·  ${row['price']:.0f}"):
                        
                        # Progress bar
                        st.markdown(f"""
                            <div class="bar-container">
                                <div class="bar-fill" style="width:{pct}%;"></div>
                            </div>
                            <div class="bar-caption" style="margin-bottom:1rem;">
                                Scores higher than ~{pct}% of CPUs in the dataset
                            </div>
                        """, unsafe_allow_html=True)

                        # Specs
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("cpuMark", f"{cpu_score:,}")
                            st.metric("Cores", int(row['cores']))
                        with col2:
                            st.metric("Single Thread Mark", int(row['threadMark']))
                            st.metric("TDP", f"{row['TDP']:.0f}W")
                        with col3:
                            st.metric("Price", f"${row['price']:.0f}")
                            st.metric("Release Year", int(row['testDate']))

                        # Tier and value score
                        st.markdown(f"""
                            <div style="margin-top:0.75rem;">
                                <span class="tier-badge" style="background:{tier_bg}; color:#fff;">
                                    {tier}
                                </span>
                                <span style="color:#555; font-size:0.85rem; margin-left:1rem;">
                                    Value Score: {row['value_score']}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
# ── MODE 2 — SEARCH BY CPU NAME ─────────────────────────────────────────
elif mode == "Search by CPU Name":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Search CPU</div>', unsafe_allow_html=True)

    query = st.text_input("Type a CPU name", placeholder="e.g. Ryzen 9 5950X")

    st.markdown('</div>', unsafe_allow_html=True)

    if query:
        cpu_names = df['cpuName'].tolist()

        matches = process.extract(
            query,
            cpu_names,
            scorer=fuzz.WRatio,
            limit=8
        )

        if matches:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

            for match in matches:
                name, score, idx = match
                row = df[df['cpuName'] == name].iloc[0]
                cpu_score = int(row['cpuMark'])
                tier, tier_bg = get_tier(cpu_score)
                pct = get_progress_pct(cpu_score)

                st.markdown(f"""
                    <div class="result-row">
                        <div class="result-cpu-name">{name}</div>
                        <div class="result-meta">
                            {row['category']} &nbsp;·&nbsp;
                            {int(row['cores'])} Cores &nbsp;·&nbsp;
                            ${row['price']:.0f} &nbsp;·&nbsp;
                            <span style="background:{tier_bg}; color:#fff;
                            padding: 2px 10px; border-radius:999px;
                            font-size:0.75rem; font-weight:700;">{tier}</span>
                        </div>
                        <div class="bar-container" style="margin-top:0.75rem;">
                            <div class="bar-fill" style="width:{pct}%;"></div>
                        </div>
                        <div class="result-score">{cpu_score:,}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#555;">No results found. Try a different search term.</p>', unsafe_allow_html=True)
