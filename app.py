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

# Load dataset
df = pd.read_csv("CPU_benchmark_cleaned.csv")

# Precompute highlights
strongest = df.loc[df['cpuMark'].idxmax()]
df_value = df[df['price'] > 0].copy()
df_value['value_score'] = df_value['cpuMark'] / df_value['price']
best_value = df_value.loc[df_value['value_score'].idxmax()]
df['efficiency'] = df['cpuMark'] / df['TDP']
most_efficient = df.loc[df['efficiency'].idxmax()]

# Page config
st.set_page_config(
    page_title="CPU Benchmark Predictor",
    page_icon="🖥️",
    layout="wide"
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

    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 0 1.5rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 2rem;
    }

    .navbar-title {
        font-size: 1.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1rem;
        color: #888;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    .stat-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 3rem;
    }

    .stat-box {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1rem 2rem;
        text-align: center;
        min-width: 140px;
    }

    .stat-number {
        font-size: 1.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #555;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .highlight-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #0072ff;
        margin-bottom: 0.5rem;
    }

    .highlight-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
    }

    .highlight-cpu {
        font-size: 1rem;
        font-weight: 700;
        color: #f0f0f0;
        margin-bottom: 0.3rem;
    }

    .highlight-score {
        font-size: 1.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .highlight-metric {
        font-size: 0.78rem;
        color: #555;
        margin-top: 0.2rem;
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
        color: #0072ff;
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
        background: transparent;
        color: #aaa;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    div[data-testid="stButton"] button:hover {
        border-color: #0072ff;
        color: #fff;
    }

    .active-btn button {
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
        color: white !important;
        border: none !important;
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

    label, .stNumberInput label, .stSelectbox label {
        color: #aaa !important;
        font-size: 0.85rem !important;
    }
    </style>
""", unsafe_allow_html=True)

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

def show_cpu_expander(row, label=None):
    cpu_score = int(row['cpuMark'])
    tier, tier_bg = get_tier(cpu_score)
    pct = get_progress_pct(cpu_score)
    value_score = round(row['cpuMark'] / row['price'], 2) if row['price'] > 0 else 'N/A'
    title = label if label else row['cpuName']

    with st.expander(f"{row['cpuName']}  —  {cpu_score:,} cpuMark  ·  ${row['price']:.0f}"):
        st.markdown(f"""
            <div class="bar-container">
                <div class="bar-fill" style="width:{pct}%;"></div>
            </div>
            <div class="bar-caption" style="margin-bottom:1rem;">
                Scores higher than ~{pct}% of CPUs in the dataset
            </div>
        """, unsafe_allow_html=True)

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

        st.markdown(f"""
            <div style="margin-top:0.75rem;">
                <span class="tier-badge" style="background:{tier_bg}; color:#fff;">
                    {tier}
                </span>
                <span style="color:#555; font-size:0.85rem; margin-left:1rem;">
                    Value Score: {value_score}
                </span>
            </div>
        """, unsafe_allow_html=True)

# Session state for mode
if 'mode' not in st.session_state:
    st.session_state.mode = 'Home'

# Navbar
col_title, col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([3, 1, 1, 1, 1, 1])

with col_title:
    st.markdown('<div class="navbar-title">CPU Benchmark Predictor</div>', unsafe_allow_html=True)

with col_nav1:
    if st.button("Home"):
        st.session_state.mode = 'Home'

with col_nav2:
    if st.button("Predict"):
        st.session_state.mode = 'Predict'

with col_nav3:
    if st.button("Search"):
        st.session_state.mode = 'Search'

with col_nav4:
    if st.button("Best Value"):
        st.session_state.mode = 'Best Value'
with col_nav5:
    if st.button("About"):
        st.session_state.mode = 'About'
        
st.markdown("<hr style='border-color:#2a2a2a; margin-bottom:2rem;'>", unsafe_allow_html=True)

mode = st.session_state.mode

# ── HOME ──────────────────────────────────────────────────────────────
if mode == 'Home':

    st.markdown("""
        <div class="hero">
            <h1>CPU Benchmark Predictor</h1>
            <p>A data-driven tool to explore, predict, and compare CPU performance<br>using real PassMark benchmark data.</p>
        </div>
    """, unsafe_allow_html=True)

    # Stats row
    st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-number">{len(df):,}</div>
                <div class="stat-label">CPUs in Database</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">0.9833</div>
                <div class="stat-label">Model R² Score</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">3</div>
                <div class="stat-label">Modes to Explore</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">XGBoost</div>
                <div class="stat-label">Model Used</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Highlights
    st.markdown('<div class="section-label" style="text-align:center; margin-bottom:1.5rem;">Did You Know?</div>', unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)

    with h1:
        cpu_score = int(strongest['cpuMark'])
        tier, tier_bg = get_tier(cpu_score)
        st.markdown(f"""
            <div class="highlight-card">
                <div class="highlight-label">Strongest CPU</div>
                <div class="highlight-cpu">{strongest['cpuName']}</div>
                <div class="highlight-score">{cpu_score:,}</div>
                <div class="highlight-metric">cpuMark score</div>
                <div style="margin-top:0.75rem;">
                    <span class="tier-badge" style="background:{tier_bg}; color:#fff; font-size:0.8rem; padding:0.3rem 1rem;">
                        {tier}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with h2:
        vs = round(best_value['value_score'], 2)
        tier, tier_bg = get_tier(int(best_value['cpuMark']))
        st.markdown(f"""
            <div class="highlight-card">
                <div class="highlight-label">Best Value CPU</div>
                <div class="highlight-cpu">{best_value['cpuName']}</div>
                <div class="highlight-score">{vs}</div>
                <div class="highlight-metric">cpuMark per dollar</div>
                <div style="margin-top:0.75rem;">
                    <span class="tier-badge" style="background:{tier_bg}; color:#fff; font-size:0.8rem; padding:0.3rem 1rem;">
                        {tier}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with h3:
        eff = round(most_efficient['efficiency'], 2)
        tier, tier_bg = get_tier(int(most_efficient['cpuMark']))
        st.markdown(f"""
            <div class="highlight-card">
                <div class="highlight-label">Most Power Efficient</div>
                <div class="highlight-cpu">{most_efficient['cpuName']}</div>
                <div class="highlight-score">{eff}</div>
                <div class="highlight-metric">cpuMark per watt</div>
                <div style="margin-top:0.75rem;">
                    <span class="tier-badge" style="background:{tier_bg}; color:#fff; font-size:0.8rem; padding:0.3rem 1rem;">
                        {tier}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Expandable specs for each highlight
    st.markdown('<div class="section-label">Click to explore each highlight</div>', unsafe_allow_html=True)
    show_cpu_expander(strongest)
    show_cpu_expander(best_value)
    show_cpu_expander(most_efficient)

# ── PREDICT BY SPECS ──────────────────────────────────────────────────
elif mode == 'Predict':

    st.markdown('<div class="section-label">Predict Benchmark Score from Specs</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

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

        st.markdown(f"""
            <div class="card">
                <div class="section-label">Score on PassMark Scale</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width:{pct}%;"></div>
                </div>
                <div class="bar-caption">Scores higher than ~{pct}% of CPUs in the dataset</div>
            </div>
        """, unsafe_allow_html=True)

        # Similar CPUs
        same_cat = df[df['category'] == category].copy()
        same_cat['diff'] = (same_cat['cpuMark'] - prediction).abs()
        similar = same_cat.sort_values('diff').head(5)

        if not similar.empty:
            st.markdown('<div class="section-label">Similar CPUs in the Same Category</div>', unsafe_allow_html=True)
            for _, row in similar.iterrows():
                show_cpu_expander(row)

# ── SEARCH BY CPU NAME ────────────────────────────────────────────────
elif mode == 'Search':

    st.markdown('<div class="section-label">Search CPU by Name</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    query = st.text_input("Type a CPU name", placeholder="e.g. Ryzen 9 5950X")
    st.markdown('</div>', unsafe_allow_html=True)

    if query:
        cpu_names = df['cpuName'].tolist()
        matches = process.extract(query, cpu_names, scorer=fuzz.WRatio, limit=8)

        if matches:
            st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
            for match in matches:
                name, score, idx = match
                row = df[df['cpuName'] == name].iloc[0]
                show_cpu_expander(row)
        else:
            st.markdown('<p style="color:#555;">No results found.</p>', unsafe_allow_html=True)

# ── BEST VALUE FINDER ─────────────────────────────────────────────────
elif mode == 'Best Value':

    st.markdown('<div class="section-label">Find Best Value CPUs Within Your Budget</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
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
            filtered = df[
                (df['price'] >= min_budget) &
                (df['price'] <= max_budget) &
                (df['price'] > 0)
            ].copy()

            if category_filter != "Any":
                filtered = filtered[filtered['category'] == category_filter]

            if filtered.empty:
                st.markdown('<p style="color:#555;">No CPUs found. Try adjusting your filters.</p>', unsafe_allow_html=True)
            else:
                filtered['value_score'] = (filtered['cpuMark'] / filtered['price']).round(2)
                top5 = filtered.nlargest(5, 'value_score').reset_index(drop=True)

                st.markdown('<div class="section-label">Top 5 Best Value CPUs</div>', unsafe_allow_html=True)
                for _, row in top5.iterrows():
                    show_cpu_expander(row)
# ── ABOUT ─────────────────────────────────────────────────────────────
elif mode == 'About':

    st.markdown('<div class="section-label">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <div class="section-label">Overview</div>
            <p style="color:#d0d0d0; line-height:1.8;">
                CPU Benchmark Predictor is a data science project that uses machine learning
                to estimate a CPU's PassMark multi-thread benchmark score from its hardware
                specifications. It was built to demonstrate an end-to-end data science workflow
                — from raw data exploration to a deployed interactive web application.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <div class="section-label">Dataset</div>
            <p style="color:#d0d0d0; line-height:1.8;">
                The dataset contains 3,659 CPUs sourced from PassMark Performance Test via Kaggle.
                Each entry includes hardware specifications such as core count, TDP, and price,
                alongside benchmark scores recorded at the time of submission.
            </p>
            <p style="color:#d0d0d0; line-height:1.8;">
                Note: Prices in this dataset reflect market prices at the time of data collection
                and may not reflect current retail prices. CPU prices tend to decrease over time
                as newer generations release.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <div class="section-label">Methodology</div>
            <p style="color:#d0d0d0; line-height:1.8;">
                The project follows a structured data science workflow across three notebooks:
            </p>
            <p style="color:#d0d0d0; line-height:1.8;">
                <strong style="color:#f0f0f0;">Notebook 1 — Exploration</strong><br>
                Loaded and inspected both datasets, identified missing values, data type issues,
                and the heavily right-skewed distribution of the target variable cpuMark.
            </p>
            <p style="color:#d0d0d0; line-height:1.8;">
                <strong style="color:#f0f0f0;">Notebook 2 — Cleaning</strong><br>
                Fixed data types, handled missing values using median imputation grouped by
                category, simplified multi-label category values, and standardized over 200
                inconsistent socket type names down to 70 clean values.
            </p>
            <p style="color:#d0d0d0; line-height:1.8;">
                <strong style="color:#f0f0f0;">Notebook 3 — Modeling</strong><br>
                Engineered three new features — price per core, TDP per core, and thread to
                core ratio. Applied a log transformation to cpuMark to handle skew. Trained
                and compared Linear Regression, Random Forest, and XGBoost. Used SHAP values
                to explain feature importance.
            </p>
            <p style="color:#d0d0d0; line-height:1.8;">
                <strong style="color:#f0f0f0;">Notebook 4 — Value Analysis</strong><br>
                Analyzed performance per dollar across the dataset to identify which CPUs
                offer the best value within different price ranges and categories.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="card">
            <div class="section-label">Model Performance</div>
            <div class="stat-row" style="margin-bottom:0;">
                <div class="stat-box">
                    <div class="stat-number">0.9833</div>
                    <div class="stat-label">R² Score</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">2,331</div>
                    <div class="stat-label">RMSE (benchmark pts)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">XGBoost</div>
                    <div class="stat-label">Final Model</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(df):,}</div>
                    <div class="stat-label">CPUs in Dataset</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <div class="section-label">Links</div>
            <p style="color:#d0d0d0; line-height:1.8;">
                View the full project including all notebooks and source code on GitHub.<br>
                <a href="https://github.com/bquan1406-droid/cpu-benchmark-predictor"
                style="color:#0072ff; text-decoration:none;">
                github.com/bquan1406-droid/cpu-benchmark-predictor
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)
