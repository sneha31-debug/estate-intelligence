"""
🏠 Estate Intelligence — Real Estate Property Price Prediction App
Built with Streamlit · Pandas · NumPy · Matplotlib · Joblib
"""

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Estate Intelligence",
    page_icon="🏠",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# MODEL LOADING  ← FIXED: correct filename best_rf_model.joblib
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_rf_model.joblib")
model = None

try:
    import joblib
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception:
    model = None  # app continues in estimate mode

# ──────────────────────────────────────────────────────────────
# REAL FEATURE IMPORTANCES from trained Random Forest
# ──────────────────────────────────────────────────────────────
FEATURE_IMPORTANCES = {
    "area":             0.478,
    "bathrooms":        0.158,
    "airconditioning":  0.068,
    "stories":          0.052,
    "parking":          0.048,
    "furnishingstatus": 0.043,
    "bedrooms":         0.040,
    "basement":         0.030,
    "prefarea":         0.028,
    "hotwaterheating":  0.018,
    "guestroom":        0.016,
    "mainroad":         0.010,
}

# Column order must match exactly what the model was trained on
FEATURE_COLUMNS = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

# ──────────────────────────────────────────────────────────────
# HELPER — PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────────
def predict_price(area, bedrooms, bathrooms, stories, mainroad,
                  guestroom, basement, hotwaterheating,
                  airconditioning, parking, prefarea, furnishingstatus):
    """
    Returns predicted price using loaded Random Forest model.
    Falls back to a hand-tuned formula if model is not available.
    Input is passed as a DataFrame to preserve column names for the model.
    """
    input_df = pd.DataFrame([{
        "area":             area,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "stories":          stories,
        "mainroad":         mainroad,
        "guestroom":        guestroom,
        "basement":         basement,
        "hotwaterheating":  hotwaterheating,
        "airconditioning":  airconditioning,
        "parking":          parking,
        "prefarea":         prefarea,
        "furnishingstatus": furnishingstatus,
    }])[FEATURE_COLUMNS]  # enforce column order to match training

    if model is not None:
        return model.predict(input_df)[0]
    else:
        # Fallback estimate formula
        return (area * 480 + bedrooms * 45000 + bathrooms * 35000
                + stories * 20000 + airconditioning * 80000
                + prefarea * 60000)


# ──────────────────────────────────────────────────────────────
# HELPER — INDIAN ₹ FORMATTER
# ──────────────────────────────────────────────────────────────
def format_inr(amount):
    """Formats a number into Indian Rupee style with commas.
    Example: 4500000 → ₹45,00,000"""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    return f"₹{result}"


# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #112240 50%, #0a1628 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f3c 0%, #162a50 100%);
        border-right: 1px solid rgba(100,160,255,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #cdd9e5 !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 8px 8px 0 0;
        color: #8babc7;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(100,160,255,0.12) !important;
        color: #64a0ff !important;
        border-bottom: 2px solid #64a0ff;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a5aff, #0d3fb8);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(26,90,255,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3373ff, #1a5aff);
        box-shadow: 0 6px 20px rgba(26,90,255,0.45);
        transform: translateY(-1px);
    }

    .price-box {
        background: linear-gradient(135deg, #0d3fb8 0%, #1a5aff 100%);
        border: 1px solid rgba(100,160,255,0.25);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 20px auto;
        max-width: 480px;
        box-shadow: 0 8px 32px rgba(13,63,184,0.4);
    }
    .price-box .label {
        color: #a3c4f3;
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .price-box .price {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
    }

    h1, h2, h3 { color: #e2eaf3 !important; }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0;
    }
    .sub-title { color: #8babc7 !important; font-size: 1rem; margin-top: 0; }

    .stSelectbox label, .stSlider label,
    .stRadio label, .stFileUploader label {
        color: #cdd9e5 !important;
        font-weight: 500;
    }

    .status-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        border: 1px solid rgba(34,197,94,0.3);
    }
    .status-estimate {
        background: rgba(250,204,21,0.15);
        color: #facc15;
        border: 1px solid rgba(250,204,21,0.3);
    }

    .insight-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100,160,255,0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #cdd9e5;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #5a7a9a;
        font-size: 0.8rem;
        margin-top: 50px;
        padding: 16px 0;
        border-top: 1px solid rgba(100,160,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Estate Intelligence")
    st.markdown("AI-powered real estate price prediction using a trained Random Forest model.")

    st.markdown("---")
    st.markdown("**Team:** Estate Intelligence")
    st.markdown("---")

    st.markdown("### 📦 Model Status")
    if model is not None:
        st.markdown('<span class="status-badge status-ready">✅ Model Ready</span>',
                    unsafe_allow_html=True)
        st.success("Tuned Random Forest loaded successfully.", icon="✅")
    else:
        st.markdown('<span class="status-badge status-estimate">⚠️ Estimate Mode</span>',
                    unsafe_allow_html=True)
        st.warning("Model file not found. Using placeholder formula.", icon="⚠️")

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Predict** — Fill in property details in Tab 1 and click *Predict Price*
    2. **Insights** — View feature importances and correlation heatmap in Tab 2
    3. **Batch** — Upload a CSV in Tab 3 to predict multiple properties at once
    4. **Download** — Export batch results as CSV
    """)


# ──────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏠 Estate Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict residential property prices with machine learning</p>',
            unsafe_allow_html=True)
st.markdown("")


# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏡 Predict Price",
    "📊 Data Insights",
    "📂 Batch Prediction"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — SINGLE PROPERTY PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("### Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
    "📐 Area (sq ft)",
    min_value=500,
    max_value=16200,
    value=3000,
    step=50,
    help="Type the property area in square feet"
)
        st.info("ℹ️ Model trained on areas **1,650 – 16,200 sq ft**. "
                "Values outside this range may give less accurate predictions.",
                icon="ℹ️")
        bedrooms        = st.selectbox("🛏️ Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms       = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4], index=0)
        stories         = st.selectbox("🏢 Stories", options=[1, 2, 3, 4], index=0)
        parking         = st.selectbox("🚗 Parking Spots", options=[0, 1, 2, 3], index=0)
        furnishing      = st.selectbox("🪑 Furnishing Status",
                                       options=["Unfurnished", "Semi-Furnished", "Furnished"],
                                       index=0)

    with col2:
        mainroad        = st.radio("🛣️ Main Road Access",  ["Yes", "No"], index=0, horizontal=True)
        guestroom       = st.radio("🚪 Guest Room",         ["Yes", "No"], index=1, horizontal=True)
        basement        = st.radio("🏗️ Basement",           ["Yes", "No"], index=1, horizontal=True)
        hotwaterheating = st.radio("♨️ Hot Water Heating",  ["Yes", "No"], index=1, horizontal=True)
        airconditioning = st.radio("❄️ Air Conditioning",   ["Yes", "No"], index=1, horizontal=True)
        prefarea        = st.radio("⭐ Preferred Area",      ["Yes", "No"], index=1, horizontal=True)

    # Encode Yes/No inputs to 1/0
    yes_no         = lambda x: 1 if x == "Yes" else 0
    furnishing_val = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}[furnishing]

    st.markdown("")

    if st.button("🔮 Predict Price", use_container_width=True, key="single_predict"):
        # Store prediction in session state so it persists across rerenders
        st.session_state["predicted"] = predict_price(
            area, bedrooms, bathrooms, stories,
            yes_no(mainroad), yes_no(guestroom), yes_no(basement),
            yes_no(hotwaterheating), yes_no(airconditioning),
            parking, yes_no(prefarea), furnishing_val,
        )

    # Show result if prediction exists in session state
    if "predicted" in st.session_state:
        predicted = st.session_state["predicted"]

        # Styled price output box
        st.markdown(
            f"""
            <div class="price-box">
                <div class="label">Estimated Property Price</div>
                <div class="price">{format_inr(predicted)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feature importance chart using REAL model values
        st.markdown("#### 📊 Feature Importance (Trained Random Forest)")
        st.caption("How much each feature contributed to the model's predictions.")

        fi_sorted = dict(sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#0a1628")
        ax.set_facecolor("#0a1628")

        colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(fi_sorted)))

        bars = ax.barh(
            list(fi_sorted.keys()),
            list(fi_sorted.values()),
            color=colors,
            edgecolor="none",
            height=0.6,
            zorder=3,
        )

        ax.set_xlim(0, 0.56)
        ax.set_xlabel("Importance Score", color="#8babc7", fontsize=10)
        ax.tick_params(colors="#8babc7", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#1e3a5f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#1e3a5f", linewidth=0.5, zorder=0)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1%}", va="center", color="#a3c4f3",
                    fontsize=8.5, fontweight=600)

        plt.tight_layout()
        st.pyplot(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — DATA INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### 📊 Model & Data Insights")

    fi_img_path   = os.path.join(os.path.dirname(__file__), "notebooks", "feature_importance.png")
    corr_img_path = os.path.join(os.path.dirname(__file__), "notebooks", "correlation_heatmap.png")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Feature Importance")
        if os.path.exists(fi_img_path):
            st.image(fi_img_path, caption="Feature Importance — Tuned Random Forest",
                     use_container_width=True)
        else:
            st.warning("feature_importance.png not found in notebooks/")

    with col_b:
        st.markdown("#### Correlation Heatmap")
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, caption="Correlation Heatmap — Feature Relationships",
                     use_container_width=True)
        else:
            st.warning("correlation_heatmap.png not found in notebooks/")

    st.markdown("#### 🔍 Key Findings")
    st.markdown("""
    <div class="insight-card">
        🏆 <strong>Area dominates (47.8%)</strong> — Property size is by far the strongest predictor of price.
        Larger homes consistently command higher prices in this dataset.
    </div>
    <div class="insight-card">
        🚿 <strong>Bathrooms are second (15.8%)</strong> — Number of bathrooms has strong influence,
        likely because it correlates with overall property quality and size.
    </div>
    <div class="insight-card">
        ❄️ <strong>Air Conditioning matters (6.8%)</strong> — Presence of AC significantly
        adds to property value, reflecting buyer preferences in the Indian market.
    </div>
    <div class="insight-card">
        🏢 <strong>Stories & Parking (5.2%, 4.8%)</strong> — Multi-storey homes and parking
        availability contribute meaningfully to price.
    </div>
    <div class="insight-card">
        🛣️ <strong>Main Road access has lowest impact (1.0%)</strong> — Despite being a common
        amenity listed, it has the least influence on price among all features.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — BATCH CSV PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### Upload CSV for Batch Prediction")
    st.caption(
        "Your CSV must contain: "
        "`area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, "
        "`basement`, `hotwaterheating`, `airconditioning`, `parking`, "
        "`prefarea`, `furnishingstatus`  "
        "(binary columns as 0/1, furnishingstatus as 0=Unfurnished / 1=Semi / 2=Furnished)"
    )

    # Sample template download
    sample_data = pd.DataFrame([{
        "area": 1500, "bedrooms": 3, "bathrooms": 2, "stories": 2,
        "mainroad": 1, "guestroom": 0, "basement": 0,
        "hotwaterheating": 0, "airconditioning": 1,
        "parking": 1, "prefarea": 0, "furnishingstatus": 1
    }])
    st.download_button(
        label="⬇️ Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="sample_input.csv",
        mime="text/csv",
    )

    st.markdown("")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### 📄 Uploaded Data")
        st.dataframe(df, use_container_width=True)

        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing columns: **{', '.join(missing_cols)}**. Please fix your CSV.")
        else:
            if st.button("🚀 Predict All", use_container_width=True, key="batch_predict"):
                with st.spinner("Running predictions..."):
                    df["predicted_price"] = df.apply(
                        lambda row: predict_price(
                            row["area"], row["bedrooms"], row["bathrooms"],
                            row["stories"], row["mainroad"], row["guestroom"],
                            row["basement"], row["hotwaterheating"],
                            row["airconditioning"], row["parking"],
                            row["prefarea"], row["furnishingstatus"],
                        ),
                        axis=1,
                    )
                    df["predicted_price_INR"] = df["predicted_price"].apply(format_inr)

                st.markdown("#### ✅ Predictions Complete")
                st.dataframe(df, use_container_width=True)

                export_df = df.drop(columns=["predicted_price"])
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=export_df.to_csv(index=False),
                    file_name="estate_intelligence_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "⚠️ Predictions are estimates only. Not financial advice. "
    "| 🏠 Estate Intelligence · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)