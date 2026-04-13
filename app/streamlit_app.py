# ==============================
# 📦 IMPORT LIBRARIES
# ==============================
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd

# ==============================
# 🎯 PAGE CONFIG
# ==============================
st.set_page_config(page_title="Dynamic Pricing AI", layout="wide")

# ==============================
# 🎨 MODERN UI STYLING
# ==============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #020617;
}
.stMetric {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #475569;
}
div.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ==============================
#  HEADER
# ==============================
st.markdown("""
# Dynamic Pricing Dashboard  
### 📊 AI-powered Pricing Intelligence System
""")

# ==============================
# 📥 LOAD MODEL & ENCODERS
# ==============================
@st.cache_resource
def load_models():
    with open("../models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("../models/encoders.pkl", "rb") as f:
        le_day, le_product = pickle.load(f)

    return model, le_day, le_product

model, le_day, le_product = load_models()

# ==============================
# 🎛️ SIDEBAR INPUTS
# ==============================
st.sidebar.markdown("## ⚙️ Controls")

price = st.sidebar.slider("💰 Price", 50, 500, 100)
discount = st.sidebar.slider("🏷️ Discount (%)", 0, 50, 10)
sentiment = st.sidebar.slider("😊 Sentiment Score", 0.0, 1.0, 0.5)
competitor_price = st.sidebar.slider("📊 Competitor Price", 50, 500, 120)

day = st.sidebar.selectbox("📅 Day of Week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# ==============================
# 🔥 PRODUCT NAME MAPPING (FIXED)
# ==============================

# Your original encoded IDs
product_ids = list(le_product.classes_)

# Map IDs → REAL PRODUCT NAMES (EDIT THIS ONCE)
product_name_map = {
    "P1": "Wireless Earbuds",
    "P2": "Smart Watch",
    "P3": "Bluetooth Speaker",
    "P4": "Laptop Stand",
    "P5": "Gaming Mouse",
    "P6": "Mechanical Keyboard",
    "P7": "Noise Cancelling Headphones",
    "P8": "Portable Charger",
    "P9": "Fitness Band",
    "P10": "USB-C Hub",
    "P11": "External SSD",
    "P12": "Webcam HD",
    "P13": "LED Monitor",
    "P14": "Tablet Stand",
    "P15": "Smartphone Case",
    "P16": "Wireless Charger",
    "P17": "Action Camera",
    "P18": "VR Headset",
    "P19": "WiFi Router",
    "P20": "Graphics Tablet"
}

# Only show mapped names (fallback if missing)
product_display_names = [product_name_map.get(pid, pid) for pid in product_ids]

# UI shows product names
selected_product_name = st.sidebar.selectbox("📦 Product", product_display_names)

# Reverse mapping: name → ID
reverse_map = {v: k for k, v in product_name_map.items()}

selected_product_id = reverse_map.get(selected_product_name, selected_product_name)

# Encode for model
product_encoded = le_product.transform([selected_product_id])[0]

# ==============================
# 🔄 FEATURE ENGINEERING
# ==============================
day_encoded = le_day.transform([day])[0]

is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
price_diff = price - competitor_price
month = 6

# ==============================
# 📊 FEATURE NAMES
# ==============================
feature_names = [
    "price","discount","sentiment_score","competitor_price",
    "day_of_week","product_id","is_weekend","price_diff","month"
]

# ==============================
# 📊 MODEL INPUT
# ==============================
input_data = np.array([[price,discount,sentiment,competitor_price,
                        day_encoded,product_encoded,is_weekend,
                        price_diff,month]])

input_df = pd.DataFrame(input_data, columns=feature_names)

# ==============================
# 🔮 DEMAND PREDICTION
# ==============================
prediction = model.predict(input_data)[0]

# ==============================
# 📊 TABS
# ==============================
tab1, tab2, tab3 = st.tabs(["📊 Demand", "📈 Optimization", "🧠 Insights"])

# ==============================
# 📊 TAB 1
# ==============================
with tab1:
    st.subheader("📊 Demand Prediction")
    st.info(f"📦 Selected Product: **{selected_product_name}**")

    col1, col2 = st.columns(2)
    col1.metric("📦 Predicted Demand", int(prediction))
    col2.metric("💰 Current Price", price)

# ==============================
# 📈 TAB 2: OPTIMIZATION
# ==============================
with tab2:

    st.subheader("📈 Optimal Pricing Strategy")

    st.info("📊 System simulates multiple prices to maximize revenue.")

    if st.button(" Find Optimal Price"):

        with st.spinner("Calculating optimal price..."):

            price_range = np.linspace(50, 500, 50)

            demands = []
            revenues = []

            for p in price_range:
                temp_price_diff = p - competitor_price

                temp_input = np.array([[p,discount,sentiment,competitor_price,
                                        day_encoded,product_encoded,
                                        is_weekend,temp_price_diff,month]])

                demand = model.predict(temp_input)[0]
                revenue = p * demand

                demands.append(demand)
                revenues.append(revenue)

            max_index = np.argmax(revenues)
            optimal_price = price_range[max_index]
            max_revenue = revenues[max_index]

        col3, col4 = st.columns(2)

        with col3:
            st.metric("🎯 Optimal Price", round(optimal_price, 2))

        with col4:
            st.metric("📈 Max Revenue", int(max_revenue))

        # Recommendation
        st.markdown(f"""
        ### 💰 Recommended Action

        **Optimal Price:** ₹{round(optimal_price,2)}  
        **Expected Revenue:** ₹{int(max_revenue)}
        """)

        if optimal_price > price:
            st.success("📈 Increase price → maximize revenue")
        else:
            st.warning("📉 Reduce price → improve demand")

        # Graph
        fig, ax = plt.subplots(figsize=(9,5))
        # Line plot
           # Line plot
        ax.plot(price_range, revenues, linewidth=3, color="#38bdf8")

        # Optimal price line
        ax.axvline(optimal_price, linestyle='--', linewidth=2, color="#f97316")

        # 🔥 FIX: Dark theme styling
        ax.set_facecolor("#020617")
        fig.patch.set_facecolor("#020617")

        # 🔥 FIX: Visible labels
        ax.set_title("📈 Revenue vs Price", color="white", fontsize=14)
        ax.set_xlabel("Price", color="white")
        ax.set_ylabel("Revenue", color="white")

        # 🔥 FIX: Tick colors
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # 🔥 FIX: Grid styling
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")

        # 🔥 Optional: highlight optimal point
        ax.scatter(optimal_price, max(revenues), color="#f97316", s=80)

        st.pyplot(fig)


# ==============================
# 🧠 TAB 3
# ==============================
with tab3:

    st.subheader(" Model Explanation")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    fig = plt.figure(figsize=(8,4.5))
    shap.plots.bar(shap_values, max_display=len(feature_names), show=False)

    plt.tight_layout()
    st.pyplot(fig)

# ==============================
# 📌 FOOTER
# ==============================
st.markdown("---")
st.markdown("AI-powered pricing system leveraging XGBoost for demand prediction and SHAP for model interpretability")