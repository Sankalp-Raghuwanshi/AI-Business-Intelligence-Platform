import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from sklearn.preprocessing import LabelEncoder

# Load all models
@st.cache_resource
def load_models():
    delay_model = joblib.load("models/delay_model.pkl")
    review_model = joblib.load("models/review_model.pkl")
    le_customer_state = joblib.load("models/le_customer_state.pkl")
    le_seller_state = joblib.load("models/le_seller_state.pkl")
    le_category = joblib.load("models/le_category.pkl")
    segment_summary = joblib.load("models/segment_summary.pkl")
    rfm_data = joblib.load("models/rfm_data.pkl")
    return delay_model, review_model, le_customer_state, le_seller_state, le_category, segment_summary, rfm_data

# Load unique values for dropdowns
@st.cache_data
def load_options():
    conn = sqlite3.connect("data/olist.db")
    states = pd.read_sql_query(
        "SELECT DISTINCT customer_state FROM master_orders WHERE customer_state IS NOT NULL ORDER BY customer_state", conn
    )['customer_state'].tolist()
    categories = pd.read_sql_query(
        "SELECT DISTINCT category FROM master_orders WHERE category IS NOT NULL ORDER BY category", conn
    )['category'].tolist()
    conn.close()
    return states, categories

delay_model, review_model, le_customer_state, le_seller_state, le_category, segment_summary, rfm_data = load_models()
states, categories = load_options()

st.title("🤖 ML Predictions")
st.markdown("Use trained machine learning models to predict delivery outcomes and customer satisfaction.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🚚 Delay Predictor",
    "⭐ Review Score Predictor", 
    "👥 Customer Segments"
])

# ── TAB 1: DELAY PREDICTOR ─────────────────────────────────────────
with tab1:
    st.markdown("### 🚚 Delivery Delay Risk Predictor")
    st.markdown("Enter order details to predict whether it will be delayed.")

    col1, col2 = st.columns(2)
    
    with col1:
        customer_state = st.selectbox("Customer State", states, key="delay_cs")
        seller_state = st.selectbox("Seller State", states, key="delay_ss")
        category = st.selectbox("Product Category", categories, key="delay_cat")
        order_month = st.slider("Order Month", 1, 12, 6, key="delay_month")
    
    with col2:
        price = st.number_input("Product Price (R$)", min_value=10.0, max_value=5000.0, value=150.0, key="delay_price")
        freight_value = st.number_input("Freight Value (R$)", min_value=5.0, max_value=500.0, value=20.0, key="delay_freight")
        payment_value = st.number_input("Payment Value (R$)", min_value=10.0, max_value=10000.0, value=170.0, key="delay_payment")
        order_year = st.selectbox("Order Year", [2017, 2018], key="delay_year")

    if st.button("🔍 Predict Delay Risk", type="primary", key="delay_btn"):
        try:
            # Engineer features
            freight_ratio = freight_value / (price + 1)
            is_holiday_season = 1 if order_month in [11, 12] else 0
            
            # Encode categoricals
            cs_encoded = le_customer_state.transform([customer_state])[0]
            ss_encoded = le_seller_state.transform([seller_state])[0]
            cat_encoded = le_category.transform([category])[0]
            seller_customer_same = 1 if customer_state == seller_state else 0

            input_data = pd.DataFrame([[
                cs_encoded, ss_encoded, cat_encoded,
                price, freight_value, payment_value,
                order_month, order_year,
                freight_ratio, is_holiday_season, seller_customer_same
            ]], columns=[
                'customer_state', 'seller_state', 'category',
                'price', 'freight_value', 'payment_value',
                'order_month', 'order_year',
                'freight_ratio', 'is_holiday_season', 'seller_customer_same_state'
            ])

            prediction = delay_model.predict(input_data)[0]
            probability = delay_model.predict_proba(input_data)[0]

            st.markdown("---")
            if prediction == 1:
                delay_pct = probability[1] * 100
                st.error(f"⚠️ HIGH DELAY RISK — {delay_pct:.1f}% probability of delay")
                st.markdown("**Factors that may contribute to delay:**")
                if is_holiday_season:
                    st.markdown("- 🎄 Holiday season order (Nov/Dec have higher delay rates)")
                if customer_state != seller_state:
                    st.markdown("- 📦 Cross-state delivery (longer distance)")
                if freight_ratio > 0.2:
                    st.markdown("- 💰 High freight-to-price ratio (remote delivery likely)")
            else:
                on_time_pct = probability[0] * 100
                st.success(f"✅ LOW DELAY RISK — {on_time_pct:.1f}% probability of on-time delivery")
                if customer_state == seller_state:
                    st.markdown("- 📍 Same-state delivery reduces delay risk")
                if not is_holiday_season:
                    st.markdown("- 📅 Non-holiday period has lower delay rates")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── TAB 2: REVIEW SCORE PREDICTOR ──────────────────────────────────
with tab2:
    st.markdown("### ⭐ Customer Review Score Predictor")
    st.markdown("Predict likely review score based on delivery performance.")

    col1, col2 = st.columns(2)
    
    with col1:
        delivery_days = st.slider("Delivery Days", 1, 60, 12, key="review_days")
        delay_flag = st.radio("Was order delayed?", ["No", "Yes"], key="review_delay")
        order_month_r = st.slider("Order Month", 1, 12, 6, key="review_month")
    
    with col2:
        price_r = st.number_input("Product Price (R$)", min_value=10.0, max_value=5000.0, value=150.0, key="review_price")
        freight_value_r = st.number_input("Freight Value (R$)", min_value=5.0, max_value=500.0, value=20.0, key="review_freight")
        payment_value_r = st.number_input("Payment Value (R$)", min_value=10.0, max_value=10000.0, value=170.0, key="review_payment")

    if st.button("⭐ Predict Review Score", type="primary", key="review_btn"):
        try:
            delay_flag_val = 1 if delay_flag == "Yes" else 0
            freight_ratio_r = freight_value_r / (price_r + 1)
            is_holiday_r = 1 if order_month_r in [11, 12] else 0

            input_data_r = pd.DataFrame([[
                delivery_days, delay_flag_val, freight_value_r,
                price_r, payment_value_r, order_month_r,
                freight_ratio_r, is_holiday_r
            ]], columns=[
                'delivery_days', 'delay_flag', 'freight_value',
                'price', 'payment_value', 'order_month',
                'freight_ratio', 'is_holiday_season'
            ])

            prediction_r = review_model.predict(input_data_r)[0]
            probability_r = review_model.predict_proba(input_data_r)[0]
            classes = review_model.classes_

            st.markdown("---")
            if prediction_r == 'High (4-5)':
                st.success(f"⭐⭐⭐⭐⭐ Predicted: {prediction_r}")
            elif prediction_r == 'Medium (3)':
                st.warning(f"⭐⭐⭐ Predicted: {prediction_r}")
            else:
                st.error(f"⭐ Predicted: {prediction_r}")

            st.markdown("**Probability breakdown:**")
            prob_df = pd.DataFrame({
                'Score Bucket': classes,
                'Probability': [f"{p:.1%}" for p in probability_r]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            if delivery_days > 30:
                st.markdown("⚠️ Delivery over 30 days significantly increases low review risk")
            if delay_flag_val == 1:
                st.markdown("⚠️ Delayed orders are 3x more likely to receive low reviews")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── TAB 3: CUSTOMER SEGMENTS ────────────────────────────────────────
with tab3:
    st.markdown("### 👥 Customer Segmentation (RFM Analysis)")
    st.markdown("Customers segmented by Recency, Frequency, and Monetary value using KMeans clustering.")

    st.markdown("#### Segment Overview")
    
    display_df = segment_summary.reset_index()[['label', 'count', 'avg_recency', 'avg_frequency', 'avg_monetary']].copy()
    display_df.columns = ['Segment', 'Orders', 'Avg Days Since Order', 'Avg Order Frequency', 'Avg Revenue (R$)']
    display_df = display_df.sort_values('Avg Revenue (R$)', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### What each segment means:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**🏆 Champions** — Ordered recently, highest frequency. Your best customers. Reward them with loyalty programs.")
        st.warning("**⚠️ At Risk** — Haven't ordered in 395+ days. Need re-engagement campaigns urgently.")
    with col2:
        st.success("**💎 Loyal Customers** — High frequency, high spend. Nurture with exclusive offers.")
        st.error("**🐋 VIP / Whale** — Only 19 orders but R$26K avg spend. Likely B2B. Assign dedicated account managers.")

    st.markdown("---")
    st.markdown("#### RFM Distribution")
    
    import plotly.express as px
    fig = px.bar(
        display_df,
        x='Segment',
        y='Orders',
        color='Avg Revenue (R$)',
        color_continuous_scale='Teal',
        title='Customer Count by Segment'
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)