"""
Interactive Streamlit Application
Hourly Bike Rental Demand Prediction via Ensemble Regression
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import custom transformer and helper so joblib can deserialize the pipeline correctly
from src.data_preparation import BikeFeatureEngineering, inverse_nonnegative

st.set_page_config(
    page_title="Bike Rental Demand Forecaster",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge {
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_best_model():
    model_path = os.path.join("models", "best_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please run training first.")
        st.stop()
    return joblib.load(model_path)

def main():
    st.markdown('<div class="main-header">🚲 Hourly Bike Rental Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ensemble Regression Inference Dashboard for Capital Bikeshare System</div>', unsafe_allow_html=True)

    # Sidebar: Input Features
    st.sidebar.header("⏱️ Calendar & Time Parameters")
    
    col_yr, col_season = st.sidebar.columns(2)
    with col_yr:
        yr_choice = st.selectbox("Year", options=[2011, 2012], index=1)
        yr = 0 if yr_choice == 2011 else 1
    with col_season:
        season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
        season = st.selectbox("Season", options=[1, 2, 3, 4], format_func=lambda x: f"{x} - {season_names[x]}", index=2)
        
    col_mnth, col_hr = st.sidebar.columns(2)
    with col_mnth:
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        mnth = st.selectbox("Month", options=list(month_names.keys()), format_func=lambda x: f"{x} ({month_names[x]})", index=7)
    with col_hr:
        hr = st.slider("Hour of Day (0–23)", min_value=0, max_value=23, value=17, step=1)

    col_day, col_flags = st.sidebar.columns(2)
    with col_day:
        weekday_names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
        weekday = st.selectbox("Day of Week", options=list(weekday_names.keys()), format_func=lambda x: weekday_names[x], index=4)
    with col_flags:
        is_holiday = st.checkbox("Holiday?", value=False)
        holiday = 1 if is_holiday else 0
        is_workingday = st.checkbox("Working Day?", value=True if weekday in [1, 2, 3, 4, 5] and not is_holiday else False)
        workingday = 1 if is_workingday else 0

    st.sidebar.markdown("---")
    st.sidebar.header("🌤️ Weather Parameters")
    
    weather_desc = {
        1: "Clear / Few Clouds",
        2: "Mist / Cloudy",
        3: "Light Snow / Light Rain",
        4: "Heavy Rain / Snow"
    }
    weathersit = st.sidebar.selectbox("Weather Condition", options=[1, 2, 3, 4], format_func=lambda x: f"{x}: {weather_desc[x]}", index=0)

    # Weather Sliders in 2 Columns for optimal layout
    col_t1, col_t2 = st.sidebar.columns(2)
    with col_t1:
        temp_c = st.slider("Temp (°C)", min_value=-8.0, max_value=39.0, value=25.0, step=0.5)
    with col_t2:
        atemp_c = st.slider("Feels-Like (°C)", min_value=-16.0, max_value=50.0, value=26.0, step=0.5)

    col_w1, col_w2 = st.sidebar.columns(2)
    with col_w1:
        hum_pct = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1)
    with col_w2:
        wind_kmh = st.slider("Wind (km/h)", min_value=0.0, max_value=67.0, value=12.0, step=0.5)

    # Normalize inputs as per UCI dataset definition
    # temp: (t - t_min) / (t_max - t_min), t_min=-8, t_max=+39
    temp_norm = (temp_c - (-8.0)) / (39.0 - (-8.0))
    # atemp: (at - at_min) / (at_max - at_min), at_min=-16, at_max=+50
    atemp_norm = (atemp_c - (-16.0)) / (50.0 - (-16.0))
    # hum: hum / 100
    hum_norm = hum_pct / 100.0
    # windspeed: wind / 67
    wind_norm = wind_kmh / 67.0

    # Build input DataFrame
    input_data = pd.DataFrame([{
        "season": season,
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": temp_norm,
        "atemp": atemp_norm,
        "hum": hum_norm,
        "windspeed": wind_norm
    }])

    # Load Model
    model = load_best_model()

    # Inference
    try:
        raw_pred = model.predict(input_data)[0]
        prediction = max(0.0, float(raw_pred))
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    # Main Dashboard Display
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.subheader("🎯 Demand Forecast")
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.95rem; color: #6B7280; text-transform: uppercase; font-weight: 600;">Predicted Hourly Rentals</span><br>
            <span style="font-size: 3.2rem; font-weight: 800; color: #1E40AF;">{prediction:.0f}</span>
            <span style="font-size: 1.2rem; color: #374151; font-weight: 600;"> bikes / hour</span>
            <hr style="margin: 0.8rem 0; border: none; border-top: 1px solid #E5E7EB;">
            <div style="font-size: 0.95rem; color: #4B5563;">
                <strong>Expected Range:</strong> {max(0, int(prediction - 49))} – {int(prediction + 49)} bikes/hr (±MAE margin)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Operational Level")
        if prediction < 100:
            status_text = "Low Demand"
            status_color = "#10B981"
            bg_color = "#D1FAE5"
            advice = "Minimal fleet redistribution needed. Ideal window for station maintenance."
        elif prediction < 350:
            status_text = "Moderate Demand"
            status_color = "#3B82F6"
            bg_color = "#DBEAFE"
            advice = "Standard operating conditions. Maintain normal dock rebalancing routes."
        elif prediction < 600:
            status_text = "High Demand"
            status_color = "#F59E0B"
            bg_color = "#FEF3C7"
            advice = "Peak operational load. Deploy mobile redistribution teams to commuter hubs."
        else:
            status_text = "Surge Demand"
            status_color = "#EF4444"
            bg_color = "#FEE2E2"
            advice = "Extreme surge. Prioritize rapid station replenishment and monitor dock saturation."

        st.markdown(f"""
        <div style="background-color: {bg_color}; border-radius: 10px; padding: 1.2rem; border-left: 5px solid {status_color};">
            <span style="color: {status_color}; font-weight: bold; font-size: 1.3rem;">{status_text}</span>
            <p style="margin-top: 0.5rem; font-size: 0.95rem; color: #1F2937;">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.subheader("📋 Context Summary")
        is_rush = (hr in [8, 9, 17, 18]) and (workingday == 1)
        st.markdown(f"""
        - **Timestamp Slot:** {yr_choice}-{mnth:02d} | Hour {hr:02d}:00
        - **Calendar Status:** {weekday_names[weekday]} {'(Working Day)' if workingday else '(Leisure Day)'}
        - **Rush Hour Commute:** {'Yes (High Demand Pattern)' if is_rush else 'No (Standard Off-Peak)'}
        - **Weather Scenario:** {weather_desc[weathersit]} ({temp_c:.1f}°C, {hum_pct}%)
        """)

    st.markdown("---")
    
    # Bottom Tabs: Model Overview, Performance, Data Dictionary
    tab1, tab2, tab3 = st.tabs(["📈 Validation & Test Performance", "🔍 Feature Breakdown", "ℹ️ System & Reproducibility"])

    with tab1:
        st.write("### Model Performance Across Evaluation Partitions")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Test MAE", "49.17 bikes/hr", "-70.9% vs Baseline", delta_color="inverse")
        mcol2.metric("Test RMSE", "73.81 bikes/hr", "-67.7% vs Baseline", delta_color="inverse")
        mcol3.metric("Test R² Score", "0.8823", "+1.01 vs Dummy")
        mcol4.metric("Validation MAE", "39.70 bikes/hr", "Best Candidate")
        st.info("Performance evaluated via strict chronological holdout partitions without temporal shuffling to ensure real-world forecasting validity.")

    with tab2:
        st.write("### Input Feature Representation")
        display_df = input_data.copy()
        display_df["temp_celsius"] = temp_c
        display_df["humidity_pct"] = hum_pct
        display_df["windspeed_kmh"] = wind_kmh
        st.dataframe(display_df, use_container_width=True)

    with tab3:
        st.write("### System Architecture & Compliance")
        st.markdown("""
        - **Ensemble Core:** Tuned LightGBM Regressor with cyclical time encoders and commuter interaction features.
        - **Leakage Prevention:** `casual` and `registered` columns strictly excluded (`cnt = casual + registered`).
        - **Bounded Output:** Predictions lower-bounded at 0 through inverse log1p transformation (`np.clip(np.expm1(y), 0, None)`).
        - **Reproducibility:** Random seed fixed at `42`, deterministic tree partitioning.
        """)

if __name__ == "__main__":
    main()
