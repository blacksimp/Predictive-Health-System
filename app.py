import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="MOT Predictive Health", page_icon="🚗")

@st.cache_resource # Caches the model so it doesn't reload on every click
def load_model():
    return joblib.load('xgb_mot_model.pkl')

@st.cache_data
def load_risk_data():
    return pd.read_csv('risk_lookup.csv')

model = load_model()
risk_df = load_risk_data()

# --- 2. FRONTEND UI (The Sidebar) ---
st.title("🚗 DVSA Predictive Maintenance System")
st.markdown("Enter the vehicle details below to assess the probability of MOT failure.")

st.sidebar.header("Vehicle Specifications")
make = st.sidebar.selectbox("Make", risk_df['make'].unique())
# Filter models based on the selected make
available_models = risk_df[risk_df['make'] == make]['model'].unique()
model_name = st.sidebar.selectbox("Model", available_models)

age = st.sidebar.slider("Vehicle Age (Years)", 1, 30, 5)
mileage = st.sidebar.number_input("Current Mileage", min_value=1000, max_value=300000, value=50000, step=1000)
fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

# --- 3. BACKEND LOGIC (When the user clicks the button) ---
if st.sidebar.button("Run Diagnostics"):
    
    # Feature Engineering
    mpy = mileage / max(age, 0.1)
    risk_row = risk_df[(risk_df['make'] == make) & (risk_df['model'] == model_name)]
    risk_index = risk_row['make_model_risk'].values[0] if not risk_row.empty else risk_df['make_model_risk'].mean()

    # Format data for XGBoost (Matching your training columns exactly)
    # Format data for XGBoost (Matching your training columns EXACTLY)
    # Notice that 'fuel_clean_Diesel' is intentionally missing!
    input_data = pd.DataFrame({
        'vehicle_age': [age],
        'test_mileage': [mileage],
        'miles_per_year': [mpy],
        'make_model_risk': [risk_index],
        'fuel_clean_Electric': [1 if fuel == 'Electric' else 0],
        'fuel_clean_Hybrid': [1 if fuel == 'Hybrid' else 0],
        'fuel_clean_Other': [1 if fuel == 'Other' else 0],
        'fuel_clean_Petrol': [1 if fuel == 'Petrol' else 0]
    })

    # Predict
    probability = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Fail)

    # --- 4. DISPLAY RESULTS ---
    st.divider()
    st.subheader("Diagnostic Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Mileage", f"{int(mpy):,} miles")
    col2.metric("Historical Brand Risk", f"{risk_index:.1%}")
    col3.metric("Predicted Failure Probability", f"{probability:.1%}")

    st.progress(float(probability)) # Visual progress bar

    if probability > 0.60:
        st.error("🔴 HIGH RISK: Immediate inspection recommended. Components are likely past standard wear limits.")
    elif probability > 0.35:
        st.warning("🟡 MODERATE RISK: Monitor wear and tear. Recommend pre-MOT servicing.")
    else:
        st.success("🟢 LOW RISK: Vehicle is within expected operational health.")