import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="L&T Asset Monitor", page_icon="🏗️", layout="wide")

# 2. Load the Model
@st.cache_resource
def load_model():
    return joblib.load('failure_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model file 'failure_model.pkl' not found. Error: {e}")

# 3. Main Interface
st.title("🏗️ Industrial Equipment Health Monitor")
st.write("Predictive Maintenance Analysis for L&T Industrial Assets")
st.markdown("---")

# 4. User Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Temperature Metrics")
    air_temp = st.number_input("Air Temperature (K)", value=300.0, format="%.2f")
    proc_temp = st.number_input("Process Temperature (K)", value=310.0, format="%.2f")
    st.subheader("Operational Metrics")
    rpm = st.number_input("Rotational Speed (RPM)", value=1500.0, format="%.2f")

with col2:
    st.subheader("Mechanical Stress")
    torque = st.number_input("Torque (Nm)", value=40.0, format="%.2f")
    tool_wear = st.number_input("Tool Wear (min)", value=100.0, format="%.2f")

# Calculate Engineeed Feature
temp_diff = proc_temp - air_temp

# 5. Prediction Logic (The Permanent Fix)
st.markdown("---")
if st.button("Run Diagnostic Analysis", use_container_width=True):
    # Step A: Define the raw names
    raw_columns = [
        'Air temperature [K]', 'Process temperature [K]', 
        'Rotational speed [rpm]', 'Torque [Nm]', 
        'Tool wear [min]', 'Temp_Diff'
    ]
    
    # Step B: Create the initial DataFrame
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear, temp_diff]], 
                            columns=raw_columns)
    
    # Step C: Clean the names to match the XGBoost Training Fix 
    # (Removes '[', ']', and '<' just like we did in Google Colab)
    input_df.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in input_df.columns]
    
    try:
        # Step D: Make the prediction
        prediction = model.predict(input_df)
        
        if prediction[0] == 1:
            st.error("🚨 **CRITICAL ALERT**: High Risk of Machine Failure detected.")
            st.warning("Recommendation: Immediate maintenance required to prevent equipment damage.")
        else:
            st.success("✅ **STABLE**: Equipment is operating within normal safety limits.")
            st.info("Recommendation: Proceed with standard operation schedule.")
            
    except ValueError as e:
        st.error("Technical Mismatch Detected. Please check the 'Manage App' logs for the specific column name XGBoost is expecting.")
        st.write(f"Current input columns: {list(input_df.columns)}")

# 6. Sidebar Project Info
st.sidebar.title("L&T Recruitment Project")
st.sidebar.info("""
**Candidate Branch:** CSE (Data Science)
**Objective:** Use Machine Learning to reduce unplanned downtime in heavy machinery.
**Model:** XGBoost Classifier optimized for Recall.
""")
