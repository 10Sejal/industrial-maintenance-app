import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration (The "Professional" touch)
st.set_page_config(page_title="L&T Asset Reliability", page_icon="🏗️")

# 2. Load the Model
model = joblib.load('failure_model.pkl')

# 3. Sidebar for Technical Info
st.sidebar.title("Project Overview")
st.sidebar.info("This system uses AI to predict industrial equipment failure, helping companies like L&T reduce downtime.")

# 4. Main UI
st.title("🏗️ Industrial Equipment Health Monitor")
st.markdown("Enter sensor readings below to assess the real-time risk of machine failure.")

# 5. User Inputs (Sliding bars for sensor data)
col1, col2 = st.columns(2)
with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    proc_temp = st.number_input("Process Temperature (K)", value=310.0)
    rpm = st.number_input("Rotational Speed (RPM)", value=1500)
with col2:
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=100)

# Calculate the feature we engineered in Colab
temp_diff = proc_temp - air_temp

# 6. Prediction Logic
if st.button("Run Diagnostic Analysis"):
    # Must match the order used during model training!
    features = np.array([[air_temp, proc_temp, rpm, torque, tool_wear, temp_diff]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("🚨 HIGH RISK: Failure Predicted! Maintenance recommended.")
    else:
        st.success("✅ OPERATIONAL: Machine is healthy.")