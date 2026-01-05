import streamlit as st
import pandas as pd
import joblib

# 1. Page Config
st.set_page_config(page_title="L&T Asset Monitor", page_icon="🏗️")

# 2. Load the Model
# Ensure failure_model.pkl is in the same GitHub folder
model = joblib.load('failure_model.pkl')

st.title("🏗️ Industrial Equipment Health Monitor")
st.write("Enter sensor readings to predict real-time machine failure risk.")

# 3. User Inputs
col1, col2 = st.columns(2)
with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    proc_temp = st.number_input("Process Temperature (K)", value=310.0)
    rpm = st.number_input("Rotational Speed (RPM)", value=1500.0)
with col2:
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=100.0)

# Calculate Temp_Diff (Mandatory: Model expects 6 features)
temp_diff = proc_temp - air_temp

# 4. Prediction Logic (The Fix)
if st.button("Run Diagnostic Analysis"):
    # We MUST use a DataFrame with the exact names from our training session
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear, temp_diff]], 
                            columns=['Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 
                                     'Tool wear min', 'Temp_Diff'])
    
    # Make the prediction
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("🚨 ALERT: Machine Failure Predicted! Maintenance required.")
    else:
        st.success("✅ NORMAL: Machine is operating within safe parameters.")