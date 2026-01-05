import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="L&T Asset Monitor", page_icon="🏗️")

# 2. Load the Model
model = joblib.load('failure_model.pkl')

st.title("🏗️ Industrial Equipment Health Monitor")
st.write("Predictive Maintenance Analysis for L&T Assets")

# 3. User Inputs (Removing Temp_Diff because your model doesn't use it)
col1, col2 = st.columns(2)
with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    proc_temp = st.number_input("Process Temperature (K)", value=310.0)
    rpm = st.number_input("Rotational Speed (RPM)", value=1500.0)
with col2:
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=100.0)

# 4. Prediction Logic (Matching the 5-column requirement)
if st.button("Run Diagnostic Analysis"):
    # We create the DataFrame with ONLY the 5 columns the model expects
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear]], 
                            columns=['Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 
                                     'Tool wear min'])
    
    # Predict using the 5-column DataFrame
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("🚨 ALERT: Machine Failure Predicted!")
    else:
        st.success("✅ NORMAL: Machine is healthy.")
