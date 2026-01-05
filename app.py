import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Setup
st.set_page_config(page_title="L&T Equipment Monitor", page_icon="🏗️")

# 2. Load the Model
# This file must be in your GitHub repository
try:
    model = joblib.load('failure_model.pkl')
except:
    st.error("Error: 'failure_model.pkl' not found. Please upload it to your GitHub repository.")

# 3. User Interface
st.title("🏗️ Industrial Equipment Health Monitor")
st.write("Predictive Maintenance System for L&T Assets")
st.markdown("---")

# Create two columns for a clean layout
col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0, step=0.1)
    proc_temp = st.number_input("Process Temperature (K)", value=310.0, step=0.1)
    rpm = st.number_input("Rotational Speed (RPM)", value=1500.0, step=1.0)

with col2:
    torque = st.number_input("Torque (Nm)", value=40.0, step=0.1)
    tool_wear = st.number_input("Tool Wear (min)", value=100.0, step=1.0)

# 4. Mandatory Feature Engineering
# This must match the features used during training in Colab
temp_diff = proc_temp - air_temp

# 5. Prediction Logic (THE FIX)
if st.button("Run Diagnostic Analysis"):
    # XGBoost requires a DataFrame with exact column names to avoid ValueError
    # These names must match the cleaned names from your training code
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear, temp_diff]], 
                            columns=['Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 
                                     'Tool wear min', 'Temp_Diff'])
    
    # Perform prediction
    prediction = model.predict(input_df)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error("🚨 **HIGH RISK ALERT**: Potential Failure Detected!")
        st.write("Recommendation: Schedule immediate inspection and pause operations if vibrations increase.")
    else:
        st.success("✅ **OPERATIONAL**: Machine is Healthy.")
        st.write("Recommendation: Continue standard operation. Next scheduled check-up in 48 hours.")

# 6. Sidebar Project Info (Looks great for interviews)
st.sidebar.header("About the Project")
st.sidebar.info("""
This AI solution predicts machine failure with high recall. 
**Key Benefits for L&T:**
- Reduces unplanned downtime.
- Optimizes maintenance scheduling.
- Increases site safety.
""")