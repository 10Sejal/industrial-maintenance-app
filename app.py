import streamlit as st
import pandas as pd
import joblib

# 1. Setup the Page
st.set_page_config(page_title="L&T Asset Monitor", page_icon="🏗️")

# 2. Load the Model 
# (Make sure failure_model.pkl is in the same folder on GitHub)
model = joblib.load('failure_model.pkl')

st.title("🏗️ Industrial Equipment Health Monitor")
st.write("Predictive Maintenance System for L&T Assets")

# 3. User Inputs (Sliding bars)
col1, col2 = st.columns(2)
with col1:
    air_temp = st.number_input("Air Temperature (K)", value=300.0)
    proc_temp = st.number_input("Process Temperature (K)", value=310.0)
    rpm = st.number_input("Rotational Speed (RPM)", value=1500.0)
with col2:
    torque = st.number_input("Torque (Nm)", value=40.0)
    tool_wear = st.number_input("Tool Wear (min)", value=100.0)

# Calculate the engineered feature
temp_diff = proc_temp - air_temp

# 4. THE FIX: Create a DataFrame with EXACT names used in Colab training
if st.button("Run Diagnostic Analysis"):
    # XGBoost refuses to work if these names are missing or different
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear, temp_diff]], 
                            columns=['Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 
                                     'Tool wear min', 'Temp_Diff'])
    
    # Send the input_df to the model (NOT 'features')
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("🚨 ALERT: Machine Failure Predicted!")
    else:
        st.success("✅ NORMAL: Machine is healthy.")
