import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import base64

# 1. Page Config & Professional Theme
st.set_page_config(page_title="L&T Asset Reliability Hub", layout="wide")

# Custom CSS for L&T Brand feel (Blue/White)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #00529b; color: white; }
    </style>
    """, unsafe_allow_html=True)

model = joblib.load('failure_model.pkl')

# --- SIDEBAR CHATBOT ---
st.sidebar.title("🤖 Maintenance Assistant")
user_ques = st.sidebar.text_input("Ask me about the machine:")
if user_ques:
    if "torque" in user_ques.lower():
        st.sidebar.write("💡 Tip: High torque usually indicates mechanical resistance or motor strain.")
    elif "temp" in user_ques.lower():
        st.sidebar.write("💡 Tip: Keep Process Temp within 10K of Air Temp for optimal cooling.")
    else:
        st.sidebar.write("I'm trained on maintenance logs. Try asking about 'Torque' or 'Temperature'.")

# --- MAIN UI ---
st.title("🏗️ L&T Industrial Asset Reliability Hub")
st.markdown("### Real-time Predictive Maintenance Dashboard")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    air_temp = st.number_input("Air Temp (K)", value=300.0)
    proc_temp = st.number_input("Process Temp (K)", value=310.0)
with col2:
    rpm = st.number_input("Rotational Speed (RPM)", value=1500.0)
    torque = st.number_input("Torque (Nm)", value=40.0)
with col3:
    tool_wear = st.number_input("Tool Wear (min)", value=100.0)

# --- PREDICTION & PDF LOGIC ---
if st.button("Run Full Diagnostic"):
    # Create DataFrame for Model
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear]], 
                            columns=['Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 'Tool wear min'])
    
    # Get Probability for the Health Score
    prob = model.predict_proba(input_df)[0][1] # Probability of failure
    health_score = int((1 - prob) * 100)
    prediction = model.predict(input_df)

    # Display Results
    st.markdown("---")
    st.subheader(f"Machine Health Score: {health_score}%")
    st.progress(health_score)

    if prediction[0] == 1:
        st.error("🚨 ALERT: CRITICAL FAILURE RISK DETECTED")
        result_text = "FAILURE PREDICTED"
    else:
        st.success("✅ SYSTEM OPERATIONAL: NO IMMEDIATE RISK")
        result_text = "HEALTHY"

    # --- GENERATE PDF REPORT ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="L&T Maintenance Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Machine Status: {result_text}", ln=True)
    pdf.cell(200, 10, txt=f"Health Score: {health_score}%", ln=True)
    pdf.cell(200, 10, txt=f"Torque: {torque} Nm | RPM: {rpm}", ln=True)
    pdf.output("report.pdf")

    with open("report.pdf", "rb") as f:
        st.download_button("📩 Download Diagnostic PDF", f, "Maintenance_Report.pdf")
