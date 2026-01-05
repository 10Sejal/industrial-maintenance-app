import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(page_title="Smart Asset Reliability Monitor", layout="wide", page_icon="🏗️")

# 2. Professional UI Styling (CSS)
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #ffffff; }
    
    /* Custom Card Style for Inputs */
    .input-card { background-color: #1f2937; padding: 20px; border-radius: 15px; border-left: 5px solid #0078d4; }
    
    /* Button Styling */
    .stButton>button { 
        background: linear-gradient(90deg, #004a99 0%, #0078d4 100%); 
        color: white; font-weight: bold; border-radius: 10px; border: none; height: 3.5em; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0px 4px 15px rgba(0, 120, 212, 0.4); }
    
    /* Headers */
    h1, h2, h3 { color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('failure_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Model file 'failure_model.pkl' not found. Please ensure it is in your repository.")

# --- SIDEBAR CHATBOT ---
with st.sidebar:
    st.title("🤖 Asset AI Assistant")
    st.markdown("---")
    user_ques = st.text_input("Ask about sensor thresholds:")
    if user_ques:
        q = user_ques.lower()
        if "torque" in q: st.success("💡 **Expert Tip:** Torque > 65Nm with high RPM is a major failure trigger.")
        elif "temp" in q: st.success("💡 **Expert Tip:** Keep Process Temp below 315K to avoid HDF failure.")
        elif "wear" in q: st.success("💡 **Expert Tip:** Replace tool bits every 200 mins for 99% safety.")
        else: st.info("I am your maintenance co-pilot. Ask me about Torque, Wear, or Temp.")

# --- MAIN DASHBOARD ---
st.title("🏗️ Industrial Asset Reliability Hub")
st.write("Leveraging XGBoost for Real-time Predictive Maintenance")

# Input Section in a Card
with st.container():
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("📡 Live Sensor Telemetry")
    c1, c2, c3 = st.columns(3)
    with c1:
        air_temp = st.number_input("Air Temp (K)", 280.0, 320.0, 300.0)
        proc_temp = st.number_input("Process Temp (K)", 280.0, 330.0, 310.0)
    with c2:
        rpm = st.number_input("Rotational Speed (RPM)", 0.0, 3000.0, 1500.0)
        torque = st.number_input("Torque (Nm)", 0.0, 100.0, 40.0)
    with c3:
        tool_wear = st.number_input("Tool Wear (min)", 0.0, 300.0, 100.0)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- DIAGNOSTIC ENGINE ---
if st.button("🚀 EXECUTE SITE-WIDE DIAGNOSTIC"):
    input_df = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear]], 
                            columns=['Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 'Tool wear min'])
    
    # Calculate Scores
    probabilities = model.predict_proba(input_df)[0]
    prob_failure = probabilities[1]
    health_score = int((1 - prob_failure) * 100)
    prediction = model.predict(input_df)

    # Visualization
    st.markdown("---")
    res_l, res_r = st.columns([2, 1])

    with res_l:
        st.subheader(f"Machine Integrity Score: {health_score}%")
        bar_color = "#00ff00" if health_score > 80 else "#ffcc00" if health_score > 40 else "#ff0000"
        st.markdown(f"""
            <div style="width: 100%; background-color: #444; border-radius: 10px;">
                <div style="width: {health_score}%; background-color: {bar_color}; height: 30px; border-radius: 10px;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.error("### 🚨 CRITICAL: High Failure Risk Detected")
            st.write("**Recommended Action:** Emergency Shutdown & Technical Inspection.")
        else:
            st.success("### ✅ NORMAL: Equipment is Healthy")
            st.write("**Recommended Action:** Continue normal duty cycle.")

    with res_r:
        st.metric("Failure Probability", f"{prob_failure:.1%}")
        st.metric("Sensor Delta", f"{proc_temp - air_temp:.1f} K")

    # --- PDF REPORT ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Maintenance Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Health Integrity: {health_score}%", ln=True)
    pdf.cell(200, 10, txt=f"Torque: {torque} Nm | Speed: {rpm} RPM", ln=True)
    pdf.cell(200, 10, txt=f"Outcome: {'Failure Risk' if prediction[0] == 1 else 'Healthy'}", ln=True)
    pdf.output("Diagnostic_Report.pdf")

    st.markdown("---")
    with open("Diagnostic_Report.pdf", "rb") as f:
        st.download_button("📂 Download Professional Report", f, "Asset_Diagnostic_Report.pdf")
