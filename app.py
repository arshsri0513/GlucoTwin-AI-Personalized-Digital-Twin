import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import io

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# Page config
st.set_page_config(page_title="GlucoTwin AI", layout="wide", page_icon="🧬")

@st.cache_resource
def load_model():
    try:
        return joblib.load('models/rf_glucose_model.pkl')
    except Exception:
        return None

def generate_personal_history(hba1c, current_glucose):
    now = datetime.now()
    times = [now - timedelta(minutes=15*(96-i)) for i in range(96)]
    df = pd.DataFrame({'timestamp': times})
    df['hour'] = df['timestamp'].dt.hour
    
    base_gl = 100 + (hba1c - 6.0) * 15
    values = np.zeros(96)
    values[0] = base_gl
    
    for i in range(1, 96):
        time_factor = 2 * np.sin(np.pi * (df.loc[i, 'hour'] - 8) / 12)
        values[i] = values[i-1] + np.random.normal(0, 4) + time_factor
        
    shift = current_glucose - values[-1]
    values = values + shift
    values = np.clip(values, 60, 350)
    
    df['glucose_reading'] = values
    df['meal_carbs'] = 0.0
    df['activity_intensity'] = 0.0
    
    meal_idx = 96 - 16
    if meal_idx > 0: df.loc[meal_idx, 'meal_carbs'] = 45.0
    act_idx = 96 - 8
    if act_idx > 0: df.loc[act_idx, 'activity_intensity'] = 0.5
        
    return df

def prepare_simulation_features(df_current, simulated_carbs, simulated_activity):
    row = df_current.iloc[-1].copy()
    future_time = row['timestamp'] + timedelta(minutes=15)
    
    features = {
        'glucose_reading': row['glucose_reading'],
        'hour': future_time.hour,
        'meal_carbs': simulated_carbs,
        'activity_intensity': simulated_activity,
        'glucose_lag_1': row['glucose_reading'],
        'glucose_lag_2': df_current.iloc[-2]['glucose_reading'],
        'glucose_lag_3': df_current.iloc[-3]['glucose_reading'],
        'glucose_lag_4': df_current.iloc[-4]['glucose_reading'],
        'meal_lag_1': row['meal_carbs'],
        'meal_lag_2': df_current.iloc[-2]['meal_carbs'],
        'activity_lag_1': row['activity_intensity'],
        'activity_lag_2': df_current.iloc[-2]['activity_intensity'],
        'glucose_roc': row['glucose_reading'] - df_current.iloc[-2]['glucose_reading']
    }
    
    hist_gl = df_current['glucose_reading'].tail(3).tolist()
    hist_gl.append(row['glucose_reading'])
    features['glucose_rolling_mean_1h'] = sum(hist_gl) / len(hist_gl)
    return pd.DataFrame([features])

def generate_pdf_report(profile, risks, df_historical):
    if not HAS_FPDF:
        return b""
        
    # 1. Generate Static Images for the PDF
    has_images = False
    try:
        import plotly.graph_objects as go
        
        # --- CGM Chart ---
        fig_cgm = go.Figure()
        fig_cgm.add_hrect(y0=70, y1=180, line_width=0, fillcolor="#00CC96", opacity=0.1, annotation_text="Optimal Target", annotation_position="top left", annotation_font_color="#00CC96")
        fig_cgm.add_trace(go.Scatter(x=df_historical['timestamp'], y=df_historical['glucose_reading'], mode='lines', name='Glucose', line=dict(color='#00CC96', width=2), fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.15)'))
        fig_cgm.add_hline(y=180, line_dash="dash", line_color="#FF4B4B")
        fig_cgm.add_hline(y=70, line_dash="dash", line_color="#FFA500")
        fig_cgm.update_layout(title="Simulated 24-Hour CGM Trend", xaxis_title="Time", yaxis_title="Glucose (mg/dL)", template="plotly_white", height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        fig_cgm.write_image("cgm_export.png", width=800, height=300)
        
        # --- Radar Chart ---
        bmi_score = max(0, min(100, 100 - abs(profile['bmi'] - 22) * 5))
        bp_score = max(0, min(100, 100 - max(0, profile['bp_sys'] - 120)))
        chol_score = max(0, min(100, 100 - max(0, profile['chol'] - 100)))
        hba1c_score = max(0, min(100, 100 - max(0, profile['hba1c'] - 5.7) * 15))
        hr_score = max(0, min(100, 100 - abs(profile['hr'] - 60)))
        
        if profile['platelets'] < 150: plt_score = max(0, (profile['platelets'] / 150) * 100)
        elif profile['platelets'] > 450: plt_score = max(0, 100 - (profile['platelets'] - 450) * 0.5)
        else: plt_score = 100
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=[bmi_score, bp_score, chol_score, hba1c_score, hr_score, plt_score, bmi_score], 
            theta=['BMI', 'Blood Pressure', 'Cholesterol', 'HbA1c', 'Heart Rate', 'Platelets', 'BMI'], 
            fill='toself', line_color='#FF4B4B'
        ))
        fig_radar.update_layout(title="Overall Wellness Index", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, template="plotly_white", height=300, margin=dict(l=40, r=40, t=40, b=20))
        fig_radar.write_image("radar_export.png", width=400, height=300)
        
        has_images = True
    except Exception as e:
        print("Image export failed:", e)

    pdf = FPDF()
    pdf.add_page()
    W = 190
    
    # --- HEADER ---
    pdf.set_fill_color(13, 110, 253) # Nice Blue
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 20)
    pdf.cell(W, 20, " GlucoTwin AI: Comprehensive Clinical Report", new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.ln(5)
    
    # --- PATIENT INFO BLOCK ---
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(W, 8, f"  Patient Name: {profile['name']}", new_x="LMARGIN", new_y="NEXT", fill=True)
    
    pdf.set_font("helvetica", "", 10)
    pdf.cell(W, 6, f"  Age: {profile['age']} yrs    |    Gender: {profile['gender']}    |    Height: {profile['height']} cm    |    Weight: {profile['weight']} kg    |    BMI: {profile['bmi']:.1f}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(5)
    
    # --- VITALS & CHARTS ---
    y_before_vitals = pdf.get_y()
    
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(13, 110, 253)
    pdf.cell(100, 10, "Clinical Vitals & Labs", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 11)
    
    pdf.cell(100, 7, f"Blood Pressure: {profile['bp_sys']}/{profile['bp_dia']} mmHg", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 7, f"Heart Rate: {profile['hr']} BPM", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 7, f"LDL Cholesterol: {profile['chol']} mg/dL", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 7, f"Platelet Count: {profile['platelets']} k/mcL", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 7, f"HbA1c: {profile['hba1c']}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 7, f"Current Glucose: {profile['current_glucose']} mg/dL", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 7, f"Medication: {profile['medication']}", new_x="LMARGIN", new_y="NEXT")
    
    if has_images:
        pdf.image("radar_export.png", x=110, y=y_before_vitals, w=90)
    
    pdf.set_y(max(pdf.get_y(), y_before_vitals + 65))
    pdf.ln(5)
    
    # --- CGM CHART ---
    if has_images:
        pdf.image("cgm_export.png", x=10, y=pdf.get_y(), w=190)
        pdf.set_y(pdf.get_y() + 75)
        pdf.ln(5)
        
    # --- RISKS ---
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(220, 53, 69) # Red
    pdf.cell(W, 10, "Flagged Indicators", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 11)
    
    if not risks:
        pdf.set_text_color(25, 135, 84) # Green
        pdf.cell(W, 8, "All clinical indicators are within standard target ranges.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    else:
        for r in risks:
            pdf.multi_cell(W, 6, text=f"- {r}", new_x="LMARGIN", new_y="NEXT")
            
    pdf.ln(5)
    
    # --- RECOMMENDATIONS ---
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(25, 135, 84) # Green
    pdf.cell(W, 10, "AI Generated Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("helvetica", "", 11)
    
    pdf.multi_cell(W, 6, text="[Activity Plan] Aim for 30 minutes of moderate aerobic exercise daily to improve insulin sensitivity.", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(W, 6, text="[Dietary] Focus on low glycemic index foods. Reduce daily carb intake to stabilize glucose baseline.", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(W, 6, text="[Follow-up] Schedule a routine follow-up in 3 months to check HbA1c progression.", new_x="LMARGIN", new_y="NEXT")
    
    # --- FOOTER ---
    pdf.set_y(-20)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(W, 10, "Disclaimer: This report is generated by a proof-of-concept AI for demonstration purposes only.", new_x="LMARGIN", new_y="NEXT", align="C")
    
    return bytes(pdf.output())

# --- UI THEME TOGGLE ---
dark_mode = st.sidebar.toggle("🌙 Toggle Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        .stApp { background-color: #121212; color: #E0E0E0; }
        [data-testid="stSidebar"] { background-color: #1E1E1E; }
        .stMetric, .stTabs { color: #FFFFFF !important; }
        h1, h2, h3, p, span { color: #E0E0E0 !important; }
        </style>
    """, unsafe_allow_html=True)
    chart_theme = "plotly_dark"
else:
    chart_theme = "plotly_white"

# --- MAIN APP ---
st.title("🧬 GlucoTwin AI: Personalized Digital Twin")
st.markdown("A comprehensive, interactive Digital Twin incorporating full clinical vitals and AI predictions.")

model = load_model()

st.sidebar.header("📝 Clinic Intake Form")
with st.sidebar.form("patient_form"):
    st.subheader("Basic Information")
    name = st.text_input("Patient Name", value="", placeholder="e.g. Jane Doe")
    age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="e.g. 45")
    gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Other"], index=0)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=None, placeholder="e.g. 165")
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=None, placeholder="e.g. 70")
    
    st.markdown("---")
    st.subheader("Clinical Vitals & Labs")
    bp_sys = st.number_input("Blood Pressure (Systolic)", 80, 250, value=None, placeholder="e.g. 120")
    bp_dia = st.number_input("Blood Pressure (Diastolic)", 50, 150, value=None, placeholder="e.g. 80")
    heart_rate = st.number_input("Heart Rate (BPM)", 40, 150, value=None, placeholder="e.g. 72")
    cholesterol = st.number_input("LDL Cholesterol (mg/dL)", 30, 300, value=None, placeholder="e.g. 110")
    platelets = st.number_input("Platelets (1000s/mcL)", 10, 1000, value=None, placeholder="e.g. 250")
    
    st.markdown("---")
    st.subheader("Diabetes Metrics")
    hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=14.0, step=0.1, value=None, placeholder="e.g. 6.5")
    current_glucose = st.number_input("Current Glucose (mg/dL)", min_value=50, max_value=400, value=None, placeholder="e.g. 130")
    medication = st.selectbox("Current Medication", ["Select...", "None", "Metformin", "Insulin", "Glipizide", "Other"], index=0)
    
    submitted = st.form_submit_button("Generate Digital Twin & Report")

if submitted:
    # Validation check to ensure no fields are empty
    missing_fields = any(v is None for v in [age, height, weight, bp_sys, bp_dia, heart_rate, cholesterol, platelets, hba1c, current_glucose])
    if not name.strip() or gender == "Select..." or medication == "Select..." or missing_fields:
        st.sidebar.error("⚠️ Please completely fill out all form fields to generate the Digital Twin.")
    else:
        st.session_state['twin_generated'] = True
        bmi = weight / ((height / 100) ** 2)
        profile = {
            'name': name, 'age': age, 'gender': gender, 
            'height': height, 'weight': weight, 'bmi': bmi,
            'bp_sys': bp_sys, 'bp_dia': bp_dia, 'hr': heart_rate, 
            'chol': cholesterol, 'platelets': platelets,
            'hba1c': hba1c, 'current_glucose': current_glucose, 'medication': medication
        }
        st.session_state['profile'] = profile
        df_hist = generate_personal_history(hba1c, current_glucose)
        st.session_state['history_df'] = df_hist
        
        # Calculate risks centrally ONCE
        risks = []
        if profile['bmi'] > 30: risks.append("Obesity (BMI > 30) - High correlation with insulin resistance.")
        elif profile['bmi'] > 25: risks.append("Overweight (BMI > 25) - Consider weight management.")
        if profile['bp_sys'] > 130 or profile['bp_dia'] > 80:
            risks.append(f"Elevated Blood Pressure ({profile['bp_sys']}/{profile['bp_dia']}) - Monitor cardiovascular risk.")
        if profile['chol'] > 100:
            risks.append(f"Elevated LDL Cholesterol ({profile['chol']} mg/dL) - Lipid management recommended.")
        if profile['platelets'] < 150:
            risks.append(f"Low Platelets ({profile['platelets']} k/mcL) - Thrombocytopenia, increased risk of bleeding.")
        elif profile['platelets'] > 450:
            risks.append(f"High Platelets ({profile['platelets']} k/mcL) - Thrombocytosis, increased risk of clotting.")
        if profile['hba1c'] >= 6.5:
            risks.append(f"HbA1c ({profile['hba1c']}%) indicates sustained hyperglycemia.")
        
        st.session_state['risks'] = risks
        
        if HAS_FPDF:
            with st.spinner("Generating AI Report & Visuals..."):
                st.session_state['pdf_bytes'] = generate_pdf_report(profile, risks, df_hist)

if st.session_state.get('twin_generated', False):
    profile = st.session_state['profile']
    df_historical = st.session_state['history_df']
    risks = st.session_state['risks']
    
    # Top Action Bar
    col_msg, col_btn = st.columns([3, 1])
    with col_msg:
        st.success(f"✅ Digital Twin Generated for **{profile['name']}**")
    with col_btn:
        if HAS_FPDF and 'pdf_bytes' in st.session_state:
            st.download_button(
                label="📄 Download PDF Report",
                data=st.session_state['pdf_bytes'],
                file_name=f"{profile['name'].replace(' ', '_')}_DigitalTwin_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.button("PDF Generator Loading...", disabled=True)
    
    # Interactive Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview & Vitals", "📈 Glucose Tracking & AI", "🏥 Clinical Insights & Risk"])
    
    with tab1:
        st.subheader("Patient Vitals Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Age / Gender", f"{profile['age']} yrs / {profile['gender'][0]}")
        col2.metric("BMI", f"{profile['bmi']:.1f}", "Normal" if 18.5 <= profile['bmi'] <= 24.9 else "High" if profile['bmi']>24.9 else "Low", delta_color="inverse")
        col3.metric("Blood Pressure", f"{profile['bp_sys']}/{profile['bp_dia']}")
        col4.metric("Heart Rate", f"{profile['hr']} BPM")
        col5.metric("Platelets", f"{profile['platelets']} k/mcL", "Normal" if 150 <= profile['platelets'] <= 450 else "Abnormal", delta_color="off")
        
        st.markdown("---")
        st.subheader("Biometric Digital Twin State")
        
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        # Glucose Gauge
        with gauge_col1:
            st.markdown("<h4 style='text-align: center;'>🩸 Current Glucose</h4>", unsafe_allow_html=True)
            fig_gl = go.Figure(go.Indicator(
                mode="gauge+number", value=profile['current_glucose'],
                number={'suffix': " mg/dL", 'font': {'size': 26}},
                gauge={
                    'axis': {'range': [40, 350]},
                    'bar': {'color': "rgba(0,0,0,0.3)"},
                    'steps': [
                        {'range': [40, 70], 'color': "#FFA500"},  # Orange
                        {'range': [70, 140], 'color': "#00CC96"}, # Green
                        {'range': [140, 180], 'color': "#FFD700"}, # Yellow
                        {'range': [180, 350], 'color': "#FF4B4B"}  # Red
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': profile['current_glucose']}
                }
            ))
            fig_gl.update_layout(template=chart_theme, height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gl, use_container_width=True)
            st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>🟢 &lt;140 Safe | 🟡 140-180 Borderline | 🔴 &gt;180 High</p>", unsafe_allow_html=True)
            
        # HbA1c Gauge
        with gauge_col2:
            st.markdown("<h4 style='text-align: center;'>📉 HbA1c Level</h4>", unsafe_allow_html=True)
            fig_hba1c = go.Figure(go.Indicator(
                mode="gauge+number", value=profile['hba1c'],
                number={'suffix': " %", 'font': {'size': 26}},
                gauge={
                    'axis': {'range': [4.0, 14.0]},
                    'bar': {'color': "rgba(0,0,0,0.3)"},
                    'steps': [
                        {'range': [4.0, 5.7], 'color': "#00CC96"}, # Green
                        {'range': [5.7, 6.4], 'color': "#FFD700"}, # Yellow (Pre-diabetic)
                        {'range': [6.4, 14.0], 'color': "#FF4B4B"} # Red (Diabetic)
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': profile['hba1c']}
                }
            ))
            fig_hba1c.update_layout(template=chart_theme, height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_hba1c, use_container_width=True)
            st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>🟢 &lt;5.7% Normal | 🟡 5.7-6.4% Pre-diabetic | 🔴 &gt;6.4% Diabetic</p>", unsafe_allow_html=True)
            
        # Cholesterol Gauge
        with gauge_col3:
            st.markdown("<h4 style='text-align: center;'>🫀 LDL Cholesterol</h4>", unsafe_allow_html=True)
            fig_chol = go.Figure(go.Indicator(
                mode="gauge+number", value=profile['chol'],
                number={'suffix': " mg/dL", 'font': {'size': 26}},
                gauge={
                    'axis': {'range': [50, 250]},
                    'bar': {'color': "rgba(0,0,0,0.3)"},
                    'steps': [
                        {'range': [50, 100], 'color': "#00CC96"},  # Green
                        {'range': [100, 159], 'color': "#FFD700"}, # Yellow
                        {'range': [159, 250], 'color': "#FF4B4B"}  # Red
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': profile['chol']}
                }
            ))
            fig_chol.update_layout(template=chart_theme, height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_chol, use_container_width=True)
            st.markdown("<p style='text-align: center; font-size: 14px; color: gray;'>🟢 &lt;100 Optimal | 🟡 100-159 Borderline | 🔴 &gt;160 High</p>", unsafe_allow_html=True)
            
        st.markdown("---")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.info("🧬 **Metabolic Digital Twin Summary**\n\n"
                    f"This digital twin is modeled after a **{profile['age']}-year-old {profile['gender'].lower()}** "
                    f"with a BMI of **{profile['bmi']:.1f}**. "
                    f"Based on the clinical inputs (HbA1c: {profile['hba1c']}%, Glucose: {profile['current_glucose']}), "
                    "the AI has synchronized a synthetic 24-hour metabolic rhythm to project future spikes and responses to lifestyle changes.")
                    
        with info_col2:
            st.success(f"💊 **Active Medication Regimen**\n\n"
                       f"**Primary:** {profile['medication']}\n\n"
                       "The digital twin model currently assumes this medication is active in the system. "
                       "Any what-if simulations involving missed doses should be calculated in the advanced parameters.")
        
    with tab2:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.subheader("Simulated 24-Hour CGM Trend")
            fig = go.Figure()
            
            # Add shaded safe zone background
            fig.add_hrect(
                y0=70, y1=180, line_width=0, fillcolor="#00CC96", opacity=0.1,
                annotation_text="Optimal Target Range", annotation_position="top left",
                annotation_font_color="#00CC96"
            )
            
            # Main Glucose Line with Area Fill
            fig.add_trace(go.Scatter(
                x=df_historical['timestamp'], y=df_historical['glucose_reading'],
                mode='lines+markers', name='Glucose', 
                line=dict(color='#00CC96', width=3),
                marker=dict(size=4),
                fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.15)',
                hovertemplate="<b>%{x|%I:%M %p}</b><br>Glucose: %{y:.1f} mg/dL<extra></extra>"
            ))
            
            # Threshold Lines
            fig.add_hline(y=180, line_dash="dash", line_color="#FF4B4B", annotation_text="Hyperglycemia (180)", annotation_font_color="#FF4B4B")
            fig.add_hline(y=70, line_dash="dash", line_color="#FFA500", annotation_text="Hypoglycemia (70)", annotation_font_color="#FFA500")
            
            # Layout & Axes styling
            fig.update_layout(
                xaxis_title="Time (Last 24 Hours)",
                yaxis_title="Blood Glucose Concentration (mg/dL)",
                hovermode="x unified", # Adds a nice vertical crosshair on hover
                template=chart_theme, 
                height=450, 
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, griddash="dot", zeroline=False, range=[40, 350])
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("AI What-If Simulation")
            sim_carbs = st.slider("Planned Meal Carbs (g)", 0, 150, 0, 5)
            sim_activity = st.slider("Planned Activity Intensity", 0.0, 1.0, 0.0, 0.1)
            sim_water = st.radio(
                "Water Intake (Renal Filtration)", 
                ["Without Water (Dehydrated)", "Normal Hydration", "With Water (Optimal Filtration)"],
                index=1,
                help="Hydration affects blood volume and how efficiently kidneys filter excess glucose."
            )
            
            if st.button("Run 2-Hour AI Prediction"):
                if model is None:
                    st.error("Model missing. Run train_model.py first.")
                else:
                    X_sim = prepare_simulation_features(df_historical, sim_carbs, sim_activity)
                    cols = ['glucose_reading', 'hour', 'meal_carbs', 'activity_intensity',
                            'glucose_lag_1', 'glucose_lag_2', 'glucose_lag_3', 'glucose_lag_4',
                            'meal_lag_1', 'meal_lag_2', 'activity_lag_1', 'activity_lag_2',
                            'glucose_roc', 'glucose_rolling_mean_1h']
                    X_sim = X_sim[cols]
                    
                    prob = model.predict_proba(X_sim)[0][1]
                    if prob > 0.5:
                        st.error(f"⚠️ High Spike Risk ({prob:.1%})")
                    else:
                        st.success(f"✅ Stable Outlook ({prob:.1%})")
                        
                    # Hydration physiological effect on estimated peak
                    hydration_mod = 0
                    if "Without" in sim_water:
                        hydration_mod = 12.0
                        st.warning("💧 Dehydration concentrates blood sugar and limits renal filtration.")
                    elif "With Water" in sim_water:
                        hydration_mod = -10.0
                        st.info("💧 Optimal hydration aids kidney filtration, naturally flushing excess glucose.")
                        
                    est_peak = profile['current_glucose'] + (sim_carbs * 0.8) - (sim_activity * 30) + hydration_mod
                    st.metric("Estimated Peak", f"{est_peak:.1f} mg/dL", f"{est_peak - profile['current_glucose']:.1f} mg/dL", delta_color="inverse")

    with tab3:
        st.subheader("Automated Clinical Risk Report")
        col_risk_left, col_risk_right = st.columns([1, 1])
        
        with col_risk_left:
            st.markdown("#### Overall Wellness Index")
            bmi_score = max(0, min(100, 100 - abs(profile['bmi'] - 22) * 5))
            bp_score = max(0, min(100, 100 - max(0, profile['bp_sys'] - 120)))
            chol_score = max(0, min(100, 100 - max(0, profile['chol'] - 100)))
            hba1c_score = max(0, min(100, 100 - max(0, profile['hba1c'] - 5.7) * 15))
            hr_score = max(0, min(100, 100 - abs(profile['hr'] - 60)))
            
            if profile['platelets'] < 150: plt_score = max(0, (profile['platelets'] / 150) * 100)
            elif profile['platelets'] > 450: plt_score = max(0, 100 - (profile['platelets'] - 450) * 0.5)
            else: plt_score = 100
            
            fig_radar = go.Figure(data=go.Scatterpolar(
              r=[bmi_score, bp_score, chol_score, hba1c_score, hr_score, plt_score, bmi_score],
              theta=['BMI', 'Blood Pressure', 'Cholesterol', 'HbA1c', 'Heart Rate', 'Platelets', 'BMI'],
              fill='toself', line_color='#FF4B4B'
            ))
            fig_radar.update_layout(
              polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
              showlegend=False, margin=dict(l=40, r=40, t=20, b=20), template=chart_theme
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_risk_right:
            st.markdown("#### Flagged Clinical Indicators")
            if risks:
                for r in risks: st.warning(f"🔸 {r}")
            else:
                st.success("All clinical indicators are within standard target ranges!")
            st.markdown("---")
            st.markdown(f"**Current Medication Plan:** `{profile['medication']}`")
            
        st.markdown("---")
        st.markdown("#### Actionable AI Recommendations")
        recs = st.columns(3)
        with recs[0]: st.info("🏃 **Activity Plan**\n\nAim for 30 minutes of moderate aerobic exercise daily to improve insulin sensitivity and lower cardiovascular risk.")
        with recs[1]: st.success("🥗 **Dietary Adjustments**\n\nFocus on low glycemic index foods. Consider reducing daily carb intake to stabilize the evening glucose baseline.")
        with recs[2]: st.error("🩺 **Clinical Follow-up**\n\nSchedule a routine follow-up in 3 months to check HbA1c progression and review the efficacy of your current medication.")
            
else:
    st.info("👈 Please enter the patient's full clinic intake details in the sidebar to generate the comprehensive report.")
