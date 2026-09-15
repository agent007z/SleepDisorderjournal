import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import pickle
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# SHAP Plot Rendering Function for Streamlit
# ---------------------------------------------------------
def st_shap(plot, height=None):
    shap_html = f"""
    <head>
        {shap.getjs()}
        <style>
            text.tick-text {{
                transform: rotate(45deg);
                transform-origin: left bottom;
                font-size: 12px !important;
                fill: #334155 !important;
                font-family: Arial, sans-serif !important;
            }}
        </style>
    </head>
    <body>
        <div style="padding: 20px;">
            {plot.html()}
        </div>
    </body>
    """
    components.html(shap_html, height=height, scrolling=True)

# ---------------------------------------------------------
# Load Saved Model & Setup Encoders (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    with open('sleep_model.pkl', 'rb') as file:
        saved_data = pickle.load(file)
        
    model = saved_data['model']
    scaler = saved_data['scaler']
    
    df = pd.read_excel('sleep_disorder_dataset.xlsx')
    df.drop_duplicates(inplace=True)
    
    df['Department'] = df['Department'].str.strip().str.lower()
    df['Daily Steps'] = df['Daily Steps'].astype(str).str.strip().str.lower()
    
    dept_mapping = {
        'cse': 'cse', 'cse ': 'cse', 'cse department': 'cse', 'ece': 'ece', 'eee': 'eee',
        'mathematics': 'mathematics', 'mathematics department': 'mathematics',
        'aie': 'agricultural and industrial engineering',
        'agricultural and industrial engineering': 'agricultural and industrial engineering',
        'civil engineering': 'civil engineering', 'civil engineering ': 'civil engineering',
        'statistics': 'statistics', 'statistic': 'statistics',
        'economics': 'economics', 'ecpnomics': 'economics',
        'software engineering': 'software engineering', 'swe': 'software engineering',
        'bba professional': 'bba', 'bba': 'bba',
        'english': 'english', 'department of english': 'english',
        'agriculture': 'agriculture'
    }
    df['Department'] = df['Department'].replace(dept_mapping)
    
    le_dept = LabelEncoder()
    df['Department'] = le_dept.fit_transform(df['Department'])
    
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    le_uni = LabelEncoder()
    df['University'] = le_uni.fit_transform(df['University'])
    
    le_target = LabelEncoder()
    df['Sleep Disorder'] = le_target.fit_transform(df['Sleep Disorder'])
    
    quality_sleep_map = {'Very Poor': 0, 'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
    sleep_duration_map = {
        'Less than 4 hours':1, '4-5 hours':2, '4–5 hours':2, '5-6 hours':3, '5–6 hours':3,
        '6-7 hours':4, '6–7 hours':4, '7-8 hours':5, '7–8 hours':5, '8 hours or more':6
    }
    stress_level_map = {'Very High': 1, 'High': 2, 'Moderate': 3, 'Low': 4, 'Very Low':5}
    bmi_category_map = {
        'Underweight (BMI < 18.5)': 1, 'Normal (BMI 18.5-24.9)': 2,
        'Overweight (BMI 25-29.9)': 3, 'Obese (BMI ≥ 30)': 4
    }
    daily_steps_map = {'less than 4,999': 1, '5,000–7,999': 2, '8,000–11,999': 3, '12,000–15,999': 4, 'more than 16,000': 5}
    academic_level_map = {'Level-1': 1, 'Level-2': 2, 'Level-3': 3, 'Level-4': 4}

    encoders = {
        'dept': le_dept, 'gender': le_gender, 'uni': le_uni, 'target': le_target,
        'sleep_dur': sleep_duration_map, 'sleep_qual': quality_sleep_map,
        'stress': stress_level_map, 'bmi': bmi_category_map,
        'steps': daily_steps_map, 'academic': academic_level_map
    }
    
    df['Quality of Sleep'] = df['Quality of Sleep'].map(quality_sleep_map)
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_duration_map)
    df['Stress Level'] = df['Stress Level'].map(stress_level_map)
    df['BMI Category'] = df['BMI Category'].map(bmi_category_map)
    df['Academic Level'] = df['Academic Level'].map(academic_level_map)
    
    def normalize_steps(x):
        if pd.isna(x): return x
        x = str(x).lower()
        if 'less than 4,999' in x or '< 3 miles' in x: return 'less than 4,999'
        elif '5,000' in x or '6,000' in x or '7,999' in x or '5,000–7,999' in x: return '5,000–7,999'
        elif '8,000' in x or '9,000' in x or '10,000' in x or '11,999' in x: return '8,000–11,999'
        elif '12,000' in x or '13,000' in x or '14,000' in x or '15,999' in x: return '12,000–15,999'
        elif '16,000' in x or 'more than 16,000' in x: return 'more than 16,000'
        else: return 'unknown'
        
    df['Daily Steps'] = df['Daily Steps'].apply(normalize_steps).map(daily_steps_map)
    
    if 'Blood Pressure' in df.columns:
        df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
        df.drop(columns=['Blood Pressure'], inplace=True)

    feature_cols = ['Department', 'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep', 
                    'Physical Activity Level', 'Stress Level', 'BMI Category', 
                    'Heart Rate (bpm)', 'Daily Steps', 'Academic Level', 'University', 
                    'Systolic BP', 'Diastolic BP']
    
    X_raw = df[feature_cols]
    
    return model, scaler, encoders, feature_cols, X_raw

# ---------------------------------------------------------
# App Layout & Execution
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced Sleep Disorder Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Distinct UI & Modern Look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%);
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        color: white;
    }
    .main-header h1 {
        margin-bottom: 10px;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: white;
    }
    .main-header p {
        font-size: 18px;
        margin: 0;
        color: #cbd5e1;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 25px;
        color: #1e293b;
        border-bottom: 3px solid #e2e8f0;
        padding-bottom: 10px;
        display: inline-block;
    }
    
    /* Distinct Modern Status Cards */
    .status-card {
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 35px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .card-green { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); color: #065f46; border-top: 6px solid #10b981; }
    .card-yellow { background: linear-gradient(135deg, #fefce8 0%, #fef08a 100%); color: #854d0e; border-top: 6px solid #eab308; }
    .card-red { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); color: #991b1b; border-top: 6px solid #ef4444; }
    
    .status-card h2 { margin: 0 0 10px 0; font-size: 32px; font-weight: 900; }
    .status-card p { margin: 0; font-size: 18px; opacity: 0.85; font-weight: 500; }
    
    [data-testid="stButton"] button {
        width: 100% !important;
        height: 55px !important;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(239, 68, 68, 0.4) !important;
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
    }
    [data-testid="stButton"] button p {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: white !important;
        line-height: 1.2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>Advanced Sleep Disorder Dashboard</h1>
    <p>Provide your details below to analyze your sleep health using Artificial Intelligence.</p>
</div>
""", unsafe_allow_html=True)

# Load resources
with st.spinner("Loading AI Engine..."):
    model, scaler, enc, feature_cols, X_raw = load_resources()

# ---------------------------------------------------------
# Input Form Section
# ---------------------------------------------------------
st.markdown("<div class='section-title'>📋 1. Enter Your Details</div>", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4 style='color:#3b82f6;'>👤 Identity</h4>", unsafe_allow_html=True)
        in_age = st.number_input("Age", min_value=15, max_value=100, value=22)
        in_gender = st.selectbox("Gender", enc['gender'].classes_)
        in_dept = st.selectbox("Department", enc['dept'].classes_)
        in_academic = st.selectbox("Academic Level", list(enc['academic'].keys()))
        in_uni = st.selectbox("University", enc['uni'].classes_)

    with col2:
        st.markdown("<h4 style='color:#10b981;'>🏃‍♂️ Lifestyle</h4>", unsafe_allow_html=True)
        in_sleep_dur = st.selectbox("Sleep Duration", list(enc['sleep_dur'].keys()))
        in_sleep_qual = st.selectbox("Quality of Sleep", list(enc['sleep_qual'].keys()))
        in_pa = st.slider("Physical Activity (0-100)", 0, 100, 50)
        in_stress = st.selectbox("Stress Level", list(enc['stress'].keys()))

    with col3:
        st.markdown("<h4 style='color:#ef4444;'>❤️ Health</h4>", unsafe_allow_html=True)
        in_bmi = st.selectbox("BMI Category", list(enc['bmi'].keys()))
        in_hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
        in_steps = st.selectbox("Daily Steps", list(enc['steps'].keys()))
        in_sys_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        in_dia_bp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)

# Process Input
user_data = {
    'Department': enc['dept'].transform([in_dept])[0],
    'Gender': enc['gender'].transform([in_gender])[0],
    'Age': in_age,
    'Sleep Duration': enc['sleep_dur'][in_sleep_dur],
    'Quality of Sleep': enc['sleep_qual'][in_sleep_qual],
    'Physical Activity Level': in_pa,
    'Stress Level': enc['stress'][in_stress],
    'BMI Category': enc['bmi'][in_bmi],
    'Heart Rate (bpm)': in_hr,
    'Daily Steps': enc['steps'][in_steps],
    'Academic Level': enc['academic'][in_academic],
    'University': enc['uni'].transform([in_uni])[0],
    'Systolic BP': in_sys_bp,
    'Diastolic BP': in_dia_bp
}

user_df = pd.DataFrame([user_data])
user_scaled = scaler.transform(user_df)

st.write("") 
st.write("") 

# Analyze Button
analyze_clicked = st.button("🔍 Analyze Now", use_container_width=True)

if analyze_clicked:
    # Get Prediction
    pred_idx = model.predict(user_scaled)[0]
    pred_class = enc['target'].inverse_transform([pred_idx])[0]
    probs = model.predict_proba(user_scaled)[0]
    
    st.markdown("---")
    
    # ---------------------------------------------------------
    # Distinct Status Card Banner
    # ---------------------------------------------------------
    if pred_class == "No Sleep Disorder":
        card_class = "card-green"
        emoji = "✨"
    elif pred_class == "Insomnia":
        card_class = "card-yellow"
        emoji = "⚠️"
    else:
        card_class = "card-red"
        emoji = "🚨"
        
    st.markdown(f"""
    <div class="status-card {card_class}">
        <h2>{emoji} AI Diagnosis Status: {pred_class}</h2>
        <p>Result generated accurately from your clinical and lifestyle indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Recommendations Box
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>💡 Personalized Recommendations</div>", unsafe_allow_html=True)
    
    if pred_class == "Insomnia":
        html_rec = """
        <div style="background-color: #fefce8; border: 1px solid #fde047; border-left: 8px solid #eab308; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <h3 style="color: #a16207; font-size: 26px; margin-top: 0; font-family: Arial, sans-serif;">⚠️ We noticed signs of Insomnia</h3>
            <p style="font-size: 20px; color: #475569; margin-bottom: 20px;">Here is how you can improve your sleep:</p>
            <ul style="font-size: 20px; color: #1e293b; line-height: 1.8; list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 12px;"><span style="color:#eab308; font-size: 24px;">✦</span> <b>Consistency:</b> Try to go to bed and wake up at the exact same time every day.</li>
                <li style="margin-bottom: 12px;"><span style="color:#eab308; font-size: 24px;">✦</span> <b>Environment:</b> Keep your bedroom dark, quiet, and cool. Avoid digital screens 1 hour before bed.</li>
                <li style="margin-bottom: 12px;"><span style="color:#eab308; font-size: 24px;">✦</span> <b>Relaxation:</b> Practice meditation or deep breathing exercises to reduce stress levels.</li>
            </ul>
        </div>
        """
        st.markdown(html_rec, unsafe_allow_html=True)
        
    elif pred_class == "Sleep Apnea":
        html_rec = """
        <div style="background-color: #fef2f2; border: 1px solid #fca5a5; border-left: 8px solid #ef4444; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <h3 style="color: #991b1b; font-size: 26px; margin-top: 0; font-family: Arial, sans-serif;">🚨 We noticed signs of Sleep Apnea</h3>
            <p style="font-size: 20px; color: #475569; margin-bottom: 20px;">Please consider these crucial steps:</p>
            <ul style="font-size: 20px; color: #1e293b; line-height: 1.8; list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 12px;"><span style="color:#ef4444; font-size: 24px;">✦</span> <b>Medical Consultation:</b> Please consult a doctor or a pulmonologist for a formal diagnosis.</li>
                <li style="margin-bottom: 12px;"><span style="color:#ef4444; font-size: 24px;">✦</span> <b>Weight Management:</b> Maintaining a healthy BMI is crucial to reducing airway pressure.</li>
                <li style="margin-bottom: 12px;"><span style="color:#ef4444; font-size: 24px;">✦</span> <b>Sleep Position:</b> Try sleeping on your side instead of your back to keep your airways open.</li>
            </ul>
        </div>
        """
        st.markdown(html_rec, unsafe_allow_html=True)
        
    else:
        html_rec = """
        <div style="background-color: #f0fdf4; border: 1px solid #86efac; border-left: 8px solid #10b981; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <h3 style="color: #065f46; font-size: 26px; margin-top: 0; font-family: Arial, sans-serif;">✅ Great job! Your sleep profile looks healthy</h3>
            <p style="font-size: 20px; color: #475569; margin-bottom: 20px;">Here is how you can maintain it:</p>
            <ul style="font-size: 20px; color: #1e293b; line-height: 1.8; list-style-type: none; padding-left: 0;">
                <li style="margin-bottom: 12px;"><span style="color:#10b981; font-size: 24px;">✦</span> <b>Consistency:</b> Try to go to bed and wake up at the same time every day, even on weekends.</li>
                <li style="margin-bottom: 12px;"><span style="color:#10b981; font-size: 24px;">✦</span> <b>Active Lifestyle:</b> Keep up with your daily physical activities and step count.</li>
                <li style="margin-bottom: 12px;"><span style="color:#10b981; font-size: 24px;">✦</span> <b>Stress Management:</b> Continue managing your stress levels through hobbies or relaxation.</li>
            </ul>
        </div>
        """
        st.markdown(html_rec, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Charts Layout
    # ---------------------------------------------------------
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<h3 style='text-align:center; color:#1e293b;'>Confidence Level</h3>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({'Condition': enc['target'].classes_, 'Probability': probs * 100})
        
        colors = ['#cbd5e1'] * len(prob_df)
        if pred_class == "No Sleep Disorder":
            colors[pred_idx] = '#10b981'
        elif pred_class == "Insomnia":
            colors[pred_idx] = '#f59e0b'
        else:
            colors[pred_idx] = '#ef4444'

        fig_bar = go.Figure(data=[go.Bar(
            x=prob_df['Condition'],
            y=prob_df['Probability'],
            marker_color=colors,
            marker_line_width=0,
            width=0.5, 
            text=[f"<b>{val:.1f}%</b>" for val in prob_df['Probability']],
            textposition='outside',
            textfont=dict(size=16, color='#1e293b', family='Arial, sans-serif')
        )])
        
        fig_bar.update_layout(
            height=450, 
            margin=dict(l=20, r=20, t=60, b=40),
            yaxis=dict(range=[0, 115], showgrid=True, gridcolor='#f1f5f9', visible=False),
            xaxis=dict(
                tickfont=dict(size=14, color='#475569', family='Arial, sans-serif'), 
                showline=True, 
                linewidth=2, 
                linecolor='#cbd5e1'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_chart2:
        st.markdown("<h3 style='text-align:center; color:#1e293b;'>Lifestyle Radar</h3>", unsafe_allow_html=True)
        
        radar_display_names = ['Sleep Duration', 'Sleep Quality', 'Physical Activity', 'Stress Level', 'Heart Rate']
        radar_features = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Heart Rate (bpm)']
        
        radar_vals = []
        for feat in radar_features:
            val = user_data[feat]
            f_min = X_raw[feat].min()
            f_max = X_raw[feat].max()
            norm_val = (val - f_min) / (f_max - f_min) if f_max > f_min else 0.5 
            radar_vals.append(max(0, min(1, norm_val))) 
            
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]], 
            theta=radar_display_names + [radar_display_names[0]],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.25)',
            line=dict(color='#3b82f6', width=3),
            marker=dict(color='#1e40af', size=8, symbol='circle', line=dict(color='white', width=2)), 
            hoverinfo='text',
            text=[f"{val:.2f} (Norm)" for val in radar_vals + [radar_vals[0]]]
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False,
                    showline=False,
                    gridcolor='#e2e8f0', 
                    gridwidth=1.5
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, color='#334155', family='Arial, sans-serif'), 
                    linecolor='#e2e8f0',
                    gridcolor='#e2e8f0',
                    gridwidth=1.5
                ),
                bgcolor='#f8fafc' 
            ),
            showlegend=False,
            height=450,
            margin=dict(l=80, r=80, t=60, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # SHAP Interpretation
    # ---------------------------------------------------------
    st.markdown("<div class='section-title'>🔬 AI Decision Logic (SHAP Interpretation)</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:16px;'>This interactive plot shows how each of your inputs pushed the AI's decision higher (red) or lower (blue).</p>", unsafe_allow_html=True)
    
    with st.spinner("Generating AI Logic..."):
        explainer = shap.KernelExplainer(model.predict_proba, np.zeros((1, user_scaled.shape[1])))
        shap_values = explainer.shap_values(user_scaled)
        
        if isinstance(shap_values, list):
            exp_val = explainer.expected_value[pred_idx]
            s_val = shap_values[pred_idx][0]
        else:
            if len(shap_values.shape) == 3:
                exp_val = explainer.expected_value[pred_idx]
                s_val = shap_values[0, :, pred_idx]
            else:
                exp_val = explainer.expected_value[pred_idx] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                s_val = shap_values[0]
        
        p = shap.force_plot(
            exp_val, 
            s_val, 
            np.round(user_df.iloc[0], 2),
            feature_names=feature_cols
        )
        
        st_shap(p, height=250)