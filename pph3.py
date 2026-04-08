import streamlit as st
import pandas as pd
import sqlite3
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import calendar

st.set_page_config(page_title="PPH Early Risk Prediction System", layout="wide")

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

/* ── FIX 1: HIDE STREAMLIT DEPLOY BUTTON, HEADER MENU, FOOTER ── */
.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

.stApp {
    background: linear-gradient(135deg, #f8f4ff 0%, #fce4f3 50%, #e8f4fd 100%);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #3b0764 0%, #6a0dad 40%, #9333ea 100%) !important;
    min-width: 240px !important; max-width: 240px !important; padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
.sidebar-logo {
    padding: 28px 20px 20px; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 8px;
}
.sidebar-logo .logo-icon { font-size: 36px; }
.sidebar-logo .logo-title { font-family: 'DM Serif Display', serif; font-size: 17px; color: #fff; line-height: 1.3; margin-top: 6px; }
.sidebar-logo .logo-sub { font-size: 11px; color: rgba(255,255,255,0.55); margin-top: 2px; letter-spacing: 0.04em; text-transform: uppercase; }
.nav-section-label { font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: rgba(255,255,255,0.4); padding: 14px 20px 4px; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%; text-align: left !important; background: transparent !important;
    border: none !important; border-radius: 10px !important; color: rgba(255,255,255,0.75) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; font-weight: 500 !important;
    padding: 10px 20px !important; margin: 1px 8px !important; width: calc(100% - 16px) !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.12) !important; color: #fff !important; }
.main-header { background: white; border-radius: 16px; padding: 22px 28px; margin-bottom: 24px; box-shadow: 0 2px 16px rgba(107,0,173,0.08); display: flex; align-items: center; gap: 14px; }
.main-header .page-icon { font-size: 32px; }
.main-header .page-title { font-family: 'DM Serif Display', serif; font-size: 24px; color: #3b0764; margin: 0; }
.main-header .page-sub { font-size: 13px; color: #6b7280; margin: 2px 0 0; }
.stat-card { background: white; border-radius: 12px; padding: 18px 20px; box-shadow: 0 2px 12px rgba(107,0,173,0.07); border-left: 4px solid #9333ea; }
.high-risk { background: #ffe5e5; padding:15px; border-radius:8px; font-size:20px; font-weight:bold; color:#b00020; text-align:center; }
.low-risk  { background: #e6ffed; padding:15px; border-radius:8px; font-size:20px; font-weight:bold; color:#006400; text-align:center; }
.mod-risk  { background: #fff3cd; padding:15px; border-radius:8px; font-size:20px; font-weight:bold; color:#856404; text-align:center; }
.msg-user { background: linear-gradient(135deg, #7c3aed, #a855f7); color: white; border-radius: 18px 18px 4px 18px; padding: 12px 16px; margin: 8px 0; max-width: 80%; margin-left: auto; font-size: 14.5px; line-height: 1.6; box-shadow: 0 4px 12px rgba(124,58,237,0.25); }
.msg-ai { background: white; color: #1a1a2e; border-radius: 18px 18px 18px 4px; padding: 14px 18px; margin: 8px 0; max-width: 85%; font-size: 14.5px; line-height: 1.7; border: 1.5px solid #ede9fe; box-shadow: 0 4px 16px rgba(0,0,0,0.07); }
.msg-label { font-size: 11px; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 5px; opacity: 0.6; }
.disclaimer { background: #fef9c3; border-left: 4px solid #fbbf24; border-radius: 8px; padding: 10px 14px; font-size: 12.5px; color: #78350f; margin-top: 16px; }
.stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div { border-radius: 8px !important; border-color: #e9d5ff !important; }
div[data-testid="stForm"] { background: white; border-radius: 14px; padding: 24px; box-shadow: 0 2px 16px rgba(107,0,173,0.06); }
.appt-card { background: white; border-radius: 14px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 2px 12px rgba(107,0,173,0.07); border-left: 5px solid #9333ea; }
.appt-card.pending  { border-left-color: #f59e0b; background: #fffdf5; }
.appt-card.confirmed { border-left-color: #16a34a; background: #f6fff9; }
.appt-card.cancelled { border-left-color: #9ca3af; background: #f9fafb; opacity: 0.7; }
.appt-badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; }
.badge-pending   { background:#fef3c7; color:#92400e; }
.badge-confirmed { background:#d1fae5; color:#065f46; }
.badge-cancelled { background:#f3f4f6; color:#6b7280; }
.checkup-banner { background: linear-gradient(135deg, #ede9fe, #fce7f3); border: 1.5px solid #c4b5fd; border-radius: 12px; padding: 18px 22px; margin: 20px 0; }
.checkup-date { font-family: 'DM Serif Display', serif; font-size: 22px; color: #6d28d9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TRANSLATIONS
# ─────────────────────────────────────────────────────────────
T = {
    "en": {
        "app_title": "PPH Early Risk Prediction System",
        "app_sub": "AI-powered maternal health platform",
        "welcome": "Welcome back",
        "sign_in": "Sign in to your account",
        "username": "Username", "password": "Password",
        "login": "Login", "no_account": "Don't have an account?",
        "create_account": "Create Account", "confirm_password": "Confirm Password",
        "role": "Role", "create": "Create", "back_login": "Back to Login",
        "invalid_login": "Invalid username or password",
        "passwords_no_match": "Passwords do not match",
        "account_created": "Account created! Please log in.",
        "logout": "Logout",
        "predict_nav": "PPH Risk Prediction", "advice_nav": "Clinical Advice",
        "chat_nav": "AI Health Assistant", "patients_nav": "All Patients",
        "highrisk_nav": "High Risk Patients", "appts_nav": "My Appointments",
        "doc_appts_nav": "Appointment Requests",
        "sched_nav": "Checkup Schedule",
        "predict_title": "PPH Risk Prediction",
        "predict_sub": "Fill in clinical details to predict postpartum haemorrhage risk",
        "patient_name": "Patient Name *", "age": "Age", "bmi": "BMI",
        "hemoglobin": "Hemoglobin Level (g/dL)", "gest_age": "Gestational Age (weeks)",
        "parity": "Parity", "anemia": "Anemia", "prev_pph": "Previous PPH",
        "multiple_preg": "Multiple Pregnancy", "hypertension": "Hypertension",
        "placenta_previa": "Placenta Previa", "gest_diabetes": "Gestational Diabetes",
        "additional_factors": "#### Additional Risk Factors",
        "prev_csection": "Previous C-Section", "blood_disorder": "Blood Disorder",
        "abnormal_placenta": "Abnormal Placenta", "polyhydramnios": "Polyhydramnios",
        "hellp": "HELLP Syndrome", "severe_preeclampsia": "Severe Preeclampsia",
        "surgery": "Surgery", "myoma": "Myoma", "yes": "Yes", "no": "No",
        "predict_btn": "Predict PPH Risk",
        "enter_name": "Please enter the patient name.",
        "low_risk": "LOW RISK", "mod_risk": "MODERATE RISK", "high_risk": "HIGH RISK",
        "feature_contributions": "Feature Contributions",
        "contribution_pct": "Contribution (%)", "risk_factor_contributions": "Risk Factor Contributions",
        "advice_title": "Clinical Advice",
        "advice_sub": "Personalised guidance based on your risk factors",
        "no_high_risk": "No high-risk features detected. Continue routine prenatal care.",
        "run_predict_first": "Run a PPH Risk Prediction first — your personalised advice will appear here.",
        "chat_title": "AI Health Assistant",
        "chat_sub": "Ask anything about PPH, pregnancy, nutrition, or postpartum care",
        "chat_greeting": "Hello! I'm your Pregnancy & PPH Health Assistant.<br><br>Ask me anything about PPH, pregnancy nutrition, symptoms, or postpartum care.",
        "you": "You", "assistant": "Assistant",
        "chat_placeholder": "Type your question here...",
        "clear_chat": "Clear Chat",
        "disclaimer": "Medical Disclaimer: This assistant provides general health information only. Always consult your doctor for personalised guidance. In emergencies, call your hospital immediately.",
        "doctor_sub": "Doctor Dashboard",
        "all_patients_title": "All Patients",
        "all_patients_sub": "Latest records — newest first.",
        "total_patients": "Total Patients", "high_risk_label": "High Risk",
        "mod_risk_label": "Moderate Risk", "low_risk_label": "Low Risk",
        "cards_view": "Cards", "table_view": "Table",
        "newest_first": "Newest First", "risk_high_low": "Risk (High to Low)", "name_az": "Name A-Z",
        "clear_all": "Clear All",
        "confirm_delete_all": "This will permanently delete all patient records. Are you sure?",
        "yes_delete": "Yes, Delete All", "cancel": "Cancel",
        "all_cleared": "All patient records cleared.",
        "download_csv": "Download CSV", "remove": "Remove ",
        "removed": "Patient removed.", "no_patients": "No patients found.",
        "no_highrisk": "No high or moderate risk patients at this time.",
        "highrisk_title": "High & Moderate Risk Patients",
        "highrisk_sub": "Patients requiring priority clinical attention",
        "clinical_risk_factors": "Clinical Risk Factors", "risk_prob": "Risk probability",
        "immediate_attention": "Immediate attention recommended — HIGH PPH risk.",
        "anaemia_alert": "Hemoglobin {} g/dL — below safe threshold (11 g/dL). Anaemia likely.",
        "bmi_alert": "BMI {} — outside normal range (18.5-30). Nutritional review advised.",
        "sort_by": "Sort by", "view": "View",
        "age_unit": "yrs", "weeks_unit": "wks",
        "main_menu": "Main Menu", "account": "Account", "reports": "Reports",
        "checkup_recommended": "Check-up Recommended",
        "checkup_desc": "Based on your risk level, a check-up is recommended within",
        "checkup_days_high": "7 days",
        "checkup_days_mod": "14 days",
        "checkup_days_low": "30 days",
        "checkup_suggested": "Suggested check-up date:",
        "request_appt_btn": "Request Appointment",
        "appt_note_label": "Add a note for your doctor (optional)",
        "appt_requested": "Appointment request sent to your doctor!",
        "appt_already": "You already have a pending appointment request.",
        "my_appts_title": "My Appointments",
        "my_appts_sub": "View and track your appointment requests",
        "no_appts": "No appointments yet. Run a prediction to request one.",
        "appt_status_pending": "PENDING",
        "appt_status_confirmed": "CONFIRMED",
        "appt_status_cancelled": "CANCELLED",
        "appt_requested_on": "Requested on",
        "appt_confirmed_time": "Confirmed appointment",
        "appt_doctor_note": "Doctor's note",
        "doc_appts_title": "Appointment Requests",
        "doc_appts_sub": "Review and confirm patient appointment requests",
        "no_appt_requests": "No pending appointment requests.",
        "confirm_appt_btn": "Confirm Appointment",
        "appt_datetime_label": "Set appointment date & time",
        "appt_doc_note_label": "Add a note for the patient (optional)",
        "appt_confirmed_success": "Appointment confirmed!",
        "cancel_appt_btn": "Cancel Request",
        "appt_cancelled_msg": "Appointment request cancelled.",
        "risk_level_label": "Risk Level",
        "patient_note": "Patient's note",
        "all_appts": "All", "pending_appts": "Pending", "confirmed_appts": "Confirmed",
        "sched_title": "Monthly Checkup Schedule",
        "sched_sub": "Your complete prenatal visit plan based on gestational age",
        "sched_no_data": "Run a PPH Risk Prediction first to see your personalised checkup schedule.",
        "sched_progress": "Pregnancy Progress",
        "sched_current_week": "You are currently at",
        "sched_visit_done": "Completed",
        "sched_visit_current": "Due Now",
        "sched_visit_upcoming": "Upcoming",
        "sched_next_visit": "Next Recommended Visit",
        "sched_overview": "Schedule Overview",
    },
    "ta": {
        "app_title": "PPH ஆரம்பகால ஆபத்து கணிப்பு அமைப்பு",
        "app_sub": "AI-இயக்கப்படும் தாய்மை சுகாதார தளம்",
        "welcome": "மீண்டும் வரவேற்கிறோம்",
        "sign_in": "உங்கள் கணக்கில் உள்நுழையவும்",
        "username": "பயனர்பெயர்", "password": "கடவுச்சொல்",
        "login": "உள்நுழை", "no_account": "கணக்கு இல்லையா?",
        "create_account": "கணக்கு உருவாக்கு", "confirm_password": "கடவுச்சொல்லை உறுதிப்படுத்தவும்",
        "role": "பங்கு", "create": "உருவாக்கு", "back_login": "உள்நுழைவுக்கு திரும்பு",
        "invalid_login": "தவறான பயனர்பெயர் அல்லது கடவுச்சொல்",
        "passwords_no_match": "கடவுச்சொற்கள் பொருந்தவில்லை",
        "account_created": "கணக்கு உருவாக்கப்பட்டது! தயவுசெய்து உள்நுழையவும்.",
        "logout": "வெளியேறு",
        "predict_nav": "PPH ஆபத்து கணிப்பு", "advice_nav": "மருத்துவ ஆலோசனை",
        "chat_nav": "AI சுகாதார உதவியாளர்", "patients_nav": "அனைத்து நோயாளிகளும்",
        "highrisk_nav": "அதிக ஆபத்து நோயாளிகள்", "appts_nav": "என் சந்திப்புகள்",
        "doc_appts_nav": "சந்திப்பு கோரிக்கைகள்",
        "sched_nav": "பரிசோதனை அட்டவணை",
        "predict_title": "PPH ஆபத்து கணிப்பு",
        "predict_sub": "பிரசவத்திற்கு பிந்தைய இரத்தப்போக்கு ஆபத்தை கணிக்க மருத்துவ விவரங்களை பூர்த்தி செய்யவும்",
        "patient_name": "நோயாளி பெயர் *", "age": "வயது", "bmi": "உடல் நிறை குறியீடு",
        "hemoglobin": "ஹீமோகுளோபின் அளவு (g/dL)", "gest_age": "கர்ப்பகால வயது (வாரங்கள்)",
        "parity": "பிரசவ எண்ணிக்கை", "anemia": "இரத்த சோகை", "prev_pph": "முந்தைய PPH",
        "multiple_preg": "பல கர்ப்பம்", "hypertension": "உயர் இரத்த அழுத்தம்",
        "placenta_previa": "நஞ்சுக்கொடி முன்னிலை", "gest_diabetes": "கர்ப்பகால நீரிழிவு",
        "additional_factors": "#### கூடுதல் ஆபத்து காரணிகள்",
        "prev_csection": "முந்தைய சிசேரியன்", "blood_disorder": "இரத்தக் கோளாறு",
        "abnormal_placenta": "அசாதாரண நஞ்சுக்கொடி", "polyhydramnios": "பாலிஹைட்ராம்னியோஸ்",
        "hellp": "HELLP நோய்க்குறி", "severe_preeclampsia": "கடுமையான முன் பிரம்பியம்",
        "surgery": "அறுவை சிகிச்சை", "myoma": "மயோமா", "yes": "ஆம்", "no": "இல்லை",
        "predict_btn": "PPH ஆபத்தை கணிக்கவும்",
        "enter_name": "தயவுசெய்து நோயாளியின் பெயரை உள்ளிடவும்.",
        "low_risk": "குறைந்த ஆபத்து", "mod_risk": "மிதமான ஆபத்து", "high_risk": "அதிக ஆபத்து",
        "feature_contributions": "காரணி பங்களிப்புகள்",
        "contribution_pct": "பங்களிப்பு (%)", "risk_factor_contributions": "ஆபத்து காரணி பங்களிப்புகள்",
        "advice_title": "மருத்துவ ஆலோசனை",
        "advice_sub": "உங்கள் ஆபத்து காரணிகளின் அடிப்படையில் தனிப்பட்ட வழிகாட்டுதல்",
        "no_high_risk": "அதிக ஆபத்து அம்சங்கள் கண்டறியப்படவில்லை. வழக்கமான பிரசவத்திற்கு முந்தைய பராமரிப்பை தொடரவும்.",
        "run_predict_first": "முதலில் PPH ஆபத்து கணிப்பை இயக்கவும் — உங்கள் தனிப்பட்ட ஆலோசனை இங்கே தோன்றும்.",
        "chat_title": "AI சுகாதார உதவியாளர்",
        "chat_sub": "PPH, கர்ப்பம், ஊட்டச்சத்து அல்லது பிரசவத்திற்கு பிந்தைய பராமரிப்பு பற்றி எதையும் கேளுங்கள்",
        "chat_greeting": "வணக்கம்! நான் உங்கள் கர்ப்பகால PPH சுகாதார உதவியாளர்.<br><br>PPH, கர்ப்பகால ஊட்டச்சத்து, அறிகுறிகள் அல்லது பிரசவத்திற்கு பிந்தைய பராமரிப்பு பற்றி எதையும் கேளுங்கள்.",
        "you": "நீங்கள்", "assistant": "உதவியாளர்",
        "chat_placeholder": "உங்கள் கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "clear_chat": "அரட்டையை அழி",
        "disclaimer": "மருத்துவ மறுப்பு: இந்த உதவியாளர் பொதுவான சுகாதார தகவல்களை மட்டுமே வழங்குகிறது. தனிப்பட்ட வழிகாட்டுதலுக்கு எப்போதும் உங்கள் மருத்துவரை அணுகவும்.",
        "doctor_sub": "மருத்துவர் டாஷ்போர்டு",
        "all_patients_title": "அனைத்து நோயாளிகளும்",
        "all_patients_sub": "சமீபத்திய பதிவுகள் — புதியது முதலில்.",
        "total_patients": "மொத்த நோயாளிகள்", "high_risk_label": "அதிக ஆபத்து",
        "mod_risk_label": "மிதமான ஆபத்து", "low_risk_label": "குறைந்த ஆபத்து",
        "cards_view": "அட்டைகள்", "table_view": "அட்டவணை",
        "newest_first": "புதியது முதலில்", "risk_high_low": "ஆபத்து (அதிகம் முதல் குறைவு)", "name_az": "பெயர் அகரவரிசை",
        "clear_all": "அனைத்தையும் அழி",
        "confirm_delete_all": "இது அனைத்து நோயாளி பதிவுகளையும் நிரந்தரமாக நீக்கும். நீங்கள் உறுதியாக இருக்கிறீர்களா?",
        "yes_delete": "ஆம், அனைத்தையும் நீக்கு", "cancel": "ரத்து செய்",
        "all_cleared": "அனைத்து நோயாளி பதிவுகளும் அழிக்கப்பட்டன.",
        "download_csv": "CSV பதிவிறக்கம்", "remove": "நீக்கு ",
        "removed": "நோயாளி நீக்கப்பட்டார்.", "no_patients": "நோயாளிகள் எவரும் இல்லை.",
        "no_highrisk": "இப்போது அதிக அல்லது மிதமான ஆபத்து நோயாளிகள் யாரும் இல்லை.",
        "highrisk_title": "அதிக மற்றும் மிதமான ஆபத்து நோயாளிகள்",
        "highrisk_sub": "முன்னுரிமை மருத்துவ கவனிப்பு தேவைப்படும் நோயாளிகள்",
        "clinical_risk_factors": "மருத்துவ ஆபத்து காரணிகள்", "risk_prob": "ஆபத்து நிகழ்தகவு",
        "immediate_attention": "உடனடி கவனிப்பு பரிந்துரைக்கப்படுகிறது — அதிக PPH ஆபத்து.",
        "anaemia_alert": "ஹீமோகுளோபின் {} g/dL — பாதுகாப்பான வரம்பிற்கு கீழே (11 g/dL). இரத்த சோகை சாத்தியம்.",
        "bmi_alert": "BMI {} — இயல்பான வரம்பிற்கு வெளியே (18.5-30). ஊட்டச்சத்து மதிப்பீடு அறிவுறுத்தப்படுகிறது.",
        "sort_by": "வரிசைப்படுத்து", "view": "காட்சி",
        "age_unit": "வயது", "weeks_unit": "வாரம்",
        "main_menu": "முதன்மை பட்டி", "account": "கணக்கு", "reports": "அறிக்கைகள்",
        "checkup_recommended": "பரிசோதனை பரிந்துரைக்கப்படுகிறது",
        "checkup_desc": "உங்கள் ஆபத்து நிலையின் அடிப்படையில், பரிசோதனை பரிந்துரைக்கப்படுகிறது",
        "checkup_days_high": "7 நாட்களுக்குள்",
        "checkup_days_mod": "14 நாட்களுக்குள்",
        "checkup_days_low": "30 நாட்களுக்குள்",
        "checkup_suggested": "பரிந்துரைக்கப்பட்ட பரிசோதனை தேதி:",
        "request_appt_btn": "சந்திப்பு கோரிக்கை அனுப்பு",
        "appt_note_label": "உங்கள் மருத்துவருக்கு குறிப்பு சேர்க்கவும் (விரும்பினால்)",
        "appt_requested": "சந்திப்பு கோரிக்கை மருத்துவரிடம் அனுப்பப்பட்டது!",
        "appt_already": "ஏற்கனவே ஒரு நிலுவையில் உள்ள சந்திப்பு கோரிக்கை உள்ளது.",
        "my_appts_title": "என் சந்திப்புகள்",
        "my_appts_sub": "உங்கள் சந்திப்பு கோரிக்கைகளை காண்க மற்றும் கண்காணிக்கவும்",
        "no_appts": "இன்னும் சந்திப்புகள் இல்லை. கோரிக்கை அனுப்ப கணிப்பை இயக்கவும்.",
        "appt_status_pending": "நிலுவையில்",
        "appt_status_confirmed": "உறுதிப்படுத்தப்பட்டது",
        "appt_status_cancelled": "ரத்து செய்யப்பட்டது",
        "appt_requested_on": "கோரிக்கை தேதி",
        "appt_confirmed_time": "உறுதிப்படுத்தப்பட்ட சந்திப்பு",
        "appt_doctor_note": "மருத்துவரின் குறிப்பு",
        "doc_appts_title": "சந்திப்பு கோரிக்கைகள்",
        "doc_appts_sub": "நோயாளி சந்திப்பு கோரிக்கைகளை மதிப்பாய்வு செய்யவும்",
        "no_appt_requests": "நிலுவையில் உள்ள சந்திப்பு கோரிக்கைகள் இல்லை.",
        "confirm_appt_btn": "சந்திப்பை உறுதிப்படுத்தவும்",
        "appt_datetime_label": "சந்திப்பு தேதி மற்றும் நேரம் அமைக்கவும்",
        "appt_doc_note_label": "நோயாளிக்கு குறிப்பு சேர்க்கவும் (விரும்பினால்)",
        "appt_confirmed_success": "சந்திப்பு உறுதிப்படுத்தப்பட்டது!",
        "cancel_appt_btn": "கோரிக்கையை ரத்து செய்",
        "appt_cancelled_msg": "சந்திப்பு கோரிக்கை ரத்து செய்யப்பட்டது.",
        "risk_level_label": "ஆபத்து நிலை",
        "patient_note": "நோயாளியின் குறிப்பு",
        "all_appts": "அனைத்தும்", "pending_appts": "நிலுவையில்", "confirmed_appts": "உறுதிப்படுத்தப்பட்டது",
        "sched_title": "மாதாந்திர பரிசோதனை அட்டவணை",
        "sched_sub": "கர்ப்பகால வயதின் அடிப்படையில் உங்கள் முழு பிரசவத்திற்கு முந்தைய வருகை திட்டம்",
        "sched_no_data": "உங்கள் தனிப்பட்ட பரிசோதனை அட்டவணையை காண முதலில் PPH ஆபத்து கணிப்பை இயக்கவும்.",
        "sched_progress": "கர்ப்ப முன்னேற்றம்",
        "sched_current_week": "நீங்கள் தற்போது இருக்கும் வாரம்",
        "sched_visit_done": "முடிந்தது",
        "sched_visit_current": "இப்போது வரவேண்டும்",
        "sched_visit_upcoming": "வரவிருக்கிறது",
        "sched_next_visit": "அடுத்த பரிந்துரைக்கப்பட்ட வருகை",
        "sched_overview": "அட்டவணை கண்ணோட்டம்",
    }
}

# ─────────────────────────────────────────────────────────────
# MONTHLY SCHEDULE DATA
# ─────────────────────────────────────────────────────────────
MONTHLY_SCHEDULE = [
    {
        "month": 1, "week_range": (1, 4),
        "title_en": "Month 1 - Pregnancy Confirmed",
        "title_ta": "மாதம் 1 - கர்ப்பம் உறுதிப்படுத்தல்",
        "visits": [{
            "week_range": (4, 8),
            "label_en": "First Prenatal Visit",
            "label_ta": "முதல் பிரசவத்திற்கு முந்தைய வருகை",
            "tests_en": ["Pregnancy confirmation (urine/blood HCG)", "Blood group and Rh typing", "CBC (Complete Blood Count)", "Urine routine and culture", "HIV, HBsAg, VDRL screening", "Thyroid function test (TSH)", "Blood pressure baseline", "BMI and weight check"],
            "tests_ta": ["கர்ப்ப உறுதிப்படுத்தல் (சிறுநீர்/இரத்த HCG)", "இரத்த வகை மற்றும் Rh சோதனை", "CBC (முழுமையான இரத்த எண்ணிக்கை)", "சிறுநீர் சோதனை மற்றும் வளர்ப்பு", "HIV, HBsAg, VDRL திரையிடல்", "தைராய்டு செயல்பாடு சோதனை (TSH)", "இரத்த அழுத்த அடிப்படை", "BMI மற்றும் எடை சோதனை"],
            "icon": "🔬", "color": "#fce7f3", "border": "#f9a8d4", "text": "#9d174d",
            "tip_en": "Start folic acid 400mcg daily. Avoid alcohol and smoking. Eat iron-rich foods.",
            "tip_ta": "தினமும் ஃபோலிக் அமிலம் 400mcg எடுக்கவும். மது மற்றும் புகையை தவிர்க்கவும்.",
        }]
    },
    {
        "month": 2, "week_range": (5, 8),
        "title_en": "Month 2 - Early Development",
        "title_ta": "மாதம் 2 - ஆரம்பகால வளர்ச்சி",
        "visits": [{
            "week_range": (8, 12),
            "label_en": "8-12 Week Dating Scan",
            "label_ta": "8-12 வார தேதி ஸ்கேன்",
            "tests_en": ["Dating ultrasound (crown-rump length)", "NT scan (Nuchal Translucency)", "NIPT / Double marker test (if advised)", "Folic acid level check", "Review blood test results"],
            "tests_ta": ["தேதி அல்ட்ராசவுண்ட் (மகுட-பிட்டம் நீளம்)", "NT ஸ்கேன் (நுகால் ட்ரான்ஸ்லூசென்சி)", "NIPT / இரட்டை குறிப்பான் சோதனை (பரிந்துரைத்தால்)", "ஃபோலிக் அமில அளவு சோதனை", "இரத்த பரிசோதனை முடிவுகள் மதிப்பாய்வு"],
            "icon": "🔊", "color": "#ffe5e5", "border": "#fca5a5", "text": "#b00020",
            "tip_en": "You may experience morning sickness. Eat small frequent meals. Stay hydrated.",
            "tip_ta": "காலை நோய் ஏற்படலாம். சிறிய அடிக்கடி உணவு சாப்பிடுங்கள். நீர்மம் பருகுங்கள்.",
        }]
    },
    {
        "month": 3, "week_range": (9, 12),
        "title_en": "Month 3 - End of First Trimester",
        "title_ta": "மாதம் 3 - முதல் மூன்று மாதங்களின் முடிவு",
        "visits": [{
            "week_range": (12, 16),
            "label_en": "12-16 Week Review",
            "label_ta": "12-16 வார மதிப்பாய்வு",
            "tests_en": ["Triple/Quad marker test review", "Anomaly scan booking (20-wk)", "Blood pressure check", "Weight gain review", "Cervical length assessment (if risk)", "Iron and calcium supplementation review"],
            "tests_ta": ["ட்ரிபிள்/குவாட் குறிப்பான் சோதனை மதிப்பாய்வு", "ஆபத்து ஸ்கேன் முன்பதிவு (20 வார)", "இரத்த அழுத்த சோதனை", "எடை அதிகரிப்பு மதிப்பாய்வு", "கர்ப்பப்பை வாய் நீள மதிப்பீடு", "இரும்பு மற்றும் கால்சியம் மாத்திரை மதிப்பாய்வு"],
            "icon": "🩺", "color": "#fff3cd", "border": "#fcd34d", "text": "#856404",
            "tip_en": "First trimester nausea usually eases now. Announce your pregnancy. Begin planning birth.",
            "tip_ta": "முதல் மூன்று மாதங்களில் குமட்டல் இப்போது குறையும். பிரசவ திட்டமிடலை தொடங்கவும்.",
        }]
    },
    {
        "month": 4, "week_range": (13, 16),
        "title_en": "Month 4 - Second Trimester Begins",
        "title_ta": "மாதம் 4 - இரண்டாம் மூன்று மாதங்கள் தொடங்கும்",
        "visits": [{
            "week_range": (16, 20),
            "label_en": "16-20 Week Anomaly Check",
            "label_ta": "16-20 வார ஆபத்து சோதனை",
            "tests_en": ["Anomaly scan (20-week morphology scan)", "OGTT (glucose tolerance test) if at risk", "Hemoglobin recheck", "Blood pressure and urine protein", "Fetal movements awareness education", "Anti-D injection if Rh-negative"],
            "tests_ta": ["ஆபத்து ஸ்கேன் (20 வார உருவியல் ஸ்கேன்)", "OGTT (குளுக்கோஸ் சகிப்பு சோதனை) ஆபத்தில் இருந்தால்", "ஹீமோகுளோபின் மறு சோதனை", "இரத்த அழுத்தம் மற்றும் சிறுநீர் புரதம்", "கரு அசைவு விழிப்புணர்வு கல்வி", "Anti-D ஊசி Rh-எதிர்மறை இருந்தால்"],
            "icon": "🫃", "color": "#e6ffed", "border": "#6ee7b7", "text": "#15803d",
            "tip_en": "You may start feeling baby movements (quickening). Sleep on your left side.",
            "tip_ta": "குழந்தை அசைவுகளை உணரத் தொடங்கலாம். இடது பக்கமாக தூங்குங்கள்.",
        }]
    },
    {
        "month": 5, "week_range": (17, 20),
        "title_en": "Month 5 - Baby Movements Begin",
        "title_ta": "மாதம் 5 - குழந்தை அசைவுகள் தொடங்கும்",
        "visits": [{
            "week_range": (20, 24),
            "label_en": "20-24 Week Growth Check",
            "label_ta": "20-24 வார வளர்ச்சி சோதனை",
            "tests_en": ["Fundal height measurement", "Fetal heart rate (Doppler)", "Blood pressure monitoring", "Urine protein check", "Iron supplementation review", "Discuss birth plan preferences"],
            "tests_ta": ["கர்ப்பப்பை அடி உயரம் அளவீடு", "கரு இதயத் துடிப்பு (டாப்ளர்)", "இரத்த அழுத்த கண்காணிப்பு", "சிறுநீர் புரத சோதனை", "இரும்பு மாத்திரை மதிப்பாய்வு", "பிரசவ திட்ட விருப்பங்கள் விவாதம்"],
            "icon": "💓", "color": "#e0f2fe", "border": "#7dd3fc", "text": "#0369a1",
            "tip_en": "Track baby kicks daily. At least 10 movements in 2 hours after meals is normal.",
            "tip_ta": "தினமும் குழந்தை உதைப்புகளை கண்காணிக்கவும். உணவிற்கு பிறகு 2 மணி நேரத்தில் குறைந்தது 10 அசைவுகள் இயல்பானது.",
        }]
    },
    {
        "month": 6, "week_range": (21, 24),
        "title_en": "Month 6 - Glucose Screening",
        "title_ta": "மாதம் 6 - குளுக்கோஸ் திரையிடல்",
        "visits": [{
            "week_range": (24, 28),
            "label_en": "24-28 Week GDM Screening",
            "label_ta": "24-28 வார கர்ப்பகால நீரிழிவு திரையிடல்",
            "tests_en": ["OGTT (75g glucose tolerance test)", "CBC recheck (anaemia screen)", "Blood pressure", "Anti-D injection (Rh-negative patients)", "Whooping cough (Tdap) vaccine", "Preeclampsia risk assessment"],
            "tests_ta": ["OGTT (75g குளுக்கோஸ் சகிப்பு சோதனை)", "CBC மறு சோதனை (இரத்த சோகை திரை)", "இரத்த அழுத்தம்", "Anti-D ஊசி (Rh-எதிர்மறை நோயாளிகள்)", "Whooping cough தடுப்பூசி", "முன் பிரம்பியம் ஆபத்து மதிப்பீடு"],
            "icon": "💉", "color": "#f3e8ff", "border": "#c4b5fd", "text": "#6d28d9",
            "tip_en": "Limit sugar intake. If GDM diagnosed, follow diet plan and monitor blood glucose.",
            "tip_ta": "சர்க்கரை உட்கொள்ளலை குறைக்கவும். GDM கண்டறியப்பட்டால், உணவு திட்டத்தை பின்பற்றவும்.",
        }]
    },
    {
        "month": 7, "week_range": (25, 28),
        "title_en": "Month 7 - Third Trimester Starts",
        "title_ta": "மாதம் 7 - மூன்றாம் மூன்று மாதங்கள் தொடங்கும்",
        "visits": [{
            "week_range": (28, 32),
            "label_en": "28-32 Week Growth Scan",
            "label_ta": "28-32 வார வளர்ச்சி ஸ்கேன்",
            "tests_en": ["Growth scan (fetal biometry)", "Fetal position check", "Placenta location check", "Amniotic fluid index (AFI)", "Blood pressure and oedema check", "Iron and haemoglobin recheck"],
            "tests_ta": ["வளர்ச்சி ஸ்கேன் (கரு பயோமெட்ரி)", "கரு நிலை சோதனை", "நஞ்சுக்கொடி இடம் சோதனை", "அம்னியோடிக் திரவ குறியீடு (AFI)", "இரத்த அழுத்தம் மற்றும் வீக்கம் சோதனை", "இரும்பு மற்றும் ஹீமோகுளோபின் மறு சோதனை"],
            "icon": "🌙", "color": "#ecfdf5", "border": "#6ee7b7", "text": "#047857",
            "tip_en": "Back pain is common. Use a pregnancy pillow. Start birth plan discussion with your doctor.",
            "tip_ta": "முதுகு வலி இயல்பானது. கர்ப்ப தலையணை பயன்படுத்தவும். பிரசவ திட்டம் பற்றி மருத்துவரிடம் பேசவும்.",
        }]
    },
    {
        "month": 8, "week_range": (29, 32),
        "title_en": "Month 8 - Pre-Labour Preparation",
        "title_ta": "மாதம் 8 - பிரசவத்திற்கு முன் தயாரிப்பு",
        "visits": [{
            "week_range": (32, 36),
            "label_en": "32-36 Week Delivery Prep",
            "label_ta": "32-36 வார பிரசவ தயாரிப்பு",
            "tests_en": ["GBS (Group B Streptococcus) swab", "Repeat growth scan if needed", "Fetal presentation (head/breech)", "Birth plan finalisation", "Discuss epidural/pain relief options", "Maternity bag checklist", "Hospital registration"],
            "tests_ta": ["GBS (குழும B ஸ்ட்ரெப்டோகாக்கஸ்) துடைப்பு", "தேவைப்பட்டால் வளர்ச்சி ஸ்கேன் மீண்டும்", "கரு முன்னிலை (தலை/பிட்டம்)", "பிரசவ திட்டம் இறுதி செய்தல்", "வலி நிவாரண விருப்பங்கள் பற்றி விவாதிக்கவும்", "மகப்பேறு பை பட்டியல்", "மருத்துவமனை பதிவு"],
            "icon": "🏥", "color": "#fef9c3", "border": "#fcd34d", "text": "#713f12",
            "tip_en": "Pack your hospital bag. Install infant car seat. Attend antenatal classes if available.",
            "tip_ta": "மருத்துவமனை பை பேக் செய்யுங்கள். குழந்தை கார் இருக்கை நிறுவுங்கள். கர்ப்பகால வகுப்புகளில் கலந்துகொள்ளுங்கள்.",
        }]
    },
    {
        "month": 9, "week_range": (33, 36),
        "title_en": "Month 9 - Final Stretch",
        "title_ta": "மாதம் 9 - இறுதி நிலை",
        "visits": [{
            "week_range": (36, 38),
            "label_en": "36-38 Week Weekly Checks",
            "label_ta": "36-38 வார வாராந்திர சோதனைகள்",
            "tests_en": ["Weekly fetal heart monitoring (CTG)", "Cervical examination", "Bishop score assessment", "Blood pressure and protein", "Membrane sweep discussion", "Placenta and fluid check (BPP)"],
            "tests_ta": ["வாராந்திர கரு இதய கண்காணிப்பு (CTG)", "கர்ப்பப்பை வாய் பரிசோதனை", "Bishop score மதிப்பீடு", "இரத்த அழுத்தம் மற்றும் புரதம்", "சவ்வு துடைப்பு விவாதம்", "நஞ்சுக்கொடி மற்றும் திரவ சோதனை (BPP)"],
            "icon": "🌟", "color": "#f0fdf4", "border": "#86efac", "text": "#166534",
            "tip_en": "Lightening (baby drops lower) may occur. You may feel urge to clean/nest. Rest as much as possible.",
            "tip_ta": "குழந்தை கீழே இறங்கலாம். முடிந்தவரை ஓய்வு எடுக்கவும்.",
        }]
    },
    {
        "month": 10, "week_range": (37, 40),
        "title_en": "Month 10 - Term and Delivery",
        "title_ta": "மாதம் 10 - காலம் மற்றும் பிரசவம்",
        "visits": [{
            "week_range": (38, 40),
            "label_en": "38-40 Week Final Visits",
            "label_ta": "38-40 வார இறுதி வருகைகள்",
            "tests_en": ["CTG (cardiotocography)", "Cervical ripening assessment", "Induction planning if overdue", "PPH prevention protocol review", "Oxytocin administration plan", "Blood grouping and crossmatch"],
            "tests_ta": ["CTG (கார்டியோடோகோகிராஃபி)", "கர்ப்பப்பை வாய் பக்குவம் மதிப்பீடு", "தாமதமான பிரசவ திட்டம்", "PPH தடுப்பு நெறிமுறை மதிப்பாய்வு", "ஆக்சிடோசின் நிர்வாக திட்டம்", "இரத்த வகை மற்றும் குறுக்கு பொருத்தம்"],
            "icon": "👶", "color": "#fce7f3", "border": "#f9a8d4", "text": "#9d174d",
            "tip_en": "Watch for labour signs: regular contractions, water breaking, bloody show. Go to hospital immediately!",
            "tip_ta": "பிரசவ அறிகுறிகளை கவனிக்கவும்: தொடர் சுருக்கங்கள், தண்ணீர் உடைதல். உடனடியாக மருத்துவமனை செல்லவும்!",
        }]
    },
]

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
conn = sqlite3.connect("users.db", check_same_thread=False, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT, role TEXT)")

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, age INTEGER, bmi REAL, hemoglobin REAL,
    gestational_age INTEGER,
    parity INTEGER, anemia INTEGER, pre_pph INTEGER,
    multiple_pregnancy INTEGER, pre_hypertension INTEGER, placenta_previa INTEGER,
    gest_diabetes INTEGER, pre_c_section INTEGER, pre_blood INTEGER, abnormal_placenta INTEGER,
    polyhydromnios INTEGER, hellp_syndrome INTEGER, severe_preeclampsia INTEGER,
    surgery INTEGER, myoma INTEGER, risk_prob REAL, risk TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime'))
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS appointments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_username TEXT NOT NULL,
    patient_name TEXT,
    risk TEXT,
    risk_prob REAL,
    gestational_age INTEGER DEFAULT 0,
    suggested_date TEXT,
    patient_note TEXT,
    status TEXT DEFAULT 'pending',
    confirmed_datetime TEXT,
    doctor_note TEXT,
    requested_at TEXT DEFAULT (datetime('now','localtime')),
    updated_at TEXT
)""")

_appt_existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(appointments)").fetchall()}
_appt_needed = {
    "patient_name": "TEXT",
    "risk": "TEXT",
    "risk_prob": "REAL",
    "gestational_age": "INTEGER DEFAULT 0",
    "suggested_date": "TEXT",
    "patient_note": "TEXT",
    "confirmed_datetime": "TEXT",
    "doctor_note": "TEXT",
    "updated_at": "TEXT",
}
for _col_name, _col_type in _appt_needed.items():
    if _col_name not in _appt_existing_cols:
        try:
            cursor.execute(f"ALTER TABLE appointments ADD COLUMN {_col_name} {_col_type}")
        except Exception:
            pass

_pat_existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(patients)").fetchall()}
_pat_needed = [
    ("gestational_age","INTEGER"), ("parity","INTEGER"), ("anemia","INTEGER"),
    ("pre_pph","INTEGER"), ("multiple_pregnancy","INTEGER"), ("pre_hypertension","INTEGER"),
    ("placenta_previa","INTEGER"), ("gest_diabetes","INTEGER"), ("pre_c_section","INTEGER"),
    ("pre_blood","INTEGER"), ("abnormal_placenta","INTEGER"), ("polyhydromnios","INTEGER"),
    ("hellp_syndrome","INTEGER"), ("severe_preeclampsia","INTEGER"), ("surgery","INTEGER"),
    ("myoma","INTEGER"), ("risk_prob","REAL"), ("created_at","TEXT"),
]
for _col_name, _col_type in _pat_needed:
    if _col_name not in _pat_existing_cols:
        try:
            cursor.execute(f"ALTER TABLE patients ADD COLUMN {_col_name} {_col_type}")
        except Exception:
            pass

conn.commit()

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
model = joblib.load("pph_model.pkl")

RAW_FEATURE_NAMES = [
    'age', 'bmi', 'hemoglobin_level', 'anemia', 'pre_pph', 'parity',
    'multiple_pregnancy', 'pre_hypertension', 'placenta_previa',
    'gest_diabetes', 'pre_c_section', 'pre_blood',
    'abnormal_placenta', 'polyhydromnios', 'hellp_syndrome',
    'severe_preeclampsia', 'surgery', 'myoma', 'gestational_age'
]

try:
    ALL_MODEL_FEATURES = model.feature_name_
except AttributeError:
    try:
        ALL_MODEL_FEATURES = list(model.feature_names_in_)
    except AttributeError:
        ALL_MODEL_FEATURES = RAW_FEATURE_NAMES

feature_names = RAW_FEATURE_NAMES

def prepare_input_for_model(patient_dict):
    row = {col: 0 for col in ALL_MODEL_FEATURES}
    for key, val in patient_dict.items():
        if key in row:
            row[key] = val
    return pd.DataFrame([row], columns=ALL_MODEL_FEATURES)

advice_map_en = {
    "anemia":             "Take iron supplements, eat iron-rich foods, check hemoglobin regularly.",
    "bmi":                "Maintain healthy weight; follow prenatal nutrition and moderate exercise.",
    "pre_pph":            "Inform your obstetrician; careful monitoring during labour is needed.",
    "pre_c_section":      "Plan delivery with an experienced obstetric team; monitor for bleeding.",
    "placenta_previa":    "Avoid strenuous activity; attend follow-up ultrasounds.",
    "gest_diabetes":      "Monitor blood sugar; follow diet and exercise plan.",
    "pre_hypertension":   "Regular BP checks; take medications as prescribed; manage stress.",
    "polyhydromnios":     "Frequent prenatal checkups; monitor fetal growth; stay hydrated.",
    "severe_preeclampsia":"Monitor BP and protein levels; hospitalisation if condition worsens.",
    "multiple_pregnancy": "Extra monitoring during pregnancy; prepare for possible complications.",
    "hemoglobin_level":   "Maintain healthy iron and vitamin levels; monitor blood counts.",
    "surgery":            "Inform doctor about previous surgeries; bleeding risk may increase.",
    "myoma":              "Monitor fibroid growth; consult doctor for labour planning.",
    "abnormal_placenta":  "Close monitoring required; follow doctor instructions for delivery.",
    "hellp_syndrome":     "Immediate hospital attention if symptoms appear; regular labs.",
}
advice_map_ta = {
    "anemia":             "இரும்புச் சத்து மாத்திரைகள் எடுக்கவும், இரும்புச் சத்து நிறைந்த உணவுகளை சாப்பிடவும், ஹீமோகுளோபினை தொடர்ந்து சரிபார்க்கவும்.",
    "bmi":                "ஆரோக்கியமான எடையை பராமரிக்கவும்; பிரசவத்திற்கு முந்தைய ஊட்டச்சத்து மற்றும் மிதமான உடற்பயிற்சியை பின்பற்றவும்.",
    "pre_pph":            "உங்கள் மகப்பேறு மருத்துவரிடம் தெரிவிக்கவும்; பிரசவத்தின் போது கவனமாக கண்காணிப்பு தேவை.",
    "pre_c_section":      "அனுபவமிக்க மகப்பேறு குழுவுடன் பிரசவத்தை திட்டமிடுங்கள்; இரத்தப்போக்கை கவனிக்கவும்.",
    "placenta_previa":    "கடினமான செயல்பாடுகளை தவிர்க்கவும்; தொடர்ச்சியான அல்ட்ராசவுண்டுகளுக்கு செல்லவும்.",
    "gest_diabetes":      "இரத்த சர்க்கரையை கண்காணிக்கவும்; உணவு மற்றும் உடற்பயிற்சி திட்டத்தை பின்பற்றவும்.",
    "pre_hypertension":   "தொடர்ந்து BP சரிபார்க்கவும்; மருந்துகளை பரிந்துரைத்தபடி எடுக்கவும்; மன அழுத்தத்தை குறைக்கவும்.",
    "polyhydromnios":     "அடிக்கடி பிரசவத்திற்கு முந்தைய பரிசோதனைகள்; கரு வளர்ச்சியை கண்காணிக்கவும்.",
    "severe_preeclampsia":"BP மற்றும் புரத அளவுகளை கவனிக்கவும்; நிலை மோசமடைந்தால் மருத்துவமனை அனுமதி தேவை.",
    "multiple_pregnancy": "கர்ப்பகாலத்தில் கூடுதல் கண்காணிப்பு; சாத்தியமான சிக்கல்களுக்கு தயாராக இருக்கவும்.",
    "hemoglobin_level":   "ஆரோக்கியமான இரும்பு மற்றும் வைட்டமின் அளவுகளை பராமரிக்கவும்.",
    "surgery":            "முந்தைய அறுவை சிகிச்சைகள் பற்றி மருத்துவரிடம் தெரிவிக்கவும்.",
    "myoma":              "நார்த்திசு வளர்ச்சியை கண்காணிக்கவும்; பிரசவ திட்டமிடலுக்கு மருத்துவரை அணுகவும்.",
    "abnormal_placenta":  "நெருக்கமான கண்காணிப்பு தேவை; மருத்துவர் அறிவுறுத்தல்களை பின்பற்றவும்.",
    "hellp_syndrome":     "அறிகுறிகள் தோன்றினால் உடனடி மருத்துவமனை கவனிப்பு; தொடர்ந்து ஆய்வக பரிசோதனைகள்.",
}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def create_user(u, p, r):
    cursor.execute("INSERT INTO users VALUES(?,?,?)", (u, p, r))
    conn.commit()

def login_user(u, p):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
    return cursor.fetchall()

def save_patient(name, age, bmi, hemoglobin, gestational_age, parity, anemia, pre_pph,
                 multiple_pregnancy, pre_hypertension, placenta_previa, gest_diabetes,
                 pre_c_section, pre_blood, abnormal_placenta, polyhydromnios, hellp_syndrome,
                 severe_preeclampsia, surgery, myoma, risk_prob, risk):
    cursor.execute("""INSERT INTO patients(
        name, age, bmi, hemoglobin, gestational_age, parity,
        anemia, pre_pph, multiple_pregnancy, pre_hypertension, placenta_previa, gest_diabetes,
        pre_c_section, pre_blood, abnormal_placenta, polyhydromnios, hellp_syndrome,
        severe_preeclampsia, surgery, myoma, risk_prob, risk
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (name, age, bmi, hemoglobin, int(gestational_age), parity, anemia, pre_pph,
         multiple_pregnancy, pre_hypertension, placenta_previa, gest_diabetes,
         pre_c_section, pre_blood, abnormal_placenta, polyhydromnios, hellp_syndrome,
         severe_preeclampsia, surgery, myoma, risk_prob, risk))
    conn.commit()

def delete_patient(pid):
    cursor.execute("DELETE FROM patients WHERE rowid=?", (pid,))
    conn.commit()

def clear_all_patients():
    cursor.execute("DELETE FROM patients")
    conn.commit()

def get_patient_appointments(username):
    try:
        return pd.read_sql(
            "SELECT * FROM appointments WHERE patient_username=? ORDER BY id DESC",
            conn, params=(username,)
        )
    except Exception:
        return pd.DataFrame()

def get_all_appointments():
    try:
        return pd.read_sql("SELECT * FROM appointments ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()

def create_appointment(patient_username, patient_name, risk, risk_prob, suggested_date, patient_note, gestational_age=0):
    try:
        cursor.execute(
            """INSERT INTO appointments(
                patient_username, patient_name, risk, risk_prob,
                gestational_age, suggested_date, patient_note, status,
                requested_at
            ) VALUES(?,?,?,?,?,?,?,'pending',datetime('now','localtime'))""",
            (
                str(patient_username),
                str(patient_name),
                str(risk),
                float(risk_prob),
                int(gestational_age),
                str(suggested_date),
                str(patient_note) if patient_note else ""
            )
        )
        conn.commit()
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return str(e)

def confirm_appointment(appt_id, confirmed_datetime, doctor_note):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute(
        "UPDATE appointments SET status='confirmed', confirmed_datetime=?, doctor_note=?, updated_at=? WHERE id=?",
        (confirmed_datetime, doctor_note, now, appt_id)
    )
    conn.commit()

def cancel_appointment(appt_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute(
        "UPDATE appointments SET status='cancelled', updated_at=? WHERE id=?",
        (now, appt_id)
    )
    conn.commit()

def has_pending_appointment(username):
    try:
        row = cursor.execute(
            "SELECT COUNT(*) FROM appointments WHERE patient_username=? AND status='pending'",
            (username,)
        ).fetchone()
        return (row[0] > 0) if row else False
    except Exception:
        return False

def get_checkup_days(risk):
    return {"HIGH RISK": 7, "MODERATE RISK": 14}.get(risk, 30)

def safe_html(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in [
    ("login", False), ("role", ""), ("username", ""), ("page", "login"),
    ("nav", "predict"), ("chat_history", []), ("last_high_risk_idx", []),
    ("lang", "en"), ("last_risk", ""), ("last_risk_prob", 0.0),
    ("last_patient_name", ""), ("last_gest_age", 0),
    ("appt_success_msg", ""), ("appt_error_msg", ""),
    ("show_appt_success", False),   # FIX 2: dedicated flag for success banner
]:
    if k not in st.session_state:
        st.session_state[k] = v

def tr(key):
    return T[st.session_state.lang].get(key, T["en"].get(key, key))

# ─────────────────────────────────────────────────────────────
# CHATBOT
# ─────────────────────────────────────────────────────────────
def get_bot_response(user_msg, lang):
    msg = user_msg.lower()
    R = {
        "pph": {
            "en": "**Postpartum Haemorrhage (PPH)**\n\nPPH means heavy bleeding after delivery (more than 500 ml).\n\n**Common causes:**\n- Uterine atony (uterus not contracting)\n- Retained placenta\n- Tears in birth canal\n- Blood clotting disorders\n\n**Warning signs:**\n- Soaking more than one pad per hour\n- Dizziness, weakness, or fainting\n- Rapid heartbeat or low blood pressure\n\n**Prevention:**\n- Regular antenatal checkups\n- Correct anaemia before delivery\n- Inform doctor of all risk factors\n\nIf heavy bleeding occurs after delivery, seek emergency care immediately.\n\nPlease consult your doctor for personalised advice.",
            "ta": "**பிரசவத்திற்கு பிந்தைய இரத்தப்போக்கு (PPH)**\n\nPPH என்பது பிரசவத்திற்கு பிறகு கடுமையான இரத்தப்போக்கு (500 மி.லி.க்கு மேல்) ஆகும்.\n\n**பொதுவான காரணங்கள்:**\n- கர்ப்பப்பை சுருக்கமின்மை\n- நஞ்சுக்கொடி தங்குதல்\n- பிரசவ வழியில் கிழிசல்\n- இரத்த உறைவு கோளாறு\n\nபிரசவத்திற்கு பிறகு கடுமையான இரத்தப்போக்கு ஏற்பட்டால், உடனடியாக அவசர சிகிச்சை பெறவும்.\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "iron": {
            "en": "**Iron and Anaemia During Pregnancy**\n\nNormal hemoglobin in pregnancy: 11 g/dL or above.\n\n**Iron-rich Indian foods:**\n- Green leafy vegetables (spinach, methi, drumstick leaves)\n- Dates, raisins, and figs\n- Rajma, chana, and lentils (dal)\n- Sesame seeds and jaggery\n- Ragi (finger millet)\n\n**Tips to absorb iron better:**\n- Eat iron-rich foods with vitamin C (lemon, amla, tomatoes)\n- Avoid tea or coffee with meals\n- Take iron tablets as prescribed\n\nPlease consult your doctor for personalised advice.",
            "ta": "**கர்ப்பகாலத்தில் இரும்பு மற்றும் இரத்த சோகை**\n\nகர்ப்பகாலத்தில் இயல்பான ஹீமோகுளோபின்: 11 g/dL அல்லது அதிகம்.\n\n**இரும்புச் சத்து நிறைந்த இந்திய உணவுகள்:**\n- பச்சை இலை காய்கறிகள் (கீரை, வெந்தயம், முருங்கை இலை)\n- பேரீச்சம்பழம், திராட்சை, அத்தி\n- ராஜ்மா, சுண்டல், பருப்பு\n- எள் மற்றும் வெல்லம்\n- கேழ்வரகு\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "nutrition": {
            "en": "**Pregnancy Nutrition Guide**\n\n**First Trimester (1-12 weeks):**\n- Folic acid is essential — green leafy vegetables, lentils\n- Small frequent meals to manage nausea\n\n**Second Trimester (13-26 weeks):**\n- Calcium — milk, curd, ragi, sesame\n- Iron — spinach, dates, rajma\n- Protein — eggs, paneer, dal, fish\n\n**Third Trimester (27-40 weeks):**\n- High energy foods — whole grains, nuts, banana\n- 8-10 glasses of water daily\n\nPlease consult your doctor for personalised advice.",
            "ta": "**கர்ப்பகால ஊட்டச்சத்து வழிகாட்டி**\n\n**முதல் மூன்று மாதங்கள்:**\n- ஃபோலிக் அமிலம் அவசியம் — பச்சை இலை காய்கறிகள், பருப்பு\n\n**இரண்டாம் மூன்று மாதங்கள்:**\n- கால்சியம், இரும்பு, புரதம் சேர்க்கவும்\n\n**மூன்றாம் மூன்று மாதங்கள்:**\n- முழு தானியங்கள், நட்ஸ், தினமும் 8-10 கிளாஸ் தண்ணீர்\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "symptom": {
            "en": "**Pregnancy Symptoms and Warning Signs**\n\n**Normal symptoms:**\n- Mild nausea, fatigue, mild back pain\n- Frequent urination\n\n**Warning signs — call your doctor immediately:**\n- Severe headache or vision changes\n- Sudden swelling of face or hands\n- Heavy vaginal bleeding\n- Reduced or no fetal movement\n\nPlease consult your doctor for personalised advice.",
            "ta": "**கர்ப்பகால அறிகுறிகள்**\n\n**எச்சரிக்கை அறிகுறிகள்:**\n- கடுமையான தலைவலி அல்லது பார்வை மாற்றங்கள்\n- முகம் அல்லது கைகளில் திடீர் வீக்கம்\n- கடுமையான யோனி இரத்தப்போக்கு\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "postpartum": {
            "en": "**Postpartum Recovery**\n\n- Rest as much as possible\n- Eat iron-rich and protein-rich foods\n- Soaking a pad in under an hour means seek medical help immediately\n\nPlease consult your doctor for personalised advice.",
            "ta": "**பிரசவத்திற்கு பிந்தைய மீட்பு**\n\n- முடிந்தவரை ஓய்வு எடுக்கவும்\n- ஒரு மணி நேரத்தில் ஒரு பேட் நனைந்தால் உடனடியாக மருத்துவ உதவி பெறவும்\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "csection": {
            "en": "**C-Section Recovery**\n\n- Avoid heavy lifting for at least 6 weeks\n- Keep incision clean and dry\n- High protein foods for wound healing\n\nPlease consult your doctor for personalised advice.",
            "ta": "**சிசேரியன் மீட்பு**\n\n- குறைந்தது 6 வாரங்களுக்கு கனமான பொருட்கள் தூக்க வேண்டாம்\n- புண் குணமாக புரத உணவுகள் சாப்பிடவும்\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "bp": {
            "en": "**Blood Pressure in Pregnancy**\n\nNormal BP: below 140/90 mmHg\n\n**Preeclampsia warning signs:**\n- Severe headache, vision changes\n- Upper abdominal pain, sudden swelling\n\nSevere preeclampsia requires immediate hospitalisation.\n\nPlease consult your doctor for personalised advice.",
            "ta": "**கர்ப்பகாலத்தில் இரத்த அழுத்தம்**\n\nஇயல்பான BP: 140/90 mmHg க்கு கீழே\n\nகடுமையான முன் பிரம்பியம் உடனடி மருத்துவமனை அனுமதி தேவைப்படுகிறது.\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "diabetes": {
            "en": "**Gestational Diabetes**\n\n- Monitor blood glucose as advised\n- Limit sugary foods, white rice, maida\n- Replace white rice with millets or brown rice\n- 30 minutes of walking daily is helpful\n\nPlease consult your doctor for personalised advice.",
            "ta": "**கர்ப்பகால நீரிழிவு**\n\n- இரத்த குளுக்கோஸை கண்காணிக்கவும்\n- வெள்ளை அரிசிக்கு பதிலாக சிறுதானியங்கள் சாப்பிடவும்\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "breastfeed": {
            "en": "**Breastfeeding Guide**\n\n- Feed on demand — every 2-3 hours for newborns\n- Methi ladoo, garlic in dal to boost milk supply\n\nPlease consult your doctor for personalised advice.",
            "ta": "**தாய்ப்பால் வழிகாட்டி**\n\n- கேட்கும் போது பால் கொடுக்கவும்\n- தாய்ப்பால் அதிகரிக்க வெந்தய லட்டு, பருப்பில் பூண்டு சேர்க்கவும்\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "exercise": {
            "en": "**Exercise During Pregnancy**\n\n**Safe:** Walking 30 min daily, prenatal yoga, swimming\n\n**Avoid:** Heavy lifting, contact sports, hot yoga\n\nPlease consult your doctor before starting any new exercise routine.",
            "ta": "**கர்ப்பகாலத்தில் உடற்பயிற்சி**\n\n**பாதுகாப்பான:** தினமும் 30 நிமிடம் நடை, யோகா\n\nபுதிய உடற்பயிற்சியை தொடங்கும் முன் மருத்துவரை அணுகவும்."
        },
        "hello": {
            "en": "Hello! Welcome to the PPH Health Assistant.\n\nI can help you with PPH risk, pregnancy nutrition, iron and anaemia, symptoms, postpartum recovery, blood pressure, gestational diabetes, breastfeeding, and safe exercise.\n\nJust type your question!\n\nPlease consult your doctor for personalised advice.",
            "ta": "வணக்கம்! PPH சுகாதார உதவியாளரிடம் வரவேற்கிறோம்.\n\nஉங்கள் கேள்வியை தட்டச்சு செய்யுங்கள்!\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "thanks": {
            "en": "You're welcome! Take care of yourself and stay well.\n\nPlease consult your doctor for personalised advice.",
            "ta": "மகிழ்ச்சியே! உங்களை நீங்களே கவனித்துக்கொள்ளுங்கள்.\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        },
        "default": {
            "en": "I can help with: PPH, iron and anaemia, pregnancy diet, symptoms, postpartum recovery, C-section, blood pressure, gestational diabetes, breastfeeding, or exercise.\n\nPlease consult your doctor for personalised advice.",
            "ta": "நான் இதில் உதவ முடியும்: PPH, இரும்பு, உணவு, அறிகுறிகள், மீட்பு, இரத்த அழுத்தம், நீரிழிவு, தாய்ப்பால், உடற்பயிற்சி.\n\nதனிப்பட்ட ஆலோசனைக்கு உங்கள் மருத்துவரை அணுகவும்."
        }
    }

    if any(w in msg for w in ["pph", "postpartum haemorrhage", "hemorrhage", "bleeding after birth", "bleeding after delivery", "இரத்தப்போக்கு"]):
        key = "pph"
    elif any(w in msg for w in ["iron", "hemoglobin", "haemoglobin", "anaemia", "anemia", "இரும்பு", "ஹீமோகுளோபின்", "இரத்த சோகை"]):
        key = "iron"
    elif any(w in msg for w in ["nutrition", "diet", "food", "eat", "meal", "trimester", "eating", "ஊட்டச்சத்து", "உணவு", "சாப்பிட"]):
        key = "nutrition"
    elif any(w in msg for w in ["symptom", "warning", "sign", "pain", "headache", "swelling", "kick", "movement", "அறிகுறி", "வலி", "தலைவலி", "வீக்கம்"]):
        key = "symptom"
    elif any(w in msg for w in ["postpartum", "after delivery", "recovery", "postnatal", "after birth", "lochia", "பிரசவத்திற்கு பிறகு", "மீட்பு"]):
        key = "postpartum"
    elif any(w in msg for w in ["c-section", "cesarean", "caesarean", "cs", "operation", "c section", "சிசேரியன்", "அறுவை சிகிச்சை"]):
        key = "csection"
    elif any(w in msg for w in ["bp", "blood pressure", "hypertension", "preeclampsia", "pressure", "இரத்த அழுத்தம்"]):
        key = "bp"
    elif any(w in msg for w in ["diabetes", "sugar", "gestational diabetes", "blood sugar", "glucose", "நீரிழிவு", "சர்க்கரை"]):
        key = "diabetes"
    elif any(w in msg for w in ["breastfeed", "breast feed", "nursing", "milk", "lactation", "feed baby", "தாய்ப்பால்", "பால்"]):
        key = "breastfeed"
    elif any(w in msg for w in ["exercise", "walk", "yoga", "activity", "workout", "உடற்பயிற்சி", "நடை", "யோகா"]):
        key = "exercise"
    elif any(w in msg for w in ["hello", "hi", "hey", "good morning", "good evening", "namaste", "vanakkam", "வணக்கம்", "ஹலோ"]):
        key = "hello"
    elif any(w in msg for w in ["thank", "thanks", "thank you", "நன்றி"]):
        key = "thanks"
    else:
        key = "default"

    return R[key][lang]

# ─────────────────────────────────────────────────────────────
# CHECKUP SCHEDULE RENDERER
# ─────────────────────────────────────────────────────────────
def render_checkup_schedule(current_ga, risk, lang):
    is_ta = (lang == "ta")
    progress_pct = min(100, int((current_ga / 40) * 100))

    if current_ga <= 13:
        trimester = "முதல் மூன்று மாதம்" if is_ta else "1st Trimester"
    elif current_ga <= 26:
        trimester = "இரண்டாம் மூன்று மாதம்" if is_ta else "2nd Trimester"
    else:
        trimester = "மூன்றாம் மூன்று மாதம்" if is_ta else "3rd Trimester"

    risk_color = {"HIGH RISK": "#b00020", "MODERATE RISK": "#d97706", "LOW RISK": "#16a34a"}.get(risk, "#6a0dad")
    risk_badge_color = "#fbbf24" if risk == "HIGH RISK" else "#86efac"

    progress_label = "கர்ப்ப முன்னேற்றம்" if is_ta else "Pregnancy Progress"
    week_label     = "வாரங்கள்" if is_ta else "Weeks"
    pph_risk_label = "PPH ஆபத்து" if is_ta else "PPH Risk"
    complete_label = "முடிந்தது" if is_ta else "Complete"
    w1_label       = "வாரம் 1" if is_ta else "Week 1"
    w40_label      = "வாரம் 40" if is_ta else "Week 40"

    st.markdown(f"""<div style="background:linear-gradient(135deg,#3b0764 0%,#7c3aed 50%,#ec4899 100%);
border-radius:20px;padding:28px 32px;margin-bottom:24px;
box-shadow:0 8px 32px rgba(107,0,173,0.25);color:white;">
<div style="font-size:13px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;opacity:0.75;margin-bottom:6px;">
{progress_label}
</div>
<div style="font-family:'DM Serif Display',serif;font-size:42px;font-weight:700;">
{current_ga} {week_label}
</div>
<div style="font-size:16px;font-weight:600;margin-top:4px;">{trimester}</div>
<div style="font-size:13px;opacity:0.75;">{pph_risk_label}:
<span style="color:{risk_badge_color};font-weight:700;">{risk}</span>
</div>
<div style="background:rgba(255,255,255,0.2);border-radius:99px;height:12px;margin-top:16px;">
<div style="background:linear-gradient(90deg,#fbbf24,#f9a8d4);border-radius:99px;height:12px;width:{progress_pct}%;"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:11px;opacity:0.65;margin-top:6px;">
<span>{w1_label}</span>
<span>{progress_pct}% {complete_label}</span>
<span>{w40_label}</span>
</div>
</div>""", unsafe_allow_html=True)

    weeks_remaining = max(0, 40 - current_ga)
    days_remaining  = weeks_remaining * 7
    month_num       = max(1, min(10, (current_ga // 4) + 1))

    sc1, sc2, sc3, sc4 = st.columns(4)
    stats = [
        (sc1, "🗓️", f"Month {month_num}" if not is_ta else f"மாதம் {month_num}",
         "Current Month" if not is_ta else "தற்போதைய மாதம்", "#6a0dad"),
        (sc2, "⏱️", f"{weeks_remaining} wks" if not is_ta else f"{weeks_remaining} வாரங்கள்",
         "Remaining" if not is_ta else "மீதமிருக்கும்", "#0369a1"),
        (sc3, "📅", f"~{days_remaining} days" if not is_ta else f"~{days_remaining} நாட்கள்",
         "To due date" if not is_ta else "தேதி வரை", "#047857"),
        (sc4, "⚠️", risk.split()[0],
         "Risk Level" if not is_ta else "ஆபத்து நிலை", risk_color),
    ]
    for _col, _icon, _val, _lbl, _clr in stats:
        with _col:
            st.markdown(f"""<div class="stat-card" style="border-left-color:{_clr};text-align:center;">
<div style="font-size:22px;">{_icon}</div>
<div style="font-size:20px;font-weight:700;color:{_clr};margin-top:4px;">{_val}</div>
<div style="font-size:12px;color:#6b7280;margin-top:2px;">{_lbl}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sched_heading = "மாதாந்திர பரிசோதனை அட்டவணை" if is_ta else "Month-by-Month Checkup Schedule"
    st.markdown(f"""<div style="font-family:'DM Serif Display',serif;font-size:22px;color:#3b0764;margin-bottom:16px;">
📋 {sched_heading}
</div>""", unsafe_allow_html=True)

    for month_data in MONTHLY_SCHEDULE:
        wk_start, wk_end = month_data["week_range"]
        title = month_data["title_ta"] if is_ta else month_data["title_en"]

        is_past    = current_ga >= wk_end
        is_current = wk_start <= current_ga < wk_end

        if is_past:
            s_icon  = "✅"; s_label = "முடிந்தது" if is_ta else "Completed"
            s_clr   = "#15803d"; h_bg = "#f0fdf4"; h_brd = "#86efac"
            badge_bg = "#d1fae5"; opacity = "0.65"; shadow = ""
        elif is_current:
            s_icon  = "▶"; s_label = "இப்போது" if is_ta else "You are here"
            s_clr   = "#7c3aed"; h_bg = "#ede9fe"; h_brd = "#a78bfa"
            badge_bg = "#ede9fe"; opacity = "1"; shadow = "box-shadow:0 4px 20px rgba(124,58,237,0.2);"
        else:
            s_icon  = "○"; s_label = "வரவிருக்கிறது" if is_ta else "Upcoming"
            s_clr   = "#6b7280"; h_bg = "white"; h_brd = "#e5e7eb"
            badge_bg = "#f3f4f6"; opacity = "0.9"; shadow = ""

        wk_lbl = "வாரம்" if is_ta else "Weeks"
        safe_title = safe_html(title)

        st.markdown(f"""<div style="background:{h_bg};border:2px solid {h_brd};border-radius:16px;
padding:16px 20px;margin-bottom:8px;opacity:{opacity};{shadow}">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
<div>
<span style="font-size:13px;font-weight:700;color:{s_clr};background:{badge_bg};
padding:3px 12px;border-radius:20px;margin-right:10px;">{s_icon} {s_label}</span>
<span style="font-family:'DM Serif Display',serif;font-size:18px;color:#1a1a2e;">{safe_title}</span>
</div>
<span style="font-size:13px;color:#6b7280;">{wk_lbl} {wk_start}-{wk_end}</span>
</div>
</div>""", unsafe_allow_html=True)

        for visit in month_data["visits"]:
            vw_start, vw_end = visit["week_range"]
            v_label  = visit["label_ta"]  if is_ta else visit["label_en"]
            v_tests  = visit["tests_ta"]  if is_ta else visit["tests_en"]
            v_tip    = visit["tip_ta"]    if is_ta else visit["tip_en"]
            v_icon   = visit["icon"]
            v_bg     = visit["color"]
            v_border = visit["border"]
            v_text   = visit["text"]

            is_v_past    = current_ga >= vw_end
            is_v_current = vw_start <= current_ga < vw_end

            card_bg    = "#f9fafb" if is_v_past else v_bg
            card_brd   = "#e5e7eb" if is_v_past else v_border
            v_opacity  = "0.55"    if is_v_past else "1"
            v_shadow   = "box-shadow:0 4px 16px rgba(124,58,237,0.18);" if is_v_current else ""
            tick       = "✅" if is_v_past else ("▶" if is_v_current else "○")
            text_clr   = "#9ca3af" if is_v_past else "#374151"
            label_clr  = "#9ca3af" if is_v_past else v_text
            tip_bg     = "#f3f4f6" if is_v_past else "rgba(255,255,255,0.8)"
            tip_clr    = "#9ca3af" if is_v_past else "#374151"
            tip_lbl    = "குறிப்பு" if is_ta else "Tip"
            wk_lbl2    = "வாரம்" if is_ta else "Weeks"

            if is_v_past:
                badge_html = "<span style=\"margin-left:auto;background:#d1fae5;color:#065f46;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;\">Done</span>"
            elif is_v_current:
                badge_html = "<span style=\"margin-left:auto;background:#ede9fe;color:#6d28d9;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;\">Now</span>"
            else:
                badge_html = ""

            tests_rows = ""
            for t in v_tests:
                safe_t = safe_html(t)
                tests_rows += (
                    f"<div style=\"display:flex;align-items:flex-start;gap:8px;padding:5px 0;"
                    f"border-bottom:1px solid {v_border}44;\">"
                    f"<span style=\"color:{v_text};margin-top:2px;flex-shrink:0;\">{tick}</span>"
                    f"<span style=\"font-size:13px;color:{text_clr};\">{safe_t}</span>"
                    f"</div>"
                )

            safe_label = safe_html(v_label)
            safe_tip   = safe_html(v_tip)

            st.markdown(
                f"<div style=\"background:{card_bg};border:1.5px solid {card_brd};"
                f"border-radius:14px;padding:18px 20px;margin-left:20px;margin-bottom:16px;"
                f"opacity:{v_opacity};{v_shadow}\">"
                f"<div style=\"display:flex;align-items:center;gap:10px;margin-bottom:12px;\">"
                f"<span style=\"font-size:24px;\">{v_icon}</span>"
                f"<div style=\"flex:1;\">"
                f"<div style=\"font-size:15px;font-weight:700;color:{label_clr};\">{safe_label}</div>"
                f"<div style=\"font-size:12px;color:#6b7280;\">{wk_lbl2} {vw_start}-{vw_end}</div>"
                f"</div>{badge_html}</div>"
                f"<div style=\"margin-bottom:12px;\">{tests_rows}</div>"
                f"<div style=\"background:{tip_bg};border-radius:8px;padding:10px 14px;"
                f"border-left:3px solid {v_border};\">"
                f"<span style=\"font-size:11px;font-weight:700;color:{v_text};"
                f"text-transform:uppercase;letter-spacing:0.06em;\">💡 {tip_lbl}</span>"
                f"<div style=\"font-size:13px;color:{tip_clr};margin-top:4px;\">{safe_tip}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

    if current_ga >= 37:
        if is_ta:
            l1 = "பிரசவ அறிகுறிகளை கவனிக்கவும்!"
            l2 = "தொடர் சுருக்கங்கள் (5 நிமிடத்திற்கு ஒருமுறை) | நீர் உடைதல் | இரத்தம் கலந்த சளி வெளியேறுதல்"
            l3 = "உடனடியாக மருத்துவமனை செல்லவும்!"
        else:
            l1 = "Watch for Labour Signs!"
            l2 = "Regular contractions every 5 min · Water breaking · Bloody show · Reduced fetal movement"
            l3 = "Go to hospital immediately if any of these occur!"

        st.markdown(f"""<div style="background:linear-gradient(135deg,#ffe5e5,#fce7f3);
border:2px solid #fca5a5;border-radius:16px;padding:20px 24px;margin-top:16px;">
<div style="font-size:22px;font-weight:700;color:#b00020;margin-bottom:8px;">🚨 {l1}</div>
<div style="font-size:14px;color:#7f1d1d;line-height:1.7;">
{l2}<br><br>
<b style="color:#b00020;">{l3}</b>
</div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═════════════════════════════════════════════════════════════
if not st.session_state.login:
    lc1, lc2, lc3 = st.columns([3, 1, 3])
    with lc2:
        lang_choice = st.radio("lang", ["English", "தமிழ்"], horizontal=True, label_visibility="collapsed")
        st.session_state.lang = "en" if lang_choice == "English" else "ta"

    st.markdown(f"""<div style="text-align:center;padding:30px 0 10px;">
<span style="font-size:56px;">🤰</span>
<div style="font-family:'DM Serif Display',serif;font-size:32px;color:#6a0dad;margin-top:8px;">{tr("app_title")}</div>
<div style="color:#6b7280;font-size:14px;margin-top:6px;">{tr("app_sub")}</div>
</div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        if st.session_state.page == "login":
            st.markdown(f"""<div style="font-family:'DM Serif Display',serif;font-size:24px;
color:#3b0764;text-align:center;margin-bottom:6px;">{tr("welcome")}</div>
<div style="font-size:13px;color:#6b7280;text-align:center;margin-bottom:16px;">{tr("sign_in")}</div>""",
                unsafe_allow_html=True)
            username = st.text_input(tr("username"), key="l_user")
            password = st.text_input(tr("password"), type="password", key="l_pass")
            if st.button(tr("login"), use_container_width=True):
                r = login_user(username, password)
                if r:
                    st.session_state.login    = True
                    st.session_state.role     = r[0][2]
                    st.session_state.username = username
                    st.session_state.nav      = "predict" if r[0][2] == "Pregnant Woman" else "patients"
                    st.rerun()
                else:
                    st.error(tr("invalid_login"))
            st.markdown(f"""<div style="text-align:center;margin-top:12px;font-size:13px;
color:#6b7280;">{tr("no_account")}</div>""", unsafe_allow_html=True)
            if st.button(tr("create_account"), use_container_width=True):
                st.session_state.page = "create"
                st.rerun()
        else:
            st.markdown(f"""<div style="font-family:'DM Serif Display',serif;font-size:24px;
color:#3b0764;text-align:center;margin-bottom:16px;">{tr("create_account")}</div>""",
                unsafe_allow_html=True)
            new_user = st.text_input(tr("username"))
            new_pass = st.text_input(tr("password"), type="password")
            confirm  = st.text_input(tr("confirm_password"), type="password")
            role     = st.selectbox(tr("role"), ["Pregnant Woman", "Doctor"])
            if st.button(tr("create"), use_container_width=True):
                if new_pass != confirm:
                    st.error(tr("passwords_no_match"))
                else:
                    create_user(new_user, new_pass, role)
                    st.success(tr("account_created"))
            if st.button(tr("back_login"), use_container_width=True):
                st.session_state.page = "login"
                st.rerun()

# ═════════════════════════════════════════════════════════════
# PATIENT DASHBOARD
# ═════════════════════════════════════════════════════════════
elif st.session_state.role == "Pregnant Woman":

    with st.sidebar:
        st.markdown("""<div class="sidebar-logo">
<div class="logo-icon">🤰</div>
<div class="logo-title">PPH Predict</div>
<div class="logo-sub">Maternal Health Platform</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='nav-section-label'>🌐 Language / மொழி</div>", unsafe_allow_html=True)
        ls = st.radio("ls", ["English", "தமிழ்"], horizontal=True, label_visibility="collapsed",
                      index=0 if st.session_state.lang == "en" else 1, key="sl1")
        st.session_state.lang = "en" if ls == "English" else "ta"

        st.markdown(f"<div class='nav-section-label'>{tr('main_menu')}</div>", unsafe_allow_html=True)
        for key, icon, label in [
            ("predict",      "📊", tr("predict_nav")),
            ("advice",       "💡", tr("advice_nav")),
            ("chat",         "🤖", tr("chat_nav")),
            ("schedule",     "📋", tr("sched_nav")),
            ("appointments", "📅", tr("appts_nav")),
        ]:
            prefix = "▶  " if st.session_state.nav == key else "   "
            if st.button(f"{prefix}{icon}  {label}", key=f"nav_{key}"):
                st.session_state.nav = key
                st.rerun()

        pending_count = 0
        if st.session_state.username:
            try:
                df_p = get_patient_appointments(st.session_state.username)
                if not df_p.empty and "status" in df_p.columns:
                    pending_count = len(df_p[df_p["status"] == "pending"])
            except Exception:
                pass
        if pending_count > 0:
            st.markdown(f"""<div style="margin:4px 20px;background:#fef3c7;border-radius:8px;
padding:6px 12px;font-size:12px;color:#92400e;">
⏳ {pending_count} appointment request pending</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='nav-section-label'>{tr('account')}</div>", unsafe_allow_html=True)
        if st.button(f"   🚪  {tr('logout')}", key="nav_logout"):
            st.session_state.login = False
            st.rerun()

    # ── PREDICT ──────────────────────────────────────────────────────────────
    if st.session_state.nav == "predict":
        logged_user = st.session_state.username
        st.markdown(f"""<div class="main-header">
<span class="page-icon">📊</span>
<div>
<p class="page-title">{tr("predict_title")}</p>
<p class="page-sub">{tr("predict_sub")}</p>
</div>
<div style="margin-left:auto;background:#f3e8ff;border-radius:10px;padding:8px 18px;text-align:right;">
<div style="font-size:11px;color:#7c3aed;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;">Logged in as</div>
<div style="font-size:16px;font-weight:700;color:#3b0764;">👤 {safe_html(logged_user)}</div>
</div>
</div>""", unsafe_allow_html=True)

        # ── FIX 2: SUCCESS / ERROR BANNERS — shown at top, OUTSIDE prediction form ──
        # These persist across reruns via session state flags
        if st.session_state.get("show_appt_success"):
            st.markdown("""<div style="background:linear-gradient(135deg,#d1fae5,#a7f3d0);
border:2px solid #34d399;border-radius:16px;padding:22px 28px;margin-bottom:20px;
display:flex;align-items:center;gap:18px;box-shadow:0 4px 20px rgba(52,211,153,0.25);">
<div style="font-size:44px;">✅</div>
<div>
  <div style="font-size:18px;font-weight:700;color:#065f46;">Appointment Request Sent Successfully!</div>
  <div style="font-size:14px;color:#047857;margin-top:4px;">
    Your doctor will review and confirm. Check <b>My Appointments</b> tab to track status.
  </div>
</div>
</div>""", unsafe_allow_html=True)
            st.session_state.show_appt_success = False

        if st.session_state.get("appt_error_msg"):
            st.error(f"❌ {st.session_state.appt_error_msg}")
            st.session_state.appt_error_msg = ""

        yn = [tr("no"), tr("yes")]
        with st.form("patient_form"):
            name = st.session_state.username
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input(tr("age"), 10, 60, value=25)
                bmi = st.number_input(tr("bmi"), 10.0, 50.0, value=22.0)
                hb  = st.number_input(tr("hemoglobin"), 5.0, 20.0, value=11.0)
            with c2:
                ga  = st.number_input(tr("gest_age"), 20, 45, value=38)
                par = st.number_input(tr("parity"), 0, 10, value=0)
                an  = st.selectbox(tr("anemia"), yn)
                pp  = st.selectbox(tr("prev_pph"), yn)
            with c3:
                mp  = st.selectbox(tr("multiple_preg"), yn)
                ht  = st.selectbox(tr("hypertension"), yn)
                ppl = st.selectbox(tr("placenta_previa"), yn)
                gd  = st.selectbox(tr("gest_diabetes"), yn)
            st.markdown(tr("additional_factors"))
            c4, c5, c6 = st.columns(3)
            with c4:
                cs  = st.selectbox(tr("prev_csection"), yn)
                bd  = st.selectbox(tr("blood_disorder"), yn)
                ap  = st.selectbox(tr("abnormal_placenta"), yn)
            with c5:
                ph  = st.selectbox(tr("polyhydramnios"), yn)
                hs  = st.selectbox(tr("hellp"), yn)
                sp  = st.selectbox(tr("severe_preeclampsia"), yn)
            with c6:
                sg  = st.selectbox(tr("surgery"), yn)
                my  = st.selectbox(tr("myoma"), yn)
            submit = st.form_submit_button(tr("predict_btn"), use_container_width=True)

        def iy(v):
            return 1 if v == tr("yes") else 0

        if submit:
            patient_dict = {
                'age': age, 'bmi': bmi, 'hemoglobin_level': hb,
                'anemia': iy(an), 'pre_pph': iy(pp), 'parity': par,
                'multiple_pregnancy': iy(mp), 'pre_hypertension': iy(ht),
                'placenta_previa': iy(ppl), 'gest_diabetes': iy(gd),
                'pre_c_section': iy(cs), 'pre_blood': iy(bd),
                'abnormal_placenta': iy(ap), 'polyhydromnios': iy(ph),
                'hellp_syndrome': iy(hs), 'severe_preeclampsia': iy(sp),
                'surgery': iy(sg), 'myoma': iy(my), 'gestational_age': ga
            }
            df_pred = prepare_input_for_model(patient_dict)
            prob    = model.predict_proba(df_pred)[0][1] * 100

            if prob < 30:
                risk = "LOW RISK"
                st.markdown(f"<div class='low-risk'>✔ {tr('low_risk')} — {prob:.1f}%</div>", unsafe_allow_html=True)
            elif prob <= 70:
                risk = "MODERATE RISK"
                st.markdown(f"<div class='mod-risk'>⚠ {tr('mod_risk')} — {prob:.1f}%</div>", unsafe_allow_html=True)
            else:
                risk = "HIGH RISK"
                st.markdown(f"<div class='high-risk'>⚠ {tr('high_risk')} — {prob:.1f}%</div>", unsafe_allow_html=True)

            st.progress(int(prob))

            save_patient(name, age, bmi, hb, ga, par, iy(an), iy(pp), iy(mp), iy(ht), iy(ppl), iy(gd),
                         iy(cs), iy(bd), iy(ap), iy(ph), iy(hs), iy(sp), iy(sg), iy(my), round(prob, 1), risk)

            st.session_state.last_risk         = risk
            st.session_state.last_risk_prob    = round(prob, 1)
            st.session_state.last_patient_name = name
            st.session_state.last_gest_age     = int(ga)

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer(df_pred)
            shap_vals_full = shap_values.values[0]

            _model_cols = list(df_pred.columns)
            raw_shap = np.zeros(len(RAW_FEATURE_NAMES))
            for fi, fname in enumerate(RAW_FEATURE_NAMES):
                if fname in _model_cols:
                    raw_shap[fi] = shap_vals_full[_model_cols.index(fname)]

            high_risk_idx = sorted([i for i, v in enumerate(raw_shap) if v > 0],
                                   key=lambda i: raw_shap[i], reverse=True)
            st.session_state.last_high_risk_idx = high_risk_idx

            feat_values    = np.array([raw_shap[i] for i in high_risk_idx])
            total          = np.sum(np.abs(feat_values)) or 1
            percent_values = (feat_values / total) * 100
            feat_labels    = [RAW_FEATURE_NAMES[i] for i in high_risk_idx]

            st.subheader(tr("feature_contributions"))
            for i, val in enumerate(percent_values):
                st.write(f"{i+1}. **{feat_labels[i]}** — {val:.1f}%")

            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(feat_labels[::-1], percent_values[::-1], color='#9333ea')
            ax.set_xlabel(tr("contribution_pct"))
            ax.set_title(tr("risk_factor_contributions"))
            for bar, val in zip(bars, percent_values[::-1]):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            days      = get_checkup_days(risk)
            suggested = (datetime.now() + timedelta(days=days)).strftime("%B %d, %Y")
            risk_clr  = {"HIGH RISK": "#b00020", "MODERATE RISK": "#d97706", "LOW RISK": "#16a34a"}.get(risk, "#6b7280")

            st.markdown(f"""<div class="checkup-banner" style="margin-top:20px;">
<div style="font-size:15px;font-weight:700;color:#6d28d9;margin-bottom:4px;">
📅 {tr("checkup_recommended")}
</div>
<div style="font-size:13px;color:#4c1d95;margin-bottom:6px;">
Gestational age: <b style="color:#3b0764;">{ga} weeks</b> &nbsp;|&nbsp;
Risk: <b style="color:{risk_clr};">{risk}</b>
</div>
<div style="font-size:12px;color:#7c3aed;margin-bottom:4px;">⏰ Suggested next visit:</div>
<div class="checkup-date">{suggested}</div>
<div style="font-size:12px;color:#6d28d9;margin-top:8px;">
📋 View your full checkup calendar in the <b>Checkup Schedule</b> tab!
</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # ── FIX 2: APPOINTMENT REQUEST — outside any nested form, uses button not form_submit ──
            # Check if already has pending appointment
            if has_pending_appointment(st.session_state.username):
                st.markdown("""<div style="background:#fef3c7;border:1.5px solid #fbbf24;border-radius:12px;
padding:14px 18px;display:flex;align-items:center;gap:12px;">
<span style="font-size:22px;">⏳</span>
<div>
<div style="font-size:13px;font-weight:700;color:#92400e;">Appointment Request Pending</div>
<div style="font-size:12px;color:#78350f;margin-top:2px;">
Your request has been sent. Check <b>My Appointments</b> for updates.
</div>
</div>
</div>""", unsafe_allow_html=True)
            else:
                # Store prediction results in session for the appt form below
                st.session_state["_pred_risk"] = risk
                st.session_state["_pred_prob"] = round(prob, 1)
                st.session_state["_pred_name"] = name
                st.session_state["_pred_ga"]   = int(ga)
                st.session_state["_pred_suggested"] = suggested

        # ── FIX 2: APPOINTMENT FORM — at top level, shown when prediction results exist ──
        # This runs on every page render, not just when submit=True
        if (not submit and
            st.session_state.get("_pred_risk") and
            not has_pending_appointment(st.session_state.username)):

            risk_for_appt      = st.session_state["_pred_risk"]
            prob_for_appt      = st.session_state["_pred_prob"]
            name_for_appt      = st.session_state["_pred_name"]
            ga_for_appt        = st.session_state["_pred_ga"]
            suggested_for_appt = st.session_state["_pred_suggested"]

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown("""<div style="font-size:14px;font-weight:700;color:#3b0764;margin-bottom:8px;">
📩 Request an Appointment with Your Doctor</div>""", unsafe_allow_html=True)

            appt_note = st.text_area(
                tr("appt_note_label"),
                height=80,
                key="appt_note_input",
                placeholder="E.g. Prefer morning slots, questions about iron levels..."
            )
            if st.button(tr("request_appt_btn"), key="appt_request_btn", type="primary", use_container_width=True):
                result = create_appointment(
                    patient_username=st.session_state.username,
                    patient_name=name_for_appt,
                    risk=risk_for_appt,
                    risk_prob=prob_for_appt,
                    suggested_date=suggested_for_appt,
                    patient_note=appt_note.strip() if appt_note else "",
                    gestational_age=ga_for_appt
                )
                if result is True:
                    # FIX 2: Set flag, clear stored prediction, then rerun
                    st.session_state.show_appt_success = True
                    st.session_state.pop("_pred_risk", None)
                    st.session_state.pop("_pred_prob", None)
                    st.session_state.pop("_pred_name", None)
                    st.session_state.pop("_pred_ga", None)
                    st.session_state.pop("_pred_suggested", None)
                    st.rerun()
                else:
                    st.session_state.appt_error_msg = f"Could not save appointment. Error: {result}"
                    st.rerun()

        elif submit and st.session_state.get("_pred_risk") and not has_pending_appointment(st.session_state.username):
            # Just ran prediction — show the appointment form inline (same run)
            risk_for_appt      = risk
            prob_for_appt      = round(prob, 1)
            name_for_appt      = name
            ga_for_appt        = int(ga)
            suggested_for_appt = suggested

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            st.markdown("""<div style="font-size:14px;font-weight:700;color:#3b0764;margin-bottom:8px;">
📩 Request an Appointment with Your Doctor</div>""", unsafe_allow_html=True)
            st.info("💡 Fill in a note below and click **Request Appointment** to send to your doctor.")

    # ── ADVICE ───────────────────────────────────────────────────────────────
    elif st.session_state.nav == "advice":
        st.markdown(f"""<div class="main-header">
<span class="page-icon">💡</span>
<div>
<p class="page-title">{tr("advice_title")}</p>
<p class="page-sub">{tr("advice_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        advice_map = advice_map_ta if st.session_state.lang == "ta" else advice_map_en
        idx_list = st.session_state.last_high_risk_idx
        if idx_list:
            shown = False
            for i in idx_list:
                feat = feature_names[i]
                if feat in advice_map:
                    st.markdown(f"""<div class="stat-card" style="margin-bottom:12px;">
<b>🔸 {feat.replace('_', ' ').title()}</b><br>
<span style="color:#374151;font-size:14px;">{advice_map[feat]}</span>
</div>""", unsafe_allow_html=True)
                    shown = True
            if not shown:
                st.success(tr("no_high_risk"))
        else:
            st.info(tr("run_predict_first"))

    # ── CHAT ─────────────────────────────────────────────────────────────────
    elif st.session_state.nav == "chat":
        st.markdown(f"""<div class="main-header">
<span class="page-icon">🤖</span>
<div>
<p class="page-title">{tr("chat_title")}</p>
<p class="page-sub">{tr("chat_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        if not st.session_state.chat_history:
            st.markdown(f"""<div class="msg-ai">
<div class="msg-label">{tr("assistant")}</div>
{tr("chat_greeting")}
</div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""<div class="msg-user">
<div class="msg-label">{tr("you")}</div>
{safe_html(msg["content"])}
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="msg-ai">
<div class="msg-label">{tr("assistant")}</div>
{msg["content"].replace(chr(10), "<br>")}
</div>""", unsafe_allow_html=True)

        user_input = st.chat_input(tr("chat_placeholder"))
        if user_input and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
            reply = get_bot_response(user_input.strip(), st.session_state.lang)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_history:
            if st.button(tr("clear_chat"), key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown(f"<div class='disclaimer'>{tr('disclaimer')}</div>", unsafe_allow_html=True)

    # ── CHECKUP SCHEDULE ─────────────────────────────────────────────────────
    elif st.session_state.nav == "schedule":
        st.markdown(f"""<div class="main-header"
style="background:linear-gradient(135deg,#ede9fe,#fce7f3);border:1.5px solid #c4b5fd;">
<span class="page-icon">📋</span>
<div>
<p class="page-title" style="color:#6d28d9;">{tr("sched_title")}</p>
<p class="page-sub" style="color:#7c3aed;">{tr("sched_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        ga_to_use   = st.session_state.get("last_gest_age", 0)
        risk_to_use = st.session_state.get("last_risk", "")

        if ga_to_use == 0:
            try:
                recent = pd.read_sql(
                    "SELECT gestational_age, risk FROM patients WHERE name=? ORDER BY rowid DESC LIMIT 1",
                    conn, params=(st.session_state.username,)
                )
                if not recent.empty:
                    ga_to_use   = int(recent.iloc[0]["gestational_age"] or 0)
                    risk_to_use = str(recent.iloc[0]["risk"] or "LOW RISK")
            except Exception:
                pass

        if ga_to_use == 0:
            st.markdown(f"""<div style="text-align:center;padding:60px 20px;background:white;
border-radius:16px;box-shadow:0 2px 12px rgba(107,0,173,0.07);">
<div style="font-size:56px;margin-bottom:16px;">🗓️</div>
<div style="font-family:'DM Serif Display',serif;font-size:22px;color:#3b0764;margin-bottom:8px;">
No Schedule Yet
</div>
<div style="font-size:14px;color:#6b7280;">{tr("sched_no_data")}</div>
</div>""", unsafe_allow_html=True)
        else:
            render_checkup_schedule(ga_to_use, risk_to_use, st.session_state.lang)

    # ── MY APPOINTMENTS (PATIENT) ─────────────────────────────────────────────
    elif st.session_state.nav == "appointments":
        st.markdown(f"""<div class="main-header">
<span class="page-icon">📅</span>
<div>
<p class="page-title">{tr("my_appts_title")}</p>
<p class="page-sub">{tr("my_appts_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        PRENATAL_SCHEDULE = [
            (4,  8,  "First Visit",   "Confirm pregnancy, blood tests, dating scan"),
            (8,  12, "8-12 Weeks",    "NT scan, genetic screening, blood pressure check"),
            (12, 16, "12-16 Weeks",   "Results review, anomaly scan booking"),
            (16, 20, "16-20 Weeks",   "Anomaly scan, OGTT if needed"),
            (20, 24, "20-24 Weeks",   "Growth check, blood pressure, fundal height"),
            (24, 28, "24-28 Weeks",   "Glucose tolerance test, anti-D if Rh-ve"),
            (28, 32, "28-32 Weeks",   "Growth scan, fetal position check"),
            (32, 36, "32-36 Weeks",   "GBS swab, birth plan discussion"),
            (36, 40, "36-40 Weeks",   "Weekly checks, cervical assessment, delivery prep"),
        ]
        SCHED_COLORS = [
            ("#fce7f3","#9d174d"), ("#ffe5e5","#b00020"), ("#fff3cd","#856404"),
            ("#e6ffed","#15803d"), ("#e0f2fe","#0369a1"), ("#f3e8ff","#6d28d9"),
            ("#ecfdf5","#047857"), ("#fef9c3","#713f12"), ("#f0fdf4","#166534"),
        ]

        df_appts = get_patient_appointments(st.session_state.username)

        if df_appts.empty:
            st.markdown("""<div style="text-align:center;padding:48px 20px;background:white;
border-radius:14px;box-shadow:0 2px 12px rgba(107,0,173,0.07);">
<div style="font-size:52px;margin-bottom:12px;">📋</div>
<div style="font-size:17px;font-weight:700;color:#3b0764;margin-bottom:8px;">No Appointments Yet</div>
<div style="font-size:13px;color:#6b7280;">
Go to <b>PPH Risk Prediction</b>, run a scan, then tap
<b>Request Appointment</b> to send a request to your doctor.
</div>
</div>""", unsafe_allow_html=True)
        else:
            filter_opt = st.radio("Filter",
                [tr("all_appts"), tr("pending_appts"), tr("confirmed_appts")],
                horizontal=True, label_visibility="collapsed")
            df_show = df_appts.copy()
            if filter_opt == tr("pending_appts"):
                df_show = df_show[df_show.status == "pending"]
            elif filter_opt == tr("confirmed_appts"):
                df_show = df_show[df_show.status == "confirmed"]

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            for _, row in df_show.iterrows():
                status     = row.get("status", "pending")
                risk       = str(row.get("risk", "-"))
                risk_color = {"HIGH RISK":"#b00020","MODERATE RISK":"#856404","LOW RISK":"#15803d"}.get(risk,"#6b7280")
                badge_cls  = {"pending":"badge-pending","confirmed":"badge-confirmed","cancelled":"badge-cancelled"}.get(status,"badge-pending")
                card_cls   = {"pending":"appt-card pending","confirmed":"appt-card confirmed","cancelled":"appt-card cancelled"}.get(status,"appt-card pending")
                status_label = tr(f"appt_status_{status}")
                ga_val       = int(row.get("gestational_age") or 0)
                ga_label     = f"{ga_val} weeks" if ga_val else "-"
                p_name       = safe_html(str(row.get("patient_name", "-")))
                req_at       = str(row.get("requested_at", ""))[:16]
                sug_date     = str(row.get("suggested_date", "-"))
                risk_prob    = row.get("risk_prob", "-")

                # ── FIX 3: CONFIRMED APPOINTMENT BLOCK ──
                # Properly shows doctor-confirmed date/time and doctor note
                confirmed_html = ""
                if status == "confirmed":
                    conf_dt = str(row.get("confirmed_datetime", "") or "")
                    doc_note_val = str(row.get("doctor_note", "") or "")
                    doc_note_html = ""
                    if doc_note_val:
                        doc_note_html = f"<div style='font-size:13px;color:#047857;margin-top:8px;'>📝 <b>Doctor's note:</b> {safe_html(doc_note_val)}</div>"

                    if conf_dt:
                        confirmed_html = f"""<div style="margin-top:14px;background:linear-gradient(135deg,#d1fae5,#a7f3d0);
border-radius:12px;padding:18px 20px;border:2px solid #34d399;">
<div style="font-size:11px;font-weight:700;color:#065f46;letter-spacing:0.08em;
text-transform:uppercase;margin-bottom:8px;">✅ Confirmed Appointment</div>
<div style="font-size:24px;font-weight:700;color:#064e3b;">🗓 {safe_html(conf_dt)}</div>
{doc_note_html}
</div>"""
                    else:
                        confirmed_html = f"""<div style="margin-top:14px;background:linear-gradient(135deg,#d1fae5,#a7f3d0);
border-radius:12px;padding:18px 20px;border:2px solid #34d399;">
<div style="font-size:11px;font-weight:700;color:#065f46;letter-spacing:0.08em;
text-transform:uppercase;margin-bottom:8px;">✅ Appointment Confirmed</div>
<div style="font-size:14px;color:#047857;">Your doctor has confirmed your appointment. Please check with them for the exact time.</div>
{doc_note_html}
</div>"""

                pending_html = ""
                if status == "pending":
                    pending_html = """<div style="margin-top:12px;background:#fef9c3;border-radius:10px;
padding:12px 16px;border:1px solid #fcd34d;">
<div style="font-size:13px;color:#78350f;">
⏳ <b>Waiting for doctor to confirm</b> — confirmed date and time will appear here once set.
</div>
</div>"""

                cancelled_html = ""
                if status == "cancelled":
                    cancelled_html = """<div style="margin-top:12px;background:#f3f4f6;border-radius:10px;
padding:12px 16px;border:1px solid #d1d5db;">
<div style="font-size:13px;color:#6b7280;">🚫 This appointment request was cancelled.</div>
</div>"""

                patient_note_html = ""
                if row.get("patient_note"):
                    patient_note_html = f"<div style='font-size:13px;color:#6b7280;margin-top:8px;'>💬 <b>Your note:</b> {safe_html(str(row['patient_note']))}</div>"

                sched_rows = ""
                for (s, e, lbl, desc), (bg, fg) in zip(PRENATAL_SCHEDULE, SCHED_COLORS):
                    is_done    = ga_val >= e
                    is_current = s <= ga_val < e
                    border     = f"2px solid {fg}" if is_current else ("1px solid #e5e7eb" if is_done else f"1px solid {fg}55")
                    opacity    = "0.4" if is_done else "1"
                    tick       = "✅ " if is_done else ("▶ " if is_current else "")
                    curr_badge = (f"<span style='background:{fg};color:white;font-size:10px;"
                                  f"font-weight:700;padding:2px 8px;border-radius:20px;margin-left:6px;'>NOW</span>") if is_current else ""
                    safe_lbl  = safe_html(lbl)
                    safe_desc = safe_html(desc)
                    sched_rows += f"""<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;
background:{bg if not is_done else "#f9fafb"};border-radius:10px;
border:{border};margin-bottom:7px;opacity:{opacity};">
<div style="min-width:46px;text-align:center;padding-top:2px;">
<div style="font-size:9px;font-weight:700;color:{fg if not is_done else "#9ca3af"};text-transform:uppercase;">Wk</div>
<div style="font-size:14px;font-weight:700;color:{fg if not is_done else "#9ca3af"};">{s}-{e}</div>
</div>
<div style="flex:1;">
<div style="font-size:13px;font-weight:700;color:{"#9ca3af" if is_done else "#1a1a2e"};">
{tick}{safe_lbl}{curr_badge}
</div>
<div style="font-size:12px;color:{"#9ca3af" if is_done else "#6b7280"};margin-top:2px;">{safe_desc}</div>
</div>
</div>"""

                st.markdown(f"""<div class="{card_cls}">
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
<div>
<div style="font-family:'DM Serif Display',serif;font-size:19px;color:#1a1a2e;">👤 {p_name}</div>
<div style="font-size:12px;color:#6b7280;margin-top:3px;">
Requested: {req_at} &nbsp;|&nbsp; Gestational age: <b>{ga_label}</b>
</div>
</div>
<div style="text-align:right;">
<span class="appt-badge {badge_cls}">{status_label}</span>
<div style="font-size:12px;font-weight:700;color:{risk_color};margin-top:5px;">
{risk} &nbsp;|&nbsp; {risk_prob}%
</div>
</div>
</div>
<div style="margin-top:10px;padding:10px 14px;background:#f5f3ff;border-radius:8px;">
<span style="font-size:12px;color:#7c3aed;font-weight:600;">⏰ Suggested visit date:</span>
<span style="font-size:13px;color:#3b0764;font-weight:700;margin-left:6px;">{sug_date}</span>
</div>
{patient_note_html}
{pending_html}
{confirmed_html}
{cancelled_html}
<div style="margin-top:16px;">
<div style="font-size:13px;font-weight:700;color:#3b0764;margin-bottom:10px;">
🗓️ Prenatal Checkup Schedule
<span style="font-size:12px;font-weight:400;color:#6b7280;margin-left:8px;">(based on {ga_label} gestational age)</span>
</div>
{sched_rows}
</div>
</div>""", unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# DOCTOR DASHBOARD
# ═════════════════════════════════════════════════════════════
elif st.session_state.role == "Doctor":

    with st.sidebar:
        st.markdown("""<div class="sidebar-logo">
<div class="logo-icon">🏥</div>
<div class="logo-title">PPH Predict</div>
<div class="logo-sub">Doctor Dashboard</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='nav-section-label'>🌐 Language / மொழி</div>", unsafe_allow_html=True)
        ls2 = st.radio("ls2", ["English", "தமிழ்"], horizontal=True, label_visibility="collapsed",
                       index=0 if st.session_state.lang == "en" else 1, key="sl2")
        st.session_state.lang = "en" if ls2 == "English" else "ta"

        st.markdown(f"<div class='nav-section-label'>{tr('reports')}</div>", unsafe_allow_html=True)
        for _key, _icon, _label in [
            ("patients",         "👥", tr("patients_nav")),
            ("highrisk",         "⚠️", tr("highrisk_nav")),
            ("doc_appointments", "📅", tr("doc_appts_nav")),
        ]:
            _prefix = "▶  " if st.session_state.nav == _key else "   "
            if st.button(f"{_prefix}{_icon}  {_label}", key=f"dnav_{_key}"):
                st.session_state.nav = _key
                st.rerun()

        try:
            _pending_count = cursor.execute(
                "SELECT COUNT(*) FROM appointments WHERE status='pending'"
            ).fetchone()[0]
        except Exception:
            _pending_count = 0
        if _pending_count > 0:
            st.markdown(f"""<div style="margin:4px 20px;background:#fef3c7;border-radius:8px;
padding:6px 12px;font-size:12px;color:#92400e;">
⏳ {_pending_count} pending request{"s" if _pending_count > 1 else ""}
</div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='nav-section-label'>{tr('account')}</div>", unsafe_allow_html=True)
        if st.button(f"   🚪  {tr('logout')}", key="dnav_logout"):
            st.session_state.login = False
            st.rerun()

    st.markdown("""<style>
.patient-card{background:white;border-radius:14px;padding:20px 24px;margin-bottom:16px;box-shadow:0 2px 14px rgba(107,0,173,0.07);border-left:5px solid #9333ea;}
.patient-card.high{border-left-color:#b00020;background:#fffafa;}
.patient-card.mod{border-left-color:#d97706;background:#fffdf5;}
.patient-card.low{border-left-color:#16a34a;background:#f6fff9;}
.risk-badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;}
.badge-high{background:#ffe5e5;color:#b00020;}
.badge-mod{background:#fff3cd;color:#856404;}
.badge-low{background:#e6ffed;color:#16a34a;}
.card-name{font-family:'DM Serif Display',serif;font-size:20px;color:#1a1a2e;margin-bottom:4px;}
.detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px;}
.detail-item{background:#f8f4ff;border-radius:8px;padding:10px 14px;}
.detail-item.warn{background:#fff0f0;}
.detail-label{font-size:11px;color:#6b7280;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;}
.detail-value{font-size:15px;font-weight:700;color:#3b0764;margin-top:2px;}
.detail-item.warn .detail-value{color:#b00020;}
.factors-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px;}
.factor-chip{border-radius:6px;padding:5px 10px;font-size:12px;font-weight:600;text-align:center;}
.factor-chip.yes{background:#ffe5e5;color:#b00020;}
.factor-chip.no{background:#f0fdf4;color:#15803d;}
.alert-banner{background:#fff0f0;border:1.5px solid #fca5a5;border-radius:8px;padding:8px 14px;margin-top:10px;font-size:13px;color:#b00020;font-weight:500;}
.section-divider{border:none;border-top:1.5px solid #ede9fe;margin:10px 0 14px;}
</style>""", unsafe_allow_html=True)

    cursor.execute("DELETE FROM patients WHERE gestational_age IS NULL AND risk_prob IS NULL")
    conn.commit()
    df_all = pd.read_sql("SELECT rowid AS id, * FROM patients ORDER BY rowid DESC", conn)

    def highlight_risk(val):
        return {"HIGH RISK": "background-color:#ffcccc", "MODERATE RISK": "background-color:#fff3cd"}.get(val, "background-color:#e6ffed")

    def risk_badge_html(risk):
        if risk == "HIGH RISK":
            return f"<span class='risk-badge badge-high'>🔴 {tr('high_risk_label')}</span>"
        if risk == "MODERATE RISK":
            return f"<span class='risk-badge badge-mod'>🟡 {tr('mod_risk_label')}</span>"
        return f"<span class='risk-badge badge-low'>🟢 {tr('low_risk_label')}</span>"

    def card_class(risk):
        return {"HIGH RISK": "high", "MODERATE RISK": "mod", "LOW RISK": "low"}.get(risk, "low")

    FACTOR_LABELS = {
        "anemia": "Anemia", "pre_pph": "Prev PPH", "multiple_pregnancy": "Multiple Preg",
        "pre_hypertension": "Hypertension", "placenta_previa": "Placenta Previa",
        "gest_diabetes": "Gest Diabetes", "pre_c_section": "Prev C-Section",
        "pre_blood": "Blood Disorder", "abnormal_placenta": "Abnml Placenta",
        "polyhydromnios": "Polyhydramnios", "hellp_syndrome": "HELLP",
        "severe_preeclampsia": "Sev Preeclampsia", "surgery": "Surgery", "myoma": "Myoma",
    }

    def render_patient_cards(df, key_prefix="card", show_remove=False):
        if df.empty:
            st.info(tr("no_patients"))
            return
        for _, row in df.iterrows():
            risk   = str(row.get("risk", "-")) if row.get("risk") else "-"
            name   = str(row.get("name", "-")) if row.get("name") else "-"
            pid    = row.get("id", None)
            age_v  = int(row["age"])                    if pd.notna(row.get("age"))             else None
            bmi_v  = round(float(row["bmi"]), 1)        if pd.notna(row.get("bmi"))             else None
            hb_v   = round(float(row["hemoglobin"]), 1) if pd.notna(row.get("hemoglobin"))      else None
            ga_v   = int(row["gestational_age"])         if pd.notna(row.get("gestational_age")) else None
            par_v  = int(row["parity"])                  if pd.notna(row.get("parity"))          else None
            prob_v = float(row["risk_prob"])             if pd.notna(row.get("risk_prob"))       else None
            created = str(row.get("created_at", ""))[:16] if row.get("created_at") else ""
            cc = card_class(risk)

            hb_flag  = (hb_v  is not None) and (hb_v  < 11.0)
            bmi_flag = (bmi_v is not None) and not (18.5 <= bmi_v <= 30)

            prob_str = f"{prob_v:.1f}%" if prob_v is not None else "-"
            age_str  = f"{age_v} {tr('age_unit')}"   if age_v  is not None else "-"
            ga_str   = f"{ga_v} {tr('weeks_unit')}"   if ga_v   is not None else "-"
            par_str  = str(par_v) if par_v is not None else "-"
            bmi_str  = f"{bmi_v} {'⚠' if bmi_flag else ''}".strip()
            hb_str   = f"{hb_v} g/dL {'⚠' if hb_flag else ''}".strip()
            bmi_cls  = "detail-item warn" if bmi_flag else "detail-item"
            hb_cls   = "detail-item warn" if hb_flag  else "detail-item"

            alert_parts = []
            if risk == "HIGH RISK":  alert_parts.append(tr("immediate_attention"))
            if hb_flag:              alert_parts.append(tr("anaemia_alert").format(hb_v))
            if bmi_flag:             alert_parts.append(tr("bmi_alert").format(bmi_v))
            alert_html = "".join(f"<div class='alert-banner'>{a}</div>" for a in alert_parts)

            chips_html = "<div class='factors-grid'>"
            for fcol, lbl in FACTOR_LABELS.items():
                fval = row.get(fcol, 0)
                try:    active = int(fval) == 1
                except: active = False
                chips_html += f"<div class='factor-chip {'yes' if active else 'no'}'>{lbl}</div>"
            chips_html += "</div>"

            safe_name = safe_html(name)
            rp_label  = tr("risk_prob")

            st.markdown(
                f"<div class='patient-card {cc}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                f"<div><div class='card-name'>👤 {safe_name}</div>"
                f"<div style='font-size:12px;color:#6b7280;margin-top:2px;'>"
                f"{age_str} | {ga_str} | {created}</div></div>"
                f"<div style='text-align:right;'>{risk_badge_html(risk)}<br>"
                f"<span style='font-size:13px;color:#6b7280;margin-top:4px;display:block;'>"
                f"{rp_label}: <b>{prob_str}</b></span></div>"
                f"</div>"
                f"<div class='detail-grid'>"
                f"<div class='{bmi_cls}'><div class='detail-label'>BMI</div><div class='detail-value'>{bmi_str}</div></div>"
                f"<div class='{hb_cls}'><div class='detail-label'>Hemoglobin</div><div class='detail-value'>{hb_str}</div></div>"
                f"<div class='detail-item'><div class='detail-label'>Gest. Age</div><div class='detail-value'>{ga_str}</div></div>"
                f"<div class='detail-item'><div class='detail-label'>Parity</div><div class='detail-value'>{par_str}</div></div>"
                f"</div><hr class='section-divider'>"
                f"<div style='font-size:11px;color:#6b7280;font-weight:700;letter-spacing:0.08em;"
                f"text-transform:uppercase;margin-bottom:6px;'>{tr('clinical_risk_factors')}</div>"
                f"{chips_html}{alert_html}</div>",
                unsafe_allow_html=True
            )

            if show_remove and pid is not None:
                if st.button(tr("remove") + name, key=f"{key_prefix}_del_{pid}"):
                    delete_patient(pid)
                    st.success(tr("removed"))
                    st.rerun()

    # ── ALL PATIENTS ──────────────────────────────────────────────────────────
    if st.session_state.nav in ("patients", ""):
        st.markdown(f"""<div class="main-header">
<span class="page-icon">👥</span>
<div>
<p class="page-title">{tr("all_patients_title")}</p>
<p class="page-sub">{tr("all_patients_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        total = len(df_all)
        high  = len(df_all[df_all.risk == "HIGH RISK"])
        mod   = len(df_all[df_all.risk == "MODERATE RISK"])
        low   = len(df_all[df_all.risk == "LOW RISK"])
        for _col, _lbl, _val, _clr in [
            (c1, tr("total_patients"),  total, "#6a0dad"),
            (c2, tr("high_risk_label"), high,  "#b00020"),
            (c3, tr("mod_risk_label"),  mod,   "#856404"),
            (c4, tr("low_risk_label"),  low,   "#006400"),
        ]:
            with _col:
                st.markdown(f"""<div class="stat-card" style="border-left-color:{_clr}">
<div style="font-size:28px;font-weight:700;color:{_clr}">{_val}</div>
<div style="font-size:13px;color:#6b7280;margin-top:2px">{_lbl}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_view, col_sort, col_clear = st.columns([2, 2, 1])
        with col_view:
            view_mode = st.radio(tr("view"), [tr("cards_view"), tr("table_view")],
                                 horizontal=True, label_visibility="collapsed")
        with col_sort:
            sort_opts = [tr("newest_first"), tr("risk_high_low"), tr("name_az")]
            sort_by   = st.selectbox(tr("sort_by"), sort_opts, label_visibility="collapsed")
        with col_clear:
            if st.button(tr("clear_all"), key="clear_all_patients"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning(tr("confirm_delete_all"))
            ca, cb = st.columns(2)
            with ca:
                if st.button(tr("yes_delete"), key="confirm_yes"):
                    clear_all_patients()
                    st.session_state["confirm_clear"] = False
                    st.success(tr("all_cleared"))
                    st.rerun()
            with cb:
                if st.button(tr("cancel"), key="confirm_no"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

        df_view = df_all.copy()
        if sort_by == tr("risk_high_low"):
            df_view["_s"] = df_view["risk"].map({"HIGH RISK": 0, "MODERATE RISK": 1, "LOW RISK": 2}).fillna(3)
            df_view = df_view.sort_values("_s").drop(columns="_s")
        elif sort_by == tr("name_az"):
            df_view = df_view.sort_values("name", key=lambda x: x.str.lower())

        if view_mode == tr("cards_view"):
            render_patient_cards(df_view, key_prefix="all", show_remove=False)
        else:
            dc = ["id", "name", "age", "bmi", "hemoglobin", "gestational_age", "parity", "risk_prob", "risk", "created_at"]
            dc = [c for c in dc if c in df_view.columns]
            st.dataframe(df_view[dc].style.map(highlight_risk, subset=["risk"]), use_container_width=True)

        csv = df_all.to_csv(index=False).encode()
        st.download_button(tr("download_csv"), csv, "patient_data.csv", "text/csv")

    # ── HIGH RISK ─────────────────────────────────────────────────────────────
    elif st.session_state.nav == "highrisk":
        st.markdown(f"""<div class="main-header"
style="background:linear-gradient(135deg,#fff0f0,#fce4e4);border:1.5px solid #fca5a5;">
<span class="page-icon">⚠️</span>
<div>
<p class="page-title" style="color:#b00020;">{tr("highrisk_title")}</p>
<p class="page-sub" style="color:#9b1c1c;">{tr("highrisk_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        df_hm = df_all[df_all.risk.isin(["HIGH RISK", "MODERATE RISK"])].copy()
        df_hm["_s"] = df_hm["risk"].map({"HIGH RISK": 0, "MODERATE RISK": 1}).fillna(2)
        df_hm = df_hm.sort_values("_s").drop(columns="_s")

        if df_hm.empty:
            st.success(tr("no_highrisk"))
        else:
            h_count = len(df_hm[df_hm.risk == "HIGH RISK"])
            m_count = len(df_hm[df_hm.risk == "MODERATE RISK"])
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"""<div class="stat-card" style="border-left-color:#b00020">
<div style="font-size:28px;font-weight:700;color:#b00020">{h_count}</div>
<div style="font-size:13px;color:#6b7280">{tr("high_risk_label")}</div>
</div>""", unsafe_allow_html=True)
            with cb:
                st.markdown(f"""<div class="stat-card" style="border-left-color:#d97706">
<div style="font-size:28px;font-weight:700;color:#d97706">{m_count}</div>
<div style="font-size:13px;color:#6b7280">{tr("mod_risk_label")}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            render_patient_cards(df_hm, key_prefix="hr", show_remove=False)
            csv = df_hm.to_csv(index=False).encode()
            st.download_button(tr("download_csv"), csv, "high_risk_patients.csv", "text/csv")

    # ── DOCTOR: APPOINTMENT REQUESTS ─────────────────────────────────────────
    elif st.session_state.nav == "doc_appointments":
        st.markdown(f"""<div class="main-header"
style="background:linear-gradient(135deg,#f0fdf4,#d1fae5);border:1.5px solid #6ee7b7;">
<span class="page-icon">📅</span>
<div>
<p class="page-title" style="color:#065f46;">{tr("doc_appts_title")}</p>
<p class="page-sub" style="color:#047857;">{tr("doc_appts_sub")}</p>
</div>
</div>""", unsafe_allow_html=True)

        df_all_appts = get_all_appointments()

        if df_all_appts.empty:
            st.markdown("""<div style="text-align:center;padding:60px 20px;background:white;
border-radius:16px;box-shadow:0 2px 12px rgba(107,0,173,0.07);">
<div style="font-size:52px;margin-bottom:12px;">📭</div>
<div style="font-size:18px;font-weight:700;color:#3b0764;margin-bottom:8px;">No Appointment Requests</div>
<div style="font-size:14px;color:#6b7280;">Patient requests will appear here once submitted.</div>
</div>""", unsafe_allow_html=True)
        else:
            filter_tab = st.radio(
                "Filter",
                [tr("all_appts"), tr("pending_appts"), tr("confirmed_appts")],
                horizontal=True,
                label_visibility="collapsed"
            )
            df_show = df_all_appts.copy()
            if filter_tab == tr("pending_appts"):
                df_show = df_show[df_show.status == "pending"]
            elif filter_tab == tr("confirmed_appts"):
                df_show = df_show[df_show.status == "confirmed"]

            total_req   = len(df_all_appts)
            pending_req = len(df_all_appts[df_all_appts.status == "pending"])
            confirmed_r = len(df_all_appts[df_all_appts.status == "confirmed"])
            s1, s2, s3  = st.columns(3)
            for _sc, _lbl, _val, _clr in [
                (s1, "Total Requests",  total_req,   "#6a0dad"),
                (s2, "Pending",         pending_req, "#d97706"),
                (s3, "Confirmed",       confirmed_r, "#16a34a"),
            ]:
                with _sc:
                    st.markdown(f"""<div class="stat-card" style="border-left-color:{_clr};margin-bottom:16px;">
<div style="font-size:26px;font-weight:700;color:{_clr}">{_val}</div>
<div style="font-size:13px;color:#6b7280">{_lbl}</div>
</div>""", unsafe_allow_html=True)

            if df_show.empty:
                st.info("No requests in this category.")
            else:
                for _, appt in df_show.iterrows():
                    appt_id    = int(appt.get("id", 0))
                    p_username = str(appt.get("patient_username", "-"))
                    p_name     = str(appt.get("patient_name", "-"))
                    risk       = str(appt.get("risk", "-"))
                    risk_prob  = appt.get("risk_prob", 0)
                    ga_val     = int(appt.get("gestational_age") or 0)
                    sug_date   = str(appt.get("suggested_date", "-"))
                    pat_note   = str(appt.get("patient_note", "") or "")
                    status     = str(appt.get("status", "pending"))
                    req_at     = str(appt.get("requested_at", ""))[:16]
                    conf_dt    = str(appt.get("confirmed_datetime", "") or "")
                    doc_note   = str(appt.get("doctor_note", "") or "")

                    risk_clr   = {"HIGH RISK":"#b00020","MODERATE RISK":"#d97706","LOW RISK":"#16a34a"}.get(risk,"#6b7280")
                    card_border = {"HIGH RISK":"#fca5a5","MODERATE RISK":"#fcd34d","LOW RISK":"#6ee7b7"}.get(risk,"#c4b5fd")
                    card_bg    = {"HIGH RISK":"#fffafa","MODERATE RISK":"#fffdf5","LOW RISK":"#f6fff9"}.get(risk,"white")

                    pat_note_html = ""
                    if pat_note:
                        pat_note_html = f"""<div style="background:#f5f3ff;border-radius:8px;padding:10px 14px;margin-top:10px;border-left:3px solid #a78bfa;">
<span style="font-size:11px;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:0.06em;">💬 Patient's Note</span>
<div style="font-size:13px;color:#374151;margin-top:4px;">{safe_html(pat_note)}</div>
</div>"""

                    confirmed_html = ""
                    if status == "confirmed" and conf_dt:
                        doc_note_html = ""
                        if doc_note:
                            doc_note_html = f"<div style='font-size:13px;color:#047857;margin-top:6px;'>📝 <b>Your note to patient:</b> {safe_html(doc_note)}</div>"
                        confirmed_html = f"""<div style="margin-top:12px;background:linear-gradient(135deg,#d1fae5,#a7f3d0);
border-radius:12px;padding:14px 18px;border:2px solid #34d399;">
<div style="font-size:11px;font-weight:700;color:#065f46;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">✅ Confirmed</div>
<div style="font-size:20px;font-weight:700;color:#064e3b;">🗓 {safe_html(conf_dt)}</div>
{doc_note_html}
</div>"""

                    cancelled_html = ""
                    if status == "cancelled":
                        cancelled_html = """<div style="margin-top:12px;background:#f3f4f6;border-radius:10px;padding:12px 16px;border:1px solid #d1d5db;">
<div style="font-size:13px;color:#6b7280;">🚫 This request has been cancelled.</div>
</div>"""

                    st.markdown(f"""<div style="background:{card_bg};border:2px solid {card_border};
border-radius:16px;padding:20px 24px;margin-bottom:16px;
box-shadow:0 2px 12px rgba(107,0,173,0.06);">
<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
<div>
<div style="font-family:'DM Serif Display',serif;font-size:20px;color:#1a1a2e;">👤 {safe_html(p_name)}</div>
<div style="font-size:12px;color:#6b7280;margin-top:3px;">
@{safe_html(p_username)} &nbsp;|&nbsp; Requested: {req_at}
</div>
</div>
<div style="text-align:right;">
<span style="background:{risk_clr};color:white;font-size:11px;font-weight:700;
padding:4px 14px;border-radius:20px;text-transform:uppercase;">{risk}</span>
<div style="font-size:13px;color:{risk_clr};font-weight:700;margin-top:4px;">{risk_prob}% risk probability</div>
</div>
</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px;">
<div style="background:white;border-radius:8px;padding:10px 14px;border:1px solid #e5e7eb;">
<div style="font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Gestational Age</div>
<div style="font-size:16px;font-weight:700;color:#3b0764;">{ga_val} weeks</div>
</div>
<div style="background:white;border-radius:8px;padding:10px 14px;border:1px solid #e5e7eb;">
<div style="font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Suggested Date</div>
<div style="font-size:14px;font-weight:700;color:#6d28d9;">{safe_html(sug_date)}</div>
</div>
<div style="background:white;border-radius:8px;padding:10px 14px;border:1px solid #e5e7eb;">
<div style="font-size:10px;color:#6b7280;font-weight:700;text-transform:uppercase;">Status</div>
<div style="font-size:14px;font-weight:700;color:{"#d97706" if status=="pending" else "#16a34a" if status=="confirmed" else "#6b7280"};">
{"⏳ Pending" if status=="pending" else "✅ Confirmed" if status=="confirmed" else "🚫 Cancelled"}
</div>
</div>
</div>
{pat_note_html}
{confirmed_html}
{cancelled_html}
</div>""", unsafe_allow_html=True)

                    # Doctor actions — only for PENDING appointments
                    if status == "pending":
                        with st.expander(f"✅ Confirm or Cancel — {p_name}", expanded=False):
                            st.markdown("""<div style="background:#f0fdf4;border-radius:10px;padding:14px 16px;
margin-bottom:12px;border:1px solid #bbf7d0;">
<div style="font-size:13px;font-weight:700;color:#065f46;margin-bottom:4px;">
Set Appointment Date & Time
</div>
<div style="font-size:12px;color:#047857;">
Choose the confirmed appointment slot for this patient.
</div>
</div>""", unsafe_allow_html=True)

                            col_date, col_time = st.columns(2)
                            with col_date:
                                appt_date = st.date_input(
                                    "Appointment Date",
                                    value=datetime.now().date() + timedelta(days=3),
                                    min_value=datetime.now().date(),
                                    key=f"appt_date_{appt_id}"
                                )
                            with col_time:
                                appt_time = st.time_input(
                                    "Appointment Time",
                                    value=datetime.strptime("10:00", "%H:%M").time(),
                                    key=f"appt_time_{appt_id}"
                                )

                            doc_note_input = st.text_area(
                                tr("appt_doc_note_label"),
                                placeholder="E.g. Please bring previous scan reports, fasting required...",
                                key=f"doc_note_{appt_id}",
                                height=80
                            )



                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button(
                                    f"✅ {tr('confirm_appt_btn')}",
                                    key=f"confirm_{appt_id}",
                                    type="primary",
                                    use_container_width=True
                                ):
                                    confirmed_dt = f"{appt_date.strftime('%B %d, %Y')} at {appt_time.strftime('%I:%M %p')}"
                                    confirm_appointment(appt_id, confirmed_dt, doc_note_input.strip())
                                    st.success(f"✅ {tr('appt_confirmed_success')} — {p_name} on {confirmed_dt}")
                                    st.rerun()

                            with btn_col2:
                                if st.button(
                                    f"🚫 {tr('cancel_appt_btn')}",
                                    key=f"cancel_{appt_id}",
                                    use_container_width=True
                                ):
                                    cancel_appointment(appt_id)
                                    st.warning(f"Request from {p_name} has been cancelled.")
                                    st.rerun()

                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)