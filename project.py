import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import io

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="PEL – Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BG_IMAGE_B64 = "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"

# ─── LANGUAGE ──────────────────────────────────────────────────
LANG = {
    "en": {
        "title": "PEL – AI Predictive Maintenance System",
        "subtitle": "Petroleum Exploration (Pvt.) Ltd. · Karachi, Pakistan",
        "login_btn": "Sign In",
        "password": "Password",
        "wrong_pw": "Incorrect password. Please try again.",
        "tabs": ["Executive Summary", "Live Dashboard", "30-Day Forecast",
                 "Alerts & Actions", "HSE & Environment", "Settings", "About PEL"],
        "logout": "Sign Out",
    },
    "ur": {
        "title": "PEL – AI مشین دیکھ بھال نظام",
        "subtitle": "پٹرولیم ایکسپلوریشن پرائیویٹ لمیٹڈ · کراچی",
        "login_btn": "لاگ ان",
        "password": "پاس ورڈ",
        "wrong_pw": "غلط پاس ورڈ۔ دوبارہ کوشش کریں۔",
        "tabs": ["خلاصہ", "لائیو ڈیش بورڈ", "پیشگوئی",
                 "الرٹس", "ماحولیات", "ترتیبات", "PEL کے بارے میں"],
        "logout": "لاگ آؤٹ",
    }
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"
L = LANG[st.session_state.lang]

# ─── ROLES ─────────────────────────────────────────────────────
USERS = {
    "admin":    {"password": "pel2025",      "role": "Admin"},
    "engineer": {"password": "engineer123",  "role": "Engineer"},
    "viewer":   {"password": "view2025",     "role": "Viewer"},
}

# ─── LOGIN ─────────────────────────────────────────────────────
def check_password():
    def do_login():
        u = st.session_state.get("username_input", "").strip().lower()
        p = st.session_state.get("password_input", "")
        if u in USERS and USERS[u]["password"] == p:
            st.session_state["authenticated"] = True
            st.session_state["current_user"]  = u
            st.session_state["current_role"]  = USERS[u]["role"]
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated"):
        return True

    # ── Login page CSS ──
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, .stApp {{
        font-family: 'Inter', sans-serif !important;
        background: #f0f2f5;
    }}
    .stApp {{
        background: linear-gradient(rgba(15,30,60,0.82),rgba(10,20,50,0.88)),
                    url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    .login-wrap {{
        display: flex; justify-content: center; align-items: center;
        min-height: 80vh; padding: 20px;
    }}
    .login-card {{
        background: #ffffff; border-radius: 12px;
        padding: 48px 44px; width: 100%; max-width: 420px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }}
    .login-logo {{ text-align: center; margin-bottom: 24px; }}
    .login-title {{ font-size: 20px; font-weight: 700; color: #1a2a4a; text-align: center; margin-bottom: 4px; }}
    .login-sub {{ font-size: 13px; color: #6b7280; text-align: center; margin-bottom: 32px; }}
    .login-divider {{ height: 1px; background: #e5e7eb; margin: 20px 0; }}
    .stTextInput label {{ font-size: 13px; font-weight: 500; color: #374151 !important; }}
    .stTextInput input {{
        border-radius: 8px !important; border: 1px solid #d1d5db !important;
        font-size: 14px !important; padding: 10px 14px !important;
        background: #f9fafb !important; color: #111827 !important;
    }}
    .stTextInput input:focus {{ border-color: #1a3a6b !important; box-shadow: 0 0 0 3px rgba(26,58,107,0.12) !important; }}
    .stButton > button {{
        background: #1a3a6b !important; color: #ffffff !important;
        border-radius: 8px !important; border: none !important;
        font-weight: 600 !important; font-size: 14px !important;
        padding: 10px !important; width: 100%;
        transition: background 0.2s ease;
    }}
    .stButton > button:hover {{ background: #15306b !important; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 30px !important; }}
    </style>""", unsafe_allow_html=True)

    st.markdown("""<div class="login-wrap"><div class="login-card">
      <div class="login-logo">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s"
             width="80" style="border-radius:8px;">
      </div>
      <div class="login-title">PEL Maintenance System</div>
      <div class="login-sub">Petroleum Exploration (Pvt.) Ltd.</div>
      <div class="login-divider"></div>
    </div></div>""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1,2,1])
    with col_c:
        st.text_input("Username", key="username_input", placeholder="e.g. admin")
        st.text_input("Password", type="password", key="password_input",
                      placeholder="Enter password", on_change=do_login)
        st.button("Sign In →", on_click=do_login, use_container_width=True)
        if "authenticated" in st.session_state and not st.session_state["authenticated"]:
            st.error("Incorrect credentials. Please try again.")
        st.markdown("""<div style='text-align:center;margin-top:16px;font-size:12px;color:#9ca3af;'>
            Admin: admin / pel2025 &nbsp;|&nbsp; Viewer: viewer / view2025</div>""",
            unsafe_allow_html=True)
    return False

if not check_password():
    st.stop()

role = st.session_state.get("current_role", "Viewer")
can_edit = role in ["Admin", "Engineer"]

# ─── MAIN CSS (Professional Light Theme) ───────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {{ font-family: 'Inter', sans-serif !important; color: #111827; }}
    .stApp {{
        background: linear-gradient(rgba(240,242,245,0.93), rgba(240,242,245,0.93)),
                    url("data:image/jpeg;base64,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");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

/* Top navy bar */
.stApp::before {{
    content:''; position:fixed; top:0; left:0; width:100%; height:4px;
    background: linear-gradient(90deg, #1a3a6b, #c0392b, #1a3a6b);
    background-size: 200% 100%; animation: topBar 6s linear infinite; z-index:9999;
}}
@keyframes topBar {{ 0%{{background-position:0%}} 100%{{background-position:200%}} }}

/* Cards */
.kpi-card {{
    background: #ffffff; border-radius: 10px; padding: 20px 22px;
    border-top: 3px solid #1a3a6b;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.12); }}
.kpi-label {{ font-size: 12px; font-weight: 500; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi-value {{ font-size: 30px; font-weight: 700; color: #111827; margin: 6px 0 4px; line-height: 1; }}
.kpi-delta {{ font-size: 12px; font-weight: 500; color: #059669; }}
.kpi-delta.warn {{ color: #d97706; }}
.kpi-delta.crit {{ color: #dc2626; }}

/* Section card */
.section-card {{
    background: #ffffff; border-radius: 10px; padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px;
}}

/* Alert banners */
.alert-crit {{
    background: #fef2f2; border-left: 4px solid #dc2626;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    color: #7f1d1d; font-size: 14px; font-weight: 500;
}}
.alert-warn {{
    background: #fffbeb; border-left: 4px solid #d97706;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    color: #78350f; font-size: 14px; font-weight: 500;
}}
.alert-ok {{
    background: #f0fdf4; border-left: 4px solid #16a34a;
    border-radius: 0 8px 8px 0; padding: 14px 18px;
    color: #14532d; font-size: 14px; font-weight: 500;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: #ffffff; border-radius: 8px; padding: 4px;
    border: 1px solid #e5e7eb; gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    color: #6b7280 !important; font-size: 13px !important;
    font-weight: 500 !important; border-radius: 6px !important;
    padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: #1a3a6b !important; color: #ffffff !important;
}}

/* Metric override */
div[data-testid="metric-container"] {{
    background: #ffffff !important; border-radius: 10px !important;
    border-top: 3px solid #1a3a6b !important; padding: 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}}
div[data-testid="metric-container"] label {{ color: #6b7280 !important; font-size: 12px !important; font-family: Inter !important; text-transform: uppercase; letter-spacing: 0.5px; }}
div[data-testid="metric-container"] > div > div:nth-child(2) {{ color: #111827 !important; font-size: 28px !important; font-weight: 700 !important; }}
div[data-testid="metric-delta"] {{ color: #059669 !important; }}

/* Buttons */
.stButton > button {{
    background: #1a3a6b !important; color: #fff !important;
    border-radius: 8px !important; border: none !important;
    font-weight: 600 !important; font-size: 13px !important;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{ background: #15306b !important; box-shadow: 0 4px 12px rgba(26,58,107,0.3) !important; }}

/* Input fields */
.stTextInput input, .stNumberInput input, .stSelectbox select {{
    border-radius: 8px !important; border: 1px solid #d1d5db !important;
    font-size: 13px !important; background: #f9fafb !important;
}}

/* Sidebar / dataframe */
.stDataFrame {{ border-radius: 8px; overflow: hidden; }}

/* Portfolio cards */
.port-card {{
    background: #ffffff; border-radius: 10px; padding: 22px;
    border-top: 3px solid #c0392b; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    transition: all 0.2s; text-align: center;
}}
.port-card:hover {{ box-shadow: 0 6px 20px rgba(0,0,0,0.12); transform: translateY(-3px); }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-thumb {{ background: #1a3a6b; border-radius: 3px; }}

#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 16px !important; max-width: 1400px; }}
</style>""", unsafe_allow_html=True)

# ─── AUTO REFRESH ──────────────────────────────────────────────
st_autorefresh(interval=15000, key="main_refresh")

# ─── HEADER ────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns([1, 4, 1.5, 1])
with h1:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s", width=90)
with h2:
    st.markdown(f"""
    <div style='padding-top:8px;'>
      <div style='font-size:20px;font-weight:700;color:#1a2a4a;'>PEL – AI Predictive Maintenance System</div>
      <div style='font-size:13px;color:#6b7280;'>Petroleum Exploration (Pvt.) Ltd. · Karachi, Pakistan</div>
    </div>""", unsafe_allow_html=True)
with h3:
    now = datetime.now()
    st.markdown(f"""
    <div style='background:#ffffff;border-radius:8px;padding:10px 14px;border:1px solid #e5e7eb;text-align:center;'>
      <div style='font-size:11px;color:#6b7280;font-weight:500;'>LIVE · Auto-refresh 15s</div>
      <div style='font-size:13px;font-weight:600;color:#1a2a4a;'>{now.strftime('%d %b %Y')}</div>
      <div style='font-size:12px;color:#6b7280;'>{now.strftime('%H:%M:%S')}</div>
    </div>""", unsafe_allow_html=True)
with h4:
    lang_choice = st.selectbox("", ["English", "اردو"], key="lang_sel", label_visibility="collapsed")
    st.session_state.lang = "ur" if lang_choice == "اردو" else "en"
    if st.button("Sign Out", use_container_width=True):
        for k in ["authenticated","current_user","current_role"]:
            st.session_state.pop(k, None)
        st.rerun()

st.markdown("<div style='height:1px;background:#e5e7eb;margin:8px 0 16px;'></div>", unsafe_allow_html=True)

# ─── USER ROLE BADGE ───────────────────────────────────────────
role_color = {"Admin":"#1a3a6b","Engineer":"#059669","Viewer":"#6b7280"}[role]
st.markdown(f"""<div style='display:flex;gap:8px;align-items:center;margin-bottom:12px;'>
  <span style='background:{role_color};color:#fff;font-size:11px;font-weight:600;
               border-radius:20px;padding:3px 12px;'>
    {role.upper()}
  </span>
  <span style='font-size:13px;color:#6b7280;'>
    Signed in as <b>{st.session_state.get("current_user","")}</b>
  </span>
</div>""", unsafe_allow_html=True)

# ─── THRESHOLD SETTINGS (session) ──────────────────────────────
if "threshold_critical" not in st.session_state:
    st.session_state.threshold_critical = 0.70
if "threshold_warning"  not in st.session_state:
    st.session_state.threshold_warning  = 0.50
CRIT = st.session_state.threshold_critical
WARN = st.session_state.threshold_warning

# ─── MACHINES DATA ─────────────────────────────────────────────
MACHINES = ["Compressor Unit A", "Pump Station B", "Gas Turbine C", "Generator D"]

def gen_machine_data(seed, days=100):
    np.random.seed(seed)
    df = pd.DataFrame({
        'Day':   range(1, days+1),
        'Vibration':    np.random.uniform(2, 9.5, days),
        'Temperature':  np.random.uniform(45, 89, days),
        'Fuel':         np.random.uniform(80, 480, days),
        'Shift':        np.random.choice(['Morning','Evening','Night'], days),
    })
    df['CO2'] = df['Fuel'] * 2.68 * (1 + df['Vibration']/12)
    df['Failure_Prob'] = np.clip(
        (df['Vibration']-4)/5.5 + (df['Temperature']-60)/32, 0, 0.96)
    return df

if "machines_data" not in st.session_state:
    st.session_state.machines_data = {
        m: gen_machine_data(i*7, 100) for i,m in enumerate(MACHINES)
    }

if "models" not in st.session_state:
    st.session_state.models    = {}
    st.session_state.acc       = {}
    for m in MACHINES:
        df = st.session_state.machines_data[m]
        X  = df[['Vibration','Temperature','Fuel']]
        y  = (df['Failure_Prob'] > CRIT).astype(int)
        Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=0.25,random_state=42)
        clf = RandomForestClassifier(n_estimators=150,random_state=42,class_weight='balanced')
        clf.fit(Xtr,ytr)
        st.session_state.models[m] = clf
        st.session_state.acc[m]    = accuracy_score(yt,clf.predict(Xt))

if "maint_log" not in st.session_state:
    st.session_state.maint_log = pd.DataFrame({
        'Date':    ['2025-01-10','2025-02-15','2025-03-20'],
        'Machine': ['Compressor Unit A','Pump Station B','Gas Turbine C'],
        'Type':    ['Preventive','Corrective','Preventive'],
        'Engineer':['Ali Hassan','Umar Farooq','Zara Khan'],
        'Notes':   ['Replaced bearings','Fixed oil leak','Blade inspection OK'],
        'Cost_PKR':[85000, 150000, 60000],
    })

if "op_notes" not in st.session_state:
    st.session_state.op_notes = []

# ── Add new live row per machine ──────────────────────────────
MAX_ROWS = 150
for m in MACHINES:
    df_m  = st.session_state.machines_data[m]
    max_d = int(df_m['Day'].max())
    nr = pd.DataFrame({
        'Day':          [max_d+1],
        'Vibration':    [np.random.uniform(2, 10.8)],
        'Temperature':  [np.random.uniform(45, 94)],
        'Fuel':         [np.random.uniform(80, 520)],
        'Shift':        [np.random.choice(['Morning','Evening','Night'])],
    })
    nr['CO2'] = nr['Fuel']*2.68*(1+nr['Vibration']/12)
    nr['Failure_Prob'] = np.clip(
        (nr['Vibration']-4)/5.5+(nr['Temperature']-60)/32, 0, 0.96)
    combined = pd.concat([df_m,nr],ignore_index=True).tail(MAX_ROWS).reset_index(drop=True)
    combined['Predicted_Risk'] = st.session_state.models[m].predict_proba(
        combined[['Vibration','Temperature','Fuel']])[:,1]
    st.session_state.machines_data[m] = combined

# ── Email helper ───────────────────────────────────────────────
def send_email(risk_pct, day, machine, vib, temp, recipient):
    try:
        sender   = st.secrets.get("alert_email","")
        pwd      = st.secrets.get("alert_email_password","")
        if not sender or not pwd:
            return False,"Email credentials not set in secrets.toml"
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"PEL ALERT — {machine} Failure Risk {risk_pct:.0f}%"
        msg["From"]    = f"PEL Maintenance AI <{sender}>"
        msg["To"]      = recipient
        html = f"""<html><body style="font-family:Inter,Arial;background:#f0f2f5;">
        <div style="max-width:580px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.1);">
          <div style="background:#1a3a6b;padding:24px;text-align:center;">
            <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s" width="60" style="border-radius:6px;">
            <h2 style="color:#fff;margin:12px 0 4px;font-size:18px;">PEL Maintenance Alert</h2>
            <p style="color:#a5c0e0;margin:0;font-size:13px;">Petroleum Exploration (Pvt.) Ltd.</p>
          </div>
          <div style="background:#dc2626;padding:14px;text-align:center;">
            <span style="color:#fff;font-weight:700;font-size:15px;">🚨 Critical Risk Detected on {machine}</span>
          </div>
          <div style="padding:28px;">
            <table width="100%" style="border-collapse:collapse;">
              <tr>
                <td style="padding:10px;background:#fef2f2;border-radius:8px;text-align:center;width:30%;">
                  <div style="font-size:28px;font-weight:700;color:#dc2626;">{risk_pct:.0f}%</div>
                  <div style="font-size:11px;color:#6b7280;margin-top:4px;">FAILURE RISK</div>
                </td>
                <td width="10"></td>
                <td style="padding:10px;background:#f9fafb;border-radius:8px;text-align:center;width:30%;">
                  <div style="font-size:24px;font-weight:700;color:#1a3a6b;">{vib:.2f}</div>
                  <div style="font-size:11px;color:#6b7280;margin-top:4px;">VIBRATION mm/s</div>
                </td>
                <td width="10"></td>
                <td style="padding:10px;background:#f9fafb;border-radius:8px;text-align:center;width:30%;">
                  <div style="font-size:24px;font-weight:700;color:#d97706;">{temp:.0f}°C</div>
                  <div style="font-size:11px;color:#6b7280;margin-top:4px;">TEMPERATURE</div>
                </td>
              </tr>
            </table>
            <div style="background:#fef2f2;border-left:4px solid #dc2626;border-radius:0 8px 8px 0;
                        padding:14px;margin-top:20px;">
              <p style="color:#7f1d1d;margin:0;font-size:13px;">
                Immediate inspection recommended for <b>{machine}</b>. 
                Failure risk has exceeded the {risk_pct:.0f}% critical threshold on Day {day}.
              </p>
            </div>
            <table style="margin-top:20px;width:100%;">
              <tr><td style="color:#6b7280;font-size:12px;padding:4px 0;">Date &amp; Time:</td>
                  <td style="color:#111827;font-size:12px;">{datetime.now().strftime('%d %B %Y — %H:%M')}</td></tr>
              <tr><td style="color:#6b7280;font-size:12px;padding:4px 0;">Machine:</td>
                  <td style="color:#111827;font-size:12px;">{machine}</td></tr>
              <tr><td style="color:#6b7280;font-size:12px;padding:4px 0;">Monitoring Day:</td>
                  <td style="color:#111827;font-size:12px;">#{day}</td></tr>
            </table>
          </div>
          <div style="background:#f9fafb;padding:16px;text-align:center;border-top:1px solid #e5e7eb;">
            <p style="color:#9ca3af;font-size:11px;margin:0;">PEL AI Predictive Maintenance v2.0 — Automated Alert System</p>
          </div>
        </div></body></html>"""
        msg.attach(MIMEText(html,"html"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as sv:
            sv.login(sender,pwd)
            sv.sendmail(sender,recipient,msg.as_string())
        return True,"Email sent successfully!"
    except Exception as e:
        return False,str(e)

def should_send(risk):
    if risk <= CRIT: return False
    last = st.session_state.get("last_alert_sent")
    return last is None or datetime.now()-last > timedelta(minutes=60)

# ─── TABS ──────────────────────────────────────────────────────
tabs = st.tabs(["📋 Executive Summary","📊 Live Dashboard","🔮 Forecast",
                "⚠️ Alerts & Actions","🌿 HSE & Environment",
                "⚙️ Settings","🏢 About PEL"])

# ══════════════════════════════════════════════════════════════
# TAB 1 – EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Executive Summary — All Machines")
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    # Fleet KPIs
    all_risks    = [float(st.session_state.machines_data[m].iloc[-1]['Predicted_Risk']) for m in MACHINES]
    avg_fleet    = np.mean(all_risks)*100
    critical_cnt = sum(r > CRIT for r in all_risks)
    total_co2    = sum(st.session_state.machines_data[m]['CO2'].sum() for m in MACHINES)/1000
    avg_health   = 100 - avg_fleet

    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        (c1,"Fleet Health Score",  f"{avg_health:.0f}%",  "Based on all machines","ok" if avg_health>70 else "crit"),
        (c2,"Critical Machines",   f"{critical_cnt}/{len(MACHINES)}", "Risk > threshold","crit" if critical_cnt>0 else "ok"),
        (c3,"Fleet Avg Risk",      f"{avg_fleet:.1f}%",   "Current average","crit" if avg_fleet>70 else "warn" if avg_fleet>50 else "ok"),
        (c4,"Total CO₂ Logged",    f"{total_co2:.1f} t",  "All machines combined","ok"),
    ]
    for col,label,val,sub,sev in kpis:
        col_class = {"ok":"#059669","warn":"#d97706","crit":"#dc2626"}[sev]
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top-color:{col_class};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{col_class};">{val}</div>
              <div class="kpi-delta">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Machine status table
    st.markdown("#### Machine Status Overview")
    rows=[]
    for m in MACHINES:
        df_m = st.session_state.machines_data[m]
        lat  = df_m.iloc[-1]
        r    = float(lat['Predicted_Risk'])
        status = "🔴 Critical" if r>CRIT else ("🟡 Warning" if r>WARN else "🟢 Normal")
        rows.append({
            "Machine": m,
            "Status": status,
            "Risk %": f"{r*100:.1f}%",
            "Vibration (mm/s)": f"{lat['Vibration']:.2f}",
            "Temp (°C)": f"{lat['Temperature']:.1f}",
            "CO₂ (kg)": f"{lat['CO2']:.0f}",
            "Health %": f"{100-r*100:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Fleet risk bar chart
    st.markdown("#### Fleet Risk Comparison")
    fig_fleet = go.Figure(go.Bar(
        x=MACHINES, y=[r*100 for r in all_risks],
        marker_color=['#dc2626' if r>CRIT else '#d97706' if r>WARN else '#059669' for r in all_risks],
        text=[f"{r*100:.1f}%" for r in all_risks], textposition='outside'
    ))
    fig_fleet.add_hline(y=CRIT*100, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"Critical {CRIT*100:.0f}%")
    fig_fleet.add_hline(y=WARN*100, line_dash="dot",  line_color="#d97706",
                        annotation_text=f"Warning {WARN*100:.0f}%")
    fig_fleet.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
        font_color='#374151', height=300,
        margin=dict(l=20,r=20,t=20,b=20),
        yaxis=dict(range=[0,110], title="Failure Risk %", gridcolor='#f3f4f6'),
        xaxis=dict(gridcolor='#f3f4f6'),
        showlegend=False
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

    # Cost savings estimate
    st.markdown("#### Estimated Cost Savings — Proactive Maintenance")
    avg_unplanned_cost = 500000
    avoided = sum(1 for r in all_risks if r > WARN)
    sav1,sav2,sav3 = st.columns(3)
    sav1.metric("Potential Downtime Avoided", f"{avoided} machines", "This month")
    sav2.metric("Estimated Savings", f"PKR {avoided*avg_unplanned_cost:,.0f}", "Unplanned repair avoided")
    sav3.metric("CO₂ Reduction Potential", f"~{avoided*150} kg", "With timely action")

# ══════════════════════════════════════════════════════════════
# TAB 2 – LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    # Machine selector + date range
    sel_col1, sel_col2, sel_col3 = st.columns([2,1,1])
    with sel_col1:
        sel_machine = st.selectbox("Select Machine", MACHINES, key="dash_machine")
    with sel_col2:
        date_range = st.selectbox("Date Range", ["Last 30 Days","Last 60 Days","Last 90 Days","All Data"], key="date_range")
    with sel_col3:
        sel_shift = st.selectbox("Shift Filter", ["All Shifts","Morning","Evening","Night"], key="shift_sel")

    df_sel = st.session_state.machines_data[sel_machine].copy()
    n_days = {"Last 30 Days":30,"Last 60 Days":60,"Last 90 Days":90,"All Data":999}[date_range]
    df_sel = df_sel.tail(n_days)
    if sel_shift != "All Shifts":
        df_sel = df_sel[df_sel['Shift']==sel_shift]

    latest = st.session_state.machines_data[sel_machine].iloc[-1]
    risk   = float(latest['Predicted_Risk'])
    health = 100 - risk*100

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    kpi_data = [
        (k1,"Health Score",f"{health:.0f}%","ok" if health>70 else "crit"),
        (k2,"Failure Risk",f"{risk*100:.1f}%","crit" if risk>CRIT else "warn" if risk>WARN else "ok"),
        (k3,"Vibration",f"{latest['Vibration']:.2f} mm/s","ok"),
        (k4,"Temperature",f"{latest['Temperature']:.1f} °C","warn" if latest['Temperature']>80 else "ok"),
        (k5,"CO₂ Emission",f"{latest['CO2']:.0f} kg","ok"),
    ]
    for col,label,val,sev in kpi_data:
        col_color = {"ok":"#1a3a6b","warn":"#d97706","crit":"#dc2626"}[sev]
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top-color:{col_color};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:22px;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    col_charts, col_side = st.columns([3,1])

    with col_charts:
        # Anomaly detection — mark anomalies on chart
        df_sel['Anomaly'] = df_sel['Predicted_Risk'] > CRIT
        normal_df  = df_sel[~df_sel['Anomaly']]
        anomaly_df = df_sel[df_sel['Anomaly']]

        # Risk trend with anomalies
        st.markdown("**Failure Risk Trend**")
        fig_r = go.Figure()
        fig_r.add_hrect(y0=CRIT, y1=1.0, fillcolor="rgba(220,38,38,0.05)", line_width=0)
        fig_r.add_hrect(y0=WARN, y1=CRIT, fillcolor="rgba(217,119,6,0.05)", line_width=0)
        fig_r.add_trace(go.Scatter(x=df_sel['Day'], y=df_sel['Predicted_Risk'],
            mode='lines', name='Risk', line=dict(color='#1a3a6b', width=2),
            fill='tozeroy', fillcolor='rgba(26,58,107,0.06)'))
        if not anomaly_df.empty:
            fig_r.add_trace(go.Scatter(x=anomaly_df['Day'], y=anomaly_df['Predicted_Risk'],
                mode='markers', name='⚠️ Anomaly',
                marker=dict(color='#dc2626', size=9, symbol='x')))
        fig_r.add_hline(y=CRIT, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"Critical ({CRIT*100:.0f}%)")
        fig_r.add_hline(y=WARN, line_dash="dot",  line_color="#d97706",
                        annotation_text=f"Warning ({WARN*100:.0f}%)")
        fig_r.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
            font_color='#374151', height=240, margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(gridcolor='#f3f4f6', title='Day'),
            yaxis=dict(gridcolor='#f3f4f6', range=[0,1.05], title='Risk'),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Vibration + Temperature dual axis
        st.markdown("**Sensor Readings — Vibration & Temperature**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df_sel['Day'], y=df_sel['Vibration'],
            name='Vibration (mm/s)', line=dict(color='#1a3a6b', width=2)))
        fig_s.add_trace(go.Scatter(x=df_sel['Day'], y=df_sel['Temperature'],
            name='Temperature (°C)', line=dict(color='#c0392b', width=2), yaxis='y2'))
        fig_s.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#374151',
            height=220, margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(gridcolor='#f3f4f6', title='Day'),
            yaxis=dict(gridcolor='#f3f4f6', title='Vibration (mm/s)', side='left'),
            yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # Shift-wise risk
        st.markdown("**Shift-wise Average Risk**")
        shift_grp = df_sel.groupby('Shift')['Predicted_Risk'].mean().reset_index()
        fig_sh = go.Figure(go.Bar(
            x=shift_grp['Shift'], y=shift_grp['Predicted_Risk']*100,
            marker_color='#1a3a6b',
            text=[f"{v:.1f}%" for v in shift_grp['Predicted_Risk']*100],
            textposition='outside'
        ))
        fig_sh.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#374151',
            height=200, margin=dict(l=10,r=10,t=10,b=10),
            yaxis=dict(gridcolor='#f3f4f6', title='Avg Risk %', range=[0,100]),
        )
        st.plotly_chart(fig_sh, use_container_width=True)

    with col_side:
        # Health gauge
        st.markdown("**Machine Health**")
        gc = '#059669' if health>70 else '#d97706' if health>40 else '#dc2626'
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            number={'suffix':'%','font':{'color':gc,'size':28,'family':'Inter'}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#6b7280'},
                'bar':{'color':gc,'thickness':0.28},
                'bgcolor':'rgba(0,0,0,0)',
                'bordercolor':'#e5e7eb',
                'steps':[
                    {'range':[0,40],'color':'rgba(220,38,38,0.1)'},
                    {'range':[40,70],'color':'rgba(217,119,6,0.08)'},
                    {'range':[70,100],'color':'rgba(5,150,105,0.08)'},
                ],
                'threshold':{'value':70,'line':{'color':'#dc2626','width':2},'thickness':0.75}
            }
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#374151',height=200,
                            margin=dict(l=10,r=10,t=20,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("**Live Readings**")
        st.metric("Monitoring Day", f"#{int(latest['Day'])}")
        st.metric("Fuel Consumption", f"{latest['Fuel']:.0f} L")
        st.metric("30-Day Avg Risk", f"{df_sel['Predicted_Risk'].mean()*100:.1f}%")
        st.metric("Model Accuracy", f"{st.session_state.acc[sel_machine]*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        risk_label = ("🔴 Critical" if risk>CRIT else "🟡 Warning" if risk>WARN else "🟢 Normal")
        alert_class = ("alert-crit" if risk>CRIT else "alert-warn" if risk>WARN else "alert-ok")
        st.markdown(f"<div class='{alert_class}'><b>{risk_label}</b><br>"
                    f"Risk: {risk*100:.1f}%</div>", unsafe_allow_html=True)

    # Maintenance history
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Maintenance History Log")
    log_df = st.session_state.maint_log
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    if can_edit:
        with st.expander("➕ Add Maintenance Record"):
            mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
            nm = mc1.selectbox("Machine", MACHINES, key="log_m")
            nd = mc2.date_input("Date", datetime.today(), key="log_d")
            nt = mc3.selectbox("Type", ["Preventive","Corrective","Emergency"], key="log_t")
            ne = mc4.text_input("Engineer", key="log_e")
            nn = mc5.text_input("Notes", key="log_n")
            nc = mc6.number_input("Cost (PKR)", min_value=0, key="log_c")
            if st.button("Add Record"):
                new_entry = pd.DataFrame([{
                    'Date':str(nd),'Machine':nm,'Type':nt,
                    'Engineer':ne,'Notes':nn,'Cost_PKR':nc
                }])
                st.session_state.maint_log = pd.concat(
                    [st.session_state.maint_log, new_entry], ignore_index=True)
                st.success("Record added!")
                st.rerun()

    # Operator Notes
    st.markdown("#### Operator Notes")
    if can_edit:
        with st.expander("✏️ Add Observation"):
            on_m = st.selectbox("Machine", MACHINES, key="note_machine")
            on_n = st.text_area("Observation", key="note_text", height=80)
            if st.button("Save Note"):
                st.session_state.op_notes.append({
                    "DateTime": datetime.now().strftime("%d %b %Y %H:%M"),
                    "Machine": on_m, "Note": on_n,
                    "Engineer": st.session_state.get("current_user","")
                })
                st.success("Note saved!")

    if st.session_state.op_notes:
        st.dataframe(pd.DataFrame(st.session_state.op_notes),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No operator notes yet.")

    # Download
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    dl_df = st.session_state.machines_data[sel_machine].tail(60)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        dl_df.to_excel(writer, index=False, sheet_name='Sensor Data')
        log_df.to_excel(writer, index=False, sheet_name='Maintenance Log')
    buf.seek(0)
    st.download_button("📥 Download Excel Report",
                       data=buf, file_name=f"PEL_{sel_machine.replace(' ','_')}_Report.xlsx",
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 – FORECAST
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    fc_machine = st.selectbox("Machine", MACHINES, key="fc_machine")
    df_fc = st.session_state.machines_data[fc_machine]
    latest_fc = df_fc.iloc[-1]
    risk_fc   = float(latest_fc['Predicted_Risk'])
    cur_day   = int(latest_fc['Day'])

    np.random.seed(cur_day % 100)
    base = [min(risk_fc + i*0.012 + np.random.uniform(-0.025,0.025), 0.98) for i in range(30)]
    fc_days = list(range(cur_day+1, cur_day+31))
    df_f = pd.DataFrame({'Day':fc_days,'Risk':base,
                         'Upper':[min(r+0.07,1) for r in base],
                         'Lower':[max(r-0.07,0) for r in base]})

    st.markdown("**30-Day Failure Risk Forecast**")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df_f['Day'], y=df_f['Upper'],
        fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
    fig_f.add_trace(go.Scatter(x=df_f['Day'], y=df_f['Lower'],
        fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
        fillcolor='rgba(26,58,107,0.08)', name='Confidence Interval'))
    fig_f.add_trace(go.Scatter(x=df_f['Day'], y=df_f['Risk'],
        mode='lines+markers', name='Forecast Risk',
        line=dict(color='#1a3a6b', width=2.5),
        marker=dict(size=5, color='#1a3a6b')))
    fig_f.add_hline(y=CRIT, line_dash="dash", line_color="#dc2626",
                    annotation_text=f"Critical ({CRIT*100:.0f}%)")
    fig_f.add_hline(y=WARN, line_dash="dot", line_color="#d97706",
                    annotation_text=f"Warning ({WARN*100:.0f}%)")
    fig_f.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#374151',
        height=350, margin=dict(l=10,r=10,t=20,b=10),
        xaxis=dict(gridcolor='#f3f4f6', title='Future Day'),
        yaxis=dict(gridcolor='#f3f4f6', range=[0,1.05], title='Predicted Risk'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Heatmap calendar
    st.markdown("**Risk Heatmap — Historical**")
    hm_data = df_fc.tail(60).copy()
    hm_data['Week'] = (hm_data['Day'] // 7).astype(int)
    hm_data['DayOfWeek'] = hm_data['Day'] % 7
    fig_hm = px.density_heatmap(hm_data, x='Week', y='DayOfWeek',
                                 z='Predicted_Risk', color_continuous_scale='RdYlGn_r',
                                 labels={'z':'Risk','Week':'Week','DayOfWeek':'Day'})
    fig_hm.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#374151',
        height=250, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Forecast summary
    crit_days = [fc_days[i] for i,r in enumerate(base) if r>CRIT]
    warn_days  = [fc_days[i] for i,r in enumerate(base) if WARN<r<=CRIT]
    fs1,fs2,fs3 = st.columns(3)
    fs1.metric("Peak Risk (30 days)", f"{max(base)*100:.1f}%", f"Day {fc_days[base.index(max(base))]}")
    fs2.metric("Critical Days", f"{len(crit_days)}")
    fs3.metric("Warning Days",  f"{len(warn_days)}")

    if crit_days:
        st.markdown(f"<div class='alert-crit'>⚠️ First critical window in "
                    f"<b>{crit_days[0]-cur_day} days</b> (Day {crit_days[0]}). "
                    f"Schedule maintenance immediately.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-ok'>✅ No critical periods forecast in next 30 days.</div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 – ALERTS & ACTIONS
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    al_machine = st.selectbox("Machine", MACHINES, key="al_machine")
    df_al  = st.session_state.machines_data[al_machine]
    lat_al = df_al.iloc[-1]
    risk_al= float(lat_al['Predicted_Risk'])

    a1,a2 = st.columns([2,1])
    with a1:
        st.markdown("**Recent Alerts (Risk > Warning)**")
        alerts_df = df_al[df_al['Predicted_Risk']>WARN].tail(15).copy()
        alerts_df['Status'] = alerts_df['Predicted_Risk'].apply(
            lambda x: '🔴 Critical' if x>CRIT else '🟡 Warning')
        if not alerts_df.empty:
            st.dataframe(
                alerts_df[['Day','Vibration','Temperature','Predicted_Risk','Status']].rename(
                    columns={'Vibration':'Vibration (mm/s)','Temperature':'Temp (°C)','Predicted_Risk':'Risk'}
                ).style.format({'Vibration (mm/s)':'{:.2f}','Temp (°C)':'{:.1f}','Risk':'{:.1%}'}),
                height=320, use_container_width=True
            )
        else:
            st.markdown("<div class='alert-ok'>✅ No recent alerts.</div>", unsafe_allow_html=True)

        st.markdown("<br>**Full Data Log (Last 30 Days)**")
        recent = df_al.tail(30)[['Day','Vibration','Temperature','Fuel','CO2','Predicted_Risk']]
        st.dataframe(
            recent.style.format({'Vibration':'{:.2f}','Temperature':'{:.1f}',
                                 'Fuel':'{:.0f}','CO2':'{:.0f}','Predicted_Risk':'{:.1%}'}),
            use_container_width=True, height=280
        )

    with a2:
        st.markdown("**Recommended Actions**")
        if risk_al > CRIT:
            actions = [("🔴","Immediate","Halt machine for inspection"),
                       ("🔴","Immediate","Replace vibration dampeners"),
                       ("🔴","Immediate","Notify maintenance team"),
                       ("🔴","Immediate","Escalate to Engineering Head")]
        elif risk_al > WARN:
            actions = [("🟡","48 Hours","Schedule maintenance inspection"),
                       ("🟡","48 Hours","Check coolant & filters"),
                       ("🟡","72 Hours","Increase monitoring frequency")]
        else:
            actions = [("🟢","Routine","Continue normal operations"),
                       ("🟢","7 Days","Next scheduled check"),
                       ("🟢","Ongoing","Log all readings")]

        for icon,timing,action in actions:
            st.markdown(f"""<div style='background:#f9fafb;border:1px solid #e5e7eb;
                        border-radius:8px;padding:10px 14px;margin:6px 0;'>
              <div style='font-size:11px;color:#6b7280;font-weight:500;'>{icon} {timing}</div>
              <div style='font-size:13px;color:#111827;margin-top:3px;'>{action}</div>
            </div>""", unsafe_allow_html=True)

        # Current status banner
        st.markdown("<br>", unsafe_allow_html=True)
        ac = "alert-crit" if risk_al>CRIT else "alert-warn" if risk_al>WARN else "alert-ok"
        lbl= "🔴 Critical — Immediate Action Required" if risk_al>CRIT else \
             "🟡 Elevated Risk — Schedule Maintenance" if risk_al>WARN else \
             "🟢 All Systems Normal"
        st.markdown(f"<div class='{ac}'><b>{lbl}</b></div>", unsafe_allow_html=True)

        # Email system
        st.markdown("<br>**Email Alert System**")
        st.markdown("""<div style='background:#f9fafb;border:1px solid #e5e7eb;
                    border-radius:8px;padding:14px;'>""", unsafe_allow_html=True)
        email_to = st.text_input("Recipient Email", placeholder="engineer@pel.com.pk",
                                  key="email_to", label_visibility="collapsed")
        eb1,eb2 = st.columns(2)
        with eb1:
            if st.button("📧 Send Alert", use_container_width=True):
                if email_to:
                    with st.spinner("Sending..."):
                        ok,msg_r = send_email(risk_al*100, int(lat_al['Day']),
                                              al_machine, float(lat_al['Vibration']),
                                              float(lat_al['Temperature']), email_to)
                    if ok:
                        st.session_state["last_alert_sent"] = datetime.now()
                        st.success("✅ Sent!")
                    else:
                        st.error(f"❌ {msg_r}")
                else:
                    st.warning("Enter email first")
        with eb2:
            auto = st.toggle("Auto Alert", value=st.session_state.get("auto_alert",False))
            st.session_state["auto_alert"] = auto

        last_s = st.session_state.get("last_alert_sent")
        if last_s:
            st.markdown(f"<div style='font-size:11px;color:#6b7280;margin-top:6px;'>"
                        f"Last sent: {last_s.strftime('%d %b %Y %H:%M')}</div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get("auto_alert") and email_to and should_send(risk_al):
            ok,_ = send_email(risk_al*100, int(lat_al['Day']), al_machine,
                              float(lat_al['Vibration']), float(lat_al['Temperature']), email_to)
            if ok:
                st.session_state["last_alert_sent"] = datetime.now()

# ══════════════════════════════════════════════════════════════
# TAB 5 – HSE & ENVIRONMENT
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### HSE & Environmental Compliance")

    total_co2_all = sum(st.session_state.machines_data[m]['CO2'].sum() for m in MACHINES)
    monthly_avg   = total_co2_all / max(1, len(st.session_state.machines_data[MACHINES[0]])) * 30
    target_monthly= 15000
    compliance_pct= min(100, (target_monthly/max(monthly_avg,1))*100)

    hse1,hse2,hse3,hse4 = st.columns(4)
    hse1.metric("Total CO₂ Logged",   f"{total_co2_all/1000:.2f} tonnes")
    hse2.metric("Monthly Average",    f"{monthly_avg:.0f} kg/month")
    hse3.metric("Monthly Target",     f"{target_monthly:,} kg")
    hse4.metric("HSE Compliance",     f"{compliance_pct:.0f}%",
                delta="On Target" if compliance_pct>=90 else "Review Needed")

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # CO2 trend all machines
    st.markdown("**CO₂ Emission Trend — All Machines (Last 60 Days)**")
    fig_co2 = go.Figure()
    colors = ['#1a3a6b','#c0392b','#059669','#d97706']
    for i,m in enumerate(MACHINES):
        df_co2 = st.session_state.machines_data[m].tail(60)
        fig_co2.add_trace(go.Scatter(
            x=df_co2['Day'], y=df_co2['CO2'],
            name=m, line=dict(color=colors[i], width=2)))
    fig_co2.add_hline(y=target_monthly/30, line_dash="dash", line_color="#dc2626",
                      annotation_text="Daily Target")
    fig_co2.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font_color='#374151',
        height=300, margin=dict(l=10,r=10,t=20,b=10),
        xaxis=dict(gridcolor='#f3f4f6', title='Day'),
        yaxis=dict(gridcolor='#f3f4f6', title='CO₂ kg/day'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_co2, use_container_width=True)

    # HSE Checklist
    st.markdown("**HSE Compliance Checklist**")
    checks = [
        ("CO₂ Emission Monitoring", True),
        ("Weekly Vibration Inspection", True),
        ("Monthly Pressure Test", True),
        ("Quarterly Safety Audit", True),
        ("Annual Environmental Review", False),
        ("Emergency Response Drill", False),
    ]
    cc1,cc2 = st.columns(2)
    for i,(chk,done) in enumerate(checks):
        col = cc1 if i%2==0 else cc2
        icon = "✅" if done else "⏳"
        bg   = "#f0fdf4" if done else "#fffbeb"
        bd   = "#16a34a" if done else "#d97706"
        col.markdown(f"""<div style='background:{bg};border:1px solid {bd};
                    border-radius:8px;padding:10px 14px;margin:4px 0;
                    font-size:13px;color:#111827;'>{icon} {chk}</div>""",
                    unsafe_allow_html=True)

    # Savings
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("**Environmental Impact & ROI Calculator**")
    roi1,roi2,roi3 = st.columns(3)
    prevented_failures = sum(1 for m in MACHINES
                             if float(st.session_state.machines_data[m].iloc[-1]['Predicted_Risk']) < CRIT)
    roi1.metric("Failures Prevented (est.)", f"{prevented_failures}", "This period")
    roi2.metric("CO₂ Avoided",  f"~{prevented_failures*180} kg", "Via proactive maintenance")
    roi3.metric("Cost Saved",   f"PKR {prevented_failures*350000:,.0f}", "Unplanned repair cost")

# ══════════════════════════════════════════════════════════════
# TAB 6 – SETTINGS
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### System Settings")

    if not can_edit:
        st.warning("🔒 Viewer role — Settings are read-only. Contact Admin to make changes.")

    set1, set2 = st.columns(2)

    with set1:
        st.markdown("**Alert Thresholds**")
        new_crit = st.slider("Critical Risk Threshold (%)", 50, 90,
                             int(CRIT*100), 5, disabled=not can_edit)
        new_warn = st.slider("Warning Risk Threshold (%)",  20, 70,
                             int(WARN*100), 5, disabled=not can_edit)
        if can_edit and st.button("Apply Thresholds"):
            st.session_state.threshold_critical = new_crit/100
            st.session_state.threshold_warning  = new_warn/100
            st.success("Thresholds updated!")
            st.rerun()

        st.markdown("<br>**Upload Real Sensor Data**")
        if can_edit:
            up_machine = st.selectbox("Machine to update", MACHINES, key="up_machine")
            uploaded   = st.file_uploader("Upload CSV/Excel", type=['csv','xlsx'], key="file_up")
            if uploaded and st.button("Load Data"):
                try:
                    if uploaded.name.endswith('.csv'):
                        new_df = pd.read_csv(uploaded)
                    else:
                        new_df = pd.read_excel(uploaded)
                    required = ['Day','Vibration','Temperature','Fuel']
                    if all(c in new_df.columns for c in required):
                        new_df['CO2'] = new_df['Fuel']*2.68*(1+new_df['Vibration']/12)
                        new_df['Failure_Prob'] = np.clip(
                            (new_df['Vibration']-4)/5.5+(new_df['Temperature']-60)/32,0,0.96)
                        new_df['Shift'] = new_df.get('Shift', pd.Series(['Morning']*len(new_df)))
                        new_df['Predicted_Risk'] = st.session_state.models[up_machine].predict_proba(
                            new_df[['Vibration','Temperature','Fuel']])[:,1]
                        st.session_state.machines_data[up_machine] = new_df
                        st.success(f"✅ Data loaded for {up_machine}!")
                        st.rerun()
                    else:
                        st.error(f"CSV must have columns: {required}")
                except Exception as e:
                    st.error(f"Error: {e}")

    with set2:
        st.markdown("**Retrain ML Model**")
        rt_machine = st.selectbox("Machine", MACHINES, key="rt_machine")
        if can_edit and st.button("🔄 Retrain Model"):
            df_rt = st.session_state.machines_data[rt_machine]
            X = df_rt[['Vibration','Temperature','Fuel']]
            y = (df_rt['Failure_Prob'] > st.session_state.threshold_critical).astype(int)
            Xtr,Xt,ytr,yt = train_test_split(X,y,test_size=0.25,random_state=42)
            clf = RandomForestClassifier(n_estimators=150,random_state=42,class_weight='balanced')
            clf.fit(Xtr,ytr)
            st.session_state.models[rt_machine] = clf
            st.session_state.acc[rt_machine]    = accuracy_score(yt,clf.predict(Xt))
            st.success(f"✅ Model retrained! New accuracy: {st.session_state.acc[rt_machine]*100:.1f}%")

        st.markdown("<br>**User Roles**")
        roles_data = [{"Username":u,"Role":d["role"]} for u,d in USERS.items()]
        st.dataframe(pd.DataFrame(roles_data), use_container_width=True, hide_index=True)

        st.markdown("<br>**Model Performance Summary**")
        acc_rows = [{"Machine":m,"Accuracy":f"{st.session_state.acc[m]*100:.1f}%"} for m in MACHINES]
        st.dataframe(pd.DataFrame(acc_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# TAB 7 – ABOUT PEL
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("""
    <div style='text-align:center;padding:30px 0 20px;'>
      <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s'
           width='100' style='border-radius:10px;border:2px solid #e5e7eb;'>
      <h2 style='color:#1a2a4a;margin:16px 0 6px;font-size:22px;'>
        Petroleum Exploration (Pvt.) Ltd.
      </h2>
      <p style='color:#6b7280;font-size:14px;margin:0;'>
        Pakistan's Premier Oil & Gas Exploration Company · Since 1981 · Karachi
      </p>
      <div style='display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:12px;'>
        <span style='background:#f0f2f5;border:1px solid #e5e7eb;border-radius:20px;
                     padding:4px 14px;font-size:12px;color:#374151;'>📍 Karachi, Pakistan</span>
        <span style='background:#f0f2f5;border:1px solid #e5e7eb;border-radius:20px;
                     padding:4px 14px;font-size:12px;color:#374151;'>🏭 Oil & Gas</span>
        <span style='background:#f0f2f5;border:1px solid #e5e7eb;border-radius:20px;
                     padding:4px 14px;font-size:12px;color:#374151;'>⚡ Est. 1981</span>
        <span style='background:#f0f2f5;border:1px solid #e5e7eb;border-radius:20px;
                     padding:4px 14px;font-size:12px;color:#374151;'>🌿 ISO 14001</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Core capabilities
    st.markdown("#### Core Capabilities")
    p1,p2,p3,p4 = st.columns(4)
    caps = [
        (p1,"🛢️","Oil Exploration","Seismic surveys, drilling operations & reservoir management across Pakistan"),
        (p2,"⚙️","Predictive AI","ML-powered failure prediction to optimize asset reliability & reduce downtime"),
        (p3,"🌿","ESG & Carbon","Real-time CO₂ monitoring, emission reduction & HSE compliance reporting"),
        (p4,"📊","Data Analytics","Live dashboards, 30-day forecasting & risk-based maintenance scheduling"),
    ]
    for col,icon,title,desc in caps:
        with col:
            st.markdown(f"""<div class='port-card'>
              <div style='font-size:36px;margin-bottom:10px;'>{icon}</div>
              <div style='font-size:13px;font-weight:600;color:#1a2a4a;margin-bottom:6px;'>{title}</div>
              <div style='font-size:12px;color:#6b7280;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # System stats
    st.markdown("#### System Statistics")
    total_days   = len(st.session_state.machines_data[MACHINES[0]])
    total_alerts = sum(len(st.session_state.machines_data[m][
        st.session_state.machines_data[m]['Predicted_Risk']>CRIT]) for m in MACHINES)
    ss1,ss2,ss3,ss4 = st.columns(4)
    ss1.metric("Days Monitored",  total_days)
    ss2.metric("Machines Tracked",len(MACHINES))
    ss3.metric("Total CO₂ Logged",f"{total_co2_all/1000:.1f} tonnes")
    ss4.metric("Critical Alerts", total_alerts)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("#### Technology Stack")
    t1,t2,t3 = st.columns(3)
    tech_groups = [
        (t1,[("🐍","Python 3.11","Core Language"),
              ("🤖","Scikit-Learn","Random Forest ML"),
              ("📊","Streamlit","Dashboard Framework")]),
        (t2,[("📈","Plotly","Interactive Charts"),
              ("🔢","NumPy · Pandas","Data Processing"),
              ("📧","SMTP Email","Live Alert System")]),
        (t3,[("☁️","Streamlit Cloud","Deployment"),
              ("📥","XlsxWriter","Excel Reports"),
              ("🔐","Role-Based Auth","Secure Access")]),
    ]
    for col,items in tech_groups:
        with col:
            for icon,name,desc in items:
                st.markdown(f"""<div style='background:#f9fafb;border:1px solid #e5e7eb;
                            border-radius:8px;padding:10px 14px;margin:5px 0;
                            display:flex;align-items:center;gap:12px;'>
                  <span style='font-size:20px;'>{icon}</span>
                  <div>
                    <div style='font-size:13px;font-weight:500;color:#1a2a4a;'>{name}</div>
                    <div style='font-size:11px;color:#6b7280;'>{desc}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # Footer credit
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1a3a6b;border-radius:10px;padding:24px;text-align:center;'>
      <div style='color:#a5c0e0;font-size:12px;letter-spacing:1px;margin-bottom:6px;'>DEVELOPED FOR</div>
      <div style='color:#ffffff;font-size:18px;font-weight:700;'>Petroleum Exploration (Pvt.) Ltd.</div>
      <div style='color:#a5c0e0;font-size:13px;margin-top:4px;'>
        AI Predictive Maintenance Platform · Version 2.0 · 2025
      </div>
    </div>""", unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
st.markdown("<div style='height:1px;background:#e5e7eb;'></div>", unsafe_allow_html=True)
fc1,fc2,fc3 = st.columns([3,3,1])
with fc1:
    st.markdown("""<p style='font-size:12px;color:#9ca3af;padding-top:8px;'>
      ⚙️ <b>PEL AI Predictive Maintenance v2.0</b> · Petroleum Exploration (Pvt.) Ltd. · Karachi
    </p>""", unsafe_allow_html=True)
with fc2:
    st.markdown("""<p style='font-size:12px;color:#9ca3af;padding-top:8px;'>
      📧 it@pel.com.pk &nbsp;·&nbsp; 🌐 www.pel.com.pk &nbsp;·&nbsp; 🔒 Secure · Live · Intelligent
    </p>""", unsafe_allow_html=True)
with fc3:
    dl_all = pd.concat([
        st.session_state.machines_data[m].tail(30).assign(Machine=m) for m in MACHINES
    ])
    buf2 = io.BytesIO()
    with pd.ExcelWriter(buf2, engine='openpyxl') as w:
        for m in MACHINES:
            st.session_state.machines_data[m].tail(60).to_excel(w, sheet_name=m[:30], index=False)
    buf2.seek(0)
    st.download_button("📥 Full Report", data=buf2,
                       file_name="PEL_Full_Report.xlsx",
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                       use_container_width=True)
