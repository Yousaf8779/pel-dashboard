import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# ──────────────── BASE64 BACKGROUND IMAGE ────────────────
BG_IMAGE_B64 = "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"

# ──────────────── PAGE CONFIG ────────────────
st.set_page_config(
    page_title="PEL – AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────── PASSWORD PROTECTION ────────────────
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets.get("password", "pel2025"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        _show_login_page(wrong=False)
        st.text_input("🔒 Password", type="password", on_change=password_entered, key="password")
        st.caption("Contact PEL IT/Admin for access credentials")
        return False
    elif not st.session_state["password_correct"]:
        _show_login_page(wrong=True)
        st.text_input("🔒 Password", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

def _show_login_page(wrong=False):
    # STEP 1: Inject CSS separately
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');

        .stApp {{
            background: linear-gradient(135deg, rgba(5,10,20,0.92) 0%, rgba(0,30,60,0.92) 100%),
                        url("data:image/jpeg;base64,{BG_IMAGE_B64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .gear {{ position:fixed; opacity:0.08; animation:rotateCW 20s linear infinite; font-size:120px; }}
        .gear-ccw {{ animation:rotateCCW 25s linear infinite !important; }}
        @keyframes rotateCW  {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}
        @keyframes rotateCCW {{ from{{transform:rotate(360deg)}} to{{transform:rotate(0deg)}} }}
        .scan-line {{ position:fixed; top:0; left:0; width:100%; height:3px;
            background:linear-gradient(90deg,transparent,#00d4ff,transparent);
            animation:scan 4s ease-in-out infinite; z-index:1; opacity:0.5; }}
        @keyframes scan {{ 0%{{top:0%}} 100%{{top:100%}} }}
        .pulse-ring {{ position:fixed; top:50%; left:50%;
            transform:translate(-50%,-50%); width:300px; height:300px;
            border:1px solid rgba(0,212,255,0.3); border-radius:50%;
            animation:pulse-expand 3s ease-out infinite; z-index:0; }}
        .pulse-ring:nth-child(2){{animation-delay:1s;}}
        .pulse-ring:nth-child(3){{animation-delay:2s;}}
        @keyframes pulse-expand {{
            0%{{transform:translate(-50%,-50%) scale(0.8);opacity:0.6;}}
            100%{{transform:translate(-50%,-50%) scale(2.5);opacity:0;}}
        }}
        .login-card {{
            background:rgba(10,20,40,0.88);
            border:1px solid rgba(0,212,255,0.4);
            border-radius:20px; padding:35px;
            backdrop-filter:blur(20px);
            box-shadow:0 0 60px rgba(0,212,255,0.2);
            max-width:500px; margin:0 auto; position:relative; z-index:2;
        }}
        .login-title {{
            font-family:'Orbitron',monospace; color:#00d4ff;
            font-size:24px; font-weight:900; text-align:center;
            text-shadow:0 0 30px rgba(0,212,255,0.8);
            margin-bottom:5px; letter-spacing:2px;
        }}
        .login-sub {{
            font-family:'Rajdhani',sans-serif; color:#a5d8ff;
            text-align:center; font-size:15px; margin-bottom:15px; letter-spacing:1px;
        }}
        .company-badge {{
            background:linear-gradient(135deg,#001a3a,#003060);
            border:1px solid rgba(0,212,255,0.3); border-radius:10px;
            padding:6px 16px; display:inline-block;
            font-family:'Rajdhani',sans-serif; color:#a5d8ff;
            font-size:12px; letter-spacing:2px;
        }}
        .corner-tl,.corner-tr,.corner-bl,.corner-br {{
            position:absolute; width:18px; height:18px;
            border-color:#00d4ff; border-style:solid; opacity:0.7;
        }}
        .corner-tl{{top:10px;left:10px;border-width:2px 0 0 2px;border-radius:3px 0 0 0;}}
        .corner-tr{{top:10px;right:10px;border-width:2px 2px 0 0;border-radius:0 3px 0 0;}}
        .corner-bl{{bottom:10px;left:10px;border-width:0 0 2px 2px;border-radius:0 0 0 3px;}}
        .corner-br{{bottom:10px;right:10px;border-width:0 2px 2px 0;border-radius:0 0 3px 0;}}
        .status-dot {{
            display:inline-block; width:8px; height:8px; border-radius:50%;
            background:#00ff88; margin-right:8px;
            animation:blink 1.5s ease-in-out infinite;
        }}
        @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.2}}}}
        .stTextInput input {{
            background:rgba(0,30,60,0.8) !important;
            border:1px solid rgba(0,212,255,0.4) !important;
            border-radius:10px !important; color:#ffffff !important;
            font-size:16px !important; padding:12px 15px !important;
        }}
        #MainMenu, footer {{ visibility:hidden; }}
        .block-container {{ padding-top:40px !important; }}
        </style>
    """, unsafe_allow_html=True)

    # STEP 2: Inject animated background elements separately
    st.markdown("""
        <div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;">
            <div class="gear" style="top:5%;left:3%;">⚙️</div>
            <div class="gear gear-ccw" style="bottom:8%;right:4%;">⚙️</div>
            <div class="gear" style="top:40%;left:1%;font-size:80px;animation-duration:15s;">⚙️</div>
            <div class="gear gear-ccw" style="top:15%;right:6%;font-size:90px;animation-duration:18s;">⚙️</div>
        </div>
        <div class="scan-line"></div>
        <div class="pulse-ring"></div>
        <div class="pulse-ring"></div>
        <div class="pulse-ring"></div>
    """, unsafe_allow_html=True)

    # STEP 3: Inject the login card separately
    wrong_msg = ""
    if wrong:
        wrong_msg = """
        <div style="max-width:500px;margin:10px auto;">
            <div style="background:rgba(255,50,50,0.15);border:1px solid rgba(255,80,80,0.5);
                        border-radius:10px;padding:12px;text-align:center;
                        color:#ff8080;font-family:Rajdhani,sans-serif;font-size:15px;">
                ❌ Incorrect Password — Please Try Again
            </div>
        </div>"""

    st.markdown(f"""
        <div class="login-card">
            <div class="corner-tl"></div>
            <div class="corner-tr"></div>
            <div class="corner-bl"></div>
            <div class="corner-br"></div>
            <div style="text-align:center;margin-bottom:18px;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s"
                     width="90" style="border-radius:10px;border:2px solid rgba(0,212,255,0.4);">
            </div>
            <div class="login-title">PEL MAINTENANCE AI</div>
            <div class="login-sub">Petroleum Exploration (Pvt.) Ltd.</div>
            <div style="text-align:center;margin-bottom:12px;">
                <span class="company-badge">⚡ PREDICTIVE ANALYTICS PLATFORM v2.0</span>
            </div>
            <div style="text-align:center;margin-bottom:8px;">
                <span class="status-dot"></span>
                <span style="color:#a5d8ff;font-size:13px;font-family:Rajdhani,sans-serif;">
                    SYSTEM ONLINE — SECURE ACCESS REQUIRED
                </span>
            </div>
        </div>
        {wrong_msg}
    """, unsafe_allow_html=True)

if not check_password():
    st.stop()

# ──────────────── EMAIL ALERT SYSTEM ────────────────
def send_alert_email(risk_pct, day, vib, temp, recipient_email):
    """Send professional HTML alert email via Gmail SMTP"""
    try:
        sender    = st.secrets.get("alert_email", "")
        password  = st.secrets.get("alert_email_password", "")
        if not sender or not password:
            return False, "Email credentials not configured in secrets.toml"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 PEL CRITICAL ALERT — Machine Failure Risk {risk_pct:.0f}% on Day {day}"
        msg["From"]    = f"PEL Maintenance AI <{sender}>"
        msg["To"]      = recipient_email

        html = f"""
        <html><body style="margin:0;padding:0;background:#0d1b2a;font-family:Arial,sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:0 auto;">
          <tr><td style="background:linear-gradient(135deg,#001a3a,#003060);padding:30px;text-align:center;">
            <h1 style="color:#00d4ff;margin:0;font-size:24px;letter-spacing:2px;">
              ⚙️ PEL MAINTENANCE AI
            </h1>
            <p style="color:#a5d8ff;margin:5px 0 0;font-size:13px;letter-spacing:1px;">
              Petroleum Exploration (Pvt.) Ltd. — Karachi
            </p>
          </td></tr>
          <tr><td style="background:#ff2020;padding:20px;text-align:center;">
            <h2 style="color:#ffffff;margin:0;font-size:20px;">
              🚨 CRITICAL MACHINE FAILURE RISK DETECTED
            </h2>
          </td></tr>
          <tr><td style="background:#0d1b2a;padding:30px;">
            <table width="100%" cellpadding="10" cellspacing="0">
              <tr>
                <td style="background:#1b263b;border:1px solid #33415c;border-radius:8px;color:#a5d8ff;text-align:center;padding:15px;">
                  <div style="font-size:32px;color:#ff4b4b;font-weight:bold;">{risk_pct:.0f}%</div>
                  <div style="font-size:12px;margin-top:5px;">FAILURE RISK</div>
                </td>
                <td width="15"></td>
                <td style="background:#1b263b;border:1px solid #33415c;border-radius:8px;color:#a5d8ff;text-align:center;padding:15px;">
                  <div style="font-size:32px;color:#00d4ff;font-weight:bold;">{vib:.2f}</div>
                  <div style="font-size:12px;margin-top:5px;">VIBRATION mm/s</div>
                </td>
                <td width="15"></td>
                <td style="background:#1b263b;border:1px solid #33415c;border-radius:8px;color:#a5d8ff;text-align:center;padding:15px;">
                  <div style="font-size:32px;color:#ffa500;font-weight:bold;">{temp:.0f}°C</div>
                  <div style="font-size:12px;margin-top:5px;">TEMPERATURE</div>
                </td>
              </tr>
            </table>
            <div style="background:#1a0000;border:1px solid #ff4b4b;border-radius:8px;padding:20px;margin-top:20px;">
              <p style="color:#ff8080;margin:0;font-size:15px;">
                ⚠️ <b>Immediate action required.</b> Compressor failure risk has exceeded the critical threshold of 70%.
                Please schedule emergency maintenance inspection.
              </p>
            </div>
            <table width="100%" style="margin-top:20px;">
              <tr>
                <td style="color:#7ec8e3;font-size:13px;">📅 Alert Generated:</td>
                <td style="color:#ffffff;font-size:13px;">{datetime.now().strftime('%d %B %Y — %H:%M:%S')}</td>
              </tr>
              <tr>
                <td style="color:#7ec8e3;font-size:13px;">📊 Monitoring Day:</td>
                <td style="color:#ffffff;font-size:13px;">Day #{day}</td>
              </tr>
              <tr>
                <td style="color:#7ec8e3;font-size:13px;">🏭 Asset:</td>
                <td style="color:#ffffff;font-size:13px;">Primary Compressor Unit</td>
              </tr>
              <tr>
                <td style="color:#7ec8e3;font-size:13px;">📍 Location:</td>
                <td style="color:#ffffff;font-size:13px;">PEL — Karachi, Pakistan</td>
              </tr>
            </table>
          </td></tr>
          <tr><td style="background:#001a3a;padding:15px;text-align:center;">
            <p style="color:#7ec8e3;font-size:12px;margin:0;">
              PEL AI Predictive Maintenance System v2.0 — Automated Alert
            </p>
          </td></tr>
        </table>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient_email, msg.as_string())
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

def should_send_alert(risk):
    """Cooldown: send alert max once every 60 minutes"""
    if risk <= 0.70:
        return False
    last_sent = st.session_state.get("last_alert_sent")
    if last_sent is None:
        return True
    return datetime.now() - last_sent > timedelta(minutes=60)

# ──────────────── MAIN DASHBOARD STYLES ────────────────
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    .stApp {{
        background: linear-gradient(160deg, rgba(4,12,28,0.97) 0%, rgba(2,18,40,0.97) 50%, rgba(0,10,25,0.97) 100%),
                    url("data:image/jpeg;base64,{BG_IMAGE_B64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }}

    /* Animated dual top border — corporate gold + cyan */
    .stApp::before {{
        content: '';
        position: fixed; top: 0; left: 0;
        width: 100%; height: 4px;
        background: linear-gradient(90deg, #c9a84c, #00d4ff, #0066cc, #c9a84c);
        background-size: 300% 100%;
        animation: borderAnim 4s linear infinite;
        z-index: 9999;
    }}
    @keyframes borderAnim {{ 0%{{background-position:0% 0%}} 100%{{background-position:300% 0%}} }}

    h1, h2, h3 {{
        font-family: 'Orbitron', monospace !important;
        color: #00d4ff !important;
        letter-spacing: 1px;
    }}
    h1 {{ text-shadow: 0 0 25px rgba(0,212,255,0.5); }}

    /* Corporate metric cards with gold accent */
    div[data-testid="metric-container"] {{
        background: linear-gradient(135deg, rgba(10,22,48,0.95), rgba(5,15,38,0.95)) !important;
        border: 1px solid rgba(201,168,76,0.25) !important;
        border-top: 2px solid rgba(201,168,76,0.6) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 8px 0 !important;
        box-shadow: 0 4px 25px rgba(0,0,0,0.4), 0 0 15px rgba(0,102,204,0.1) !important;
        transition: all 0.3s ease;
        backdrop-filter: blur(15px);
    }}
    div[data-testid="metric-container"]:hover {{
        border-color: rgba(201,168,76,0.6) !important;
        border-top-color: #c9a84c !important;
        box-shadow: 0 8px 35px rgba(0,0,0,0.5), 0 0 25px rgba(201,168,76,0.15) !important;
        transform: translateY(-4px);
    }}
    div[data-testid="metric-container"] label {{
        color: #8aabce !important;
        font-size: 11px !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }}
    div[data-testid="metric-container"] > div > div:nth-child(2) {{
        color: #ffffff !important;
        font-size: 34px !important;
        font-weight: bold !important;
        font-family: 'Orbitron', monospace !important;
    }}
    div[data-testid="metric-delta"] {{ color: #c9a84c !important; font-size: 13px !important; font-weight:600 !important; }}

    /* Corporate buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #0055a5, #003d7a) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(201,168,76,0.3) !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 12px !important;
    }}
    .stButton > button:hover {{
        background: linear-gradient(135deg, #0066cc, #0055a5) !important;
        border-color: rgba(201,168,76,0.7) !important;
        box-shadow: 0 0 20px rgba(0,102,204,0.4) !important;
        transform: translateY(-2px);
    }}

    /* Progress bar */
    .stProgress > div > div > div {{ background: linear-gradient(90deg, #00b4d8, #ff4b4b) !important; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(10,25,50,0.8);
        border-radius: 10px;
        padding: 5px;
        border: 1px solid rgba(0,212,255,0.2);
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #7ec8e3 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(0,212,255,0.15) !important;
        color: #00d4ff !important;
        border-radius: 8px;
    }}

    /* Dataframe */
    .stDataFrame {{ border: 1px solid rgba(0,212,255,0.2); border-radius: 10px; }}

    /* Section cards */
    .section-card {{
        background: linear-gradient(135deg, rgba(10,25,50,0.85), rgba(0,20,45,0.85));
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    }}

    /* Alert boxes */
    .alert-critical {{
        background: linear-gradient(135deg, rgba(180,0,0,0.3), rgba(100,0,0,0.3));
        border: 1px solid rgba(255,75,75,0.6);
        border-radius: 12px;
        padding: 15px 20px;
        color: #ff8080;
        font-family: 'Rajdhani', sans-serif;
        font-size: 16px;
        font-weight: 600;
        animation: alertPulse 2s ease-in-out infinite;
    }}
    @keyframes alertPulse {{ 0%,100%{{box-shadow:0 0 10px rgba(255,75,75,0.3)}} 50%{{box-shadow:0 0 30px rgba(255,75,75,0.7)}} }}

    .alert-warning {{
        background: linear-gradient(135deg, rgba(180,130,0,0.2), rgba(100,70,0,0.2));
        border: 1px solid rgba(255,180,0,0.4);
        border-radius: 12px;
        padding: 12px 18px;
        color: #ffd166;
        font-family: 'Rajdhani', sans-serif;
    }}
    .alert-ok {{
        background: linear-gradient(135deg, rgba(0,150,80,0.2), rgba(0,80,40,0.2));
        border: 1px solid rgba(0,255,130,0.3);
        border-radius: 12px;
        padding: 12px 18px;
        color: #00ff88;
        font-family: 'Rajdhani', sans-serif;
    }}

    /* Portfolio cards */
    .portfolio-card {{
        background: linear-gradient(135deg, rgba(15,30,60,0.9), rgba(0,20,45,0.9));
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }}
    .portfolio-card:hover {{
        border-color: rgba(0,212,255,0.7);
        box-shadow: 0 0 25px rgba(0,212,255,0.3);
        transform: translateY(-5px);
    }}
    .portfolio-icon {{ font-size: 40px; margin-bottom: 10px; }}
    .portfolio-title {{
        font-family: 'Orbitron', monospace;
        color: #00d4ff;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    .portfolio-desc {{
        font-family: 'Rajdhani', sans-serif;
        color: #a5d8ff;
        font-size: 14px;
        margin-top: 8px;
    }}

    /* Hide streamlit footer & menu */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: #0d1b2a; }}
    ::-webkit-scrollbar-thumb {{ background: #00d4ff; border-radius: 3px; }}
    </style>
""", unsafe_allow_html=True)

# ──────────────── AUTO REFRESH ────────────────
st_autorefresh(interval=10000, key="refresh_data")

# ──────────────── HEADER ────────────────
col_logo, col_title, col_status = st.columns([1, 5, 2])
with col_logo:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s", width=110)
with col_title:
    st.markdown("""
        <h1 style='margin:0; font-size:36px;'>PEL – AI PREDICTIVE MAINTENANCE</h1>
        <p style='color:#7ec8e3; font-size:15px; margin:5px 0 0 0; font-family:Rajdhani,sans-serif; letter-spacing:2px;'>
            ⚡ PROACTIVE ASSET RELIABILITY &nbsp;|&nbsp; 🌍 CARBON REDUCTION &nbsp;|&nbsp; 📊 LIVE MONITORING
        </p>
    """, unsafe_allow_html=True)
with col_status:
    from datetime import datetime
    now = datetime.now().strftime("%d %b %Y | %H:%M:%S")
    st.markdown(f"""
        <div style='background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.3);
                    border-radius:10px;padding:12px;text-align:center;margin-top:5px;'>
            <div style='color:#00ff88;font-size:11px;font-family:Rajdhani,sans-serif;letter-spacing:2px;'>● LIVE</div>
            <div style='color:#a5d8ff;font-size:12px;font-family:Rajdhani,sans-serif;'>{now}</div>
            <div style='color:#7ec8e3;font-size:11px;'>Auto-refresh: 10s</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid rgba(0,212,255,0.2);margin:10px 0;'>", unsafe_allow_html=True)

# ──────────────── KEY METRICS ────────────────
st.markdown("<h3 style='font-size:16px;'>📌 KEY BUSINESS METRICS</h3>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Downtime Reduction", "Up to 40%", delta="↑ High Confidence")
m2.metric("CO₂ Savings", "~200–600 kg/month", delta="↑ Sustainability")
m3.metric("Maintenance Efficiency", "Risk-Based", delta="Optimized")
m4.metric("System Uptime", "99.7%", delta="↑ Stable")

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────── DATA & MODEL ────────────────
MAX_DATA_ROWS = 150  # Cap to prevent infinite growth

if 'data' not in st.session_state:
    np.random.seed(42)
    num_days = 100
    st.session_state.data = pd.DataFrame({
        'Day': range(1, num_days + 1),
        'Compressor_Vibration': np.random.uniform(2, 9.5, num_days),
        'Compressor_Temperature': np.random.uniform(45, 89, num_days),
        'Fuel_Consumption': np.random.uniform(80, 480, num_days),
    })
    st.session_state.data['Carbon_Emission'] = (
        st.session_state.data['Fuel_Consumption'] * 2.68 *
        (1 + st.session_state.data['Compressor_Vibration'] / 12)
    )
    st.session_state.data['Failure_Probability'] = np.clip(
        (st.session_state.data['Compressor_Vibration'] - 4) / 5.5 +
        (st.session_state.data['Compressor_Temperature'] - 60) / 32, 0, 0.96
    )

if 'model' not in st.session_state:
    X = st.session_state.data[['Compressor_Vibration', 'Compressor_Temperature', 'Fuel_Consumption']]
    y = (st.session_state.data['Failure_Probability'] > 0.6).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    st.session_state.model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
    st.session_state.model.fit(X_train, y_train)
    st.session_state.accuracy = accuracy_score(y_test, st.session_state.model.predict(X_test))

# Add new row — but cap the data at MAX_DATA_ROWS
current_max_day = int(st.session_state.data['Day'].max())
new_day = current_max_day + 1
new_vib = np.random.uniform(2, 10.8)
new_temp = np.random.uniform(45, 94)
new_fuel = np.random.uniform(80, 520)
new_row = pd.DataFrame({
    'Day': [new_day],
    'Compressor_Vibration': [new_vib],
    'Compressor_Temperature': [new_temp],
    'Fuel_Consumption': [new_fuel],
})
new_row['Carbon_Emission'] = new_row['Fuel_Consumption'] * 2.68 * (1 + new_row['Compressor_Vibration'] / 12)
new_row['Failure_Probability'] = np.clip(
    (new_row['Compressor_Vibration'] - 4) / 5.5 +
    (new_row['Compressor_Temperature'] - 60) / 32, 0, 0.96
)

df_combined = pd.concat([st.session_state.data, new_row], ignore_index=True)
# Keep only last MAX_DATA_ROWS rows to prevent infinite growth
if len(df_combined) > MAX_DATA_ROWS:
    df_combined = df_combined.tail(MAX_DATA_ROWS).reset_index(drop=True)
st.session_state.data = df_combined

X_current = st.session_state.data[['Compressor_Vibration', 'Compressor_Temperature', 'Fuel_Consumption']]
st.session_state.data['Predicted_Risk'] = st.session_state.model.predict_proba(X_current)[:, 1]

latest = st.session_state.data.iloc[-1]
current_day = int(latest['Day'])
risk = float(latest['Predicted_Risk'])
health_score = 100 - (risk * 100)
avg_risk = float(st.session_state.data['Predicted_Risk'].tail(30).mean())

# ──────────────── TABS ────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Live Dashboard",
    "🔮 Forecasting",
    "⚠️ Alerts & Actions",
    "🏢 About PEL"
])

# ══════════════════════════════════════
# TAB 1 – LIVE DASHBOARD
# ══════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        # Carbon Emission Chart
        st.markdown("<h3 style='font-size:16px;'>🌍 CARBON EMISSION TREND (Last 80 Days)</h3>", unsafe_allow_html=True)
        df_plot = st.session_state.data.tail(80).copy()
        fig_em = go.Figure()
        fig_em.add_trace(go.Scatter(
            x=df_plot['Day'], y=df_plot['Carbon_Emission'],
            mode='lines+markers',
            name='CO₂ Emission (kg)',
            line=dict(color='#00d4ff', width=2.5),
            marker=dict(size=4, color='#00d4ff'),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.06)'
        ))
        fig_em.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(10,25,50,0.5)',
            font_color='#a5d8ff', height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Day'),
            yaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='kg CO₂'),
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_em, use_container_width=True)

        # Failure Risk Chart
        st.markdown("<h3 style='font-size:16px;'>⚠️ FAILURE RISK TREND (Live)</h3>", unsafe_allow_html=True)
        fig_risk = go.Figure()
        # Color zones
        fig_risk.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(255,75,75,0.08)", line_width=0)
        fig_risk.add_hrect(y0=0.5, y1=0.7, fillcolor="rgba(255,165,0,0.06)", line_width=0)
        fig_risk.add_trace(go.Scatter(
            x=df_plot['Day'], y=df_plot['Predicted_Risk'],
            mode='lines+markers',
            name='Failure Risk',
            line=dict(color='#ff4b4b', width=2.5),
            marker=dict(size=4, color='#ff4b4b'),
            fill='tozeroy',
            fillcolor='rgba(255,75,75,0.06)'
        ))
        fig_risk.add_hline(y=0.7, line_dash="dash", line_color="#ff4b4b",
                           annotation_text="Critical Threshold (70%)", annotation_font_color="#ff4b4b")
        fig_risk.add_hline(y=0.5, line_dash="dot", line_color="#ffa500",
                           annotation_text="Warning Level (50%)", annotation_font_color="#ffa500")
        fig_risk.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(10,25,50,0.5)',
            font_color='#a5d8ff', height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Day'),
            yaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Risk Score', range=[0,1]),
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Sensor Readings
        st.markdown("<h3 style='font-size:16px;'>🌡️ SENSOR READINGS (Last 80 Days)</h3>", unsafe_allow_html=True)
        fig_sensor = go.Figure()
        fig_sensor.add_trace(go.Scatter(x=df_plot['Day'], y=df_plot['Compressor_Vibration'],
                                         name='Vibration (mm/s)', line=dict(color='#ffd166', width=2)))
        fig_sensor.add_trace(go.Scatter(x=df_plot['Day'], y=df_plot['Compressor_Temperature'],
                                         name='Temperature (°C)', line=dict(color='#ff6b6b', width=2),
                                         yaxis='y2'))
        fig_sensor.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(10,25,50,0.5)',
            font_color='#a5d8ff', height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Day'),
            yaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Vibration (mm/s)', side='left'),
            yaxis2=dict(title='Temperature (°C)', overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'),
            legend=dict(bgcolor='rgba(10,25,50,0.8)', bordercolor='rgba(0,212,255,0.3)', borderwidth=1)
        )
        st.plotly_chart(fig_sensor, use_container_width=True)

    with col2:
        # Health Score Gauge
        st.markdown("<h3 style='font-size:14px; text-align:center;'>💪 COMPRESSOR HEALTH</h3>", unsafe_allow_html=True)
        gauge_color = '#00ff88' if health_score > 70 else ('#ffa500' if health_score > 40 else '#ff4b4b')
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={'suffix': '%', 'font': {'color': gauge_color, 'size': 36, 'family': 'Orbitron'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#a5d8ff'},
                'bar': {'color': gauge_color, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': 'rgba(0,212,255,0.3)',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255,75,75,0.15)'},
                    {'range': [40, 70], 'color': 'rgba(255,165,0,0.12)'},
                    {'range': [70, 100], 'color': 'rgba(0,255,136,0.12)'},
                ],
                'threshold': {'value': 70, 'line': {'color': '#ff4b4b', 'width': 2}, 'thickness': 0.75}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#a5d8ff',
            height=220,
            margin=dict(l=15, r=15, t=20, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Current Readings
        st.markdown("<h3 style='font-size:14px;'>📟 CURRENT READINGS</h3>", unsafe_allow_html=True)
        st.metric("Monitoring Day", f"#{current_day}")
        st.metric("CO₂ Emission", f"{latest['Carbon_Emission']:.1f} kg")
        st.metric("Vibration", f"{latest['Compressor_Vibration']:.2f} mm/s")
        st.metric("Temperature", f"{latest['Compressor_Temperature']:.1f} °C")
        st.metric("Fuel Consumption", f"{latest['Fuel_Consumption']:.1f} L")
        st.metric("30-Day Avg Risk", f"{avg_risk*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        # Risk Progress Bar
        st.markdown(f"<div style='font-family:Rajdhani,sans-serif;color:#7ec8e3;font-size:13px;letter-spacing:1px;margin-bottom:5px;'>CURRENT FAILURE RISK</div>", unsafe_allow_html=True)
        st.progress(min(risk, 1.0))
        risk_label = "🔴 CRITICAL" if risk > 0.75 else ("🟡 ELEVATED" if risk > 0.5 else "🟢 NORMAL")
        st.markdown(f"<div style='text-align:center;font-family:Orbitron,monospace;font-size:20px;color:{gauge_color};font-weight:bold;'>{risk*100:.1f}% — {risk_label}</div>", unsafe_allow_html=True)

        # Model info
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.2);
                        border-radius:10px;padding:12px;font-family:Rajdhani,sans-serif;'>
                <div style='color:#7ec8e3;font-size:11px;letter-spacing:1px;'>🤖 MODEL PERFORMANCE</div>
                <div style='color:#00ff88;font-size:18px;font-weight:bold;margin-top:5px;'>
                    {st.session_state.accuracy*100:.1f}% Accuracy
                </div>
                <div style='color:#a5d8ff;font-size:12px;'>Random Forest · 150 Estimators</div>
                <div style='color:#a5d8ff;font-size:12px;'>Trained on 100 days baseline</div>
            </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════
# TAB 2 – FORECASTING
# ══════════════════════════════════════
with tab2:
    st.markdown("<h3 style='font-size:16px;'>🔮 30-DAY RISK FORECAST</h3>", unsafe_allow_html=True)

    # More realistic forecast with some variance
    np.random.seed(current_day)
    base_trend = [min(risk + (i * 0.012) + np.random.uniform(-0.03, 0.03), 0.98) for i in range(30)]
    forecast_days = list(range(current_day + 1, current_day + 31))
    df_forecast = pd.DataFrame({'Future_Day': forecast_days, 'Forecast_Risk': base_trend})
    df_forecast['Upper_CI'] = np.clip([r + 0.08 for r in base_trend], 0, 1)
    df_forecast['Lower_CI'] = np.clip([r - 0.08 for r in base_trend], 0, 1)

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=df_forecast['Future_Day'], y=df_forecast['Upper_CI'],
        fill=None, mode='lines', line_color='rgba(255,165,0,0)', showlegend=False
    ))
    fig_fc.add_trace(go.Scatter(
        x=df_forecast['Future_Day'], y=df_forecast['Lower_CI'],
        fill='tonexty', mode='lines', line_color='rgba(255,165,0,0)',
        fillcolor='rgba(255,165,0,0.1)', name='Confidence Interval'
    ))
    fig_fc.add_trace(go.Scatter(
        x=df_forecast['Future_Day'], y=df_forecast['Forecast_Risk'],
        mode='lines+markers', name='Predicted Risk',
        line=dict(color='#ffa500', width=3),
        marker=dict(size=5)
    ))
    fig_fc.add_hline(y=0.7, line_dash="dash", line_color="#ff4b4b",
                     annotation_text="⚠️ Critical Threshold", annotation_font_color="#ff4b4b")
    fig_fc.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(10,25,50,0.5)',
        font_color='#a5d8ff', height=350,
        xaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Future Day'),
        yaxis=dict(gridcolor='rgba(0,212,255,0.1)', title='Predicted Risk', range=[0,1]),
        legend=dict(bgcolor='rgba(10,25,50,0.8)')
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast Summary
    critical_days = [forecast_days[i] for i, r in enumerate(base_trend) if r > 0.7]
    warning_days  = [forecast_days[i] for i, r in enumerate(base_trend) if 0.5 < r <= 0.7]

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("<h3 style='font-size:15px;'>📋 FORECAST SUMMARY</h3>", unsafe_allow_html=True)
        peak_risk = max(base_trend)
        peak_day  = forecast_days[base_trend.index(peak_risk)]
        st.metric("Peak Predicted Risk", f"{peak_risk*100:.1f}%", delta=f"Day {peak_day}")
        st.metric("Critical Days (>70%)", len(critical_days))
        st.metric("Warning Days (50-70%)", len(warning_days))

    with fc2:
        st.markdown("<h3 style='font-size:15px;'>🗓️ ACTION CALENDAR</h3>", unsafe_allow_html=True)
        if critical_days:
            days_until = critical_days[0] - current_day
            st.markdown(f"""
                <div class='alert-critical'>
                    🚨 CRITICAL ACTION NEEDED IN {days_until} DAYS<br>
                    <small>First critical window: Day {critical_days[0]}</small>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>**High-risk days:**", unsafe_allow_html=True)
            cols_days = st.columns(3)
            for i, d in enumerate(critical_days[:9]):
                cols_days[i % 3].markdown(f"<div style='background:rgba(255,75,75,0.1);border:1px solid rgba(255,75,75,0.3);border-radius:8px;padding:8px;text-align:center;margin:3px;color:#ff8080;font-family:Rajdhani;'>Day {d}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-ok'>✅ No critical periods in next 30 days</div>", unsafe_allow_html=True)

    # Estimated CO2 & Cost Savings
    st.markdown("<br><h3 style='font-size:15px;'>💰 ESTIMATED SAVINGS (If Maintenance Done on Time)</h3>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    saved_co2 = round(len(critical_days) * 45.5, 0)
    s1.metric("CO₂ Prevented", f"{saved_co2} kg", delta="Environmental Benefit")
    s2.metric("Downtime Averted", f"{len(critical_days)} days", delta="Operational")
    s3.metric("Maintenance Priority", "HIGH" if critical_days else "LOW", delta="Risk-Based")

# ══════════════════════════════════════
# TAB 3 – ALERTS & ACTIONS
# ══════════════════════════════════════
with tab3:
    col_al1, col_al2 = st.columns([2, 1])
    with col_al1:
        st.markdown("<h3 style='font-size:15px;'>🔔 RECENT HIGH-RISK ALERTS</h3>", unsafe_allow_html=True)
        alerts = st.session_state.data[st.session_state.data['Predicted_Risk'] > 0.5].tail(15).copy()
        if not alerts.empty:
            alerts['Risk_Level'] = alerts['Predicted_Risk'].apply(
                lambda x: '🔴 CRITICAL' if x > 0.75 else ('🟡 WARNING' if x > 0.5 else '🟢 Normal')
            )
            display_cols = ['Day', 'Compressor_Vibration', 'Compressor_Temperature', 'Predicted_Risk', 'Risk_Level']
            st.dataframe(
                alerts[display_cols].rename(columns={
                    'Compressor_Vibration': 'Vibration (mm/s)',
                    'Compressor_Temperature': 'Temp (°C)',
                    'Predicted_Risk': 'Risk Score'
                }).style
                .format({'Vibration (mm/s)': '{:.2f}', 'Temp (°C)': '{:.1f}', 'Risk Score': '{:.1%}'}),
                height=350, use_container_width=True
            )
        else:
            st.markdown("<div class='alert-ok'>✅ No recent high-risk alerts — System is performing normally</div>", unsafe_allow_html=True)

    with col_al2:
        st.markdown("<h3 style='font-size:15px;'>📋 RECOMMENDED ACTIONS</h3>", unsafe_allow_html=True)
        actions = []
        if risk > 0.75:
            actions = [
                ("🚨", "URGENT", "Shutdown compressor for inspection", "#ff4b4b"),
                ("🔧", "URGENT", "Replace vibration dampeners immediately", "#ff4b4b"),
                ("📞", "URGENT", "Notify maintenance team NOW", "#ff4b4b"),
            ]
        elif risk > 0.5:
            actions = [
                ("⚙️", "WARNING", "Schedule maintenance within 48 hours", "#ffa500"),
                ("🌡️", "WARNING", "Check coolant system & filters", "#ffa500"),
                ("📊", "WARNING", "Increase monitoring frequency", "#ffa500"),
            ]
        else:
            actions = [
                ("✅", "OK", "Continue standard operations", "#00ff88"),
                ("📅", "OK", "Next scheduled check: 7 days", "#00ff88"),
                ("📄", "OK", "Log readings — all sensors nominal", "#00ff88"),
            ]
        for icon, level, action, color in actions:
            st.markdown(f"""
                <div style='background:rgba(0,0,0,0.2);border-left:3px solid {color};
                            border-radius:8px;padding:12px;margin:8px 0;'>
                    <div style='color:{color};font-size:11px;font-family:Rajdhani;letter-spacing:1px;'>{icon} {level}</div>
                    <div style='color:#ffffff;font-size:14px;font-family:Rajdhani;margin-top:3px;'>{action}</div>
                </div>
            """, unsafe_allow_html=True)

        # Overall status alert
        st.markdown("<br>", unsafe_allow_html=True)
        if risk > 0.75:
            st.markdown("<div class='alert-critical'>🚨 CRITICAL RISK DETECTED — IMMEDIATE ACTION REQUIRED</div>", unsafe_allow_html=True)
        elif risk > 0.5:
            st.markdown("<div class='alert-warning'>🟡 ELEVATED RISK — Schedule maintenance soon</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-ok'>🟢 ALL SYSTEMS NORMAL — No immediate action needed</div>", unsafe_allow_html=True)

        # ── EMAIL ALERT SECTION ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.25);
                        border-radius:12px;padding:18px;'>
                <div style='color:#00d4ff;font-family:Orbitron,monospace;font-size:13px;
                            font-weight:700;letter-spacing:1px;margin-bottom:12px;'>
                    📧 EMAIL ALERT SYSTEM
                </div>
        """, unsafe_allow_html=True)

        alert_email = st.text_input(
            "Recipient Email",
            placeholder="engineer@pel.com.pk",
            key="alert_email_input",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📧 Send Alert Now", use_container_width=True):
                if alert_email:
                    with st.spinner("Sending..."):
                        ok, msg_result = send_alert_email(
                            risk * 100, current_day,
                            float(latest['Compressor_Vibration']),
                            float(latest['Compressor_Temperature']),
                            alert_email
                        )
                    if ok:
                        st.session_state["last_alert_sent"] = datetime.now()
                        st.success("✅ Alert email sent!")
                    else:
                        st.error(f"❌ Failed: {msg_result}")
                else:
                    st.warning("Enter recipient email first")

        with col_btn2:
            auto_alert = st.toggle("🔔 Auto Alert", value=st.session_state.get("auto_alert_on", False), key="auto_alert_toggle")
            st.session_state["auto_alert_on"] = auto_alert

        # Show last sent time
        last_sent = st.session_state.get("last_alert_sent")
        if last_sent:
            st.markdown(f"<div style='color:#7ec8e3;font-size:12px;font-family:Rajdhani;margin-top:5px;'>Last sent: {last_sent.strftime('%d %b %Y %H:%M')}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-send logic
        if st.session_state.get("auto_alert_on") and alert_email and should_send_alert(risk):
            ok, _ = send_alert_email(
                risk * 100, current_day,
                float(latest['Compressor_Vibration']),
                float(latest['Compressor_Temperature']),
                alert_email
            )
            if ok:
                st.session_state["last_alert_sent"] = datetime.now()

    # Recent Data Table
    st.markdown("<br><h3 style='font-size:15px;'>📂 FULL DATA LOG (Last 30 Days)</h3>", unsafe_allow_html=True)
    recent_data = st.session_state.data.tail(30)[
        ['Day', 'Compressor_Vibration', 'Compressor_Temperature', 'Fuel_Consumption', 'Carbon_Emission', 'Predicted_Risk']
    ].copy()
    st.dataframe(
        recent_data.style.format({
            'Compressor_Vibration': '{:.2f}',
            'Compressor_Temperature': '{:.1f}',
            'Fuel_Consumption': '{:.1f}',
            'Carbon_Emission': '{:.1f}',
            'Predicted_Risk': '{:.1%}'
        }),
        use_container_width=True, height=300
    )

# ══════════════════════════════════════
# TAB 4 – ABOUT PEL (PORTFOLIO)
# ══════════════════════════════════════
with tab4:
    # Company Header
    st.markdown("""
        <div style='text-align:center;padding:30px 0;'>
            <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s'
                 width='120' style='border-radius:15px;border:2px solid rgba(0,212,255,0.5);
                 box-shadow:0 0 30px rgba(0,212,255,0.3);'>
            <h1 style='margin:20px 0 5px;font-size:28px;'>PETROLEUM EXPLORATION (PVT.) LTD.</h1>
            <p style='color:#a5d8ff;font-size:16px;font-family:Rajdhani;letter-spacing:3px;'>
                PAKISTAN'S PREMIER OIL & GAS EXPLORATION COMPANY
            </p>
            <div style='display:flex;justify-content:center;gap:15px;flex-wrap:wrap;margin-top:15px;'>
                <span style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
                             border-radius:20px;padding:5px 15px;color:#7ec8e3;font-size:13px;font-family:Rajdhani;'>
                    📍 Karachi, Pakistan
                </span>
                <span style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
                             border-radius:20px;padding:5px 15px;color:#7ec8e3;font-size:13px;font-family:Rajdhani;'>
                    🏭 Oil & Gas Industry
                </span>
                <span style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
                             border-radius:20px;padding:5px 15px;color:#7ec8e3;font-size:13px;font-family:Rajdhani;'>
                    ⚡ Since 1981
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Portfolio Cards
    st.markdown("<h3 style='font-size:16px;text-align:center;margin-bottom:20px;'>🏆 CORE CAPABILITIES</h3>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    cards = [
        (p1, "🛢️", "OIL EXPLORATION", "Seismic surveys, drilling operations, and reservoir management across Pakistan"),
        (p2, "⚙️", "PREDICTIVE AI", "Machine learning models to prevent failures and optimize compressor performance"),
        (p3, "🌍", "ESG & CARBON", "Real-time CO₂ monitoring and emission reduction through smart maintenance"),
        (p4, "📊", "DATA ANALYTICS", "Live dashboards, 30-day forecasts, and risk-based maintenance scheduling"),
    ]
    for col, icon, title, desc in cards:
        with col:
            st.markdown(f"""
                <div class='portfolio-card'>
                    <div class='portfolio-icon'>{icon}</div>
                    <div class='portfolio-title'>{title}</div>
                    <div class='portfolio-desc'>{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System Stats
    st.markdown("<h3 style='font-size:16px;text-align:center;margin-bottom:15px;'>📈 SYSTEM STATISTICS</h3>", unsafe_allow_html=True)
    total_emissions = st.session_state.data['Carbon_Emission'].sum()
    total_alerts    = len(st.session_state.data[st.session_state.data['Predicted_Risk'] > 0.7])
    avg_health      = 100 - (st.session_state.data['Predicted_Risk'].mean() * 100)

    ss1, ss2, ss3, ss4 = st.columns(4)
    ss1.metric("Days Monitored", len(st.session_state.data))
    ss2.metric("Total CO₂ Logged", f"{total_emissions/1000:.1f} tonnes")
    ss3.metric("Critical Alerts", total_alerts)
    ss4.metric("Avg System Health", f"{avg_health:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Technology Stack
    st.markdown("<h3 style='font-size:16px;text-align:center;margin-bottom:15px;'>🛠️ TECHNOLOGY STACK</h3>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    tech_cards = [
        (t1, [("🐍", "Python 3.11", "Core Language"),
              ("🤖", "Scikit-Learn", "ML / Random Forest"),
              ("📊", "Streamlit", "Dashboard Framework")]),
        (t2, [("📈", "Plotly", "Interactive Charts"),
              ("🔢", "NumPy / Pandas", "Data Processing"),
              ("🔐", "Secrets.toml", "Secure Auth")]),
        (t3, [("🌐", "Streamlit Cloud", "Deployment Platform"),
              ("⚙️", "Auto-Refresh", "Live Monitoring"),
              ("📥", "CSV Export", "Report Download")]),
    ]
    for col, items in tech_cards:
        with col:
            for icon, name, desc in items:
                st.markdown(f"""
                    <div style='background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                                border-radius:10px;padding:10px 15px;margin:5px 0;display:flex;align-items:center;gap:10px;'>
                        <span style='font-size:22px;'>{icon}</span>
                        <div>
                            <div style='color:#00d4ff;font-family:Rajdhani;font-weight:600;font-size:14px;'>{name}</div>
                            <div style='color:#7ec8e3;font-size:12px;font-family:Rajdhani;'>{desc}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Developer Credit
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background:linear-gradient(135deg,rgba(0,40,80,0.6),rgba(0,20,50,0.6));
                    border:1px solid rgba(0,212,255,0.2);border-radius:15px;padding:25px;text-align:center;'>
            <div style='color:#7ec8e3;font-size:13px;font-family:Rajdhani;letter-spacing:2px;margin-bottom:8px;'>
                DEVELOPED FOR
            </div>
            <div style='color:#00d4ff;font-size:20px;font-family:Orbitron;font-weight:bold;'>
                Petroleum Exploration (Pvt.) Ltd.
            </div>
            <div style='color:#a5d8ff;font-size:14px;font-family:Rajdhani;margin-top:5px;'>
                AI Predictive Maintenance Platform · Version 2.0 · 2025
            </div>
        </div>
    """, unsafe_allow_html=True)

# ──────────────── FOOTER ────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid rgba(0,212,255,0.15);'>", unsafe_allow_html=True)

footer_c1, footer_c2, footer_c3 = st.columns([2, 2, 1])
with footer_c1:
    st.markdown("""
        <p style='color:#7ec8e3;font-size:13px;font-family:Rajdhani;'>
            ⚙️ <b>PEL AI Predictive Maintenance v2.0</b><br>
            Petroleum Exploration (Pvt.) Ltd. | Karachi, Pakistan
        </p>
    """, unsafe_allow_html=True)
with footer_c2:
    st.markdown("""
        <p style='color:#7ec8e3;font-size:13px;font-family:Rajdhani;'>
            📧 it@pel.com.pk &nbsp;|&nbsp; 🌐 www.pel.com.pk<br>
            🔒 Secure • Live • Intelligent
        </p>
    """, unsafe_allow_html=True)
with footer_c3:
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(st.session_state.data.tail(60))
    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name=f"PEL_Report_Day_{current_day}.csv",
        mime='text/csv',
        use_container_width=True
    )
