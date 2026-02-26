import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ──────────────── PASSWORD PROTECTION WITH PEL LOGO ────────────────
def check_password():
    """Simple password check using secrets.toml"""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Password screen with PEL logo
        st.markdown("""
            <div style="text-align: center; margin: 50px 0;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s" width="150" alt="PEL Logo">
            </div>
            <h2 style="text-align: center; color: #00d4ff;">PEL Predictive Maintenance Dashboard</h2>
            <p style="text-align: center; color: #a5d8ff;">Enter password to access</p>
        """, unsafe_allow_html=True)
        
        st.text_input("🔒 Password", type="password", on_change=password_entered, key="password")
        st.caption("Contact PEL IT/Admin for password")
        return False
    
    elif not st.session_state["password_correct"]:
        st.markdown("""
            <div style="text-align: center; margin: 50px 0;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s" width="150" alt="PEL Logo">
            </div>
            <h2 style="text-align: center; color: #00d4ff;">PEL Predictive Maintenance Dashboard</h2>
        """, unsafe_allow_html=True)
        
        st.text_input("🔒 Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Wrong password. Try again.")
        return False
    
    else:
        return True

if not check_password():
    st.stop()  # Stop execution until correct password

# ──────────────── POLISHED DARK THEME ────────────────
st.markdown("""
    <style>
    .main { background-color: #0d1b2a; color: #ffffff; }
    h1, h2, h3 { color: #00d4ff; }
    .stButton > button { background-color: #00b4d8; color: #0d1b2a; font-weight: bold; border: none; border-radius: 8px; }
    .stProgress > div > div > div { background: linear-gradient(to right, #00b4d8, #48cae4); }
    
    div[data-testid="metric-container"] {
        background-color: #1b263b !important;
        border: 1px solid #33415c !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 8px 0 !important;
        box-shadow: 0 0 15px rgba(0,180,216,0.4) !important;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 0 30px rgba(0,180,216,0.7) !important;
        transform: translateY(-3px);
    }
    div[data-testid="metric-container"] label {
        color: #a5d8ff !important;
        font-size: 16px !important;
    }
    div[data-testid="metric-container"] > div > div:nth-child(2) {
        color: #ffffff !important;
        font-size: 44px !important;
        font-weight: bold !important;
    }
    div[data-testid="metric-delta"] {
        color: #90ff90 !important;
        font-size: 20px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────── HEADER WITH PEL LOGO ────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFQ3PjNzDquakJIda7FDzsH32tqqD-_vomTQ&s", width=120)
with col_title:
    st.markdown("""
        <h1 style='margin: 0; font-size: 42px;'>PEL – AI Predictive Maintenance</h1>
        <p style='color: #a5d8ff; font-size: 20px; margin: 10px 0 0 0;'>Proactive Asset Reliability • Reduced Downtime • Carbon Reduction</p>
    """, unsafe_allow_html=True)

st_autorefresh(interval=10000, key="refresh")

# ──────────────── KEY HIGHLIGHTS ────────────────
st.subheader("Key Business Benefits")
col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Downtime Reduction", "Up to 40%", delta="High Confidence")
col_h2.metric("CO₂ Savings", "~200-600 kg/month", delta="Sustainability")
col_h3.metric("Maintenance Efficiency", "Risk-Based", delta="Optimized")

# ──────────────── DATA & MODEL (same) ────────────────
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

current_max_day = int(st.session_state.data['Day'].max())
new_day = current_max_day + 1
new_row = pd.DataFrame({
    'Day': [new_day],
    'Compressor_Vibration': [np.random.uniform(2, 10.8)],
    'Compressor_Temperature': [np.random.uniform(45, 94)],
    'Fuel_Consumption': [np.random.uniform(80, 520)],
})
new_row['Carbon_Emission'] = new_row['Fuel_Consumption'] * 2.68 * (1 + new_row['Compressor_Vibration'] / 12)
new_row['Failure_Probability'] = np.clip(
    (new_row['Compressor_Vibration'] - 4) / 5.5 + (new_row['Compressor_Temperature'] - 60) / 32, 0, 0.96
)
st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)

X_current = st.session_state.data[['Compressor_Vibration', 'Compressor_Temperature', 'Fuel_Consumption']]
st.session_state.data['Predicted_Risk'] = st.session_state.model.predict_proba(X_current)[:, 1]

latest = st.session_state.data.iloc[-1]
current_day = int(latest['Day'])
risk = latest['Predicted_Risk']
health_score = 100 - (risk * 100)

# ──────────────── MAIN LAYOUT ────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Carbon Emission Trend (Last 80 Days) 🌍")
    fig_em = px.line(st.session_state.data.tail(80), x='Day', y='Carbon_Emission', markers=True)
    fig_em.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
    fig_em.update_traces(line_color='#00d4ff', line_width=3)
    st.plotly_chart(fig_em, use_container_width=True)

    st.subheader("Failure Risk Trend (Live) ⚠️")
    fig_risk = px.line(st.session_state.data.tail(80), x='Day', y='Predicted_Risk', markers=True)
    fig_risk.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
    fig_risk.update_traces(line_color='#ff4b4b', line_width=3)
    fig_risk.update_yaxes(range=[0, 1])
    fig_risk.add_hline(y=0.7, line_dash="dash", line_color="#ff4b4b")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.subheader("30-Day Risk Forecast 🔮")
    forecast_risk = [min(risk + (i * 0.016), 0.98) for i in range(30)]
    forecast_days = range(current_day + 1, current_day + 31)
    df_forecast = pd.DataFrame({'Future_Day': forecast_days, 'Forecast_Risk': forecast_risk})
    fig_fc = px.line(df_forecast, x='Future_Day', y='Forecast_Risk')
    fig_fc.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff')
    fig_fc.update_traces(line_color='#ffa500', line_width=3)
    fig_fc.add_hline(y=0.7, line_dash="dash", line_color="#ff4b4b")
    st.plotly_chart(fig_fc, use_container_width=True)

with col2:
    st.subheader("Compressor Health 💪")
    st.metric("Health Score", f"{health_score:.0f}%", delta="Good" if health_score > 70 else "Critical")

    st.subheader("Current Readings 📊")
    st.metric("Current Day", current_day)
    st.metric("Emission (kg CO₂) 🌍", f"{latest['Carbon_Emission']:.1f}")
    st.progress(risk)
    st.caption(f"**Failure Risk: {risk*100:.1f}%**")

    if risk > 0.75:
        st.markdown("<div style='background-color:#ff4b4b; color:white; padding:14px; border-radius:10px; text-align:center; font-weight:bold; font-size:18px;'>URGENT: Critical Risk Detected – Immediate Action Required!</div>", unsafe_allow_html=True)
    elif risk > 0.50:
        st.warning("🟡 Elevated Risk – Schedule Maintenance Soon")
    else:
        st.success("🟢 Normal Operation – All Good")

    st.subheader("Maintenance Schedule (Next 30 Days)")
    high_risk_days = [current_day + i for i in range(1, 31) if min(risk + (i * 0.016), 0.98) > 0.7]
    if high_risk_days:
        st.error(f"**First action needed in {high_risk_days[0] - current_day} days (Day {high_risk_days[0]})**")
        st.write("**High-risk days:**")
        for d in high_risk_days[:6]:
            st.write(f"• Day {d}")
    else:
        st.success("No critical period in next 30 days")

    st.subheader("Estimated Impact")
    saved_co2 = round(np.random.uniform(280, 650), 0) if high_risk_days else 0
    if saved_co2 > 0:
        st.success(f"Timely action can save **~{saved_co2} kg CO₂**")
    else:
        st.info("Low risk – minimal savings opportunity now")

    st.subheader("Recent Alerts")
    alerts = st.session_state.data[st.session_state.data['Predicted_Risk'] > 0.6].tail(8)
    if not alerts.empty:
        st.dataframe(
            alerts[['Day', 'Predicted_Risk']].style
                .format({"Predicted_Risk": "{:.1%}"})
                .background_gradient(cmap='OrRd', subset=['Predicted_Risk']),
            height=160
        )
    else:
        st.success("No recent high-risk alerts")

    st.subheader("Model Performance")
    st.info(f"Accuracy: {st.session_state.accuracy*100:.1f}% | Trained on 100 days")

# ──────────────── FOOTER ────────────────
st.markdown("---")
st.markdown("<p style='text-align: center; color: #a5d8ff; font-size: 14px;'>Developed for PEL – Petroleum Exploration (Pvt.) Ltd. | 2025</p>", unsafe_allow_html=True)

# ──────────────── DOWNLOAD ────────────────
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(st.session_state.data.tail(60))
st.download_button(
    label="📥 Download Report (Last 60 Days)",
    data=csv,
    file_name=f"PEL_Compressor_Report_Day_{current_day}.csv",
    mime='text/csv',
    use_container_width=True
)