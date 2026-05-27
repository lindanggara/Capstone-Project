# =====================================================
# WELLGUARD DASHBOARD — FIXED & ALIGNED WITH DATA
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="WellGuard Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }
.main { background-color: #f0f4f8; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: white; }

div[data-testid="metric-container"] {
    background: white;
    border: none;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 14px;
    padding: 12px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,0.4);
}

.section-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

h1, h2, h3 { color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# =====================================================
# MODEL ARCHITECTURE (sesuai wellguard_model.pt)
# =====================================================

class FourierEmbedder(nn.Module):
    def __init__(self, n_features=12, freq_dim=16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_features, freq_dim))
        self.linear = nn.Linear(n_features * freq_dim * 2, n_features * freq_dim * 2 // 2)

    def forward(self, x):
        angles = x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        emb = emb.view(emb.size(0), -1)
        return self.linear(emb)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.fc2  = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.act  = nn.GELU()

    def forward(self, x):
        res = self.proj(x)
        x   = self.act(self.bn1(self.fc1(x)))
        x   = self.bn2(self.fc2(x))
        return self.act(x + res)


class WellGuardModel(nn.Module):
    def __init__(self, n_features=12, freq_dim=16):
        super().__init__()
        embed_dim = n_features * freq_dim * 2 // 2   # 192
        self.embedder  = FourierEmbedder(n_features, freq_dim)
        self.input_bn  = nn.BatchNorm1d(embed_dim)
        self.blocks    = nn.ModuleList([
            ResBlock(192, 512),
            ResBlock(512, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
        ])
        self.head = nn.Linear(128, 3)

    def forward(self, x):
        x = self.embedder(x)
        x = self.input_bn(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    m = WellGuardModel()
    ckpt = torch.load("wellguard_model.pt", map_location="cpu")
    m.load_state_dict(ckpt)
    m.eval()
    return m

model = load_model()

st.markdown("""
<div style="
    background: linear-gradient(90deg, #dcfce7, #f0fdf4);
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 12px 20px;
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 10px;
">
<span style="font-size:20px;">✅</span>
<span style="color:#166534; font-weight:600; font-size:14px;">
    Model AI WellGuard (Fourier + ResNet) berhasil dimuat dan siap digunakan
</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD & CLEAN DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("wellguard.csv")

    # sex_label sudah ada di CSV — gunakan langsung, jangan re-map
    # Isi NaN sex_label dengan 'Unknown'
    df['sex_label'] = df['sex_label'].fillna('Unknown')

    # age_group sudah ada di CSV — gunakan langsung
    df['age_group'] = df['age_group'].astype(str)

    # Feature engineering
    df['stress_sleep_ratio'] = df['stress'] / (df['sleep'] + 1)
    df['sleep_deficit']      = 7 - df['sleep']
    df['age_stress']         = (df['age'] * df['stress']) / 100

    return df

df = load_data()

# =====================================================
# STATS UNTUK NORMALISASI PREDIKSI
# Menggunakan 12 fitur sesuai arsitektur model
# =====================================================

@st.cache_data
def compute_stats(_df):
    """Hitung stats normalisasi dari full dataset (tanpa NaN sex)."""
    d = _df[_df['sex_label'] != 'Unknown'].copy()
    # fitur tambahan
    d['sex_num']     = d['sex'].astype(float)
    d['sleep_sq']    = d['sleep'] ** 2
    d['stress_sq']   = d['stress'] ** 2
    d['age_sq']      = d['age'] ** 2
    d['sex_stress']  = d['sex_num'] * d['stress']

    feat_cols = ['age','sex_num','sleep','stress','shift',
                 'stress_sleep_ratio','sleep_deficit','age_stress',
                 'sleep_sq','stress_sq','age_sq','sex_stress']
    stats = {}
    for c in feat_cols:
        stats[c] = {'mean': float(d[c].mean()), 'std': float(d[c].std()) + 1e-8,
                    'min': float(d[c].min()), 'max': float(d[c].max())}
    return stats

STATS = compute_stats(df)

FEAT_COLS = ['age','sex_num','sleep','stress','shift',
             'stress_sleep_ratio','sleep_deficit','age_stress',
             'sleep_sq','stress_sq','age_sq','sex_stress']

def build_features(age, sex_num, sleep, stress, shift=1):
    """Buat vektor 12 fitur, normalisasi min-max dari full dataset."""
    ssr     = stress / (sleep + 1)
    sd      = 7 - sleep
    as_     = (age * stress) / 100
    slp_sq  = sleep ** 2
    str_sq  = stress ** 2
    age_sq  = age ** 2
    ss      = sex_num * stress

    raw = [age, sex_num, sleep, stress, shift,
           ssr, sd, as_, slp_sq, str_sq, age_sq, ss]

    normed = []
    for val, col in zip(raw, FEAT_COLS):
        mn = STATS[col]['min']
        mx = STATS[col]['max']
        normed.append((val - mn) / (mx - mn + 1e-8))
    return normed


def rule_based_pred(sleep, stress):
    """Rule-based override berdasarkan threshold empiris dari data."""
    if sleep >= 7.0 and stress <= 4.5:
        return 0   # Rendah
    elif sleep <= 4.8 and stress >= 7.0:
        return 2   # Tinggi
    else:
        return None  # biarkan model


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #0ea5e9 100%);
    padding: 36px 40px;
    border-radius: 24px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(37,99,235,0.3);
    position: relative; overflow: hidden;
">
<div style="position:absolute; top:-40px; right:-40px; width:200px; height:200px;
background:rgba(255,255,255,0.05); border-radius:50%;"></div>
<div style="position:absolute; bottom:-60px; left:30%; width:150px; height:150px;
background:rgba(255,255,255,0.04); border-radius:50%;"></div>
<h1 style="color:white; text-align:center; font-size:36px; font-weight:900;
margin:0 0 8px 0; letter-spacing:-0.5px;">🛡️ WellGuard Dashboard</h1>
<p style="color:rgba(255,255,255,0.85); text-align:center; font-size:16px;
margin:0; letter-spacing:1px;">AI-BASED EMPLOYEE FATIGUE MONITORING SYSTEM</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR — FILTER
# =====================================================

st.sidebar.markdown("""
<div style="text-align:center; padding:16px 0 8px 0;
border-bottom:1px solid rgba(255,255,255,0.1); margin-bottom:16px;">
<div style="font-size:28px;">🛡️</div>
<div style="font-weight:800; font-size:16px; letter-spacing:1px;">WellGuard</div>
<div style="font-size:10px; color:#94a3b8; letter-spacing:2px;">AI FATIGUE MONITOR</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin-bottom:6px;">FILTER DATA</div>""", unsafe_allow_html=True)

# Gender filter — nilai dari CSV: Female, Male, Unknown
gender_options = sorted([g for g in df['sex_label'].unique() if g != 'Unknown'])
gender_options_all = gender_options + (['Unknown'] if 'Unknown' in df['sex_label'].values else [])

selected_gender = st.sidebar.multiselect(
    "👤 Gender",
    options=gender_options_all,
    default=gender_options   # default: Female + Male (exclude Unknown)
)

selected_fatigue = st.sidebar.multiselect(
    "⚡ Level Fatigue",
    options=df['fatigue_label'].unique().tolist(),
    default=df['fatigue_label'].unique().tolist()
)

# =====================================================
# SIDEBAR — PREDIKSI AI
# =====================================================

st.sidebar.markdown("""
<div style="border-top:1px solid rgba(255,255,255,0.1); margin:16px 0;"></div>
<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin-bottom:12px;">🧠 PREDIKSI FATIGUE AI</div>
""", unsafe_allow_html=True)

# Input fields
st.sidebar.markdown("""<div style="background:rgba(255,255,255,0.06); border-radius:12px;
padding:4px 10px 10px 10px; margin-bottom:8px;">
<div style="font-size:10px; color:#94a3b8; letter-spacing:1px; margin:6px 0 2px;">
👤 USIA KARYAWAN</div>""", unsafe_allow_html=True)
input_age = st.sidebar.slider("Usia", 18, 69, 30, label_visibility="collapsed")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="background:rgba(255,255,255,0.06); border-radius:12px;
padding:4px 10px 10px 10px; margin-bottom:8px;">
<div style="font-size:10px; color:#94a3b8; letter-spacing:1px; margin:6px 0 2px;">
⚥ GENDER</div>""", unsafe_allow_html=True)
input_gender = st.sidebar.selectbox("Gender", ["Female", "Male"],
                                    label_visibility="collapsed")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="background:rgba(255,255,255,0.06); border-radius:12px;
padding:4px 10px 10px 10px; margin-bottom:8px;">
<div style="font-size:10px; color:#94a3b8; letter-spacing:1px; margin:6px 0 2px;">
🌙 DURASI TIDUR (JAM)</div>""", unsafe_allow_html=True)
input_sleep = st.sidebar.slider("Tidur", 3.0, 10.5, 6.0, step=0.1,
                                label_visibility="collapsed")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("""<div style="background:rgba(255,255,255,0.06); border-radius:12px;
padding:4px 10px 10px 10px; margin-bottom:12px;">
<div style="font-size:10px; color:#94a3b8; letter-spacing:1px; margin:6px 0 2px;">
😰 TINGKAT STRESS (1–10)</div>""", unsafe_allow_html=True)
input_stress = st.sidebar.slider("Stress", 1.0, 10.0, 5.0, step=0.1,
                                 label_visibility="collapsed")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    btn_predict  = st.button("🔍 Prediksi",  use_container_width=True)
with col_btn2:
    btn_simulate = st.button("⚡ Simulasi", use_container_width=True)

if btn_simulate:
    st.sidebar.markdown("""
    <div style="background:#7f1d1d; border-radius:10px; padding:10px 12px;
    margin-top:8px; font-size:11px; color:#fca5a5; line-height:1.6;">
    ⚡ <b>Skenario Ekstrem:</b><br>
    Tidur: 4 jam | Stress: 9/10<br>
    → Kemungkinan: <b>FATIGUE TINGGI</b>
    </div>
    """, unsafe_allow_html=True)

# ── Logika prediksi ──
if btn_predict:
    with st.spinner("🧠 Model AI sedang menganalisis..."):
        sex_num = 0.0 if input_gender == "Female" else 1.0

        # Cek rule-based override dulu
        rule_pred = rule_based_pred(input_sleep, input_stress)

        # Jalankan model untuk confidence scores
        feats       = build_features(input_age, sex_num, input_sleep, input_stress)
        input_t     = torch.tensor([feats], dtype=torch.float32)
        with torch.no_grad():
            logits  = model(input_t)
            prob    = torch.softmax(logits, dim=1)
        prob_np     = prob.numpy()[0]
        ai_pred     = int(torch.argmax(logits, dim=1).item())

        # Gunakan rule-based jika kondisi ekstrem, else gunakan AI
        pred = rule_pred if rule_pred is not None else ai_pred

    label_map = {
        0: ('🟢', 'RENDAH', '#22c55e', '#dcfce7', 16),
        1: ('🟡', 'SEDANG', '#f59e0b', '#fef9c3', 50),
        2: ('🔴', 'TINGGI', '#ef4444', '#fee2e2', 84),
    }
    icon, label, color, bg, gauge_val = label_map[pred]

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_val,
        number={'suffix': "", 'font': {'size': 1, 'color': 'rgba(0,0,0,0)'}},
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar':  {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0,  33], 'color': '#dcfce7'},
                {'range': [33, 66], 'color': '#fef9c3'},
                {'range': [66,100], 'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.85, 'value': gauge_val
            }
        }
    ))
    fig_gauge.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.sidebar.plotly_chart(fig_gauge, use_container_width=True)

    st.sidebar.markdown(f"""
    <div style="background:{bg}; border:2px solid {color}; border-radius:16px;
    padding:14px 12px; margin-top:-10px; text-align:center;">
        <div style="font-size:32px; line-height:1;">{icon}</div>
        <div style="font-size:10px; color:#6b7280; font-weight:700;
        letter-spacing:2px; margin-top:4px;">LEVEL FATIGUE</div>
        <div style="font-size:24px; font-weight:900; color:{color};
        letter-spacing:3px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    st.sidebar.markdown("""<div style="font-size:10px; font-weight:700; color:#64748b;
    letter-spacing:2px; margin:14px 0 8px 0;">CONFIDENCE SCORE</div>""",
    unsafe_allow_html=True)

    for i, (lbl, clr) in enumerate(
            zip(['Rendah','Sedang','Tinggi'], ['#22c55e','#f59e0b','#ef4444'])):
        pct    = round(float(prob_np[i]) * 100, 1)
        active = "⬅" if i == pred else ""
        st.sidebar.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between;
            font-size:11px; color:#cbd5e1; margin-bottom:3px;">
                <span>{lbl} {active}</span>
                <span style="color:{clr}; font-weight:700;">{pct}%</span>
            </div>
            <div style="background:#1e293b; border-radius:99px; height:7px; overflow:hidden;">
                <div style="width:{pct}%; background:{clr}; height:100%; border-radius:99px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Rekomendasi
    rek_map = {
        0: ("✅ Kondisi Baik",
            "Pertahankan pola tidur 7–8 jam dan jaga stress di level rendah.",
            "#166534", "#dcfce7", "#86efac"),
        1: ("⚠️ Perlu Perhatian",
            "Tingkatkan kualitas tidur dan lakukan teknik relaksasi setiap hari.",
            "#92400e", "#fef9c3", "#fde047"),
        2: ("🚨 Risiko Tinggi",
            "Segera konsultasikan ke tim kesehatan kerja. Kurangi beban kerja.",
            "#991b1b", "#fee2e2", "#fca5a5"),
    }
    rek_title, rek_msg, rek_text, rek_bg, rek_border = rek_map[pred]
    st.sidebar.markdown(f"""
    <div style="background:{rek_bg}; border:1px solid {rek_border};
    border-radius:12px; padding:12px; margin-top:10px;">
        <div style="font-size:12px; font-weight:800; color:{rek_text};
        margin-bottom:4px;">{rek_title}</div>
        <div style="font-size:11px; color:{rek_text}; line-height:1.5; opacity:0.85;">
        {rek_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Simpan riwayat
    st.session_state.prediction_history.append({
        'Waktu':   datetime.now().strftime("%H:%M:%S"),
        'Usia':    input_age,
        'Gender':  input_gender,
        'Tidur':   input_sleep,
        'Stress':  input_stress,
        'Hasil':   label,
    })
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history.pop(0)

# Riwayat prediksi
if st.session_state.prediction_history:
    st.sidebar.markdown("""<div style="font-size:10px; font-weight:700; color:#64748b;
    letter-spacing:2px; margin:14px 0 8px 0;">📋 RIWAYAT PREDIKSI</div>""",
    unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state.prediction_history)
    st.sidebar.dataframe(hist_df, use_container_width=True, hide_index=True)
    if st.sidebar.button("🗑️ Hapus Riwayat", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()

# =====================================================
# FILTER DATA
# =====================================================

filtered_df = df[
    (df['sex_label'].isin(selected_gender)) &
    (df['fatigue_label'].isin(selected_fatigue))
]

# Guard kosong
if len(filtered_df) == 0:
    st.warning("⚠️ Tidak ada data yang sesuai dengan filter aktif.")
    st.stop()

# =====================================================
# KPI CARDS
# =====================================================

st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin-bottom:12px;">📊 RINGKASAN DATA</div>""",
unsafe_allow_html=True)

total_data  = len(filtered_df)
avg_sleep   = round(filtered_df['sleep'].mean(),  2)
avg_stress  = round(filtered_df['stress'].mean(), 2)
avg_age     = round(filtered_df['age'].mean(),    1)

fatigue_counts_kpi = filtered_df['fatigue_label'].value_counts()
most_fatigue_kpi   = fatigue_counts_kpi.idxmax()
most_pct_kpi       = round(fatigue_counts_kpi.max() / total_data * 100, 1)

sleep_pct    = min(round((avg_sleep  / 10.5) * 100), 100)
stress_pct   = min(round((avg_stress / 10.0) * 100), 100)
sleep_color  = '#22c55e' if avg_sleep >= 7 else ('#f59e0b' if avg_sleep >= 5 else '#ef4444')
stress_color = '#ef4444' if avg_stress >= 7 else ('#f59e0b' if avg_stress >= 5 else '#22c55e')

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="section-card" style="padding:20px;">
        <div style="font-size:11px; color:#64748b; font-weight:700;
        letter-spacing:1px; margin-bottom:6px;">👥 TOTAL KARYAWAN</div>
        <div style="font-size:36px; font-weight:900; color:#0f172a;
        line-height:1;">{total_data:,}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">data terekam</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="section-card" style="padding:20px;">
        <div style="font-size:11px; color:#64748b; font-weight:700;
        letter-spacing:1px; margin-bottom:6px;">🌙 RATA-RATA TIDUR</div>
        <div style="font-size:36px; font-weight:900; color:#0f172a;
        line-height:1;">{avg_sleep}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">
        jam/malam (target: 7 jam)</div>
        <div style="background:#f1f5f9; border-radius:99px; height:6px;
        overflow:hidden; margin-top:8px;">
            <div style="width:{sleep_pct}%; background:{sleep_color};
            height:100%; border-radius:99px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="section-card" style="padding:20px;">
        <div style="font-size:11px; color:#64748b; font-weight:700;
        letter-spacing:1px; margin-bottom:6px;">😰 RATA-RATA STRESS</div>
        <div style="font-size:36px; font-weight:900; color:#0f172a;
        line-height:1;">{avg_stress}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">
        skala 1–10 (batas aman: ≤5)</div>
        <div style="background:#f1f5f9; border-radius:99px; height:6px;
        overflow:hidden; margin-top:8px;">
            <div style="width:{stress_pct}%; background:{stress_color};
            height:100%; border-radius:99px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="section-card" style="padding:20px;">
        <div style="font-size:11px; color:#64748b; font-weight:700;
        letter-spacing:1px; margin-bottom:6px;">⚡ FATIGUE DOMINAN</div>
        <div style="font-size:28px; font-weight:900; color:#0f172a;
        line-height:1; margin-top:4px;">{most_fatigue_kpi}</div>
        <div style="font-size:12px; color:#64748b; margin-top:4px;">
        {most_pct_kpi}% dari total data</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# VISUALISASI
# =====================================================

COLOR_MAP = {
    'Rendah': '#22c55e',
    'Sedang': '#f59e0b',
    'Tinggi': '#ef4444'
}

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
    letter-spacing:2px; margin-bottom:10px;">📌 DISTRIBUSI FATIGUE</div>""",
    unsafe_allow_html=True)
    fig_pie = px.pie(
        filtered_df, names='fatigue_label', hole=0.5,
        color='fatigue_label', color_discrete_map=COLOR_MAP
    )
    fig_pie.update_traces(
        textposition='outside',
        textinfo='percent+label',
        marker=dict(line=dict(color='white', width=3))
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True, margin=dict(t=20, b=20, l=20, r=20), height=320
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
    letter-spacing:2px; margin-bottom:10px;">📈 SLEEP VS STRESS</div>""",
    unsafe_allow_html=True)
    fig_scatter = px.scatter(
        filtered_df.sample(min(3000, len(filtered_df)), random_state=42),
        x='sleep', y='stress',
        color='fatigue_label', size_max=8,
        hover_data=['age', 'sex_label'],
        color_discrete_map=COLOR_MAP,
        labels={'sleep': 'Durasi Tidur (jam)', 'stress': 'Tingkat Stress'}
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafafa',
        margin=dict(t=20, b=20, l=20, r=20), height=320,
        xaxis=dict(gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
    letter-spacing:2px; margin-bottom:10px;">👨‍💼 FATIGUE PER GENDER</div>""",
    unsafe_allow_html=True)
    # Filter hanya Female & Male untuk chart ini
    gender_df = filtered_df[filtered_df['sex_label'].isin(['Female','Male'])]
    gender_analysis = (
        gender_df.groupby(['sex_label', 'fatigue_label'])
        .size().reset_index(name='count')
    )
    fig_gender = px.bar(
        gender_analysis, x='sex_label', y='count',
        color='fatigue_label', barmode='group',
        color_discrete_map=COLOR_MAP,
        labels={'sex_label': 'Gender', 'count': 'Jumlah'}
    )
    fig_gender.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafafa',
        margin=dict(t=20, b=20, l=20, r=20), height=300,
        xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with col_right2:
    st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
    letter-spacing:2px; margin-bottom:10px;">👥 FATIGUE PER KELOMPOK USIA</div>""",
    unsafe_allow_html=True)
    # Urutkan age_group secara logis
    age_order = ['Muda (<30)', 'Dewasa (30-45)', 'Senior (>45)']
    age_analysis = (
        filtered_df.groupby(['age_group', 'fatigue_label'])
        .size().reset_index(name='count')
    )
    age_analysis['age_group'] = pd.Categorical(
        age_analysis['age_group'], categories=age_order, ordered=True
    )
    age_analysis = age_analysis.sort_values('age_group')
    fig_age = px.bar(
        age_analysis, x='age_group', y='count',
        color='fatigue_label', color_discrete_map=COLOR_MAP,
        labels={'age_group': 'Kelompok Usia', 'count': 'Jumlah'}
    )
    fig_age.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafafa',
        margin=dict(t=20, b=20, l=20, r=20), height=300,
        xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig_age, use_container_width=True)

# ── Feature Engineering Histogram ──
st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin-bottom:10px;">🧠 DISTRIBUSI FEATURE ENGINEERING</div>""",
unsafe_allow_html=True)

feat_labels = {
    'stress_sleep_ratio': 'Stress/Sleep Ratio',
    'sleep_deficit':      'Sleep Deficit (jam kurang dari 7)',
    'age_stress':         'Age × Stress Index'
}
selected_feature = st.selectbox("Pilih Feature", list(feat_labels.keys()),
                                format_func=lambda x: feat_labels[x])
fig_feature = px.histogram(
    filtered_df, x=selected_feature,
    color='fatigue_label', barmode='overlay',
    color_discrete_map=COLOR_MAP, opacity=0.75,
    labels={selected_feature: feat_labels[selected_feature]}
)
fig_feature.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafafa',
    margin=dict(t=20, b=20, l=20, r=20), height=300,
    xaxis=dict(gridcolor='#f1f5f9'), yaxis=dict(gridcolor='#f1f5f9'),
)
st.plotly_chart(fig_feature, use_container_width=True)

# ── Heatmap Korelasi ──
st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin-bottom:10px;">🔥 HEATMAP KORELASI</div>""",
unsafe_allow_html=True)

numeric_cols = filtered_df.select_dtypes(include=np.number).drop(
    columns=['id', 'sex', 'shift', 'fatigue'], errors='ignore'
)
corr = numeric_cols.corr()
fig_hm, ax = plt.subplots(figsize=(10, 5))
fig_hm.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')
sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={"size": 9}, fmt='.2f')
plt.tight_layout()
st.pyplot(fig_hm)

# ── Preview Dataset ──
st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin:20px 0 10px 0;">📄 PREVIEW DATASET</div>""",
unsafe_allow_html=True)

preview_cols = ['id', 'age', 'sex_label', 'sleep', 'stress', 'age_group',
                'fatigue_label', 'stress_sleep_ratio', 'sleep_deficit', 'age_stress']
st.dataframe(filtered_df[preview_cols].head(20), use_container_width=True)

# =====================================================
# INSIGHT & KESIMPULAN
# =====================================================

st.markdown("""<div style="font-size:11px; font-weight:700; color:#64748b;
letter-spacing:2px; margin:24px 0 16px 0;">📝 INSIGHT & KESIMPULAN</div>""",
unsafe_allow_html=True)

avg_sleep_i  = round(filtered_df['sleep'].mean(),  2)
avg_stress_i = round(filtered_df['stress'].mean(), 2)
avg_age_i    = round(filtered_df['age'].mean(),    1)
total_i      = len(filtered_df)

fatigue_counts_i = filtered_df['fatigue_label'].value_counts()
most_fatigue_i   = fatigue_counts_i.idxmax()
least_fatigue_i  = fatigue_counts_i.idxmin() if len(fatigue_counts_i) > 1 else most_fatigue_i
fatigue_pct_i    = {lbl: round(cnt / total_i * 100, 2) for lbl, cnt in fatigue_counts_i.items()}

# KPI insight cards
ic1, ic2, ic3, ic4 = st.columns(4)
insight_cards = [
    ("👥", "Total Karyawan",   f"{total_i:,}",       "data dianalisis",           "#2563eb", "#eff6ff"),
    ("🌙", "Rata-rata Tidur",  f"{avg_sleep_i} jam",  "per malam",                 "#0891b2", "#ecfeff"),
    ("😰", "Rata-rata Stress", f"{avg_stress_i}",     "skala 1–10",                "#d97706", "#fffbeb"),
    ("⚡", "Fatigue Dominan",  most_fatigue_i,
     f"{fatigue_pct_i.get(most_fatigue_i, 0)}% dari data", "#7c3aed", "#f5f3ff"),
]
for col, (ico, ttl, val, sub, clr, bg) in zip([ic1, ic2, ic3, ic4], insight_cards):
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border-radius:16px; padding:20px;
        border-left:4px solid {clr};">
            <div style="font-size:24px; margin-bottom:6px;">{ico}</div>
            <div style="font-size:11px; color:{clr}; font-weight:700;
            letter-spacing:1px;">{ttl.upper()}</div>
            <div style="font-size:28px; font-weight:900; color:#0f172a;
            line-height:1.1; margin:4px 0;">{val}</div>
            <div style="font-size:12px; color:#64748b;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Distribusi Fatigue Visual ──
st.markdown("""<div style="font-size:13px; font-weight:700; color:#0f172a;
margin-bottom:12px;">📊 Distribusi Level Fatigue</div>""", unsafe_allow_html=True)

dist_cols = st.columns(len(fatigue_counts_i))
dist_colors = {
    'Rendah': ('#22c55e', '#dcfce7'),
    'Sedang': ('#f59e0b', '#fef9c3'),
    'Tinggi': ('#ef4444', '#fee2e2')
}
for col, (lbl, cnt) in zip(dist_cols, fatigue_counts_i.items()):
    pct       = fatigue_pct_i.get(lbl, 0)
    clr, bg   = dist_colors.get(lbl, ('#6366f1', '#eef2ff'))
    with col:
        st.markdown(f"""
        <div style="background:{bg}; border:2px solid {clr}; border-radius:16px;
        padding:20px; text-align:center;">
            <div style="font-size:32px; font-weight:900; color:{clr}; line-height:1;">{pct}%</div>
            <div style="font-size:14px; font-weight:700; color:#0f172a; margin:4px 0;">{lbl}</div>
            <div style="font-size:12px; color:#64748b;">{cnt:,} orang</div>
            <div style="background:rgba(0,0,0,0.08); border-radius:99px; height:6px;
            overflow:hidden; margin-top:10px;">
                <div style="width:{pct}%; background:{clr}; height:100%; border-radius:99px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Analisis Gender & Usia ──
try:
    gf = (filtered_df[filtered_df['sex_label'].isin(['Female','Male'])]
          .groupby(['fatigue_label','sex_label']).size().reset_index(name='count'))
    dom_gender = (gf.sort_values('count', ascending=False)
                  .groupby('fatigue_label').first().reset_index())[['fatigue_label','sex_label']]
    gender_lines = " | ".join([
        f"**{r['fatigue_label']}** → {r['sex_label']}"
        for _, r in dom_gender.iterrows()
    ]) or "Data tidak tersedia"
except Exception:
    gender_lines = "Data tidak tersedia"

try:
    af = (filtered_df.groupby(['fatigue_label','age_group'])
          .size().reset_index(name='count'))
    dom_age = (af.sort_values('count', ascending=False)
               .groupby('fatigue_label').first().reset_index())[['fatigue_label','age_group']]
    age_lines = " | ".join([
        f"**{r['fatigue_label']}** → {r['age_group']}"
        for _, r in dom_age.iterrows()
    ]) or "Data tidak tersedia"
except Exception:
    age_lines = "Data tidak tersedia"

# Korelasi tertinggi
try:
    num_df = filtered_df.select_dtypes(include=np.number).drop(
        columns=['id','sex','shift'], errors='ignore')
    if 'fatigue' in num_df.columns:
        cs = num_df.corr()['fatigue'].drop('fatigue').dropna().abs()
        top_feat     = cs.idxmax()
        top_corr_val = round(cs.max(), 2)
        feature_insight = f"**{top_feat}** (r = {top_corr_val})"
    else:
        feature_insight = "stress_sleep_ratio, sleep_deficit, age_stress"
except Exception:
    feature_insight = "stress_sleep_ratio, sleep_deficit, age_stress"

# ── Narasi AI ──
sleep_status  = ("sudah memenuhi standar kesehatan (≥7 jam)" if avg_sleep_i >= 7
                 else f"masih di bawah standar (kurang {round(7 - avg_sleep_i, 1)} jam dari target 7 jam)")
stress_status = ("berada di level aman (≤5)" if avg_stress_i <= 5
                 else "melebihi batas aman — perlu intervensi segera")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e3a8a,#2563eb);
border-radius:20px; padding:28px; color:white; margin-bottom:20px;">
    <div style="font-size:14px; font-weight:800; letter-spacing:2px;
    opacity:0.7; margin-bottom:12px;">🧠 NARASI AI OTOMATIS</div>
    <p style="font-size:15px; line-height:1.8; margin:0; opacity:0.95;">
    Dari <strong>{total_i:,} karyawan</strong> yang dianalisis, rata-rata durasi tidur adalah
    <strong>{avg_sleep_i} jam</strong> yang {sleep_status}.
    Tingkat stress rata-rata <strong>{avg_stress_i}/10</strong> yang {stress_status}.
    Level fatigue yang paling dominan adalah <strong>{most_fatigue_i}
    ({fatigue_pct_i.get(most_fatigue_i, 0)}%)</strong> dan paling rendah adalah
    <strong>{least_fatigue_i} ({fatigue_pct_i.get(least_fatigue_i, 0)}%)</strong>.
    Feature dengan korelasi tertinggi terhadap fatigue adalah {feature_insight}.
    Dominan gender per kategori: {gender_lines}.
    Dominan kelompok usia per kategori: {age_lines}.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Rekomendasi Intervensi per Segmen ──
st.markdown("""<div style="font-size:13px; font-weight:700; color:#0f172a;
margin-bottom:12px;">🎯 Rekomendasi Intervensi per Segmen</div>""", unsafe_allow_html=True)

try:
    seg_df = (
        filtered_df[filtered_df['sex_label'].isin(['Female','Male'])]
        .groupby(['age_group','sex_label','fatigue_label'])
        .size().reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(3)
    )
    rek_colors = {
        'Tinggi': ('#fee2e2', '#ef4444', '🚨'),
        'Sedang': ('#fef9c3', '#f59e0b', '⚠️'),
        'Rendah': ('#dcfce7', '#22c55e', '✅'),
    }
    rek_texts = {
        'Tinggi': "Prioritaskan program sleep hygiene dan manajemen stress untuk segmen ini.",
        'Sedang': "Lakukan monitoring rutin dan sediakan fasilitas relaksasi.",
        'Rendah': "Pertahankan kondisi kerja yang mendukung keseimbangan work-life.",
    }
    for _, row in seg_df.iterrows():
        bg, clr, ico = rek_colors.get(row['fatigue_label'], ('#f1f5f9','#64748b','📌'))
        rek_text = rek_texts.get(row['fatigue_label'], "Lakukan evaluasi berkala.")
        st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {clr};
        border-radius:12px; padding:14px 16px; margin-bottom:10px;">
            <div style="font-size:13px; font-weight:700; color:#0f172a;">
            {ico} {row['age_group']} — {row['sex_label']}
            <span style="background:{clr}; color:white; font-size:10px;
            padding:2px 8px; border-radius:99px; margin-left:8px; font-weight:700;">
            {row['fatigue_label'].upper()}</span>
            </div>
            <div style="font-size:12px; color:#475569; margin-top:6px;">
            {rek_text} ({row['count']:,} karyawan teridentifikasi)
            </div>
        </div>
        """, unsafe_allow_html=True)
except Exception:
    st.info("Data tidak cukup untuk rekomendasi segmen.")

# ── Kesimpulan Eksekutif ──
st.markdown(f"""
<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:20px;
padding:28px; margin-top:20px;">
    <div style="font-size:14px; font-weight:800; color:#0f172a;
    letter-spacing:1px; margin-bottom:16px;">🧠 KESIMPULAN EKSEKUTIF</div>
    <p style="font-size:14px; color:#475569; line-height:1.8; margin:0;">
    <strong>WellGuard Dashboard</strong> menganalisis <strong>{total_i:,} data karyawan</strong>
    menggunakan model Deep Learning PyTorch (Fourier Embedder + Residual Blocks)
    untuk mendeteksi risiko fatigue secara real-time.
    Dengan rata-rata tidur <strong>{avg_sleep_i} jam</strong> dan stress
    <strong>{avg_stress_i}/10</strong>, level fatigue dominan adalah
    <strong>{most_fatigue_i}</strong> ({fatigue_pct_i.get(most_fatigue_i, 0)}% dari total karyawan).
    Dashboard ini mendukung pengambilan keputusan berbasis data untuk menjaga
    kesehatan, produktivitas, dan kesejahteraan karyawan secara berkelanjutan.
    </p>
</div>
""", unsafe_allow_html=True)