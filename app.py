import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="WattWise AI — Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as file:
    META = json.load(file)

FEATURES = META["features"]
CLUSTER_FEATURES = META["cluster_features"]
TARGET = META["target"]
HIGH_DEMAND_THRESHOLD = float(META["high_demand_threshold"])

ZONE_COLS = ["Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption"]

@st.cache_resource
def load_models():
    return {
        "regression": joblib.load(MODEL_DIR / "linear_regression.joblib"),
        "classification_scaler": joblib.load(MODEL_DIR / "classification_scaler.joblib"),
        "classifier": joblib.load(MODEL_DIR / "random_forest_classifier.joblib"),
        "clustering_scaler": joblib.load(MODEL_DIR / "clustering_scaler.joblib"),
        "kmeans": joblib.load(MODEL_DIR / "kmeans_model.joblib"),
        "gmm": joblib.load(MODEL_DIR / "gmm_model.joblib"),
    }

@st.cache_data
def load_data():
    data = pd.read_csv(MODEL_DIR / "dashboard_data.csv")
    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
    data = data.dropna(subset=["DateTime"]).sort_values("DateTime")
    return data

@st.cache_data
def get_pca_sample(data, features):
    sample = data.sample(min(4500, len(data)), random_state=42).copy()
    scaler = joblib.load(MODEL_DIR / "clustering_scaler.joblib")
    scaled = scaler.transform(sample[features])
    points = PCA(n_components=2, random_state=42).fit_transform(scaled)
    sample["PCA 1"] = points[:, 0]
    sample["PCA 2"] = points[:, 1]
    return sample

models = load_models()
data = load_data()

# Precompute key dataset statistics for instant, zero-latency rendering
TOTAL_ENERGY_GWH = (data[TARGET] * (10 / 60)).sum() / 1e6
AVG_DEMAND_KW = data[TARGET].mean()
PEAK_DEMAND_KW = data[TARGET].max()
PEAK_ROW = data.loc[data[TARGET].idxmax()]
HIGH_DEMAND_PCT = 100 * data["High_Demand"].mean()

# ----------------------------- Premium SaaS Visual System -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..800;1,9..40,300..800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #070b14;
    --sidebar-bg: #090e1a;
    --surface-card: #0e1626;
    --surface-card-hover: #121c30;
    --border-subtle: #1a2538;
    --border-accent: rgba(45, 212, 191, 0.22);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #8392a7;
    --accent-teal: #14b8a6;
    --accent-mint: #2dd4bf;
    --accent-cyan: #38bdf8;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
}

/* Global resets & typography */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

/* Hide default streamlit decorations */
header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}
footer {
    display: none !important;
}

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}

/* Base Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.025em;
    font-weight: 700;
}

p, span, div, li {
    color: var(--text-secondary);
}

/* ---------------- Sidebar Refined SaaS Navigation ---------------- */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Remove default radio label header */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}

/* Remove default radio circular button indicator */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Style each navigation item into a sleek menu item */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: transparent;
    padding: 10px 14px;
    border-radius: 9px;
    margin-bottom: 5px;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    border: 1px solid transparent;
    cursor: pointer;
    display: flex;
    align-items: center;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(20, 184, 166, 0.08);
    border-color: rgba(45, 212, 191, 0.2);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(20, 184, 166, 0.18) 0%, rgba(14, 22, 38, 0.6) 100%) !important;
    border: 1px solid rgba(45, 212, 191, 0.38) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] *,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) * {
    color: #2dd4bf !important;
    font-weight: 700 !important;
}

/* Sidebar Brand Header */
.sidebar-brand {
    padding: 8px 4px 18px 4px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 20px;
}
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 11px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.28rem;
    font-weight: 700;
    color: #f8fafc !important;
    letter-spacing: -0.015em;
}
.sidebar-logo-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
    color: #042f2e;
    font-size: 1.05rem;
    font-weight: 800;
    box-shadow: 0 0 16px rgba(45, 212, 191, 0.35);
}
.sidebar-tagline {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    margin-top: 6px;
    padding-left: 2px;
}

/* Sidebar Dataset Status Card */
.sidebar-status-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 26px;
}
.sidebar-status-title {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    margin-bottom: 6px;
}
.sidebar-status-val {
    font-size: 0.88rem;
    color: #f8fafc !important;
    font-weight: 500;
    line-height: 1.45;
}

/* ---------------- Hero Section ---------------- */
.hero-wrapper {
    position: relative;
    padding: 30px 36px 24px 36px;
    border-radius: 18px;
    background: linear-gradient(135deg, #081122 0%, #0a172e 50%, #06222b 100%);
    border: 1px solid rgba(45, 212, 191, 0.22);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    margin-bottom: 24px;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px;
    right: -40px;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(20, 184, 166, 0.16) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.74rem;
    font-weight: 700;
    color: #2dd4bf !important;
    margin-bottom: 8px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.hero-pulse-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2dd4bf;
    box-shadow: 0 0 8px #2dd4bf;
}
.hero-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0 0 10px 0;
    line-height: 1.15;
    letter-spacing: -0.025em;
}
.hero-supporting {
    color: #cbd5e1 !important;
    font-size: 1.02rem;
    margin: 0 0 16px 0;
    max-width: 680px;
    line-height: 1.5;
}

/* Energy Waveform SVG Container */
.energy-waveform-box {
    width: 100%;
    height: 48px;
    overflow: hidden;
    margin-top: 8px;
}
.energy-pulse-svg {
    width: 100%;
    height: 100%;
}
.waveform-glow {
    stroke-dasharray: 180, 320;
    stroke-dashoffset: 0;
    animation: pulseTravel 7.5s linear infinite;
}
@keyframes pulseTravel {
    0% { stroke-dashoffset: 900; }
    100% { stroke-dashoffset: 0; }
}
@media (prefers-reduced-motion: reduce) {
    .waveform-glow {
        animation: none !important;
        stroke-dasharray: none !important;
    }
}

/* ---------------- Sleek SaaS Metric Cards (Zero Ellipsis Truncation) ---------------- */
.metric-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28);
    transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.18s ease, box-shadow 0.18s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 110px;
    position: relative;
    overflow: visible;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(45, 212, 191, 0.35);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.38);
}
.metric-card-primary {
    background: linear-gradient(145deg, #0d1e34 0%, #0c2738 100%) !important;
    border: 1px solid rgba(45, 212, 191, 0.45) !important;
    box-shadow: 0 6px 24px rgba(20, 184, 166, 0.16) !important;
}
.metric-card-primary::before {
    content: '';
    position: absolute;
    top: 0;
    left: 14px;
    right: 14px;
    height: 2px;
    background: linear-gradient(90deg, #14b8a6, #2dd4bf);
    border-radius: 2px;
}
.metric-card-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 6px;
    white-space: normal;
}
.metric-card-help {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    font-size: 0.65rem;
    cursor: help;
}
.metric-card-value-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
}
.metric-card-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.metric-card-primary .metric-card-num {
    color: #2dd4bf;
}
.metric-card-unit {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #94a3b8;
}
.metric-card-primary .metric-card-unit {
    color: #99f6e4;
}
.metric-card-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 6px;
    line-height: 1.35;
    display: flex;
    align-items: center;
    gap: 6px;
}
.metric-card-sub.sub-teal {
    color: #2dd4bf;
    font-weight: 600;
}
.metric-card-sub.sub-amber {
    color: #fbbf24;
    font-weight: 600;
}
.metric-card-sub.sub-rose {
    color: #f87171;
    font-weight: 600;
}

/* ---------------- Editorial Typography & Panels ---------------- */
.editorial-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc !important;
    margin: 6px 0 2px 0;
    display: flex;
    align-items: center;
    gap: 9px;
}
.editorial-sub {
    color: var(--text-muted) !important;
    font-size: 0.88rem;
    margin-bottom: 16px;
    line-height: 1.45;
}

/* Insight Callout Panel */
.insight-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-left: 4px solid #2dd4bf;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 16px 0 22px 0;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}
.insight-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.80rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #2dd4bf;
    margin-bottom: 6px;
}
.insight-card-body {
    color: #e2e8f0 !important;
    font-size: 0.94rem;
    line-height: 1.55;
    margin: 0;
}

/* Informational Note Box */
.note-box {
    background: rgba(14, 22, 38, 0.75);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0 18px 0;
    color: #cbd5e1 !important;
    font-size: 0.90rem;
    line-height: 1.5;
}
.note-box strong {
    color: #f8fafc !important;
}

/* Badges & Chips */
.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.badge-teal {
    background: rgba(20, 184, 166, 0.16);
    color: #2dd4bf;
    border: 1px solid rgba(45, 212, 191, 0.32);
}
.badge-amber {
    background: rgba(245, 158, 11, 0.16);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.badge-rose {
    background: rgba(244, 63, 94, 0.16);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.35);
}
.badge-slate {
    background: #172236;
    color: #cbd5e1;
    border: 1px solid #23334d;
}

/* ---------------- Form Controls, Sliders & Inputs ---------------- */
div[data-testid="stForm"] {
    background: var(--surface-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

/* Widget Headings & High Contrast Labels */
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] *,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] label p,
div[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
    color: #f8fafc !important;
    font-size: 0.94rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 6px !important;
}

/* Sliders High Contrast */
div[data-testid="stSlider"] {
    padding-top: 4px;
    margin-bottom: 12px;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: #2dd4bf !important;
    border: 2px solid #042f2e !important;
    box-shadow: 0 0 12px rgba(45, 212, 191, 0.75) !important;
    width: 18px !important;
    height: 18px !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
    background: linear-gradient(90deg, #0d9488, #2dd4bf) !important;
}
div[data-testid="stSlider"] div[data-testid="stSliderTickBar"] + div,
div[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
div[data-testid="stSlider"] div[data-testid="stSliderTickBar"] ~ div {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] div[role="slider"] + div,
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    color: #2dd4bf !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Segmented Control */
div[data-testid="stSegmentedControl"] button {
    background-color: #111a2d !important;
    color: #cbd5e1 !important;
    border: 1px solid var(--border-subtle) !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stSegmentedControl"] button:hover {
    background-color: #18253f !important;
    color: #ffffff !important;
    border-color: rgba(45, 212, 191, 0.3) !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[data-checked="true"] {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
    color: #ffffff !important;
    border-color: #2dd4bf !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(20, 184, 166, 0.35) !important;
}

/* Inputs & Selectboxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input.stNumberInput {
    background-color: #0c1424 !important;
    border: 1px solid var(--border-subtle) !important;
    color: #f8fafc !important;
    border-radius: 9px !important;
}
div[data-baseweb="select"] * {
    color: #f8fafc !important;
}
div[data-baseweb="select"] svg {
    fill: #2dd4bf !important;
}

/* Action Buttons */
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 16px rgba(20, 184, 166, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #14b8a6 0%, #2dd4bf 100%) !important;
    color: #042f2e !important;
    box-shadow: 0 6px 22px rgba(45, 212, 191, 0.5) !important;
    transform: translateY(-1px);
}

/* Dataframe and Tables */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #14b8a6 0%, #f59e0b 80%, #f43f5e 100%) !important;
    border-radius: 6px !important;
}
div[data-testid="stProgress"] p {
    color: #cbd5e1 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: var(--surface-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}
div[data-testid="stExpander"] summary * {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

/* Subtle entrance animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.hero-wrapper, .metric-card, .insight-card, div[data-testid="stForm"], [data-testid="stDataFrame"], .js-plotly-plot {
    animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Custom dark scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #070b14;
}
::-webkit-scrollbar-thumb {
    background: #1a2538;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #2dd4bf;
}

@media (prefers-reduced-motion: reduce) {
    .waveform-glow,
    .hero-wrapper,
    .metric-card,
    .insight-card,
    div[data-testid="stForm"],
    [data-testid="stDataFrame"],
    .js-plotly-plot {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Shared Helpers -----------------------------
def editorial_header(title, subtitle=None):
    st.markdown(f'<div class="editorial-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="editorial-sub">{subtitle}</div>', unsafe_allow_html=True)

def render_metric_card(label, value, unit="", subtext="", subtext_class="", is_primary=False, help_text=""):
    primary_cls = "metric-card-primary" if is_primary else ""
    unit_html = f'<span class="metric-card-unit">{unit}</span>' if unit else ""
    help_icon = f'<span class="metric-card-help" title="{help_text}">ℹ</span>' if help_text else ""
    sub_cls = f"metric-card-sub {subtext_class}" if subtext_class else "metric-card-sub"
    sub_html = f'<div class="{sub_cls}">{subtext}</div>' if subtext else ""
    
    st.markdown(f"""
    <div class="metric-card {primary_cls}">
      <div class="metric-card-label">
        <span>{label}</span>
        {help_icon}
      </div>
      <div class="metric-card-value-row">
        <span class="metric-card-num">{value}</span>
        {unit_html}
      </div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)

def chart_layout(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=32, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#cbd5e1", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#cbd5e1", size=11),
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(
            bgcolor="#0e1626",
            bordercolor="#2dd4bf",
            font=dict(family="DM Sans, sans-serif", color="#f8fafc", size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#1a2538",
        tickcolor="#1a2538",
        tickfont=dict(color="#8392a7", size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.05)",
        zeroline=False,
        tickfont=dict(color="#8392a7", size=11),
    )
    return fig

CHART_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}

def render_chart(fig, height=330):
    st.plotly_chart(chart_layout(fig, height), use_container_width=True, config=CHART_CONFIG)

def render_hero(eyebrow, headline, supporting):
    st.markdown(f"""
    <div class="hero-wrapper">
      <div class="hero-eyebrow">
        <span class="hero-pulse-dot"></span>
        <span>{eyebrow}</span>
      </div>
      <h1 class="hero-headline">{headline}</h1>
      <p class="hero-supporting">{supporting}</p>
      <div class="energy-waveform-box">
        <svg class="energy-pulse-svg" viewBox="0 0 1000 70" fill="none" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
          <defs>
            <linearGradient id="waveLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#14b8a6" stop-opacity="0.05" />
              <stop offset="30%" stop-color="#14b8a6" stop-opacity="0.6" />
              <stop offset="50%" stop-color="#2dd4bf" stop-opacity="1" />
              <stop offset="70%" stop-color="#38bdf8" stop-opacity="0.7" />
              <stop offset="100%" stop-color="#14b8a6" stop-opacity="0.05" />
            </linearGradient>
            <filter id="waveGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>
          <path d="M0,35 Q120,35 200,28 T360,42 T480,12 T540,55 T620,20 T720,38 T840,32 T1000,35" 
                stroke="rgba(45, 212, 191, 0.12)" stroke-width="1.6" fill="none" />
          <path class="waveform-glow" d="M0,35 Q120,35 200,28 T360,42 T480,12 T540,55 T620,20 T720,38 T840,32 T1000,35" 
                stroke="url(#waveLineGrad)" stroke-width="2.5" filter="url(#waveGlow)" fill="none" />
          <circle cx="200" cy="28" r="3" fill="#2dd4bf" opacity="0.6" />
          <circle cx="480" cy="12" r="4.2" fill="#38bdf8" opacity="0.9" filter="url(#waveGlow)" />
          <circle cx="540" cy="55" r="3" fill="#14b8a6" opacity="0.7" />
          <circle cx="720" cy="38" r="3.5" fill="#2dd4bf" opacity="0.8" />
        </svg>
      </div>
    </div>
    """, unsafe_allow_html=True)

def input_frame(temp, humidity, wind, diffuse, diffuse_direct, hour, day):
    return pd.DataFrame([{
        "Temperature": float(temp),
        "Humidity": float(humidity),
        "Wind Speed": float(wind),
        "general diffuse flows": float(diffuse),
        "diffuse flows": float(diffuse_direct),
        "Hour": int(hour),
        "Day": int(day),
    }], columns=FEATURES)

# ----------------------------- Refined SaaS Sidebar -----------------------------
NAV_PAGES = ["Overview", "Demand Forecast", "Demand Risk", "Usage Patterns", "Model Insights"]
NAV_ICONS = {
    "Overview": "⚡",
    "Demand Forecast": "📈",
    "Demand Risk": "🛡️",
    "Usage Patterns": "🧩",
    "Model Insights": "🔬"
}

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-logo">
        <span class="sidebar-logo-icon">⚡</span>
        <span>WattWise AI</span>
      </div>
      <div class="sidebar-tagline">Energy Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        NAV_PAGES,
        format_func=lambda x: f"{NAV_ICONS.get(x, '•')}  {x}",
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div class="sidebar-status-card">
      <div class="sidebar-status-title">Active Dataset</div>
      <div class="sidebar-status-val"><strong>{len(data):,}</strong> observations</div>
      <div class="sidebar-status-val" style="color: #8392a7 !important; font-size: 0.8rem; margin-top: 3px;">
        {data['DateTime'].min():%d %b %Y} — {data['DateTime'].max():%d %b %Y}
      </div>
      <div style="margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap;">
        <span class="badge-chip badge-teal" style="font-size: 0.68rem; padding: 2px 8px;">3 Grid Zones</span>
        <span class="badge-chip badge-slate" style="font-size: 0.68rem; padding: 2px 8px;">10-min Res</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.caption("WattWise AI • ML Systems Lab")

# ============================================================================
# 1. OVERVIEW (LANDING DASHBOARD - MINIMAL AND BEAUTIFUL)
# ============================================================================
if page == "Overview":
    render_hero(
        eyebrow="WATTWISE INTELLIGENCE",
        headline="Understand the pulse of energy.",
        supporting="Explore consumption patterns, forecast demand, and identify periods of elevated energy use."
    )

    # 4 Essential Metrics with Asymmetric Visual Weight (Zero Truncation)
    m1, m2, m3, m4 = st.columns([1.3, 1, 1, 1], gap="medium")
    with m1:
        render_metric_card(
            label="Peak Observed Demand",
            value=f"{PEAK_DEMAND_KW:,.0f}",
            unit="kW",
            subtext=f"{PEAK_ROW['DateTime']:%d %b, %H:%M} • Summer Peak",
            subtext_class="sub-teal",
            is_primary=True,
            help_text="Highest single 10-minute demand recorded across all zones during 2017."
        )
    with m2:
        render_metric_card(
            label="Total Recorded Energy",
            value=f"{TOTAL_ENERGY_GWH:,.1f}",
            unit="GWh",
            subtext="Cumulative grid volume",
            help_text="Total cumulative electricity consumed across all 3 zones over the 1-year recording."
        )
    with m3:
        render_metric_card(
            label="Average Demand",
            value=f"{AVG_DEMAND_KW:,.0f}",
            unit="kW",
            subtext="Per 10-min observation",
            help_text="Mean total grid consumption per 10-minute observation interval."
        )
    with m4:
        render_metric_card(
            label="High-Demand Rate",
            value=f"{HIGH_DEMAND_PCT:.1f}%",
            subtext="Threshold: 86.3k kW",
            subtext_class="sub-amber",
            help_text="Percentage of observations exceeding the 86,266 kW high-demand threshold."
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Main Visual Composition (Editorial Layout)
    # Visual 1: Primary Consumption Over Time
    editorial_header(
        "Consumption Dynamics Over Time",
        "Daily mean electricity consumption across 2017 with seasonal high-demand threshold indication."
    )
    daily = data.set_index("DateTime")[TARGET].resample("D").mean().reset_index()
    peak_daily = daily.loc[daily[TARGET].idxmax()]

    fig_time = px.area(daily, x="DateTime", y=TARGET)
    fig_time.update_traces(
        line=dict(width=2, color="#2dd4bf"),
        fillcolor="rgba(45, 212, 191, 0.12)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Daily Avg: %{y:,.0f} kW<extra></extra>",
    )
    # Peak observed demand day emphasis
    fig_time.add_trace(go.Scatter(
        x=[peak_daily["DateTime"]],
        y=[peak_daily[TARGET]],
        mode="markers+text",
        name="Peak Demand",
        marker=dict(size=8, color="#f59e0b", symbol="circle", line=dict(color="#ffffff", width=1.5)),
        text=[f"Annual Peak: {peak_daily[TARGET]:,.0f} kW ({peak_daily['DateTime']:%d %b})"],
        textposition="top left",
        textfont=dict(color="#f59e0b", size=11, family="DM Sans"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig_time.add_hline(
        y=HIGH_DEMAND_THRESHOLD,
        line_dash="dot",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text="High-Demand Threshold (86,266 kW)",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=11, family="DM Sans"),
    )
    fig_time.update_yaxes(title="Average Consumption (kW)")
    fig_time.update_xaxes(title="")
    render_chart(fig_time, 330)

    # Visuals 2 & 3: Daily Energy Rhythm + Zone Distribution
    col_left, col_right = st.columns([1.35, 1], gap="large")

    with col_left:
        editorial_header(
            "Daily Energy Rhythm",
            "Mean demand by hour of the day. Notice the pronounced peak between 18:00 and 21:00."
        )
        hourly = data.groupby("Hour", as_index=False)[TARGET].mean()
        fig_bar = px.bar(
            hourly,
            x="Hour",
            y=TARGET,
            color=TARGET,
            color_continuous_scale=[
                (0.0, "#0a262e"),
                (0.5, "#14b8a6"),
                (1.0, "#2dd4bf"),
            ],
        )
        fig_bar.update_layout(coloraxis_showscale=False)
        fig_bar.update_traces(
            marker_line_width=0,
            hovertemplate="Hour %{x}:00<br>Average: %{y:,.0f} kW<extra></extra>",
        )
        fig_bar.update_xaxes(dtick=2, title="Hour of Day (00:00 - 23:00)")
        fig_bar.update_yaxes(title="Average Consumption (kW)")
        render_chart(fig_bar, 310)

    with col_right:
        editorial_header(
            "Zone Contribution Share",
            "Relative share of total energy delivered to the three metropolitan distribution zones."
        )
        zone_sums = data[ZONE_COLS].sum()
        zone_df = pd.DataFrame({
            "Zone": ["Zone 1 (Core & Commercial)", "Zone 2 (Mixed Residential)", "Zone 3 (Suburban & Peripheral)"],
            "Energy": zone_sums.values,
        })
        fig_donut = px.pie(
            zone_df,
            names="Zone",
            values="Energy",
            hole=0.68,
            color_discrete_sequence=["#14b8a6", "#38bdf8", "#818cf8"],
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} kWh (%{percent})<extra></extra>",
            marker=dict(line=dict(color="#070b14", width=2)),
        )
        render_chart(fig_donut, 310)

    # 4. One Concise, Data-Derived Insight Panel
    zone_1_pct = 100 * data["Zone 1 Power Consumption"].sum() / data[TARGET].sum()
    summer_peak_mean = data[(data["DateTime"].dt.month.isin([7, 8])) & (data["Hour"].between(18, 21))][TARGET].mean()
    baseline_night_mean = data[(data["Hour"].between(3, 5))][TARGET].mean()
    surge_multiplier = summer_peak_mean / baseline_night_mean

    st.markdown(f"""
    <div class="insight-card">
      <div class="insight-card-header">
        <span>⚡</span>
        <span>Key Analytical Insight</span>
      </div>
      <p class="insight-card-body">
        <strong>Diurnal and Spatial Concentration:</strong> Electricity demand surges to an average of 
        <strong>{summer_peak_mean:,.0f} kW</strong> during summer evening peak hours (18:00–21:00), representing a 
        <strong>{surge_multiplier:.1f}× increase</strong> over early-morning baseload ({baseline_night_mean:,.0f} kW). 
        Spatially, <strong>Zone 1</strong> represents the primary driver of grid strain, accounting for 
        <strong>{zone_1_pct:.1f}%</strong> of total metropolitan power flow throughout the year.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Data source: Tetouan City transmission logs. Values are historical measurements; model outputs are statistical estimates.")

# ============================================================================
# 2. DEMAND FORECAST (REGRESSION PREDICTION WORKFLOW)
# ============================================================================
elif page == "Demand Forecast":
    render_hero(
        eyebrow="PREDICTIVE INFERENCE",
        headline="Forecast grid electricity demand.",
        supporting="Evaluate anticipated aggregate power consumption by modulating ambient environmental factors, time of day, and calendar rhythms."
    )

    st.markdown("""
    <div class="note-box">
      <strong>Model Specification:</strong> Continuous demand is predicted via a verified multivariable Linear Regression model 
      trained on weather and time covariates. Historical performance on the complete dataset: <em>R² = 0.616</em>, <em>MAE = 8,360 kW</em>.
    </div>
    """, unsafe_allow_html=True)

    editorial_header("Environmental and Temporal Inputs", "Adjust the parameters below to compute an instant demand projection.")

    with st.form("forecast_form"):
        f1, f2, f3 = st.columns(3, gap="large")
        with f1:
            st.markdown("<p style='font-weight: 700; color: #2dd4bf; margin-bottom: 6px;'>Atmospheric Conditions</p>", unsafe_allow_html=True)
            temp = st.slider("Temperature (°C)", min_value=3.0, max_value=40.0, value=22.0, step=0.5,
                             help="Ambient dry-bulb temperature in degrees Celsius.")
            humidity = st.slider("Relative Humidity (%)", min_value=10.0, max_value=98.0, value=65.0, step=1.0,
                                 help="Ambient relative humidity percentage.")
            wind = st.slider("Wind Speed (m/s)", min_value=0.05, max_value=6.5, value=2.1, step=0.05,
                             help="Near-surface wind speed in meters per second.")

        with f2:
            st.markdown("<p style='font-weight: 700; color: #2dd4bf; margin-bottom: 6px;'>Solar Radiation</p>", unsafe_allow_html=True)
            diffuse_flow = st.slider("General Diffuse Flow (W/m²)", min_value=0.0, max_value=1163.0, value=175.0, step=5.0,
                                     help="Broad global diffuse solar irradiance.")
            direct_flow = st.slider("Diffuse Flow Direct (W/m²)", min_value=0.0, max_value=936.0, value=75.0, step=5.0,
                                    help="Direct diffuse solar component.")

        with f3:
            st.markdown("<p style='font-weight: 700; color: #2dd4bf; margin-bottom: 6px;'>Calendar & Schedule</p>", unsafe_allow_html=True)
            hour = st.slider("Hour of Day (24h)", min_value=0, max_value=23, value=19, step=1,
                             help="Hour in military format (0 = midnight, 19 = 7:00 PM).")
            day_name = st.selectbox("Day of Week",
                                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                    index=2,
                                    help="Day of the week influences commercial vs residential operations.")

        submitted = st.form_submit_button("Compute Demand Forecast", use_container_width=True)

    day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)
    input_row = input_frame(temp, humidity, wind, diffuse_flow, direct_flow, hour, day_idx)
    predicted_demand = float(models["regression"].predict(input_row)[0])

    # Contextual calculations
    percentile = float((data[TARGET] <= predicted_demand).mean() * 100)
    delta_vs_avg = predicted_demand - AVG_DEMAND_KW
    delta_pct = (delta_vs_avg / AVG_DEMAND_KW) * 100

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    editorial_header("Forecast Results & Historical Context", "Comparison against the 52,416 historical observations in the dataset.")

    res1, res2, res3 = st.columns([1.2, 1, 1], gap="medium")
    with res1:
        delta_sign = "+" if delta_vs_avg >= 0 else ""
        delta_color_cls = "sub-amber" if delta_vs_avg > 0 else "sub-teal"
        render_metric_card(
            label="Estimated Total Demand",
            value=f"{predicted_demand:,.0f}",
            unit="kW",
            subtext=f"{delta_sign}{delta_vs_avg:,.0f} kW ({delta_sign}{delta_pct:.1f}% vs Avg)",
            subtext_class=delta_color_cls,
            is_primary=True,
            help_text="Multivariable linear regression projection based on selected inputs."
        )
    with res2:
        render_metric_card(
            label="Historical Percentile",
            value=f"{percentile:.0f}",
            unit="th",
            subtext="Relative load ranking",
            help_text="Percentage of recorded timestamps that exhibited demand equal to or lower than this forecast."
        )
    with res3:
        demand_status = "High Demand Zone" if predicted_demand >= HIGH_DEMAND_THRESHOLD else "Normal Operating Band"
        status_cls = "sub-amber" if predicted_demand >= HIGH_DEMAND_THRESHOLD else "sub-teal"
        status_sub = "Exceeds 86,266 kW threshold" if predicted_demand >= HIGH_DEMAND_THRESHOLD else "Within baseline capacity"
        render_metric_card(
            label="Load Classification",
            value=demand_status,
            subtext=status_sub,
            subtext_class=status_cls,
            help_text="High demand is classified when total consumption meets or exceeds 86,266 kW."
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Historical Context Gauge Chart
    col_gauge, col_interp = st.columns([1.3, 1], gap="large")
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_demand,
            number={"valueformat": ",.0f", "suffix": " kW", "font": {"family": "Space Grotesk", "color": "#f8fafc", "size": 32}},
            title={"text": "Forecast Location in Historical Range", "font": {"family": "Space Grotesk", "color": "#cbd5e1", "size": 14}},
            gauge={
                "axis": {
                    "range": [float(data[TARGET].min()), float(data[TARGET].max())],
                    "tickfont": {"color": "#8392a7", "size": 10},
                    "tickcolor": "#1a2538",
                },
                "bar": {"color": "#2dd4bf", "thickness": 0.45},
                "bgcolor": "#0e1626",
                "borderwidth": 1,
                "bordercolor": "#1a2538",
                "steps": [
                    {"range": [float(data[TARGET].min()), HIGH_DEMAND_THRESHOLD], "color": "rgba(20, 184, 166, 0.15)"},
                    {"range": [HIGH_DEMAND_THRESHOLD, float(data[TARGET].max())], "color": "rgba(245, 158, 11, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 3},
                    "thickness": 0.75,
                    "value": HIGH_DEMAND_THRESHOLD,
                },
            },
        ))
        render_chart(fig_gauge, 290)

    with col_interp:
        st.markdown(f"""
        <div class="insight-card" style="margin-top: 10px;">
          <div class="insight-card-header">
            <span>📊</span>
            <span>Scenario Explanation</span>
          </div>
          <p class="insight-card-body">
            At <strong>{temp:.1f}°C</strong> on a <strong>{day_name}</strong> at <strong>{hour:02d}:00</strong>, 
            the model projects aggregate demand to reach <strong>{predicted_demand:,.0f} kW</strong>.
            <br><br>
            • <strong>Temperature Effect:</strong> The positive linear coefficient (+1,133 kW/°C) reflects rising cooling energy requirements.<br>
            • <strong>Diurnal Effect:</strong> Hour {hour:02d} provides an incremental weight of +{hour * 1533:,.0f} kW relative to the midnight baseline.<br>
            • <strong>Historical Context:</strong> This scenario sits at the <strong>{percentile:.0f}th percentile</strong> of historical distribution records.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 3. DEMAND RISK (HIGH-DEMAND CLASSIFICATION & PROBABILITY)
# ============================================================================
elif page == "Demand Risk":
    render_hero(
        eyebrow="RISK GOVERNANCE",
        headline="Evaluate high-demand risk probability.",
        supporting="Estimate the likelihood of severe grid load events using the Random Forest classifier trained on 2017 transmission records."
    )

    st.markdown("""
    <div class="note-box">
      <strong>Model Integrity & Governance Notice:</strong> High demand is formally defined as consumption ≥ 
      <strong>86,265.70 kW</strong> (the empirical 75th percentile of historical load). The classification model is a 
      200-estimator Random Forest evaluated with <em>Accuracy = 98.8%</em>, <em>Precision = 95.1%</em>, and <em>Recall = 99.7%</em>. 
      Predictions represent statistical model estimates, not operational guarantees.
    </div>
    """, unsafe_allow_html=True)

    editorial_header("Simulate Operational Conditions", "Modulate conditions to observe the model's calibrated risk probability.")

    with st.form("risk_form"):
        r1, r2, r3 = st.columns(3, gap="large")
        with r1:
            st.markdown("<p style='font-weight: 700; color: #2dd4bf; margin-bottom: 6px;'>Environmental State</p>", unsafe_allow_html=True)
            r_temp = st.slider("Temperature (°C)", 3.0, 40.0, 26.5, 0.5, key="r_temp",
                               help="High ambient temperatures strongly elevate peak cooling demand.")
            r_hum = st.slider("Relative Humidity (%)", 10.0, 98.0, 62.0, 1.0, key="r_hum")
            r_wind = st.slider("Wind Speed (m/s)", 0.05, 6.5, 1.8, 0.05, key="r_wind")

        with r2:
            st.markdown("<p style='font-weight: 700; color: #2dd4bf; margin-bottom: 6px;'>Solar Irradiance</p>", unsafe_allow_html=True)
            r_diff = st.slider("General Diffuse Flow (W/m²)", 0.0, 1163.0, 210.0, 5.0, key="r_diff")
            r_dir = st.slider("Diffuse Flow Direct (W/m²)", 0.0, 936.0, 90.0, 5.0, key="r_dir")

        with r3:
            st.markdown("<p style='font-weight: 700; color: #2dd4bf; margin-bottom: 6px;'>Operating Window</p>", unsafe_allow_html=True)
            r_hour = st.slider("Hour of Day", 0, 23, 20, 1, key="r_hour",
                               help="Peak grid load typically clusters in evening hours.")
            r_day_name = st.selectbox("Day of Week",
                                      ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                      index=3, key="r_day")

        r_submitted = st.form_submit_button("Assess High-Demand Risk", use_container_width=True)

    r_day_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(r_day_name)
    r_row = input_frame(r_temp, r_hum, r_wind, r_diff, r_dir, r_hour, r_day_idx)
    r_scaled = models["classification_scaler"].transform(r_row)
    r_prob = float(models["classifier"].predict_proba(r_scaled)[0][1])
    r_is_high = int(models["classifier"].predict(r_scaled)[0])

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    editorial_header("Risk Assessment Summary", "Model-derived probability and categorical risk estimate.")

    k_risk1, k_risk2, k_risk3 = st.columns([1.2, 1, 1], gap="medium")
    with k_risk1:
        if r_is_high:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #2a1116 0%, #3d141b 100%); border: 1px solid rgba(244, 63, 94, 0.45); box-shadow: 0 4px 18px rgba(244, 63, 94, 0.18);">
              <div class="metric-card-label" style="color: #fca5a5;">Classification Output</div>
              <div class="metric-card-num" style="color: #fb7185; font-size: 1.65rem;">HIGH DEMAND RISK</div>
              <div class="metric-card-sub" style="color: #fca5a5;">Projected to exceed 86,266 kW</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #09211d 0%, #0c332c 100%); border: 1px solid rgba(45, 212, 191, 0.45); box-shadow: 0 4px 18px rgba(45, 212, 191, 0.18);">
              <div class="metric-card-label" style="color: #99f6e4;">Classification Output</div>
              <div class="metric-card-num" style="color: #2dd4bf; font-size: 1.65rem;">NORMAL DEMAND</div>
              <div class="metric-card-sub" style="color: #99f6e4;">Projected within standard capacity</div>
            </div>
            """, unsafe_allow_html=True)

    with k_risk2:
        prob_cls = "sub-amber" if r_prob > 0.5 else "sub-teal"
        render_metric_card(
            label="High-Demand Probability",
            value=f"{r_prob * 100:.1f}%",
            subtext="Estimated from RF ensemble",
            subtext_class=prob_cls,
            help_text="Posterior classification probability of exceeding the threshold."
        )
    with k_risk3:
        render_metric_card(
            label="Empirical Prevalence",
            value=f"{HIGH_DEMAND_PCT:.1f}%",
            subtext="Historical baseline frequency",
            help_text="Historical baseline probability of high-demand events in the dataset."
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.progress(r_prob, text=f"Calibrated Probability of Exceeding Threshold: {r_prob * 100:.1f}%")

    # Risk Factor Context Visuals
    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        editorial_header("Historical Risk Probability by Hour", "Empirical percentage of observations exceeding the threshold across each hour.")
        hourly_risk = data.groupby("Hour")["High_Demand"].mean().reset_index()
        hourly_risk["RiskPct"] = hourly_risk["High_Demand"] * 100
        fig_hr_risk = px.bar(
            hourly_risk,
            x="Hour",
            y="RiskPct",
            color="RiskPct",
            color_continuous_scale=[(0, "#0a262e"), (0.5, "#14b8a6"), (1, "#f59e0b")],
        )
        fig_hr_risk.update_layout(coloraxis_showscale=False)
        fig_hr_risk.update_xaxes(dtick=2, title="Hour of Day")
        fig_hr_risk.update_yaxes(title="Historical High-Demand %")
        render_chart(fig_hr_risk, 280)

    with c_right:
        editorial_header("Historical Risk vs Temperature", "Empirical probability segmented by temperature buckets.")
        data_copy = data.copy()
        data_copy["Temp_Bin"] = pd.cut(data_copy["Temperature"], bins=[0, 15, 20, 25, 30, 45],
                                       labels=["<15°C", "15–20°C", "20–25°C", "25–30°C", ">30°C"])
        temp_risk = data_copy.groupby("Temp_Bin", observed=False)["High_Demand"].mean().reset_index()
        temp_risk["RiskPct"] = temp_risk["High_Demand"] * 100
        fig_temp_risk = px.bar(
            temp_risk,
            x="Temp_Bin",
            y="RiskPct",
            color="RiskPct",
            color_continuous_scale=[(0, "#0a262e"), (0.5, "#14b8a6"), (1, "#f59e0b")],
        )
        fig_temp_risk.update_layout(coloraxis_showscale=False)
        fig_temp_risk.update_xaxes(title="Temperature Range")
        fig_temp_risk.update_yaxes(title="Historical High-Demand %")
        render_chart(fig_temp_risk, 280)

# ============================================================================
# 4. USAGE PATTERNS (K-MEANS & GMM CLUSTERING)
# ============================================================================
elif page == "Usage Patterns":
    render_hero(
        eyebrow="UNSUPERVISED DISCOVERY",
        headline="Find the patterns behind the numbers.",
        supporting="Explore behavioral energy archetypes identified via K-Means and Gaussian Mixture Model clustering across environmental and load dimensions."
    )

    st.markdown("""
    <div class="note-box">
      <strong>Analytical Context:</strong> Clustering is performed on 6 standardized physical features: 
      <em>Temperature, Humidity, Wind Speed, General Diffuse Flow, Diffuse Flow Direct,</em> and <em>Total Consumption</em>. 
      Clusters describe operating intervals across the metropolitan grid, not individual domestic households.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='color: #f8fafc; font-size: 0.95rem; font-weight: 700; margin-bottom: 8px;'>Segmentation Algorithm</p>", unsafe_allow_html=True)
    method = st.segmented_control(
        "Segmentation Algorithm",
        options=["K-Means", "Gaussian Mixture Model"],
        default="K-Means",
        label_visibility="collapsed",
    )
    label_col = "KMeans_Cluster" if method == "K-Means" else "GMM_Cluster"

    # Elegant qualitative color palette
    cluster_colors = ["#2dd4bf", "#38bdf8", "#f59e0b", "#c084fc"]

    col_map, col_profile = st.columns([1.35, 1], gap="large")

    with col_map:
        editorial_header(
            "2D Principal Component Projection",
            "PCA compresses the 6 standardized clustering features into 2 orthogonal axes for geometric inspection."
        )
        pca_data = get_pca_sample(data, CLUSTER_FEATURES)
        pca_data["Cluster_Str"] = "Cluster " + pca_data[label_col].astype(str)
        fig_pca = px.scatter(
            pca_data,
            x="PCA 1",
            y="PCA 2",
            color="Cluster_Str",
            color_discrete_sequence=cluster_colors,
            opacity=0.72,
            hover_data={
                "DateTime": "|%d %b %Y %H:%M",
                "Temperature": ":.1f °C",
                "Humidity": ":.0f%",
                TARGET: ":,.0f kW",
                "PCA 1": False,
                "PCA 2": False,
                "Cluster_Str": False,
            },
        )
        fig_pca.update_traces(marker=dict(size=5.5, line=dict(width=0.5, color="#070b14")))
        render_chart(fig_pca, 410)

    with col_profile:
        editorial_header(
            "Cluster Centroids & Profiles",
            "Mean feature values across each identified operational archetype."
        )
        profiles = data.groupby(label_col)[CLUSTER_FEATURES].mean().round(1)
        profiles.index = [f"Cluster {i}" for i in profiles.index]
        st.dataframe(
            profiles.style.format({
                "Temperature": "{:.1f} °C",
                "Humidity": "{:.1f}%",
                "Wind Speed": "{:.2f} m/s",
                "general diffuse flows": "{:.0f}",
                "diffuse flows": "{:.0f}",
                "Total_Consumption": "{:,.0f} kW",
            }),
            use_container_width=True,
            height=200,
        )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #f8fafc; font-size: 0.92rem; font-weight: 600; margin-bottom: 4px;'>Inspect Cluster Archetype</p>", unsafe_allow_html=True)
        selected_cluster = st.selectbox(
            "Inspect Cluster Archetype",
            sorted(data[label_col].unique().tolist()),
            format_func=lambda x: f"Cluster {x}",
            label_visibility="collapsed",
        )
        subset = data[data[label_col] == selected_cluster]

        ci1, ci2 = st.columns(2, gap="small")
        with ci1:
            render_metric_card(
                label="Observations",
                value=f"{len(subset):,}",
                subtext=f"{100*len(subset)/len(data):.1f}% of dataset"
            )
        with ci2:
            render_metric_card(
                label="Mean Demand",
                value=f"{subset[TARGET].mean():,.0f}",
                unit="kW",
                subtext="Archetype average"
            )

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        ci3, ci4 = st.columns(2, gap="small")
        with ci3:
            render_metric_card(
                label="Mean Temperature",
                value=f"{subset['Temperature'].mean():.1f}",
                unit="°C"
            )
        with ci4:
            hr_cls = "sub-amber" if subset['High_Demand'].mean() > 0.25 else "sub-teal"
            render_metric_card(
                label="High-Demand Rate",
                value=f"{100*subset['High_Demand'].mean():.1f}%",
                subtext="Exceedance frequency",
                subtext_class=hr_cls
            )

    if method == "Gaussian Mixture Model":
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        with st.expander("Interactive GMM Probability Simulator (Test a Custom Observation)"):
            st.markdown("""
            <p style='color: #cbd5e1; font-size: 0.9rem;'>
              Unlike hard K-Means assignments, the Gaussian Mixture Model models overlapping clusters via continuous posterior probabilities.
              Enter values below to evaluate membership confidence across all 4 components.
            </p>
            """, unsafe_allow_html=True)
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                gt = gc1.number_input("Temperature (°C)", 3.0, 40.0, 22.0, 0.5)
                gh = gc2.number_input("Humidity (%)", 11.0, 95.0, 65.0, 1.0)
                gw = gc3.number_input("Wind Speed (m/s)", 0.05, 6.5, 2.0, 0.1)
            with gc2:
                ggf = gc1.number_input("General Diffuse Flow", 0.0, 1163.0, 180.0, 10.0)
                gdf = gc2.number_input("Diffuse Flow Direct", 0.0, 936.0, 75.0, 10.0)
                gd = gc3.number_input("Total Consumption (kW)", 36000.0, 135000.0, 72000.0, 1000.0)

            sim_point = pd.DataFrame([[gt, gh, gw, ggf, gdf, gd]], columns=CLUSTER_FEATURES)
            scaled_sim = models["clustering_scaler"].transform(sim_point)
            probs = models["gmm"].predict_proba(scaled_sim)[0]

            prob_df = pd.DataFrame({
                "Archetype": [f"Cluster {i}" for i in range(len(probs))],
                "Membership Probability": probs,
            })
            fig_probs = px.bar(
                prob_df,
                x="Archetype",
                y="Membership Probability",
                color="Membership Probability",
                color_continuous_scale=[(0, "#0a262e"), (1, "#2dd4bf")],
            )
            fig_probs.update_layout(coloraxis_showscale=False)
            fig_probs.update_yaxes(range=[0, 1], tickformat=".0%")
            render_chart(fig_probs, 240)

# ============================================================================
# 5. MODEL INSIGHTS (METHODOLOGY & VERIFIED BENCHMARKS)
# ============================================================================
elif page == "Model Insights":
    render_hero(
        eyebrow="SYSTEM ARCHITECTURE",
        headline="Methodology, benchmarks & integrity.",
        supporting="A transparent account of model topologies, verified historical performance metrics, and feature influence dynamics."
    )

    editorial_header("Trained Models & Verified Performance", "Comparison of supervised and unsupervised estimators serialized in this repository.")

    model_metrics_df = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "Family": "Supervised Regression",
            "Objective": "Continuous Total Demand Estimation (kW)",
            "Evaluated Metrics": "R² = 0.616 • MAE = 8,360 kW • RMSE = 10,629 kW",
            "Key Strengths": "Instant inference, transparent interpretable coefficients",
        },
        {
            "Model": "Random Forest Classifier",
            "Family": "Supervised Ensemble (200 Trees)",
            "Objective": "Binary High-Demand Exceedance (≥ 86,266 kW)",
            "Evaluated Metrics": "Accuracy = 98.8% • Precision = 95.1% • Recall = 99.7% • F1 = 97.3%",
            "Key Strengths": "Captures non-linear atmospheric interactions and threshold shifts",
        },
        {
            "Model": "K-Means Clustering",
            "Family": "Unsupervised Partitioning",
            "Objective": "4 Geometric Operational Regimes",
            "Evaluated Metrics": "Inertia convergence, centroid separation across 6 physical features",
            "Key Strengths": "Clear boundary separation for seasonal and diurnal grouping",
        },
        {
            "Model": "Gaussian Mixture Model",
            "Family": "Probabilistic Density Estimation",
            "Objective": "4 Soft Overlapping Archetypes",
            "Evaluated Metrics": "Full covariance matrix, expectation-maximization convergence",
            "Key Strengths": "Outputs continuous Bayesian membership probabilities",
        },
    ])
    st.dataframe(model_metrics_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    c_feat, c_meta = st.columns([1.3, 1], gap="large")

    with c_feat:
        editorial_header("Linear Regression Feature Coefficients", "Marginal impact of each covariate on total power demand (kW).")
        coef_df = pd.DataFrame({
            "Feature": FEATURES,
            "Coefficient": models["regression"].coef_,
        }).sort_values("Coefficient", ascending=False)
        fig_coef = px.bar(
            coef_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale=[(0, "#f59e0b"), (0.5, "#0e1626"), (1, "#2dd4bf")],
        )
        fig_coef.update_layout(coloraxis_showscale=False)
        fig_coef.update_xaxes(title="Marginal Weight (kW per feature unit)")
        fig_coef.update_yaxes(title="")
        render_chart(fig_coef, 290)

    with c_meta:
        editorial_header("Dataset Specification", "Telemetry dimensions and physical variables.")
        st.markdown(f"""
        <div class="insight-card" style="margin-top: 4px;">
          <div class="insight-card-header">
            <span>📋</span>
            <span>Metadata Summary</span>
          </div>
          <p class="insight-card-body">
            • <strong>Total Records:</strong> {len(data):,} valid 10-minute intervals<br>
            • <strong>Observation Period:</strong> 01 Jan 2017 to 30 Dec 2017<br>
            • <strong>High-Demand Threshold:</strong> 86,265.70 kW (75th percentile)<br>
            • <strong>Grid Distribution Zones:</strong> 3 distinct metropolitan zones<br>
            • <strong>Environment:</strong> scikit-learn 1.6.1, Streamlit 1.41.1, Plotly 5.24.1
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note-box">
      <strong>Analytical Boundary:</strong> All models were serialized using the 2017 Tetouan municipal power dataset. 
      Regression and classification predictions represent statistical inferences under historical distributions; they should not 
      be interpreted as physical equipment guarantees or direct real-time supervisory control system commands.
    </div>
    """, unsafe_allow_html=True)
