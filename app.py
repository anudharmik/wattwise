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
    page_title="WattWise AI | Energy Intelligence",
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
    # PCA is used only to visualize the existing usage segments.
    sample = data.sample(min(5000, len(data)), random_state=42).copy()
    scaler = joblib.load(MODEL_DIR / "clustering_scaler.joblib")
    scaled = scaler.transform(sample[features])
    points = PCA(n_components=2, random_state=42).fit_transform(scaled)
    sample["PCA 1"] = points[:, 0]
    sample["PCA 2"] = points[:, 1]
    return sample

models = load_models()
data = load_data()

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fb; color: #172033; }
[data-testid="stSidebar"] { background: #111c31; }
[data-testid="stSidebar"] * { color: #edf3ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 7px 10px; border-radius: 9px; }
.hero {
    padding: 27px 30px; border-radius: 22px; color: white;
    background: linear-gradient(115deg, #13213b 0%, #203f65 56%, #287d88 100%);
    margin-bottom: 20px; box-shadow: 0 12px 32px rgba(25, 49, 82, .14);
}
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; margin: 0 0 7px 0; color: white; }
.hero p { color: #d9e8f4; margin: 0; font-size: 1rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; color: #72e0cf; margin-bottom: 9px; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; margin: 10px 0 2px 0; color: #172033; }
.section-sub { color: #64748b; margin-bottom: 14px; font-size: .92rem; }
.note {
    background: #eaf5f4; border-left: 4px solid #218b84; border-radius: 9px;
    padding: 12px 15px; color: #244653; margin: 8px 0 16px 0; font-size: .91rem;
}
div[data-testid="stMetric"] {
    background: white; padding: 16px 18px; border: 1px solid #e6eaf1;
    border-radius: 16px; box-shadow: 0 3px 12px rgba(26, 43, 68, .035);
}
div[data-testid="stMetricLabel"] { color: #64748b; }
div[data-testid="stMetricValue"] { color: #172033; font-family: 'Space Grotesk', sans-serif; }
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
    background: #167f7a; color: white; border: 0; border-radius: 10px;
    padding: .55rem 1rem; font-weight: 700;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    background: #116b67; color: white; border: 0;
}
.small-muted { color: #64748b; font-size: .83rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Shared helpers -----------------------------
def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

def chart_layout(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e9edf4", zeroline=False)
    return fig

def input_frame(temp, humidity, wind, diffuse, diffuse_direct, hour, day):
    return pd.DataFrame([{
        "Temperature": temp,
        "Humidity": humidity,
        "Wind Speed": wind,
        "general diffuse flows": diffuse,
        "diffuse flows": diffuse_direct,
        "Hour": hour,
        "Day": day,
    }], columns=FEATURES)

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## ⚡ WattWise AI")
    st.caption("ENERGY INTELLIGENCE")
    page = st.radio(
        "Navigate",
        ["Overview", "Demand predictor", "Usage patterns", "Model notes"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"{len(data):,} observations")
    st.caption(f"{data['DateTime'].min():%d %b %Y} — {data['DateTime'].max():%d %b %Y}")
    st.markdown("---")
    st.caption("Mini-project • Machine Learning")

# ----------------------------- Overview -----------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Powering better decisions</div>
      <h1>Understand the pulse of energy.</h1>
      <p>Explore electricity demand, discover usage patterns, and estimate what conditions could mean for the grid.</p>
    </div>
    """, unsafe_allow_html=True)

    total_avg = data[TARGET].mean()
    peak = data[TARGET].max()
    high_pct = 100 * data["High_Demand"].mean()
    zone_cols = ["Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average total demand", f"{total_avg:,.0f} kW")
    k2.metric("Peak observed demand", f"{peak:,.0f} kW")
    k3.metric("High-demand observations", f"{high_pct:.1f}%")
    k4.metric("Distribution zones", "03")

    st.markdown("")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section("Demand through the year", "Observed total consumption across the recorded period.")
        daily = data.set_index("DateTime")[TARGET].resample("D").mean().reset_index()
        fig = px.area(daily, x="DateTime", y=TARGET, color_discrete_sequence=["#198c88"])
        fig.update_traces(line=dict(width=1.8), fillcolor="rgba(25,140,136,0.14)")
        fig.update_yaxes(title="Average consumption (kW)")
        fig.update_xaxes(title="")
        st.plotly_chart(chart_layout(fig, 365), use_container_width=True)
    with right:
        section("Where demand is distributed", "Share of total recorded zone consumption.")
        zone_totals = data[zone_cols].sum()
        zone_view = pd.DataFrame({"Zone": ["Zone 1", "Zone 2", "Zone 3"], "Consumption": zone_totals.values})
        fig = px.pie(zone_view, names="Zone", values="Consumption", hole=0.68,
                     color_discrete_sequence=["#198c88", "#f1a65b", "#5d77b8"])
        fig.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(chart_layout(fig, 315), use_container_width=True)
        st.markdown('<div class="note">Zone totals are historical measurements. The model predicts combined demand from weather and time features.</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        section("Daily rhythm", "Average demand by hour of day.")
        hourly = data.groupby("Hour", as_index=False)[TARGET].mean()
        fig = px.bar(hourly, x="Hour", y=TARGET, color=TARGET,
                     color_continuous_scale=["#bde8df", "#16827b"])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(dtick=2, title="Hour of day")
        fig.update_yaxes(title="Average consumption (kW)")
        st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
    with right:
        section("Weather and consumption", "Temperature against observed demand, sampled for readability.")
        sample = data.sample(min(2500, len(data)), random_state=42)
        fig = px.scatter(sample, x="Temperature", y=TARGET, color="High_Demand",
                         color_discrete_map={0: "#9aa8bb", 1: "#e58a48"},
                         labels={"High_Demand": "High demand", TARGET: "Total consumption (kW)"}, opacity=0.62)
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(chart_layout(fig, 300), use_container_width=True)

    st.caption("Source: Tetouan City power-consumption dataset. Values shown are historical observations; predictions are estimates, not operational grid instructions.")

# ----------------------------- Demand predictor -----------------------------
elif page == "Demand predictor":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Interactive machine learning</div>
      <h1>What might demand look like?</h1>
      <p>Change weather and time conditions to get an estimated consumption level and high-demand risk.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="note">The prediction uses the saved regression and Random Forest models. Enter realistic values from the dataset range for the most meaningful estimate.</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature = st.slider("Temperature (°C)", 3.0, 40.0, 20.0, 0.5)
            humidity = st.slider("Humidity (%)", 11.0, 95.0, 68.0, 1.0)
        with c2:
            wind = st.slider("Wind speed", 0.05, 6.5, 2.0, 0.05)
            hour = st.slider("Hour of day", 0, 23, 18)
        with c3:
            day_label = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
            diffuse_flow = st.slider("General diffuse flow", 0.0, 1163.0, 180.0, 5.0)
            direct_flow = st.slider("Diffuse flow", 0.0, 936.0, 75.0, 5.0)
        submitted = st.form_submit_button("Estimate demand", use_container_width=True)

    if submitted:
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_label)
        row = input_frame(temperature, humidity, wind, diffuse_flow, direct_flow, hour, day)
        estimated = float(models["regression"].predict(row)[0])
        scaled = models["classification_scaler"].transform(row)
        high_prob = float(models["classifier"].predict_proba(scaled)[0][1])
        is_high = int(models["classifier"].predict(scaled)[0])

        st.markdown("")
        a, b, c = st.columns(3)
        a.metric("Estimated total demand", f"{estimated:,.0f} kW")
        b.metric("High-demand probability", f"{high_prob:.1%}")
        b.caption("Model probability; not a calibrated guarantee.")
        c.metric("Model classification", "High demand" if is_high else "Normal demand")
        st.progress(min(max(high_prob, 0.0), 1.0), text="Estimated high-demand probability")

        if is_high:
            st.warning("The classifier flags these conditions as high demand. Consider this a model-based signal, not a grid operation recommendation.")
        else:
            st.success("The classifier flags these conditions as normal demand.")

        # Compare the estimate with the historical distribution.
        percentile = float((data[TARGET] <= estimated).mean() * 100)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=estimated,
            number={"valueformat": ",.0f", "suffix": " kW"},
            title={"text": "Predicted demand against historical range"},
            gauge={
                "axis": {"range": [float(data[TARGET].min()), float(data[TARGET].max())]},
                "bar": {"color": "#16827b"},
                "steps": [
                    {"range": [float(data[TARGET].min()), HIGH_DEMAND_THRESHOLD], "color": "#e5f4f0"},
                    {"range": [HIGH_DEMAND_THRESHOLD, float(data[TARGET].max())], "color": "#fff0df"},
                ],
                "threshold": {"line": {"color": "#d97735", "width": 3},
                              "thickness": 0.8, "value": HIGH_DEMAND_THRESHOLD},
            },
        ))
        st.plotly_chart(chart_layout(fig, 280), use_container_width=True)
        st.caption(f"This estimate is around the {percentile:.0f}th percentile of historical total consumption. The gauge threshold is the training-period 75th percentile.")

    else:
        st.info("Adjust the inputs and select **Estimate demand** to run the saved models.")

# ----------------------------- Usage patterns -----------------------------
elif page == "Usage patterns":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Unsupervised learning</div>
      <h1>Find the patterns behind the numbers.</h1>
      <p>Explore K-Means groups and GMM assignments across weather and total demand.</p>
    </div>
    """, unsafe_allow_html=True)

    method = st.segmented_control("Segmentation method", options=["K-Means", "Gaussian Mixture Model"], default="K-Means")
    label_col = "KMeans_Cluster" if method == "K-Means" else "GMM_Cluster"
    colors = px.colors.qualitative.Safe

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        section("Usage map", "PCA compresses six input features into two axes for visualization.")
        pca_data = get_pca_sample(data, CLUSTER_FEATURES)
        pca_data["Cluster"] = pca_data[label_col].astype(str)
        fig = px.scatter(
            pca_data, x="PCA 1", y="PCA 2", color="Cluster",
            color_discrete_sequence=colors,
            labels={"Cluster": "Cluster"},
            opacity=0.68,
            hover_data=["DateTime", "Temperature", "Humidity", TARGET],
        )
        fig.update_traces(marker=dict(size=5))
        st.plotly_chart(chart_layout(fig, 420), use_container_width=True)
    with right:
        section("Cluster fingerprints", "Average conditions and consumption within each group.")
        profile = data.groupby(label_col)[CLUSTER_FEATURES].mean().round(2)
        profile.index = [f"Cluster {i}" for i in profile.index]
        st.dataframe(profile, use_container_width=True, height=315)
        selected_cluster = st.selectbox("Inspect a cluster", sorted(data[label_col].unique().tolist()))
        chosen = data[data[label_col] == selected_cluster]
        st.metric("Observations in cluster", f"{len(chosen):,}")
        st.metric("Average total demand", f"{chosen[TARGET].mean():,.0f} kW")
        st.metric("Average temperature", f"{chosen['Temperature'].mean():.1f} °C")

    if method == "Gaussian Mixture Model":
        st.markdown('<div class="note">GMM can represent overlapping groups using membership probabilities. The stored dataset contains the final GMM cluster labels; membership probabilities are calculated for new inputs only when the GMM model is applied.</div>', unsafe_allow_html=True)
        with st.expander("Try a new observation"):
            c1, c2, c3 = st.columns(3)
            t = c1.number_input("Temperature", 3.0, 40.0, 20.0)
            h = c2.number_input("Humidity", 11.0, 95.0, 68.0)
            w = c3.number_input("Wind speed", 0.05, 6.5, 2.0)
            gf = c1.number_input("General diffuse flow", 0.0, 1163.0, 180.0)
            dfv = c2.number_input("Diffuse flow", 0.0, 936.0, 75.0)
            demand = c3.number_input("Total consumption", 36785.0, 134208.0, 71000.0)
            if st.button("Estimate cluster membership"):
                new_point = pd.DataFrame([[t, h, w, gf, dfv, demand]], columns=CLUSTER_FEATURES)
                scaled_point = models["clustering_scaler"].transform(new_point)
                probabilities = models["gmm"].predict_proba(scaled_point)[0]
                prob_df = pd.DataFrame({"Cluster": [f"Cluster {i}" for i in range(len(probabilities))],
                                        "Membership probability": probabilities})
                fig = px.bar(prob_df, x="Cluster", y="Membership probability",
                             color="Membership probability", color_continuous_scale=["#bde8df", "#16827b"])
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(chart_layout(fig, 260), use_container_width=True)

    st.caption("Cluster numbers are algorithm labels, not a ranking. Interpret groups by their feature profiles; they do not represent individual customers.")

# ----------------------------- Model notes -----------------------------
else:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Behind the interface</div>
      <h1>Simple models. Clear explanations.</h1>
      <p>How WattWise AI turns the lab practicals into one coherent energy analytics project.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Models in this project")
    model_info = pd.DataFrame([
        ["Linear Regression", "Estimate continuous total electricity consumption.", "MAE, MSE, RMSE and R²"],
        ["Random Forest Classifier", "Flag normal vs high-demand observations.", "Accuracy, precision, recall and F1"],
        ["K-Means", "Assign observations to one of four usage groups.", "Cluster profiles and PCA plot"],
        ["Gaussian Mixture Model", "Model overlapping usage groups and membership probabilities.", "Cluster profiles and posterior probabilities"],
        ["GridSearchCV / RandomizedSearchCV", "Compare hyperparameter search strategies in the Colab notebook.", "Cross-validation score and runtime"],
    ], columns=["Technique", "Purpose", "How it is evaluated"])
    st.dataframe(model_info, use_container_width=True, hide_index=True)

    st.subheader("What the prediction means")
    st.write(
        "The regression model estimates combined demand from weather and time features. "
        "The classifier estimates whether the same conditions resemble high-demand observations. "
        "Neither model establishes causation or should be treated as a live utility control system."
    )

    st.subheader("Data snapshot")
    a, b, c = st.columns(3)
    a.metric("Observations", f"{len(data):,}")
    b.metric("Input features", len(FEATURES))
    c.metric("Recorded period", "2017")
    st.code(", ".join(FEATURES), language="text")
    st.caption("Models were serialized from a scikit-learn 1.6.1 environment. The included requirements pin this version to preserve compatibility.")
