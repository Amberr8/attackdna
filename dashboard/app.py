
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AttackDNA Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom styling ---
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        border: 1px solid #2d3348;
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
""", unsafe_allow_html=True)


TOTAL_ENTITIES_PATH = "data/processed/wednesday_combined.csv"
ALERTS_PATH = "data/processed/wednesday_mitre_final.csv"
# Assign business criticality per host (0-100). Customize this to your
# actual network's asset importance - defaults to 50 for anything not
# listed. This is the "Asset Criticality" term in the risk formula.
ENTITY_CRITICALITY = {
    "192.168.10.50": 90,   # Web server - primary DoS target, business-critical
}
DEFAULT_CRITICALITY = 50

# Severity weight per MITRE technique (0-100), used in the risk formula.
MITRE_SEVERITY = {
    "T1498 - Network Denial of Service": 85,
    "T1499.002 - Endpoint DoS: Service Exhaustion Flood": 75,
    "T1046 - Network Service Scanning": 70,
    "T1499 - Network Denial of Service (general)": 50,
    "T1499 - Network Denial of Service (default)": 30,
}

@st.cache_data
def load_total_entity_count():
    """Entities Monitored should reflect everything the system watches,
    not just entities that happen to have a flagged anomaly today."""
    df = pd.read_csv(TOTAL_ENTITIES_PATH, usecols=["entity"])
    return df["entity"].nunique()


@st.cache_data
def load_alerts():
    return pd.read_csv(ALERTS_PATH)


def compute_risk_score(df, threat_intel_score=0):
    """Weighted risk score combining four signals, per the project's
    risk-scoring design: 40% anomaly severity, 30% asset criticality,
    20% MITRE technique severity, 10% threat intelligence (defaults to
    0 - no live feed wired in yet, documented as future work)."""
    df = df.copy()

    min_score, max_score = df["anomaly_score"].min(), df["anomaly_score"].max()
    spread = max_score - min_score
    if spread == 0:
        anomaly_component = pd.Series(50.0, index=df.index)
    else:
        anomaly_component = 100 * (max_score - df["anomaly_score"]) / spread

    criticality_component = df["entity"].map(ENTITY_CRITICALITY).fillna(DEFAULT_CRITICALITY)
    mitre_component = df["mitre_technique"].map(MITRE_SEVERITY).fillna(40)

    df["risk_score"] = (
        0.40 * anomaly_component +
        0.30 * criticality_component +
        0.20 * mitre_component +
        0.10 * threat_intel_score
    )
    return df


def risk_band(score):
    if score <= 30:
        return "Normal"
    elif score <= 60:
        return "Suspicious"
    elif score <= 80:
        return "High"
    return "Critical"


BAND_COLORS = {"Normal": "#2ecc71", "Suspicious": "#f1c40f",
               "High": "#e67e22", "Critical": "#e74c3c"}

# --- Load & prep ---
total_entities = load_total_entity_count()
df = load_alerts()
df = compute_risk_score(df)
df["risk_band"] = df["risk_score"].apply(risk_band)

# --- Header ---
st.title("🧬 AttackDNA — Behavioral Anomaly Detection")
st.caption(
    "Per-entity Isolation Forest models, trained on Monday/Tuesday/Thursday "
    "benign baselines, evaluated on Wednesday's DoS traffic."
)

# --- Sidebar filters ---
st.sidebar.header("Filters")
entities = sorted(df["entity"].unique())
selected_entity = st.sidebar.selectbox("Entity", ["All"] + entities)
selected_bands = st.sidebar.multiselect(
    "Risk Band", ["Normal", "Suspicious", "High", "Critical"],
    default=["Suspicious", "High", "Critical"]
)
selected_techniques = st.sidebar.multiselect(
    "MITRE Technique", sorted(df["mitre_technique"].unique()),
    default=sorted(df["mitre_technique"].unique())
)

view_df = df.copy()
if selected_entity != "All":
    view_df = view_df[view_df["entity"] == selected_entity]
if selected_bands:
    view_df = view_df[view_df["risk_band"].isin(selected_bands)]
if selected_techniques:
    view_df = view_df[view_df["mitre_technique"].isin(selected_techniques)]

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Entities Monitored", total_entities)
col2.metric("Flagged Anomalies (filtered)", f"{len(view_df):,}")
col3.metric("Critical Risk Alerts", f"{(view_df['risk_band'] == 'Critical').sum():,}")
col4.metric("MITRE Techniques Seen", view_df["mitre_technique"].nunique())

st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Overview", "Top Entities", "Alert Table"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Risk Band Distribution")
        band_counts = view_df["risk_band"].value_counts().reindex(
            ["Normal", "Suspicious", "High", "Critical"]
        ).fillna(0).reset_index()
        band_counts.columns = ["Risk Band", "Count"]
        fig = px.bar(band_counts, x="Risk Band", y="Count", color="Risk Band",
                     color_discrete_map=BAND_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("MITRE Technique Distribution")
        mitre_counts = view_df["mitre_technique"].value_counts().reset_index()
        mitre_counts.columns = ["Technique", "Count"]
        fig = px.bar(mitre_counts, x="Count", y="Technique", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("True Label Breakdown")
    st.caption("Ground truth from CICIDS2017 - shown for validation, not visible in a real deployment.")
    label_counts = view_df["Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    fig = px.pie(label_counts, names="Label", values="Count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Entities Ranked by Average Risk Score")
    entity_summary = (
        view_df.groupby("entity")
        .agg(avg_risk=("risk_score", "mean"), alerts=("risk_score", "count"))
        .sort_values("avg_risk", ascending=False)
        .reset_index()
    )
    fig = px.bar(entity_summary, x="entity", y="avg_risk", color="avg_risk",
                 color_continuous_scale="Reds", labels={"avg_risk": "Avg Risk Score"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(entity_summary, use_container_width=True)

with tab3:
    st.subheader("Flagged Alerts (highest risk first)")
    display_cols = ["entity", "risk_score", "risk_band", "Label",
                 "mitre_technique", "mitre_reasoning", "scan_flag", "explanation"]
    display_df = view_df.sort_values("risk_score", ascending=False)[display_cols].head(300)

    def highlight_band(row):
        color = BAND_COLORS.get(row["risk_band"], "white")
        return [f"background-color: {color}20"] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_band, axis=1).format({"risk_score": "{:.1f}"}),
        use_container_width=True, height=550
    )

    st.download_button(
        "Download filtered alerts as CSV",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="attackdna_filtered_alerts.csv",
        mime="text/csv"
    )

st.caption(
    "Risk score = 40% anomaly severity + 30% asset criticality + 20% MITRE "
    "severity + 10% threat intelligence (no live feed configured; defaults to 0)."
)
