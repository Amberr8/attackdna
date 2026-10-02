import sys
import os

# Add src/ and root directories to Python path so modules like config can be loaded
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(BASE_DIR, 'src')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import CFG, DATASET
from evaluate import compute_metrics

st.set_page_config(
    page_title="AttackDNA Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --accent: #4a9eff;
    --accent-soft: rgba(74, 158, 255, 0.12);
    --panel: #171a26;
    --panel-border: #2a2f42;
    --text-dim: #8b93a7;
}

.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(160deg, #1c2030 0%, #161925 100%);
    border: 1px solid var(--panel-border);
    padding: 18px 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.78em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim) !important;
}
div[data-testid="stMetricValue"] { font-weight: 700; }

/* Header block */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 4px;
}
.app-header .icon {
    font-size: 2.1em;
    line-height: 1;
    background: var(--accent-soft);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 10px 14px;
}
.app-header h1 {
    margin: 0;
    font-size: 2.1em;
    font-weight: 800;
    letter-spacing: -0.01em;
}
.app-subtitle {
    color: var(--text-dim);
    font-size: 0.95em;
    margin: 2px 0 18px 0;
}

/* Migration banner */
.migration-banner {
    background: linear-gradient(90deg, var(--accent-soft), transparent);
    border-left: 3px solid var(--accent);
    padding: 14px 20px;
    border-radius: 6px;
    margin-bottom: 22px;
    font-size: 0.92em;
    line-height: 1.5;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--panel-border);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    font-weight: 600;
    color: var(--text-dim);
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
}

/* Section headers */
h2, h3 { font-weight: 700 !important; letter-spacing: -0.005em; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12141d;
    border-right: 1px solid var(--panel-border);
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 18px 0;
    border-bottom: 1px solid var(--panel-border);
    margin-bottom: 18px;
}
.sidebar-brand .icon { font-size: 1.6em; }
.sidebar-brand .name { font-weight: 700; font-size: 1.05em; }
.sidebar-brand .sub { color: var(--text-dim); font-size: 0.75em; }
</style>
""", unsafe_allow_html=True)

# Sidebar branding - renders once, above whatever filters each tab adds below it
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="icon">🧬</div>
    <div>
        <div class="name">AttackDNA</div>
        <div class="sub">GeNIS deployment</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data source paths resolved relative to project root.
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
ALL_ENTITIES_PATH = os.path.join(DATA_DIR, "genis_test_tw.csv")       # every entity in the test set, trained or not
IF_SCORED_PATH = os.path.join(DATA_DIR, "genis_test_scored.csv")       # Isolation Forest alone
COMBINED_PATH = os.path.join(DATA_DIR, "genis_test_combined.csv")      # IF + scan detector
LOF_SCORED_PATH = os.path.join(DATA_DIR, "genis_test_scored_lof.csv")  # LOF alone
ALERTS_PATH = os.path.join(DATA_DIR, "genis_test_mitre.csv")           # combined system's flagged anomalies + MITRE mapping
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
MODELS_LOF_DIR = os.path.join(BASE_DIR, "src", "models_lof")

label_col = CFG["label_col"]
benign_value = CFG["benign_value"]

ENTITY_CRITICALITY = {
    CFG["default_target_entity"]: 90,
}
DEFAULT_CRITICALITY = 50

MITRE_SEVERITY = {
    "T1498 - Network Denial of Service": 85,
    "T1499.002 - Endpoint DoS: Service Exhaustion Flood": 75,
    "T1046 - Network Service Scanning": 70,
    "T1499 - Network Denial of Service (general)": 50,
    "T1499 - Network Denial of Service (default)": 30,
}

SEED_STABILITY = pd.DataFrame([
    {"seed": 1,   "precision": 0.971, "recall": 0.432, "f1": 0.598},
    {"seed": 7,   "precision": 0.965, "recall": 0.419, "f1": 0.584},
    {"seed": 21,  "precision": 0.966, "recall": 0.402, "f1": 0.568},
    {"seed": 99,  "precision": 0.963, "recall": 0.365, "f1": 0.530},
    {"seed": 123, "precision": 0.961, "recall": 0.392, "f1": 0.557},
])


@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def get_entity_coverage():
    all_df = load_csv(ALL_ENTITIES_PATH)
    total_entities = all_df["entity"].nunique() if not all_df.empty and "entity" in all_df.columns else 0

    if_models = set()
    if os.path.isdir(MODELS_DIR):
        if_models = {f.replace(".pkl", "") for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")}
    lof_models = set()
    if os.path.isdir(MODELS_LOF_DIR):
        lof_models = {f.replace(".pkl", "") for f in os.listdir(MODELS_LOF_DIR) if f.endswith(".pkl")}

    return {
        "total_entities": total_entities,
        "if_trained": len(if_models),
        "lof_trained": len(lof_models),
    }


@st.cache_data
def get_model_comparison():
    rows = []
    for name, path in [
        ("Isolation Forest (alone)", IF_SCORED_PATH),
        ("IF + Scan Detector (combined)", COMBINED_PATH),
        ("LOF (alone)", LOF_SCORED_PATH),
    ]:
        if not os.path.exists(path):
            continue
        df = load_csv(path)
        if not df.empty:
            m = compute_metrics(df, label_col, benign_value)
            rows.append({"Model": name, **m})
    return pd.DataFrame(rows)


def compute_risk_score(df, threat_intel_score=0):
    if df.empty or "anomaly_score" not in df.columns:
        return df
    df = df.copy()
    min_score, max_score = df["anomaly_score"].min(), df["anomaly_score"].max()
    spread = max_score - min_score
    if spread == 0:
        anomaly_component = pd.Series(50.0, index=df.index)
    else:
        anomaly_component = 100 * (max_score - df["anomaly_score"]) / spread

    criticality_component = df["entity"].map(ENTITY_CRITICALITY).fillna(DEFAULT_CRITICALITY) if "entity" in df.columns else DEFAULT_CRITICALITY
    mitre_component = df["mitre_technique"].map(MITRE_SEVERITY).fillna(40) if "mitre_technique" in df.columns else 40

    df["risk_score"] = (
        0.40 * anomaly_component +
        0.30 * criticality_component +
        0.20 * mitre_component +
        0.10 * threat_intel_score
    )
    return df


def risk_band(score):
    if score <= 30: return "Normal"
    elif score <= 60: return "Suspicious"
    elif score <= 80: return "High"
    return "Critical"


BAND_COLORS = {"Normal": "#2ecc71", "Suspicious": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}

# ---------------------------------------------------------------------------
# Data Initialization
# ---------------------------------------------------------------------------
coverage = get_entity_coverage()
df = load_csv(ALERTS_PATH)

if not df.empty:
    df = compute_risk_score(df)
    df["risk_band"] = df["risk_score"].apply(risk_band)

st.markdown("""
<div class="app-header">
    <div class="icon">🧬</div>
    <h1>AttackDNA</h1>
</div>
<div class="app-subtitle">Behavioral Anomaly Detection &middot; Per-entity Isolation Forest + scan detector, trained on GeNIS benign baseline traffic.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="migration-banner">
<b>About this dataset:</b> AttackDNA was originally built and validated on CICIDS2017 (2017 traffic).
This deployment uses <b>GeNIS</b> (2025) to test whether the same per-entity behavioral approach
generalizes to newer network data. See the <b>Migration Notes</b> tab for what changed and why.
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Entities in Dataset", coverage["total_entities"])
col2.metric("Entities with IF Model", f"{coverage['if_trained']}/{coverage['total_entities']}")
col3.metric("Flagged Anomalies", f"{len(df):,}")
col4.metric("Critical Risk Alerts", f"{(df['risk_band'] == 'Critical').sum():,}" if not df.empty and 'risk_band' in df.columns else "0")
col5.metric("MITRE Techniques Seen", df["mitre_technique"].nunique() if not df.empty and "mitre_technique" in df.columns else 0)

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Model Comparison", "Top Entities", "Alert Table", "Diagnostics", "Migration Notes"]
)

# ---------------------------------------------------------------------------
with tab1:
    if not df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Risk Band Distribution")
            band_counts = df["risk_band"].value_counts().reindex(
                ["Normal", "Suspicious", "High", "Critical"]).fillna(0).reset_index()
            band_counts.columns = ["Risk Band", "Count"]
            fig = px.bar(band_counts, x="Risk Band", y="Count", color="Risk Band",
                         color_discrete_map=BAND_COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.subheader("MITRE Technique Distribution")
            if "mitre_technique" in df.columns:
                mitre_counts = df["mitre_technique"].value_counts().reset_index()
                mitre_counts.columns = ["Technique", "Count"]
                fig = px.bar(mitre_counts, x="Count", y="Technique", orientation="h")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("True Label Breakdown")
        st.caption("Ground truth from GeNIS - for validation only, not visible in a real deployment.")
        if label_col in df.columns:
            label_counts = df[label_col].value_counts().reset_index()
            label_counts.columns = ["Label", "Count"]
            fig = px.pie(label_counts, names="Label", values="Count", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No alerts data found at `{ALERTS_PATH}`. Run the pipeline (`mitre_map.py`) to populate.")

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Isolation Forest vs Scan Detector vs LOF")
    st.caption("Computed live from each scored file - always matches evaluate.py's output.")

    comparison = get_model_comparison()
    if comparison.empty:
        st.warning("No scored files found. Run detect.py, combine_detectors.py, and detect_lof.py first.")
    else:
        metric_cols = st.columns(len(comparison))
        for i, row in comparison.iterrows():
            with metric_cols[i]:
                st.markdown(f"**{row['Model']}**")
                st.metric("Precision", f"{row['Precision']:.3f}")
                st.metric("Recall", f"{row['Recall']:.3f}")
                st.metric("F1", f"{row['F1']:.3f}")
                st.metric("FPR", f"{row['FPR']:.3f}")

        melted = comparison.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1", "FPR"],
                                  var_name="Metric", value_name="Value")
        fig = px.bar(melted, x="Metric", y="Value", color="Model", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Coverage")
    st.caption("How many of the dataset's entities each model could actually train a baseline for.")
    cov_df = pd.DataFrame([
        {"Model": "Isolation Forest", "Trained": coverage["if_trained"], "Total": coverage["total_entities"]},
        {"Model": "LOF", "Trained": coverage["lof_trained"], "Total": coverage["total_entities"]},
    ])
    cov_df["Coverage %"] = (cov_df["Trained"] / cov_df["Total"] * 100).round(1) if coverage["total_entities"] > 0 else 0
    st.dataframe(cov_df, use_container_width=True)

    st.subheader("Seed Stability (Isolation Forest, 5 seeds)")
    st.caption("From seed_stability.py. F1 std here is notably higher than the <1% relative "
               "variation found on CICIDS2017 - smaller per-entity baselines make GeNIS results "
               "more seed-sensitive.")
    st.dataframe(SEED_STABILITY, use_container_width=True)
    st.write(f"F1: mean={SEED_STABILITY['f1'].mean():.3f}, std={SEED_STABILITY['f1'].std():.4f}")

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Entities Ranked by Average Risk Score")
    if not df.empty and "entity" in df.columns:
        entity_summary = (df.groupby("entity")
                           .agg(avg_risk=("risk_score", "mean"), alerts=("risk_score", "count"))
                           .sort_values("avg_risk", ascending=False).reset_index())
        fig = px.bar(entity_summary.head(20), x="entity", y="avg_risk", color="avg_risk",
                     color_continuous_scale="Reds", labels={"avg_risk": "Avg Risk Score"})
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(entity_summary, use_container_width=True)

# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Flagged Alerts (highest risk first)")
    if not df.empty:
        st.sidebar.markdown("### Filters")
        entities = sorted(df["entity"].unique()) if "entity" in df.columns else []
        selected_entity = st.sidebar.selectbox("Entity", ["All"] + [str(e) for e in entities])
        selected_bands = st.sidebar.multiselect(
            "Risk Band", ["Normal", "Suspicious", "High", "Critical"],
            default=["Suspicious", "High", "Critical"])

        techs = sorted(df["mitre_technique"].unique()) if "mitre_technique" in df.columns else []
        selected_techniques = st.sidebar.multiselect("MITRE Technique", techs, default=techs)

        view_df = df.copy()
        if selected_entity != "All" and "entity" in view_df.columns:
            view_df = view_df[view_df["entity"].astype(str) == selected_entity]
        if selected_bands and "risk_band" in view_df.columns:
            view_df = view_df[view_df["risk_band"].isin(selected_bands)]
        if selected_techniques and "mitre_technique" in view_df.columns:
            view_df = view_df[view_df["mitre_technique"].isin(selected_techniques)]

        display_cols = [c for c in ["entity", "risk_score", "risk_band", label_col,
                        "mitre_technique", "mitre_reasoning", "scan_flag", "explanation"] if c in view_df.columns]
        display_df = view_df.sort_values("risk_score", ascending=False)[display_cols].head(300)

        def highlight_band(row):
            color = BAND_COLORS.get(row.get("risk_band"), "white")
            return [f"background-color: {color}20"] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_band, axis=1).format({"risk_score": "{:.1f}"}),
            use_container_width=True, height=550)

        st.download_button(
            "Download filtered alerts as CSV",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="attackdna_genis_filtered_alerts.csv",
            mime="text/csv")

# ---------------------------------------------------------------------------
with tab5:
    st.subheader("False Positive Root Cause")
    if os.path.exists(COMBINED_PATH):
        full_df = load_csv(COMBINED_PATH)
        if not full_df.empty and label_col in full_df.columns and "predicted" in full_df.columns:
            fp = full_df[(full_df[label_col] == benign_value) & (full_df["predicted"] == -1)]
            tn = full_df[(full_df[label_col] == benign_value) & (full_df["predicted"] == 1)]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**False positive rate by entity (top 10)**")
                total_by_entity = full_df[full_df[label_col] == benign_value]["entity"].value_counts()
                fp_by_entity = fp["entity"].value_counts()
                fpr_series = (fp_by_entity / total_by_entity).dropna().sort_values(ascending=False).head(10)
                fpr_plot_df = fpr_series.reset_index()
                fpr_plot_df.columns = ["entity", "fpr"]
                fig = px.bar(fpr_plot_df, x="entity", y="fpr")
                fig.update_xaxes(type="category")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**Anomaly score: false positives vs true negatives**")
                if "anomaly_score" in full_df.columns:
                    hist_df = pd.concat([
                        fp.assign(group="False Positive")[["anomaly_score", "group"]],
                        tn.assign(group="True Negative")[["anomaly_score", "group"]],
                    ])
                    fig = px.histogram(hist_df, x="anomaly_score", color="group", barmode="overlay",
                                        nbins=40, opacity=0.6)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"{COMBINED_PATH} not found.")

    st.subheader("Scan Detector Coverage")
    st.caption(
        "14.0% recon recall network-wide, but 98.7% on the subset of entities the scan "
        "detector actually has a calibrated threshold for (7 of 22 recon-generating entities). "
        "This mirrors the coverage-gap pattern above, not a weaker detector.")

# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Why GeNIS, and What Changed From CICIDS2017")
    st.markdown("""
**Why migrate.** CICIDS2017 is from 2017. The project was asked to validate the same
per-entity approach against newer traffic, while keeping the CICIDS2017 results as the
original baseline (see the project report's discussion of dataset age).

**Why GeNIS specifically.** Most newer public NIDS datasets strip IP addresses before
release, since they're built for generic ML classification, not per-host behavioral
modeling - the entity concept this project is built around requires them. GeNIS (2025)
was confirmed to retain host-identifying fields, via HERA-exported flow context features.

**What had to change:**
- **No internal/external subnet.** GeNIS is a single closed testbed, so `entity` is
  built from both `Ssaddr`/`Sdaddr` directly, with no `192.168.10.x`-style filtering.
- **No real timestamp.** GeNIS has no wall-clock timestamp column, only `Offset`
  (confirmed to be an Argus record-position value, not elapsed time). Time-window burst
  features bucket by `Offset` ranges instead of real seconds - a documented substitution,
  not literal per-minute bucketing.
- **Mixed train/test, not day-separated.** CICIDS2017's benign baseline came from entire
  clean days with zero attack traffic. GeNIS's train file is a random split, so a
  benign-labeled flow's own time bucket can contain concurrent attack traffic. Scan
  detector calibration filters to buckets that are 100% benign to avoid this contamination.
- **Feature set.** GeNIS's 87 HERA/Argus-based columns share no names with CICFlowMeter's
  79. Zero-variance and correlated-feature drop lists, and the MITRE timing/volume feature
  categories, were rebuilt from scratch against GeNIS's actual columns.

**What stayed the same:** the core architecture (per-entity Isolation Forest with a
calibrated threshold, LOF comparison, scan detector, rule-based MITRE mapping, risk
scoring) is unchanged. Every number on this dashboard comes from that same pipeline,
just pointed at a different dataset.
""")
