# AttackDNA — Organization-Specific Behavioral Detection Engine

An anomaly-based network intrusion detection system that learns a **per-host behavioral baseline** and flags deviations from each host's own normal pattern, instead of applying one generic rule set to every host on the network.

Built on the CICIDS2017 dataset, using per-entity Isolation Forest models, a full detection pipeline (cleaning → training → detection → explanation → MITRE ATT&CK mapping), and an interactive Streamlit dashboard.

---

## Why per-entity detection?

Traditional signature-based IDS tools (Sigma, Snort, YARA) apply one static rule set to every host. This causes two well-known problems:
- **High false positive rates** — a rule tuned for a quiet workstation misfires constantly on a busy file server, and vice versa.
- **Blind spots to novel attacks** — signature tools only catch what they already have a rule for.

AttackDNA instead learns what "normal" looks like *per host*, so a deviation is judged against that host's own history rather than a network-wide average.

## Objectives

- Build a genuinely trained ML pipeline — not an API wrapper.
- Detect attacks without ever training on attack-labeled data (fully unsupervised).
- Produce human-readable explanations for every flagged anomaly, not just a binary alert.
- Map detected anomalies to standardized MITRE ATT&CK techniques.
- Present results through an interactive, analyst-facing dashboard.
- Run entirely on CPU, with free/open-source tools, no live infrastructure dependency.

---

## Dataset

**CICIDS2017** (Canadian Institute for Cybersecurity, UNB) — labeled network traffic captured over 5 days, processed by CICFlowMeter into ~79 numeric flow-level statistics per row (duration, packet counts, byte rates, inter-arrival timing, TCP flags) plus a ground-truth `Label`.

This project uses the **GeneratedLabelledFlows** version (not `MachineLearningCSV`), since it retains `Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`, `Protocol`, and `Timestamp` — required for per-entity modeling.

| Day / File | Role | Content |
|---|---|---|
| Monday-WorkingHours | Baseline (training) | 100% benign |
| Tuesday-WorkingHours | Baseline (BENIGN rows only) | Brute force present, filtered out |
| Thursday-Morning-WebAttacks | Baseline (BENIGN rows only) | Web attacks present, filtered out |
| Thursday-Afternoon-Infiltration | Baseline (BENIGN rows only) | Infiltration present, filtered out |
| Wednesday-WorkingHours | Held-out test | DoS: Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed |
| Friday-Morning | Held-out test | Botnet traffic |
| Friday-Afternoon-PortScan | Held-out test | Port scanning |
| Friday-Afternoon-DDoS | Held-out test | DDoS traffic |

Training baseline: **2,106,675 benign rows** across 15 internal entities (Monday + BENIGN-only rows from Tuesday and both Thursday files). No attack-labeled row and no row from Wednesday/Friday is used in training, avoiding data leakage.

---

## Pipeline

```
Core pipeline:
  data_prep.py          → clean raw CSV, derive per-flow "entity" (internal host)
  feature_selection.py  → identify zero-variance / redundant features
  train.py               → train one Isolation Forest per entity on benign baseline
  detect.py              → score held-out traffic against saved per-entity models
  evaluate.py             → precision / recall / F1 / FPR, network-wide and per-entity
  explain.py             → plain-language, z-score-based explanation per anomaly
  mitre_map.py           → rule-based MITRE ATT&CK technique mapping
  dashboard/app.py        → interactive Streamlit dashboard

Extended pipeline (supervisor-requested improvements):
  time_window_features.py → per-entity connections/min & distinct-ports/min
  scan_detector.py        → dedicated rule-based burst/scan detector
  combine_detectors.py     → merges Isolation Forest + scan detector into one flag
  diagnose_entity.py       → compare one entity's baseline vs. attack-period stats
  diagnose_fpr.py          → break down false-positive rate by entity / score distribution
  train_lof.py / detect_lof.py → Local Outlier Factor, benchmarked against Isolation Forest
  seed_stability.py        → repeats training/eval across multiple random seeds
```

Environment: Ubuntu 24.04 LTS, Python virtual environment, CPU-only, 16GB RAM.

---

## Key design decisions

**Defining an "entity":** CICFlowMeter assigns Source/Destination IP based on which side sent the first packet — not internal vs. external. This means a single host's traffic is split across both columns. `data_prep.py` derives an `entity` column pointing to whichever IP is internal (`192.168.10.0/24`), regardless of which raw column it's in. Flows with both sides internal are counted for both entities; flows with neither side internal (~0.06% of data, a capture artifact) are dropped.

**Per-entity calibrated thresholds:** rather than Isolation Forest's default `contamination='auto'` (which flags ~10% of any dataset regardless of actual cleanliness), each entity's threshold is calibrated on a held-out validation split of its own baseline, targeting a 5% false-positive rate.

**Data quality fixes applied:** leading whitespace in column headers; `inf` values in `Flow Bytes/s`/`Flow Packets/s` when duration is 0; a corrupted non-UTF-8 byte (`0x96`) in some Label values (fixed by reading `Thursday-Morning-WebAttacks` with `latin1` encoding); exact duplicate rows.

---

## Results

### Initial remediation (12 identified limitations)

| # | Limitation | Status |
|---|---|---|
| 1 | Limited baseline data (1 day only) | Addressed — expanded to 4 days (mixed result) |
| 2 | Static baseline, never updates | Addressed by design — retraining process defined |
| 3 | Same threshold for every host | Addressed — calibrated per entity, trade-off documented |
| 4 | Only one ML algorithm tested | Addressed — LOF comparison added |
| 5 | No feature selection | Addressed — 79 → 57 features |
| 6 | Weak MITRE mapping logic | Addressed — evidence-based scoring across all features |
| 7 | Explanations are statistical only | Partially addressed — SHAP/LIME remains future work |
| 8 | No real-time detection | Documented as future work |
| 9 | Tested on 1 day, 1 attack type | Addressed — expanded to 2 days, 5 attack types |
| 10 | Risk score used only one factor | Addressed — weighted 4-factor score |
| 11 | No multi-flow pattern detection | Proven limitation, directly addressed |
| 12 | No formal accuracy metrics | Addressed — precision/recall/F1 added |

### Supervisor-requested improvements

| Request | Result |
|---|---|
| Time-window features (fix Bot/PortScan gap) | PortScan recall: 0.2% → **99.2%** (combined system) |
| LOF algorithm comparison | LOF outperforms Isolation Forest on every attack type tested; 99.4% PortScan recall standalone |
| Split/seed stability validation | F1 std. dev. < 1% relative variation across 5 random seeds (DoS/DDoS) |
| False-positive rate root cause | Traced to 2 of 15 entities, two distinct, individually explainable causes |

**Combined detection system (Isolation Forest + scan detector), network-wide F1:**

| Attack | Isolation Forest alone | Combined system |
|---|---|---|
| DoS (Wednesday) | 0.743 | **0.913** |
| DDoS (Friday) | 0.708 | **0.872** |
| PortScan (Friday) | 0.003 | **0.962** |
| Bot (Friday) | 0.005 | 0.005 (unchanged — see below) |

**Known, honestly-reported limitation:** Bot/C2 traffic remains essentially undetected across every approach tested (Isolation Forest, LOF, and the scan detector). This is a confirmed structural gap, not an oversight — bot traffic is deliberately low-volume and non-bursty, so it evades every volume/density-based signal used in this project. Detecting it requires a beaconing-interval/periodicity detector, which is scoped as future work.

---

## Dashboard

An interactive Streamlit dashboard presents all results: top-level KPIs, MITRE technique distribution, risk-band breakdown, an entity leaderboard, and a filterable, color-coded alert table with a CSV export.

```bash
streamlit run dashboard/app.py
```

Risk score formula: `40% anomaly severity + 30% asset criticality + 20% MITRE technique severity + 10% threat intelligence` (threat intel defaults to 0 — no live feed configured).

---

## Setup

```bash
git clone https://github.com/Amberr8/attackdna.git
cd attackdna
python3 -m venv attackdna-env
source attackdna-env/bin/activate
pip install pandas numpy scikit-learn streamlit plotly --break-system-packages
```

Download the CICIDS2017 **GeneratedLabelledFlows** CSVs into `data/raw/` (not included in this repo — see `.gitignore`), then run the pipeline scripts in order as listed above.

---

## Known caveats / future work

- **Bot/C2 detection** needs a periodicity-based detector — the one gap that persisted across every method tried.
- **Fixed 60-second time-window buckets** have an edge effect at bucket boundaries; a true sliding window would remove this.
- **Model-native explanations** (SHAP/LIME) would improve on the current statistical z-score explanations.
- **Real-time detection** (Zeek/Suricata/Kafka) is out of scope for this CPU-only, no-live-infrastructure project.
- **Asset criticality values** in the dashboard are currently placeholders — should be sourced from an actual asset inventory.
- **LOF vs. Isolation Forest comparison** used a wider feature set for LOF (81 features) than Isolation Forest's feature-selected set (57–60 features) — worth re-running on matched feature sets for a stricter apples-to-apples comparison.

---

## Project Report

Full methodology, all source code listings, and detailed results are documented in the accompanying project report PDF.
