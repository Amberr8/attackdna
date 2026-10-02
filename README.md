# 🧬 AttackDNA

### Organization-Specific Behavioral Detection Engine

**AttackDNA** is an unsupervised, machine-learning-based network anomaly detection system that learns the normal behavioral profile of individual network entities and identifies deviations from those profiles.

Unlike traditional signature-based detection approaches that apply the same rules across an entire network, AttackDNA builds **entity-specific behavioral baselines**. This allows suspicious activity to be evaluated relative to the normal behavior of the host generating the traffic.

The project combines:

* Per-entity behavioral modeling
* Unsupervised anomaly detection
* Isolation Forest
* Local Outlier Factor (LOF) comparison
* Time-window burst detection
* Statistical anomaly explanations
* MITRE ATT&CK technique mapping
* Risk scoring
* Interactive Streamlit dashboard
* Cross-dataset validation using CICIDS2017 and GeNIS (2025)

---

## 🎯 Problem Statement

Traditional intrusion detection systems often rely on predefined signatures or generic detection rules.

This creates two important challenges:

1. **High false positives** — behavior that is normal for one host may be unusual for another.
2. **Detection blind spots** — signature-based systems primarily detect patterns that have already been defined.

AttackDNA addresses this problem by learning what **normal behavior looks like for each individual network entity** and identifying significant deviations from that entity's own baseline.

The project is designed as a genuinely trained machine-learning pipeline rather than an API wrapper.

---

## 💡 Core Idea

The central concept of AttackDNA is:

> **Normal behavior is entity-specific.**

Instead of creating one global model:

```text
Network Traffic
       │
       ▼
 ┌──────────────┐
 │ Global Model │
 └──────────────┘
       │
       ▼
   Detection
```

AttackDNA creates behavioral models for individual entities:

```text
                    Network Traffic
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Entity A     Entity B     Entity C
             │            │            │
             ▼            ▼            ▼
        Behavioral   Behavioral   Behavioral
          Model        Model        Model
             │            │            │
             └────────────┼────────────┘
                          ▼
                  Anomaly Detection
                          │
                          ▼
                Explanation + MITRE
                          │
                          ▼
                     Risk Score
                          │
                          ▼
                    Dashboard
```

Each entity receives its own behavioral baseline and calibrated anomaly threshold.

---

# 🔬 Main Features

## 1. Per-Entity Behavioral Modeling

AttackDNA identifies individual network entities from source and destination host identifiers.

Each entity receives its own:

* Feature baseline
* Machine-learning model
* Scaler
* Anomaly threshold

This allows the system to evaluate traffic relative to the entity's own historical behavior.

---

## 2. Unsupervised Anomaly Detection

The main detection model is **Isolation Forest**.

Attack-labeled traffic is not used to train the primary anomaly detector.

The model learns from benign behavioral data and identifies observations that deviate from the learned baseline.

The project uses:

```text
StandardScaler
      ↓
Isolation Forest
      ↓
Entity-specific threshold
      ↓
Anomaly / Normal
```

A custom threshold is calibrated using validation data with a target false-positive rate of approximately **5%**.

---

## 3. Time-Window Burst Detection

Isolation Forest showed a structural weakness when dealing with dense, repetitive reconnaissance traffic.

To address this, AttackDNA introduced two behavioral burst features:

```text
connections_per_minute
distinct_ports_per_minute
```

These features capture activity that may appear normal at the individual-flow level but becomes suspicious when many similar connections occur within a short capture window.

For GeNIS, which does not contain a real wall-clock timestamp, the system uses fixed ranges of the `Offset` field as a documented capture-order approximation.

---

## 4. Local Outlier Factor Comparison

AttackDNA also implements **Local Outlier Factor (LOF)** as an alternative anomaly detection approach.

The project compares:

* Isolation Forest
* LOF
* Isolation Forest + scan detector

This allows the behavior of different unsupervised approaches to be evaluated rather than assuming that one algorithm is universally optimal.

---

## 5. Explainable Anomalies

AttackDNA does not stop at:

```text
ANOMALY = TRUE
```

For flagged traffic, the system examines deviations from the entity's normal feature statistics.

Example explanation:

```text
Destination-related feature is unusually high
(z = 5.4)

Packet-related feature is 3.2x higher than normal
(z = 4.1)
```

The explanation system uses per-entity baseline statistics and feature z-scores to identify the strongest deviations.

---

## 6. MITRE ATT&CK Mapping

Detected anomalies are mapped to relevant **MITRE ATT&CK techniques** using evidence-based feature patterns.

Examples include:

* Network Service Scanning
* Network Denial of Service
* Endpoint Denial of Service
* Other network-related techniques supported by observed behavior

The purpose of the mapping is to make anomaly results more understandable from a cybersecurity analyst's perspective.

---

## 7. Risk Scoring

AttackDNA combines multiple factors into a weighted risk score:

```text
Risk Score =
    40% Anomaly Score
  + 30% Entity Criticality
  + 20% MITRE Severity
  + 10% Threat Intelligence
```

This allows the system to provide more contextual information than a simple anomaly/normal classification.

---

# 📊 Datasets

AttackDNA was evaluated on two datasets.

## CICIDS2017

CICIDS2017 was used as the original development and validation dataset.

The project uses the version containing:

* Source IP
* Destination IP
* Source Port
* Destination Port
* Protocol
* Timestamp
* Flow statistics
* Attack labels

The final CICIDS2017 training baseline contained:

* **2,106,675 benign training rows**
* **15 internal entities**
* No attack-labeled rows in training

The Wednesday and Friday traffic was reserved for evaluation to reduce data leakage.

---

## GeNIS 2025

The methodology was subsequently adapted to **GeNIS**, a newer 2025 network intrusion dataset.

GeNIS was selected because it retained anonymized host identifiers, source/destination ports, and multiple label granularities, making it suitable for per-entity behavioral modeling.

### GeNIS characteristics

| Property                             | GeNIS              |
| ------------------------------------ | ------------------ |
| Dataset year                         | 2025               |
| Flow exporter                        | HERA / Argus-based |
| Host identifiers                     | `Ssaddr`, `Sdaddr` |
| Destination port                     | `Dport`            |
| Main label                           | `CategoryLabel`    |
| Entities                             | 116                |
| Entities with sufficient benign data | 80                 |
| Training rows                        | 589,688            |
| Test rows                            | 147,424            |

Unlike CICIDS2017, GeNIS does not provide a real wall-clock timestamp. Its `Offset` field was therefore used as a capture-order proxy for the burst detector.

---

# 🏗️ System Architecture

```text
                    Raw Network Flow Data
                              │
                              ▼
                    ┌──────────────────┐
                    │  Data Preprocess │
                    │                  │
                    │ • Cleaning       │
                    │ • Entity         │
                    │   extraction     │
                    │ • Label cleanup  │
                    │ • Deduplication  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature Selection│
                    │                  │
                    │ • Low variance   │
                    │ • Correlation    │
                    │ • Leakage removal│
                    └────────┬─────────┘
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
       ┌────────────────┐         ┌────────────────┐
       │ Isolation      │         │ Time-Window    │
       │ Forest         │         │ Features       │
       │                │         │                │
       │ Per Entity     │         │ Connections/min│
       │ Calibration    │         │ Distinct ports │
       └───────┬────────┘         └───────┬────────┘
               │                          │
               └────────────┬─────────────┘
                            ▼
                   ┌──────────────────┐
                   │ Scan Detector    │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Combined Results │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Explanation      MITRE        Risk Score
                         Mapping
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                   ┌──────────────────┐
                   │ Streamlit        │
                   │ Dashboard        │
                   └──────────────────┘
```

---

# 📁 Project Structure

```text
AttackDNA/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── feature_selection.py
│   ├── train.py
│   ├── detect.py
│   ├── train_lof.py
│   ├── detect_lof.py
│   ├── time_window_features.py
│   ├── scan_detector.py
│   ├── combine_detectors.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── mitre_map.py
│   ├── diagnose_entity.py
│   ├── diagnose_fpr.py
│   └── seed_stability.py
│
├── models/
│   └── entity-specific models
│
├── models_lof/
│   └── LOF models
│
└── README.md
```

The pipeline is configuration-driven through `config.py`, allowing dataset-specific settings to be changed without maintaining separate codebases.

---

# ⚙️ Pipeline

The main processing pipeline is:

```text
1. Raw CSV
     ↓
2. Data preprocessing
     ↓
3. Entity extraction
     ↓
4. Feature selection
     ↓
5. Time-window feature generation
     ↓
6. Per-entity model training
     ↓
7. Anomaly detection
     ↓
8. Scan detection
     ↓
9. Detector combination
     ↓
10. Evaluation
     ↓
11. Explanation generation
     ↓
12. MITRE ATT&CK mapping
     ↓
13. Risk scoring
     ↓
14. Streamlit dashboard
```

---

# 🧪 Model Training

For each entity with sufficient benign traffic:

```text
Benign Entity Traffic
        │
        ▼
   Feature Matrix
        │
        ▼
   StandardScaler
        │
        ▼
  Isolation Forest
        │
        ▼
Validation Scores
        │
        ▼
Entity-Specific
Anomaly Threshold
        │
        ▼
Saved Model Bundle
```

Each saved model contains:

```python
{
    "model": model,
    "scaler": scaler,
    "threshold": threshold,
    "feature_cols": feature_cols,
    "entity_ip": entity
}
```

For GeNIS, the final training process produced models for **80 of 116 entities** because 36 entities did not have the minimum required benign traffic.

---

# 📈 Results

## GeNIS Combined System

The final combined system uses:

```text
Isolation Forest
       +
Scan Detector
       ↓
Combined Detection
```

Network-wide GeNIS results:

| Metric              |    Result |
| ------------------- | --------: |
| Precision           | **0.973** |
| Recall              | **0.471** |
| F1-score            | **0.635** |
| False Positive Rate | **0.066** |

For Entity 1:

| Metric              |    Result |
| ------------------- | --------: |
| Precision           | **0.990** |
| Recall              | **0.537** |
| F1-score            | **0.696** |
| False Positive Rate | **0.054** |

These results are reported directly from the final evaluation described in the project report.

---

# 🔎 Reconnaissance Detection

The scan detector was specifically designed to address dense reconnaissance traffic that Isolation Forest struggled to identify.

For GeNIS:

```text
Network-wide reconnaissance recall:
14.0%

Coverage-restricted recall:
98.7%
```

The difference is primarily caused by entity coverage: only **7 of the 22 reconnaissance-generating entities** had sufficient clean benign data to produce a calibrated scan threshold.

This highlights an important practical limitation of per-entity behavioral detection:

> The system needs sufficient benign history for an entity before it can reliably establish its behavioral baseline.

---

# 🆚 CICIDS2017 vs GeNIS

| Metric                               | CICIDS2017 |  GeNIS |
| ------------------------------------ | ---------: | -----: |
| Dataset year                         |       2017 |   2025 |
| Entities                             |         15 |    116 |
| Isolation Forest coverage            |      15/15 | 80/116 |
| LOF coverage                         |      15/15 | 48/116 |
| DoS-family F1                        |      0.913 |  0.635 |
| Scan/recon recall — full population  |      99.2% |  14.0% |
| Scan/recon recall — covered entities |      99.2% |  98.7% |
| Network-wide precision               |          — |  0.973 |

The comparison shows that the core architecture transferred to GeNIS, while entity coverage became a major limiting factor because of the newer dataset's attack-heavy, randomly split structure.

---

# 📊 Dashboard

AttackDNA includes an interactive **Streamlit dashboard** designed for security analysis.

The dashboard provides:

### Overview

* Detection metrics
* Entity-level information
* Risk scores
* Alert statistics

### Model Comparison

Compares:

* Isolation Forest
* Combined Isolation Forest + scan detector
* LOF

### Alerts

Provides a filterable view of detected anomalies.

### MITRE ATT&CK

Displays the distribution of mapped techniques.

### Diagnostics

Provides information about:

* False-positive behavior
* Entity coverage
* Scan detector coverage

### Migration Notes

Documents the changes required to adapt AttackDNA from CICIDS2017 to GeNIS.

The GeNIS dashboard was rebuilt specifically to include model comparison, entity coverage, diagnostics, and migration information.

---

# 🛠️ Technology Stack

| Technology           | Purpose                  |
| -------------------- | ------------------------ |
| Python               | Core development         |
| Pandas               | Data processing          |
| NumPy                | Numerical operations     |
| Scikit-learn         | Machine learning         |
| Isolation Forest     | Main anomaly detector    |
| Local Outlier Factor | Model comparison         |
| Streamlit            | Interactive dashboard    |
| Plotly               | Dashboard visualizations |
| Pickle               | Model serialization      |
| MITRE ATT&CK         | Threat technique mapping |

The project was designed to operate using CPU hardware and free/open-source tools without requiring always-on infrastructure.

---

# 🚀 Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/Amberr8/attackdna.git
cd attackdna
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure the repository's `requirements.txt` matches the dependencies used by the current source code.

---

# 🔄 Dataset Configuration

Dataset selection is controlled through:

```text
src/config.py
```

The project supports:

```python
DATASET = "genis"
```

or:

```python
DATASET = "cicids2017"
```

The configuration contains dataset-specific information such as:

* Source host column
* Destination host column
* Label column
* Benign label
* Timestamp/time field
* Destination port
* Entity configuration

This allows the same pipeline to operate across both datasets.

---

# 🧹 Data Preprocessing

Example:

```bash
python src/data_prep.py input.csv output.csv
```

The preprocessing stage performs:

* Column normalization
* Label normalization
* Duplicate removal
* Entity extraction
* Infinite/NaN handling
* Boolean-to-integer conversion where required

GeNIS specifically required Boolean feature columns to be converted to integers so that the feature list used during training remained consistent with the features used during detection.

---

# 🔬 Feature Selection

```bash
python src/feature_selection.py processed.csv
```

Feature selection removes:

* Identity fields
* Labels
* Ports
* Capture-order fields
* Near-zero-variance features
* Highly correlated features

For GeNIS, the original CICIDS2017 feature-drop list was not reused. The feature-selection process was rebuilt against the GeNIS schema.

---

# 🤖 Train Models

```bash
python src/train.py
```

The training script:

1. Loads benign baseline traffic.
2. Groups traffic by entity.
3. Selects numerical features.
4. Splits training and validation data.
5. Scales the features.
6. Trains an Isolation Forest for each entity.
7. Calibrates an entity-specific threshold.
8. Saves the model bundle.

---

# 🔍 Detect Anomalies

```bash
python src/detect.py input.csv output.csv
```

The detector loads the appropriate entity model and generates:

```text
predicted
anomaly_score
```

where:

```text
1  = normal
-1 = anomaly
```

---

# 📡 Scan Detection

Generate burst features first:

```bash
python src/time_window_features.py input.csv output.csv
```

Then run the scan detector:

```bash
python src/scan_detector.py baseline.csv,test.csv,output.csv
```

The scan detector uses entity-specific behavioral thresholds for:

```text
connections_per_minute
distinct_ports_per_minute
```

---

# 🔗 Combine Detectors

```bash
python src/combine_detectors.py \
    isolation_forest_results.csv \
    scan_results.csv \
    combined_results.csv
```

A flow is considered anomalous if either:

```text
Isolation Forest → anomaly
```

or:

```text
Scan Detector → scan
```

---

# 📏 Evaluation

```bash
python src/evaluate.py scored_results.csv
```

The evaluation pipeline calculates:

* True Positives
* False Positives
* True Negatives
* False Negatives
* Precision
* Recall
* F1-score
* False Positive Rate

---

# 🧠 Explanations and MITRE Mapping

The explanation stage analyzes the strongest statistical deviations from the entity's baseline.

The resulting anomalies can then be mapped to MITRE ATT&CK techniques.

The project therefore produces a more analyst-oriented output:

```text
Anomaly
   ↓
Why was it anomalous?
   ↓
Which behavior changed?
   ↓
Which MITRE technique is relevant?
   ↓
What is the resulting risk?
```

---

# 🖥️ Launch Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard loads the processed detection results and provides interactive analysis of:

* Entity behavior
* Alerts
* Risk scores
* Detection metrics
* MITRE techniques
* Model comparison
* Entity coverage
* Diagnostics

---

# ⚠️ Limitations

AttackDNA has several documented limitations.

### 1. Entity Coverage

The most important limitation on GeNIS is the availability of sufficient benign traffic for individual entities.

Only:

```text
80 / 116
```

entities had enough benign traffic for Isolation Forest training.

---

### 2. No Real-Time Detection

The current system operates on prepared flow datasets rather than a live network stream.

Real-time deployment remains future work.

---

### 3. GeNIS Has No Real Timestamp

GeNIS provides `Offset`, which was confirmed to represent capture position rather than elapsed time.

Therefore, the project's burst detector uses fixed `Offset` ranges rather than literal real-time minute windows.

This is explicitly treated as a documented approximation rather than a real timestamp.

---

### 4. Bot/Beaconing Detection

Volume-based burst features are not effective against every attack pattern.

In particular, beaconing traffic can avoid producing obvious connection or destination-port bursts.

The report therefore observed weak Bot detection even after introducing time-window features.

---

### 5. Dataset Dependency

Detection performance depends strongly on:

* Dataset structure
* Amount of benign history
* Entity coverage
* Feature availability
* Attack distribution

The GeNIS evaluation demonstrates that the availability of clean per-entity behavioral history can become a major practical constraint.

---

# 🔮 Future Work

Potential future extensions include:

* Real-time network traffic ingestion
* Continuous behavioral baseline updates
* Improved entity coverage
* Better detection of low-and-slow attacks
* Improved bot/beaconing detection
* Additional anomaly detection algorithms
* More advanced explainable ML techniques
* Live threat-intelligence integration
* Production SOC integration
* Automated alert response

---

# 📚 Research Contribution

AttackDNA demonstrates that an organization-specific behavioral detection approach can be built using:

```text
Per-Entity Modeling
        +
Unsupervised ML
        +
Behavioral Burst Detection
        +
Explainability
        +
MITRE ATT&CK Mapping
        +
Risk Scoring
        +
Interactive Visualization
```

The project also provides a cross-dataset evaluation showing that the core architecture can be adapted from CICIDS2017 to a substantially different 2025 dataset without redesigning the complete system.

One of the key findings is that **having enough clean per-entity behavioral history is critical to practical deployment**.

---

# 👩‍💻 Author

**Amber Waseem**

AttackDNA — Organization-Specific Behavioral Detection Engine

GitHub:

https://github.com/Amberr8/attackdna

---

# 📄 Project Report

The complete technical report documents:

* Original CICIDS2017 implementation
* GeNIS migration
* Dataset analysis
* Data preprocessing
* Feature engineering
* Isolation Forest
* LOF comparison
* Scan detection
* Explainability
* MITRE mapping
* Risk scoring
* Dashboard
* Cross-dataset evaluation
* Limitations and future work

---

## ⭐ Acknowledgement

This project was developed as a cybersecurity and machine-learning research project focused on unsupervised, organization-specific behavioral anomaly detection.

**AttackDNA  Learn the normal. Detect the abnormal.**
