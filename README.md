# TRITON-ML -- Predictive Maintenance for Ship Machinery

[![Research Preview](https://img.shields.io/badge/status-research%20preview-orange)](https://github.com/TRITON-ML)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006600.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## Problem Statement

Unplanned downtime of marine diesel engines costs vessel operators between **$50,000 and $500,000 per day**, depending on vessel class and charter rates. Traditional time-based maintenance schedules -- replacing components at fixed intervals regardless of actual condition -- lead to two costly outcomes: premature replacement of healthy parts, and unexpected failures of components that deteriorate faster than the schedule assumes.

Condition-based predictive maintenance driven by machine learning addresses both failure modes. By continuously monitoring sensor data from shipboard machinery, TRITON-ML detects early-stage degradation patterns and estimates remaining useful life, enabling maintenance to be scheduled precisely when needed. Industry data shows that ML-based condition monitoring **reduces maintenance costs by 30--50%** and cuts unplanned downtime by up to 70%.

TRITON-ML is an end-to-end pipeline purpose-built for the maritime domain: from raw sensor ingestion through feature engineering, fault classification, remaining useful life estimation, and explainable alerting -- all designed to meet classification society expectations for transparent, auditable decision support.

---

## Architecture

```
+---------------------+       +----------------------+       +---------------------+
|   DATA INGESTION    |       |  FEATURE ENGINEERING |       |    MODEL LAYER      |
|                     |       |                      |       |                     |
|  NMEA 2000 / J1939  +------>+  Vibration: FFT,     +------>+  XGBoost Classifier |
|  Modbus RTU (SCADA)  |       |   RMS, Kurtosis,    |       |   (Fault Detection) |
|  CSV / Parquet logs  |       |   Crest Factor,     |       |                     |
|  AIS context data    |       |   Envelope Analysis  |       |  DNN Regressor      |
|                     |       |                      |       |   (RUL Estimation)  |
+---------------------+       |  Temperature: Trend  |       |                     |
                              |   Slopes, Delta-T,   |       |  Isolation Forest   |
                              |   Thermal Gradients  |       |   (Anomaly Detect.) |
                              |                      |       |                     |
                              |  Pressure: Oil Press  |       +----------+----------+
                              |   vs RPM, Injection  |                  |
                              |   Pressure Profiles  |                  v
                              |                      |       +---------------------+
                              |  Operational: Load   |       |   EXPLAINABILITY    |
                              |   Factor, SFOC,      |       |                     |
                              |   Power Output       |       |  SHAP Values for    |
                              +----------------------+       |  Every Prediction   |
                                                             |                     |
                                                             +----------+----------+
                                                                        |
                                                                        v
                                                             +---------------------+
                                                             |     ALERTING        |
                                                             |                     |
                                                             |  NORMAL --> WATCH   |
                                                             |  WATCH  --> ALARM   |
                                                             |  ALARM  --> SHUTDOWN|
                                                             +---------------------+
```

---

## Core Pipeline

### 1. Data Ingestion

TRITON-ML consumes sensor streams from multiple shipboard sources in real time and batch modes:

| Source | Protocol | Data |
|---|---|---|
| Engine control systems | NMEA 2000 / J1939-76 (CAN bus) | RPM, exhaust temps, turbo speed, load |
| PLC / SCADA | Modbus RTU | Pressures, flow rates, tank levels |
| Historical logs | CSV / Parquet | Maintenance records, voyage data |
| AIS transponder | AIS (NMEA 0183) | Speed over ground, heading, operational context |

The ingestion layer handles timestamp alignment across sources with different sampling rates (vibration at 25.6 kHz, process parameters at 1 Hz, AIS at 0.1 Hz) and manages data quality checks including gap detection, outlier flagging, and sensor drift compensation.

### 2. Feature Engineering

**72+ engineered features** are extracted from raw sensor data, organized by physical domain:

**Vibration Analysis**
- FFT spectral decomposition with characteristic frequency tracking (bearing BPFO/BPFI/BSF/FTF)
- RMS velocity and acceleration (ISO 10816 / ISO 20816 bands)
- Kurtosis and crest factor for impulse detection
- Envelope analysis (amplitude demodulation) for early bearing fault identification
- Spectral kurtosis for optimal band selection

**Temperature Monitoring**
- Linear trend slopes over configurable windows (1h, 6h, 24h, 7d)
- Delta-T across component pairs (e.g., cooling water inlet vs. outlet)
- Thermal gradient rates for detecting fouling and blockage
- Exhaust gas temperature spread across cylinders

**Pressure Analysis**
- Lube oil pressure vs. RPM characteristic curves with deviation scoring
- Fuel injection pressure profiles and timing drift
- Compression pressure trends (peak pressure and rate of pressure rise)
- Scavenge air pressure relative to turbocharger speed

**Operational Context**
- Load factor (percentage of MCR)
- Specific fuel oil consumption (SFOC) trend and deviation from baseline
- Power output vs. hull resistance (early fouling detection)
- Cumulative running hours since last overhaul per component

### 3. Models

**XGBoost Classifier -- Fault Detection**

Multi-label classification for identifying specific degradation modes:

- Bearing wear (main, crankpin, crosshead)
- Cylinder liner scoring and scuffing
- Turbocharger surge and bearing deterioration
- Fuel injector failure (atomization degradation, needle seat wear)
- Exhaust valve recession and blow-by
- Cooling water system fouling

Trained on labeled maintenance records mapped to pre-failure sensor windows. Class imbalance handled via SMOTE and cost-sensitive learning, since failure events are rare relative to normal operation.

**DNN Regressor -- Remaining Useful Life (RUL)**

A deep neural network that estimates hours remaining before a component reaches its maintenance threshold:

- Architecture: fully connected network with batch normalization and dropout
- Input: rolling feature windows (configurable 24h--168h lookback)
- Output: point estimate + prediction interval (aleatoric uncertainty via MC Dropout)
- Loss: asymmetric loss function penalizing over-prediction (predicting more remaining life than actual) more heavily than under-prediction

**Isolation Forest -- Anomaly Detection**

Unsupervised detection of novel failure modes not present in training data:

- Operates on the full feature space to catch unexpected multivariate deviations
- Contamination parameter tuned per equipment class
- Flags novel patterns for engineering review and potential model retraining

### 4. Explainability

Every prediction is accompanied by SHAP (SHapley Additive exPlanations) values. This is not optional -- it is a core design requirement.

- **Fault detection**: SHAP waterfall plots show which sensor features drove the classification
- **RUL estimation**: SHAP dependence plots reveal which operational conditions are accelerating degradation
- **Audit trail**: all SHAP outputs are logged with timestamps for post-incident review

Classification societies increasingly require that automated decision-support systems provide transparent reasoning. SHAP values directly satisfy this requirement by attributing each prediction to measurable physical quantities that engineers can verify against domain knowledge.

### 5. Alerting

Four-tier severity system aligned with standard engine room alarm conventions:

| Level | Condition | Action |
|---|---|---|
| **NORMAL** | All parameters within baseline | Continue monitoring |
| **WATCH** | Early degradation detected; RUL > maintenance planning horizon | Schedule inspection at next port |
| **ALARM** | Significant degradation; RUL approaching maintenance threshold | Initiate maintenance planning, increase monitoring frequency |
| **SHUTDOWN** | Critical fault probability exceeds safety threshold | Recommend immediate load reduction or controlled shutdown |

Thresholds are configurable per equipment class and can be tuned based on operator risk tolerance and operational constraints (e.g., mid-ocean vs. coastal sailing).

---

## Supported Equipment

| Equipment | Typical Sensors | Key Failure Modes |
|---|---|---|
| Main engines (2-stroke low-speed) | Cylinder pressure, exhaust temps, bearing temps, scavenge air | Liner wear, bearing damage, exhaust valve failure |
| Main engines (4-stroke medium-speed) | Vibration, cylinder pressure, lube oil analysis | Piston ring wear, injector degradation, turbo damage |
| Auxiliary engines and generators | Vibration, temperature, load, frequency | Bearing wear, governor faults, alternator issues |
| Turbochargers | Speed, vibration, inlet/outlet temps, pressure ratio | Surge, fouling, bearing deterioration |
| Pumps (CW, LO, FO) | Vibration, discharge pressure, flow rate, motor current | Impeller erosion, seal leakage, cavitation |
| Compressors | Vibration, pressure stages, temperature, valve condition | Valve failure, ring wear, intercooler fouling |
| Heat exchangers | Inlet/outlet temps, pressure drop, flow rates | Fouling, tube leakage, gasket deterioration |

---

## Data Sources

### Real-Time

- **NMEA 2000 / SAE J1939-76**: Standard marine CAN bus protocol. TRITON-ML reads PGNs for engine parameters (RPM, temperatures, pressures, fuel rate) via a SocketCAN or USB-CAN gateway.
- **Modbus RTU**: Serial communication with PLCs, SCADA systems, and standalone sensor transmitters. Configurable register maps per installation.

### Historical / Batch

- **CSV and Parquet files**: Import historical operational logs, maintenance records, and condition reports. Parquet preferred for large datasets due to columnar compression.
- **AIS data**: Automatic Identification System records provide operational context -- vessel speed, heading, and draft -- which correlates with engine load and environmental conditions.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch 2.x |
| Gradient boosting | XGBoost 2.x |
| ML utilities | scikit-learn 1.x |
| Explainability | SHAP 0.43+ |
| Data processing | pandas, NumPy |
| Signal processing | SciPy (FFT, filtering, envelope) |
| Visualization | Matplotlib, Plotly |
| Experiment tracking | MLflow |
| Configuration | Hydra / OmegaConf |
| Edge export | ONNX Runtime |
| Testing | pytest |

---

## Project Structure

```
TRITON-ML/
├── configs/                  # Hydra configuration files
│   ├── data/                 # Data source configs per vessel
│   ├── model/                # Model hyperparameters
│   ├── features/             # Feature engineering configs
│   └── alerting/             # Threshold definitions
├── src/
│   └── triton/
│       ├── ingestion/        # Data readers (NMEA, Modbus, CSV/Parquet)
│       ├── features/         # Feature engineering pipeline
│       ├── models/           # XGBoost, DNN, Isolation Forest
│       ├── explain/          # SHAP integration and reporting
│       ├── alerting/         # Severity classification and notifications
│       ├── export/           # ONNX conversion and edge packaging
│       └── utils/            # Logging, metrics, data validation
├── notebooks/                # Exploratory analysis and model development
├── tests/                    # Unit and integration tests
├── data/
│   ├── raw/                  # Raw sensor dumps (not committed)
│   ├── processed/            # Engineered feature sets
│   └── models/               # Trained model artifacts
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/<org>/TRITON-ML.git
cd TRITON-ML

# Create environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e ".[dev]"

# Run feature engineering on sample data
python -m triton.features.pipeline --config configs/features/default.yaml

# Train fault detection model
python -m triton.models.train_xgb --config configs/model/xgboost.yaml

# Train RUL estimator
python -m triton.models.train_dnn --config configs/model/dnn_rul.yaml

# Generate SHAP explanations
python -m triton.explain.report --model-path data/models/xgb_fault.json

# Export to ONNX for edge deployment
python -m triton.export.to_onnx --model-path data/models/dnn_rul.pt
```

---

## Edge Deployment

Shipboard systems typically lack reliable internet connectivity and run on constrained hardware. TRITON-ML supports edge deployment through ONNX export:

1. **Model export**: Trained PyTorch and XGBoost models are converted to ONNX format with full operator coverage verification.
2. **Runtime**: ONNX Runtime provides cross-platform inference on x86 and ARM hardware without requiring PyTorch or XGBoost at runtime.
3. **Quantization**: INT8 quantization available for further size and latency reduction on resource-constrained edge devices.
4. **Inference container**: Minimal Docker image (~200 MB) containing only ONNX Runtime, feature engineering logic, and alerting module.

Target hardware: industrial PCs (e.g., Advantech, Kontron) commonly found in engine control rooms, as well as NVIDIA Jetson modules for installations requiring GPU-accelerated signal processing.

Inference latency target: < 500 ms per equipment unit per inference cycle on CPU-only hardware.

---

## Classification Society Alignment

TRITON-ML is designed with classification society requirements in mind. While the software itself is not class-approved, its architecture supports the documentation and transparency expectations of major societies:

| Society | Relevant Framework | How TRITON-ML Aligns |
|---|---|---|
| **DNV** | DNV-RU-SHIP Pt.4 Ch.3, DNV-CG-0264 (data-driven methods) | SHAP explainability, audit logging, uncertainty quantification |
| **Lloyd's Register** | ShipRight Predictive: Digital Maintenance | Equipment-specific models, condition trend tracking, alert severity levels |
| **Bureau Veritas** | NR 467 (smart ship notation), NR 641 (cyber) | Data provenance tracking, model versioning, input validation |

Key compliance features:
- Full prediction audit trail with timestamps, input features, model version, and SHAP attributions
- Model performance monitoring with drift detection (data and concept drift)
- Human-in-the-loop design: the system recommends, the engineer decides
- No autonomous control actions -- TRITON-ML is advisory only

---

## Configuration

All pipeline parameters are managed through Hydra configuration files. Example for a 4-stroke auxiliary engine:

```yaml
# configs/data/aux_engine_4s.yaml
equipment:
  type: auxiliary_engine
  stroke: 4
  cylinders: 6

sensors:
  vibration:
    sampling_rate: 25600   # Hz
    channels: [de_bearing, nde_bearing, cylinder_head]
  temperature:
    sampling_rate: 1       # Hz
    channels: [exhaust_1, exhaust_2, exhaust_3, exhaust_4, exhaust_5, exhaust_6,
               lo_inlet, lo_outlet, cw_inlet, cw_outlet]
  pressure:
    sampling_rate: 1
    channels: [lo_main_gallery, fo_rail, cw_pump_discharge]

features:
  vibration_window: 1.0    # seconds per FFT window
  trend_windows: [3600, 21600, 86400, 604800]   # 1h, 6h, 24h, 7d in seconds
  rpm_bins: 10             # bins for pressure-vs-RPM curves

alerting:
  thresholds:
    watch:  0.35           # fault probability
    alarm:  0.65
    shutdown: 0.90
  rul_horizon: 720         # hours -- maintenance planning window
```

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=triton --cov-report=html

# Run only model tests
pytest tests/test_models/ -v

# Run integration tests (requires sample data)
pytest tests/integration/ -v --run-integration
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## About the Author

Marine engineer with 3+ years of hands-on maintenance experience on seagoing vessels -- from overhauling main engine cylinder units to troubleshooting turbocharger bearing failures at 2 AM in the Indian Ocean. Now applying machine learning to the domain I know from inside the engine room.

TRITON-ML exists because I have seen firsthand what unplanned breakdowns cost: not just money, but safety margins, crew rest hours, and operational reliability. The gap between what modern ML can detect in sensor data and what traditional watch-keeping catches is enormous. This project aims to close that gap with tools that are transparent, auditable, and built by someone who understands both the machinery and the math.

---

*TRITON-ML is a research preview. It is not a certified safety system and must not be used as the sole basis for maintenance decisions on safety-critical equipment.*
