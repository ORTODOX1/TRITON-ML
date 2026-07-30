# TRITON-ML -- Predictive Maintenance for Ship Machinery

[![Research Preview](https://img.shields.io/badge/status-research%20preview-orange)](https://github.com/hermandoronin/TRITON-ML)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006600.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **Scope, up front.** This is a small research preview written by a marine engineer
> learning applied ML -- not a product. What exists today is a feature extraction
> library, three model wrappers, an alert engine and an ONNX exporter, all covered by
> unit tests. There is **no bundled dataset, no trained model weights, no data
> ingestion layer, and no end-to-end training pipeline.** The
> [Not implemented yet](#not-implemented-yet) section lists exactly what is missing.

---

## Problem Statement

Unplanned failures of marine machinery are expensive and unsafe. Time-based
maintenance -- replacing components at fixed intervals regardless of their actual
condition -- gets it wrong in both directions: healthy parts are discarded early,
while parts degrading faster than the schedule assumes fail in service, often far
from a port.

Condition-based maintenance attacks both problems by deciding from sensor data
rather than from a calendar. TRITON-ML is my attempt to build that decision layer
for shipboard machinery: turn raw vibration, temperature and performance telemetry
into interpretable condition indicators, feed them to models that flag a fault mode
and estimate remaining useful life, and surface the result as a graded alert an
engineer can check against what the machine is actually doing.

I have deliberately left savings percentages and cost-per-day figures out of this
README. Numbers like that circulate widely without traceable sourcing, and this
project has measured none of them.

---

## What Is Implemented

```
                    +-------------------------------------+
   raw arrays  -->  |          FeaturePipeline            |  --> 15 features
   (NumPy)          |  vibration | thermal | operational  |      (flat dict)
                    +-------------------------------------+
                                     |
                 +-------------------+-------------------+
                 v                   v                   v
        +----------------+  +----------------+  +------------------+
        | FaultClassifier|  |  RULEstimator  |  | AnomalyDetector  |
        |    XGBoost     |  |  PyTorch MLP   |  | Isolation Forest |
        |  + SHAP values |  |  + MC-dropout  |  |                  |
        +----------------+  +----------------+  +------------------+
                 |                   |                   |
                 +-------------------+-------------------+
                                     v
                        +----------------------+     +------------------+
                        |     AlertEngine      |     |   ONNXExporter   |
                        | NORMAL/WATCH/ALARM/  |     | RUL net -> .onnx |
                        |       SHUTDOWN       |     +------------------+
                        |  -> WebSocket (JSON) |
                        +----------------------+
```

Inputs are plain NumPy arrays supplied by the caller. Getting sensor data into those
arrays is out of scope for the current code.

### 1. Feature Engineering

`FeaturePipeline.run()` produces **15 features**, five per domain, as a flat dict
keyed `<domain>__<feature>`:

| Domain | Features |
|---|---|
| `vib__` | `rms`, `kurtosis` (excess), `crest_factor`, `dominant_freq_hz`, `envelope_peak` |
| `therm__` | `mean_temp_c`, `max_delta_t`, `gradient_c_per_min`, `trend_slope`, `cylinder_spread` |
| `ops__` | `load_factor`, `sfoc_g_per_kwh`, `power_output_kw`, `scavenge_pressure_bar`, `torque_deviation_pct` |

How they are computed:

- **Vibration** (`features/vibration.py`) -- RMS, excess kurtosis and crest factor
  from the time series; dominant frequency from a Welch power spectral density;
  envelope peak from the magnitude of the Hilbert transform. The full PSD array is
  also returned on the dataclass as `spectral_energy`, but the pipeline skips array
  fields, so it is not one of the 15 scalar features.
- **Thermal** (`features/thermal.py`) -- mean temperature, maximum and mean
  inter-sensor spread per timestep, mean temporal gradient in degC/min, and an
  ordinary-least-squares trend slope over the window.
- **Operational** (`features/operational.py`) -- shaft power from `P = 2*pi*n*T/60`,
  load factor against a configurable MCR, SFOC in g/kWh, mean scavenge air pressure,
  and torque deviation from a 10-sample running mean.

### 2. Models

**`FaultClassifier` -- XGBoost, single-label multiclass**

Six mutually exclusive classes: `NORMAL`, `BEARING_WEAR`, `MISALIGNMENT`,
`IMBALANCE`, `FOULING`, `INJECTOR_FAULT` (`objective="multi:softprob"`, 400 trees,
depth 6, learning rate 0.05, `tree_method="hist"`). `train()` also fits a SHAP
`TreeExplainer`, so `explain()` returns per-feature attributions for any batch.
Class imbalance is **not** handled -- there is no resampling and no class weighting.

**`RULEstimator` -- PyTorch MLP with MC-dropout**

Fully connected network `input -> 128 -> 64 -> 1`, with ReLU and dropout (p = 0.20)
after the first two layers, trained with Adam and plain `nn.MSELoss`. There is no
batch normalisation and no asymmetric loss.

`predict()` keeps dropout active and runs 50 stochastic forward passes, returning
`(mean, std)` across them. That standard deviation is **epistemic** (model)
uncertainty -- it measures disagreement between dropout sub-networks. It is not
aleatoric uncertainty, which would require the network to predict its own noise
variance.

Input is one flat feature vector per sample, not a rolling time window.

**`AnomalyDetector` -- Isolation Forest**

scikit-learn `IsolationForest` (200 estimators, `contamination=0.02`,
`random_state=42`) for behaviour the supervised classifier has never seen. `score()`
returns `score_samples()` output; `is_anomalous()` flags anything below the
configured limit (default `-0.35`).

### 3. Explainability

SHAP is wired up for the **fault classifier only**, via `TreeExplainer`. Calling
`explain()` before `train()` raises. The RUL network has no SHAP integration, and
there is no automatic audit-log persistence -- explanations are returned to the
caller, not written anywhere.

### 4. Alerting

`AlertEngine.evaluate()` maps a remaining-useful-life estimate onto four tiers:

| Level | Trigger (default thresholds) | Intended action |
|---|---|---|
| **NORMAL** | RUL > 720 h | Continue monitoring |
| **WATCH** | RUL <= 720 h (30 days) | Schedule inspection at next port |
| **ALARM** | RUL <= 168 h (7 days) | Start maintenance planning |
| **SHUTDOWN** | RUL <= 24 h | Recommend load reduction or controlled shutdown |

An anomaly score below `anomaly_score_limit` escalates `NORMAL`/`WATCH` to `ALARM`.
These thresholds are project defaults, configurable via `AlertThresholds`; they are
not taken from any standard. `send()` serialises the alert to JSON and pushes it
over a WebSocket connection (default `ws://aegis-monitor:8400/ws/alerts`, matching
the companion [AEGIS-MONITOR](https://github.com/hermandoronin/AEGIS-MONITOR)
dashboard); dispatch failures are logged, not raised.

### 5. ONNX Export

`ONNXExporter.export_rul()` converts the RUL network with `torch.onnx.export` at
opset 17 with a dynamic batch axis, then validates the graph using `onnx.checker`.
`verify()` runs the exported graph through ONNX Runtime so the export can be checked
against the source model. XGBoost models are saved as JSON through XGBoost's own
serialiser, not converted to ONNX, and there is no quantisation step.

---

## Tech Stack

Everything below is an actual dependency in `pyproject.toml` and is imported by the
code:

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch 2.1+ |
| Gradient boosting | XGBoost 2.0+ |
| Anomaly detection | scikit-learn 1.3+ (Isolation Forest) |
| Explainability | SHAP 0.43+ |
| Numerics | NumPy 1.24+ |
| Signal processing | SciPy 1.11+ (Welch PSD, Hilbert transform) |
| Alert transport | websockets 12+ |
| Configuration | PyYAML 6+ |
| Edge export | onnx 1.15+, onnxruntime 1.17+ |
| Dev tooling | pytest, ruff, mypy |

No pandas, Matplotlib, Plotly, MLflow or Hydra -- the pipeline works on NumPy arrays
and dataclasses, and configuration is a single YAML file.

---

## Project Structure

```
TRITON-ML/
├── triton_ml/
│   ├── __init__.py              # Package exports: Settings, AlertEngine, ONNXExporter
│   ├── __main__.py              # CLI entry point (argument / config validation)
│   ├── config.py                # Frozen dataclass settings + Settings.from_yaml()
│   ├── alerting.py              # AlertEngine, Alert, Severity + WebSocket dispatch
│   ├── export.py                # ONNXExporter
│   ├── features/
│   │   ├── __init__.py          # FeaturePipeline (merges the three extractors)
│   │   ├── vibration.py         # VibrationFeatureExtractor
│   │   ├── thermal.py           # ThermalFeatureExtractor
│   │   └── operational.py       # OperationalFeatureExtractor
│   └── models/
│       ├── __init__.py
│       ├── fault_classifier.py  # XGBoost + SHAP
│       ├── rul_estimator.py     # PyTorch MLP + MC-dropout
│       └── anomaly.py           # Isolation Forest
├── tests/                       # pytest suite
├── .github/workflows/ci.yml     # ruff + mypy + pytest on Python 3.11
├── config.example.yaml          # Example config for Settings.from_yaml()
├── docker-compose.yml           # TimescaleDB + Grafana for local development
├── .env.example                 # Credentials template for docker-compose
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hermandoronin/TRITON-ML.git
cd TRITON-ML

# Create environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -e ".[dev]"

# Run the test suite
pytest tests/ -v
```

The CLI exposes three subcommands:

```bash
python -m triton_ml train   --config config.yaml
python -m triton_ml predict --model model.onnx --input data.csv
python -m triton_ml export  --model model.pt   --output model.onnx
```

**These are not a training pipeline yet.** `train` loads and validates the YAML
config through `Settings.from_yaml()` and logs the resulting parameters; `predict`
and `export` check that their file arguments exist. All three then log a warning and
stop -- the orchestration is not written (see `triton_ml/__main__.py`). Use the
library API directly in the meantime:

```python
import numpy as np
from triton_ml.features import FeaturePipeline

rng = np.random.default_rng(0)
features = FeaturePipeline().run(
    vib_raw=rng.standard_normal(4096),             # accelerometer window
    thermal_readings=rng.normal(350, 5, (60, 6)),  # (timesteps, sensors), degC
    rpm=rng.normal(95, 1, 60),
    fuel_flow=rng.normal(1800, 20, 60),            # kg/h
    torque=rng.normal(900_000, 5_000, 60),         # Nm
    scav_pressure=rng.normal(2.6, 0.05, 60),       # bar
)
print(len(features))   # 15
```

---

## Configuration

`Settings.from_yaml()` reads a single YAML file. Recognised sections are `paths`,
`model`, `alerts` and `onnx`; anything you omit falls back to the dataclass defaults
in `triton_ml/config.py`. See [`config.example.yaml`](config.example.yaml) for the
full layout.

```python
from triton_ml.config import Settings

settings = Settings.from_yaml("config.yaml")
print(settings.alerts.shutdown_rul_hours)   # 24.0
```

For local development, `docker-compose.yml` brings up TimescaleDB and Grafana. Copy
`.env.example` to `.env` and set credentials first -- compose fails fast if they are
unset. TRITON-ML itself is not containerised; there is no Dockerfile in this
repository.

---

## Testing

```bash
pytest tests/ -v
```

29 tests covering vibration feature extraction (dominant frequency, RMS, kurtosis,
envelope), fault classifier training / prediction / SHAP, alert severity
transitions, and YAML config loading. CI additionally runs `ruff check` and `mypy`
on Python 3.11.

No coverage tooling is configured and there are no integration tests -- the suite is
unit-level only and runs on synthetic signals, not recorded machinery data.

---

## Not Implemented Yet

Listed here because earlier versions of this README described these as if they
already existed. They do not. Roughly in the order I would like to build them:

**Data ingestion** -- there is no ingestion module at all. NMEA 2000 / SAE J1939 CAN
decoding, Modbus RTU polling, AIS context, and CSV/Parquet readers with timestamp
alignment across differing sample rates are all planned, none written. Today the
caller passes NumPy arrays in directly.

**Richer vibration analysis** -- bearing characteristic defect frequencies
(BPFO / BPFI / BSF / FTF) and order tracking against shaft speed; spectral kurtosis
for demodulation band selection; RMS velocity in the bands used by vibration
severity standards. The current extractor is broadband only.

**Pressure analysis** -- lube oil pressure versus RPM characteristic curves,
injection pressure profiles, compression peak pressure and rate of pressure rise. No
pressure-domain extractor exists; scavenge air pressure is the only pressure signal
used, and only as a mean.

**Multi-window trending** -- 1 h / 6 h / 24 h / 7 d trend slopes and cumulative
running hours since last overhaul. Extractors currently see one window at a time and
hold no state.

**Model work** -- class-imbalance handling (resampling or class weights); an
asymmetric RUL loss penalising over-prediction of remaining life more heavily than
under-prediction; rolling-window inputs; SHAP for the RUL network.

**Deployment** -- a Dockerfile and inference container, INT8 quantisation, ONNX
conversion of the XGBoost model, and latency benchmarks on real edge hardware.
Nothing here has been measured, so this README quotes no image size or latency
figures.

---

## On Classification Societies

Class societies expect automated decision-support tools to be transparent and
auditable, which is why SHAP attribution and explicit uncertainty output are in the
design from the start rather than bolted on afterwards. That is a design principle I
am working toward -- not a compliance claim. TRITON-ML has not been submitted to,
reviewed by, or approved by any classification society, and I cite no specific class
rules here because I have not verified this code against them.

The one property the code does enforce is advisory-only behaviour: it emits alerts
and explanations and takes no control action.

---

## License

MIT. See [LICENSE](LICENSE).

---

## About the Author

Marine engineer with hands-on maintenance experience on seagoing vessels -- from
overhauling main engine cylinder units to troubleshooting turbocharger bearing
failures at 2 AM in the Indian Ocean. Now learning to apply machine learning to the
domain I know from inside the engine room.

TRITON-ML exists because I have seen what unplanned breakdowns cost: not just money,
but safety margins, crew rest hours, and operational reliability. This repository is
where I work out how much of that intuition can be expressed as code -- kept honest
by writing down what is missing alongside what is finished.

---

*TRITON-ML is a research preview. It is not a certified safety system and must not
be used as the basis for maintenance decisions on safety-critical equipment.*
