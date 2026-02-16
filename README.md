# Volatility Models: Pricing & Calibration Framework

A modular Python framework for pricing and calibrating option pricing models,
with an initial focus on stochastic volatility models.

The current implementation includes a full Heston (1993) and SABR (2002) pipelines,
serving as the reference models for the framework design.

The project is designed for robust calibration, numerical stability, and practical use in equity derivatives research.

---

### Features
- Modular pricing and calibration framework
- Vectorized pricing engines
- Robust calibration pipelines (DE + L-BFGS-B)
- Diagnostic plots and calibration reports


### Current model support
- Heston (1993)
  - Gauss–Laguerre pricer
  - Sinh-acceleration pricer
  - Full implied-volatility calibration

- SABR (2002)
  - Hagan approximation pricer
  - Dynamic (per-maturity) SABR calibration with parameter continuity across tenors.


### Roadmap
- Add calibration accuracy and benchmark routines
- Add new models
    - Local volatility (Dupire)
    - Stochastic local volatility (SLV)
    - Rough Heston / Bergomi

---

## Repository Structure

```text
volcal/                                     (repo root)
├── pyproject.toml
├── README.md
├── data/                                   (example market data, not packaged)
│   ├── spx/
│   |   └── SPX_17_10_25.xlsx
|   └── googl/
|       └── GOOGL_16_12_25.xlsx
├── examples/                               (usage examples / notebooks)
└── src/                                    (Python package contents)
    └── volcal/                             (volatility calibration package)
        |
        ├── heston/
        |   ├── calibrator/
        |   │   └── main.py
        |   └── pricer/
        |       ├── laguerre/
        |       |   └── price.py
        |       └── sinh/
        |           └── price.py
        ├── sabr/
        |   ├── calibrator/
        |   └── pricer/
        |       └── hagan/
        |           └── price.py
        ├── utils/                          (shared utilities)
        |   └── black_scholes.py
        └── market_data/                    (data adapters & preprocessing)
            └── preprocessing.py

```

## Installation

From the repository root:

pip install -e .

This installs the package in editable mode and makes `volcal` importable from any location.

---

## Example Usage

from volcal.heston.calibrator import calibrate_heston

params = calibrate_heston(...)

---

## Data

The data/ directory contains examples of implied volatility datasets used for calibration and validation.

In practical workflows, this data is expected to be replaced or loaded dynamically from external sources.

---

## References

Model-specific references are provided in the corresponding module README files.


