# Volatility Models: Pricing & Calibration Framework

![CI](https://img.shields.io/github/actions/workflow/status/qjant/volatility-calibration/ci.yml?branch=main)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/github/license/qjant/volatility-calibration)
![Last commit](https://img.shields.io/github/last-commit/qjant/volatility-calibration)
![Repo size](https://img.shields.io/github/repo-size/qjant/volatility-calibration)

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
├── data/                                   (example market data)
├───docs/
├── examples/                               (usage examples / notebooks)
└── src/                                    (Python package contents)
    └── volcal/                             (volatility calibration package)
        ├── heston/                         (heston calibrator and pricer module)
        ├── sabr/                           (sabr calibrator and pricer module)
        ├── utils/                          (shared utilities)
        └── market_data/                    (data adapters & preprocessing)

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


