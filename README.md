# Volatility Models: Pricing & Calibration Framework

A modular Python framework for pricing and calibrating option pricing models,
with an initial focus on stochastic volatility models.

The current implementation includes a full Heston (1993) pipeline,
serving as the reference model for the framework design.

The project is designed for robust calibration, numerical stability, and practical use in equity derivatives research.

---

### Features
- Modular pricing and calibration framework
- Vectorized pricing engines
- Robust calibration pipelines (DE + L-BFGS-B)
- Diagnostic plots and calibration reports

---

### Current model support
- Heston (1993)
  - Gauss–Laguerre pricer
  - Sinh-acceleration pricer
  - Full implied-volatility surface calibration

---

### Roadmap
- Rough Heston
- SABR / Dynamic SABR
- Local volatility (Dupire)
- Stochastic local volatility (SLV)

---

## Repository Structure

```text
volcal/
├── pyproject.toml
├── README.md
├── data/
│   ├── spx/
│   |   └── SPX_17_10_25.xlsx
|   └── googl/
|       └── GOOGL_16_12_25.xlsx
├── examples/
└── src/
    ├── heston/
    |   ├── __init__.py
    |   ├── calibrator/
    |   │   ├── __init__.py
    |   │   └── main.py
    |   ├── pricer/
    |   │   ├── __init__.py
    |   │   ├── laguerre.py
    |   │   └── sinh.py
    ├── utils/
    |   └── black_scholes.py
    └── market_data/
        ├── __init__.py
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

Heston, S. L. (1993).
A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.
The Review of Financial Studies, 6(2), 327–343.
https://doi.org/10.1093/rfs/6.2.327

Gatheral, J. (2006).
The Volatility Surface: A Practitioner’s Guide.
Wiley Finance Series.
https://doi.org/10.1002/9781119202073

Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).
The Little Heston Trap.
Wilmott Magazine, January, 83–92.

Ortiz Ramírez, A., Venegas Martínez, F., & Martínez Palacios, M. T. V. (2021).
Parameter calibration of stochastic volatility Heston’s model: constrained optimization vs. differential evolution.
Accounting and Management, 67(1), 309.
https://doi.org/10.22201/fca.24488410e.2022.2789


Boyarchenko, S. & Levendorskii, S. (2019).
Sinh-acceleration: Efficient Evaluation of Probability Distributions, Option Pricing, and Monte Carlo Simulations.
International Journal of Theoretical and Applied Finance, 03(22), 1950011.
https://doi.org/10.1142/S0219024919500110
