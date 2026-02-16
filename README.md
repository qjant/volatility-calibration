# Volatility Models: Pricing & Calibration Framework

This repository currently implements the Heston (1993) stochastic volatility model,
with a modular design intended to support additional volatility models.

Vectorized Heston (1993) option pricer and full calibration pipeline for implied volatility surfaces, implemented as a reusable Python package.

The project is designed for robust calibration, numerical stability, and practical use in equity derivatives research.

---

### Features
- Vectorized Heston (1993) vanilla pricer (Gauss–Laguerre)
- Black–Scholes utilities (price, vega, implied volatility)
- IV surface ingestion from Bloomberg-style tables
- Robust calibration (DE + L-BFGS-B)
- Diagnostic plots and calibration reports

---

### Planned extensions
- Rough Heston
- SABR / Dynamic SABR
- Local volatility (Dupire)
- Stochastic local volatility (SLV)

---

## Repository Structure

```text
heston-model-calibration/
├── pyproject.toml
├── README.md
├── data/
│   └── spx/
│       └── SPX_17_10_25.xlsx
├── examples/
└── src/
    └── heston_model_calibration/
        ├── __init__.py
        ├── calibrator/
        │   ├── __init__.py
        │   └── heston.py
        ├── pricer/
        │   ├── __init__.py
        │   ├── black_scholes.py
        │   ├── laguerre.py
        │   └── sinh.py
        └── market_data/
            ├── __init__.py
            └── preprocessing.py
```

## Installation

From the repository root:

pip install -e .

This installs the package in editable mode and makes `heston_model_calibration` importable from any location.

---

## Example Usage

from heston_model_calibration.pricing.heston import heston_price
from heston_model_calibration.pricing.heston_sinh import heston_price_sinh

Both Gauss–Laguerre and sinh-accelerated pricing methods are available for vanilla options.

---

## Data

The data/spx/ directory contains an example implied volatility dataset (SPX options) used for calibration and validation.

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
