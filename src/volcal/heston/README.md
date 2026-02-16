# Heston model

Implementation of the Heston (1993) stochastic volatility model.

## Pricing engines
- Gauss–Laguerre quadrature
- Sinh-accelerated contour integration

## Calibration
- Global + local optimization on implied-volatility surfaces

## Conventions
- Log-forward formulation
- Continuous dividend yield

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