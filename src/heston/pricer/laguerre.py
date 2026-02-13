"""
Heston Vanilla Pricer
---------------------

Vectorized vanilla option pricer under the Heston (1993) model using
Gauss-Laguerre quadrature and the P1/P2 formulation.

References
----------
- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options".
  Review of Financial Studies, 6(2), 327-343.
"""

import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.laguerre import laggauss


# ============================ LIGHT OPTIMIZATIONS ============================
@lru_cache(maxsize=None)
def laggauss_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Laguerre nodes/weights (laggauss is deterministic but expensive)."""
    return laggauss(n)


# =============================================================================
def heston_cf(u, s0: float, T: float, r: float, q: float, params: dict, which: int):
    """
    Log-price characteristic function under the Heston model (P1/P2 formulation),
    using common stabilizations (little Heston trap style):
      - flip Re(d) < 0
      - invert g when |g| > 1
      - log1p for log(1 - g e^{dT}) - log(1 - g)
    """
    v0 = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho = params["rho"]

    i = 1j
    u_bar = 0.5 if which == +1 else -0.5
    a = kappa * theta
    b = kappa - (rho * sigma if which == +1 else 0.0)

    u = np.asarray(u, dtype=np.complex128)
    log_s0 = np.log(s0)

    d = np.sqrt((rho * sigma * i * u - b) ** 2 - sigma**2 * (2.0 * u_bar * i * u - u**2))
    d = np.where(np.real(d) < 0, -d, d)

    g_plus = b - rho * sigma * i * u + d
    g_minus = b - rho * sigma * i * u - d
    g = g_plus / g_minus

    unstable = np.abs(g) > 1.0
    if np.any(unstable):
        g = np.where(unstable, 1.0 / g, g)
        d = np.where(unstable, -d, d)

    exp_dT = np.exp(d * T)
    log_term = np.log1p(-g * exp_dT) - np.log1p(-g)

    C = (r - q) * i * u * T + (a / sigma**2) * ((b - rho * sigma * i * u + d) * T - 2.0 * log_term)
    D = ((b - rho * sigma * i * u + d) / sigma**2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    return np.exp(C + D * v0 + i * u * log_s0)


def vanilla_price(
    T: float,
    K: np.ndarray,
    option_params: tuple,  # (s0, r, q)
    heston_params: dict,   # {v0, kappa, theta, sigma, rho}
    option_type=None,      # None -> all calls, else elementwise 'call'/'put'
    N: int = 128,
) -> np.ndarray:
    """
    Vectorized Heston vanilla pricing (Gauss-Laguerre) for multiple strikes at one maturity.
    """
    s0, r, q = option_params
    K = np.asarray(K, dtype=float)

    x, w = laggauss_cached(N)
    u = x.astype(np.complex128)

    const = np.exp(x) / (1j * u)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    cf1 = heston_cf(u, s0, T, r, q, heston_params, which=+1)
    cf2 = heston_cf(u, s0, T, r, q, heston_params, which=-1)

    logK = np.log(K)
    exp_term = np.exp(-1j * np.outer(logK, u))

    int_p1 = np.real(exp_term @ (w * (const * cf1))).astype(float)
    int_p2 = np.real(exp_term @ (w * (const * cf2))).astype(float)

    P1 = 0.5 + int_p1 / np.pi
    P2 = 0.5 + int_p2 / np.pi

    calls = s0 * disc_q * P1 - K * disc_r * P2
    if option_type is None:
        return np.maximum(calls, 0.0)

    puts = K * disc_r * (1.0 - P2) - s0 * disc_q * (1.0 - P1)
    is_call = np.asarray(option_type) == "call"
    return np.maximum(np.where(is_call, calls, puts), 0.0)


# =============================================================================
if __name__ == "__main__":
    heston_params = {"kappa": 2, "theta": 0.0314, "sigma": 1.2, "v0": 0.04125, "rho": -0.73}

    s0 = 1.0
    T = 30.0 / 365.0
    r = 0.03
    q = 0.01
    kmin, kmax = 0.7, 1.3

    fwd = s0 * np.exp((r - q) * T)
    K = np.linspace(kmin * fwd, kmax * fwd, 100)
    option_params = (s0, r, q)

    nruns = 10_000
    n_opts = K.size

    t0 = time.perf_counter()
    for _ in range(nruns):
        prices = vanilla_price(T, K, option_params, heston_params, N=185)
    t1 = time.perf_counter()

    dt = t1 - t0
    mean_per_run = dt / nruns
    mean_per_opt = mean_per_run / n_opts

    print(f"Total time: {dt:.6f} s for {nruns} runs")
    print(f"Mean time per run: {mean_per_run*1e3:.2f} ms")
    print(f"Mean time per option: {mean_per_opt*1e6:.3f} µs/option  (n_opts={n_opts})")

    plt.figure(figsize=(7, 4))
    plt.plot(K, prices, "o-", color="tab:blue", label="Heston (Gauss-Laguerre)")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Heston Model - European Call Prices")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
