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

import numpy as np
from numpy.polynomial.laguerre import laggauss
from functools import lru_cache
import matplotlib.pyplot as plt
import cProfile, pstats, io


# ============================ LIGHT OPTIMIZATIONS ============================
@lru_cache(maxsize=None)
def _laggauss_cached(N: int):
    """
    Return cached Gauss-Laguerre nodes and weights.

    Generating them via 'laggauss(n)' is deterministic but expensive,
    so caching avoids recomputation when using the same order repeatedly.

    Parameters
    ----------
    N : int
        Order of the Gauss-Laguerre quadrature (typically 32-256).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Nodes and weights, both of length 'N'.
    """
    return laggauss(N)


# ============================================================================
def heston_cf(u, s0, T, r, q, v0, kappa, theta, sigma, rho, lambd, which):
    """
    Log-price characteristic function under the Heston model.

    Implements the CF for P1/P2 with standard numerical stabilizations (little Heston trap):
    - Flip the sign of 'd' to avoid cancellation errors ("flip_d").
    - Conditionally invert 'g' when |g| > 1 to improve logarithmic stability.
    - Use 'log1p' to compute log(1 - g e^{dT}) - log(1 - g) safely near g ≈ 1.

    Parameters
    ----------
    u : array-like of complex
        Integration points.
    s0 : float
        Current spot price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuously compounded dividend yield.
    v0, kappa, theta, sigma, rho, lambd : float
        Heston parameters and volatility risk premium (λ).
    which : {+1, -1}
        Selects P1 (+1) or P2 (-1) according to the standard convention.

    Returns
    -------
    np.ndarray of complex
        Characteristic function evaluated at each 'u'.
    """
    i = 1j
    u_bar = 0.5 if which == +1 else -0.5
    a = kappa * theta
    b = (kappa + lambd) - (rho * sigma if which == +1 else 0.0)

    u = np.asarray(u, dtype=np.complex128)
    log_s0 = np.log(s0)

    d = np.sqrt((rho * sigma * i * u - b)**2 - sigma**2 * (2.0 * u_bar * i * u - u**2))
    flip_d = np.real(d) < 0
    d = np.where(flip_d, -d, d)

    g_plus = b - rho * sigma * i * u + d
    g_minus = b - rho * sigma * i * u - d
    g = g_plus / g_minus

    # Conditional inversion for numerical stability
    unstable = np.abs(g) > 1.0
    if np.any(unstable):
        g = np.where(unstable, 1.0 / g, g)
        d = np.where(unstable, -d, d)

    exp_dt = np.exp(d * T)
    log_term = np.log1p(-g * exp_dt) - np.log1p(-g)

    C = (r - q) * i * u * T + (a / sigma**2) * ((b - rho * sigma * i * u + d) * T - 2.0 * log_term)
    D = ((b - rho * sigma * i * u + d) / sigma**2) * (1.0 - exp_dt) / (1.0 - g * exp_dt)

    return np.exp(C + D * v0 + i * u * log_s0)


def vanilla_price(
    T: float,
    K: np.ndarray,
    option_params: tuple,   # (s0, r, q)
    heston_params: tuple,   # (v0, kappa, theta, sigma, rho, lambd)
    option_type=None,       # None -> all calls
    N: int = 128
    ) -> np.ndarray:
    """
    Vectorized Heston vanilla option pricing using Gauss-Laguerre quadrature.

    Computes call (and optionally put) prices for multiple K at a single
    maturity, sharing the characteristic function and quadrature grid.

    Parameters
    ----------
    T : float
        Time to maturity (years).
    K : np.ndarray
        1D array of K.
    option_params : tuple
        (s0, r, q) for this maturity.
    heston_params : tuple
        (v0, kappa, theta, sigma, rho, lambd).
    option_type : {'call', 'put'} or array-like, optional
        If None, returns calls. Otherwise mixes calls/puts element-wise.
    N : int
        Gauss-Laguerre order (default 128).

    Returns
    -------
    np.ndarray
        1D array of option prices.
    """
    s0, r, q = option_params
    v0, kappa, theta, sigma, rho, lambd = heston_params

    # Shared quadrature nodes
    x, w = _laggauss_cached(N)
    u = x.astype(np.complex128)
    const = np.exp(x) / (1j * u)

    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    # Characteristic functions (evaluated once)
    cf1 = heston_cf(u, s0, T, r, q, v0, kappa, theta, sigma, rho, lambd, which=+1)
    cf2 = heston_cf(u, s0, T, r, q, v0, kappa, theta, sigma, rho, lambd, which=-1)

    # Outer product for all K
    log_k = np.log(np.asarray(K, dtype=float))
    exp_term = np.exp(-1j * np.outer(log_k, u))

    # Vectorized integrals: Re{ E * (const * cf) } @ w
    b1 = const * cf1
    b2 = const * cf2
    int_p1 = np.real(exp_term @ (w * b1)).astype(float)
    int_p2 = np.real(exp_term @ (w * b2)).astype(float)

    P1 = 0.5 + int_p1 / np.pi
    P2 = 0.5 + int_p2 / np.pi

    calls = s0 * disc_q * P1 - K * disc_r * P2
    if option_type is None:
        return np.maximum(calls, 0.0)

    puts = K * disc_r * (1.0 - P2) - s0 * disc_q * (1.0 - P1)
    is_call = np.asarray(option_type) == 'call'
    out = np.where(is_call, calls, puts)
    return np.maximum(out, 0.0)


# ============================================================================
if __name__ == "__main__":
    params = {
        'kappa': 2,
        'theta': 0.0314,
        'sigma': 1.2,
        'v0': 0.04125,
        'rho': -0.73
    }

    # Example benchmark
    s0 = 1
    T = 2
    r = 0.03
    q = 0.01
    kmin = 0.7
    kmax = 1.3
    fwd = s0*np.exp((r-q)*T)
    K = np.linspace(kmin*fwd,kmax*fwd,100)
    option_params = (s0, r, q)
    heston_params = (params['v0'], params['kappa'], params['theta'], params['sigma'], params['rho'], 0.0)

    pr = cProfile.Profile()
    pr.enable()
    prices = vanilla_price(T, K, option_params, heston_params, N=185)
    pr.disable()
    s = io.StringIO()
    (pstats.Stats(pr, stream=s)
        .strip_dirs()
        .sort_stats("tottime")
        .print_stats(5))
    print(s.getvalue())

    plt.figure(figsize=(7, 4))
    plt.plot(K, prices, "o-", color="tab:blue", label="Heston (Gauss-Laguerre)")
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Heston Model - European Call Prices")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
