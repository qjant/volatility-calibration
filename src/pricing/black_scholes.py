import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

########################################################################
################### BLACK-SCHOLES ANALYTIC PRICE #######################
########################################################################
def price(iv, T, K, option_params, option_type='call'):
    """
    Black-Scholes-Merton price for a European call/put with continuous dividends.

    Parameters
    ----------
    iv : float
        Annualized volatility (decimal, > 0).
    T : float
        Time to maturity in years (> 0).
    K : float
        Strike.
    option_params : tuple
        (S0, r, q): spot, risk-free rate, dividend/convenience yield (cont. comp.).
    option_type : {'call','put'}
        Option type.

    Returns
    -------
    float
        Theoretical BSM price.
    """
    S0, r, q = option_params
    d1 = (np.log(S0 / K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)

    if option_type == 'call':
        return S0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * np.exp(-q*T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type (must be 'call' or 'put').")

########################################################################
########################### BLACK-SCHOLES VEGA #########################
########################################################################
def vega(iv, T, K, option_params):
    """
    Black-Scholes vega: ∂Price/∂vol.

    Parameters
    ----------
    iv : float
        Volatility (decimal, > 0).
    T : float
        Time to maturity (years).
    K : float
        Strike.
    option_params : tuple
        (S0, r, q).

    Returns
    -------
    float
        Price sensitivity to volatility.
    """
    S0, r, q = option_params
    d1 = (np.log(S0 / K) + (r - q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

########################################################################
###################### BLACK-SCHOLES IV OBJECTIVE ######################
########################################################################
def iv_objective(iv, mkt_price, T, K, option_params, option_type='call'):
    """
    Root-finding target for IV: market_price - BS_price(iv).

    Parameters
    ----------
    iv : float
        Candidate volatility.
    mkt_price : float
        Observed market price.
    T, K, option_params, option_type
        Passed to `price`.

    Returns
    -------
    float
        Market - model price difference.
    """
    return mkt_price - price(iv, T, K, option_params, option_type)

########################################################################
######################## BLACK-SCHOLES IV SOLVER #######################
########################################################################
def iv_solver(mkt_price, T, K, option_params, option_type,
              sigma_lo=1e-9, sigma_hi=5.0, max_expand=3, tol=1e-12):
    """
    Implied volatility via Brent's method (`scipy.optimize.root_scalar`),
    calling `iv_objective`. Expands the upper bracket if needed.

    Parameters
    ----------
    mkt_price : float
        Observed option price.
    T : float
        Time to maturity (years).
    K : float
        Strike.
    option_params : tuple
        (S0, r, q).
    option_type : {'call','put'}
        Option type.
    sigma_lo, sigma_hi : float
        Initial volatility bracket.
    max_expand : int
        Max upper-bound doublings if no sign change.
    tol : float
        Absolute tolerance for the root.

    Returns
    -------
    float
        Implied volatility (>= 0).

    Raises
    ------
    ValueError
        If no sign change is found after expansions.
    """

    a, b = sigma_lo, sigma_hi
    fa = iv_objective(a, mkt_price, T, K, option_params, option_type)
    fb = iv_objective(b, mkt_price, T, K, option_params, option_type)

    # Expand the upper bracket if no sign change
    expand = 0
    while fa * fb > 0 and expand < max_expand:
        b *= 2.0
        fb = iv_objective(b, mkt_price, T, K, option_params, option_type)
        expand += 1

    # If still no sign change, return saturated vol
    if fa * fb > 0:
        raise ValueError("IV solver failed to bracket a root (no sign change after expansions).")

    sol = root_scalar(
        iv_objective,
        method="brentq",
        bracket=[a, b],
        xtol=tol,
        rtol=1e-10,
        maxiter=200,
        args=(mkt_price, T, K, option_params, option_type),
    )

    return max(float(sol.root), 0.0)
