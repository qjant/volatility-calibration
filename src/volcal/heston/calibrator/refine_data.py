import pandas as pd
import numpy as np
from volcal.utils import black_scholes as bs
from typing import Literal


def refine_data(
        S0: float,
        act_date: str,
        df: pd.DataFrame,
        mn_low: float = 0.5,
        mn_high: float = 1.5,
        trading_days = 252,
        vega: bool = True,
        market_quote: Literal["call", "put", "otm"] = "otm"
    ) -> pd.DataFrame:
    """
    Clean and enrich an options dataset with derived quantities needed for
    implied-volatility surface work (expiry, forward/strike construction, BS prices)
    and apply basic data-quality filters. Optionally compute Black-Scholes vega for
    later weighting in calibration.

    Steps performed
    --------------
    1) Parse `act_date` and 'Exp Date' to datetime and compute:
           'To expiry' = (Exp Date - act_date) / 252
       using an ACT/252 approximation (252 trading days).
    2) Compute a theoretical forward under continuous compounding:
           'CalcFwd' = S0 * exp((r - q) * T)
       and enforce consistency by setting:
           'ImplFwd' = 'CalcFwd'
    3) Build the effective strike from moneyness:
           'Strike' = 'ImplFwd' * 'Moneyness'
    4) Filter out rows with non-positive implied volatility ('IV' <= 0).
    5) Compute Black-Scholes market call/put prices using the quoted IV.
    6) Filter out rows with non-positive computed prices.
    7) Check put-call parity:
           P_parity = C + K*exp(-rT) - S0*exp(-qT)
       and print a warning if the maximum residual exceeds 1e-10.
    8) Apply a moneyness liquidity filter, keeping rows with
       'Moneyness' in [mn_low, mn_high].
    9) If `vega=True`, compute Black-Scholes vega and store it in 'Vega'.
    10) Reset the index.

    Parameters
    ----------
    S0 : float
        Spot price of the underlying.
    act_date : str
        Valuation/as-of date. Parsed with `pandas.to_datetime`.
    df : pandas.DataFrame
        Input options dataset. Required columns (case-sensitive):
          - 'Exp Date'      : expiration date (string/datetime-like)
          - 'Risk Free'     : continuously-compounded risk-free rate r
          - 'Impl (Yld)'    : continuously-compounded dividend yield q
          - 'Moneyness'     : moneyness used to reconstruct strike from forward
          - 'IV'            : implied volatility (decimal, e.g. 0.25)

        Columns added/overwritten:
          - 'To expiry', 'CalcFwd', 'ImplFwd', 'Strike',
            'Market call price', 'Market put price',
            and optionally 'Vega' (if `vega=True`).
    mn_low : float, default 0.5
        Lower bound for the moneyness filter.
    mn_high : float, default 1.5
        Upper bound for the moneyness filter.
    vega : bool, default True
        If True, compute Black-Scholes vega and store it in the 'Vega' column.

    Returns
    -------
    pandas.DataFrame
        Filtered and reindexed DataFrame with derived fields.

    Notes
    -----
    - Time-to-expiry uses ACT/252 (trading-day approximation).
    - Rates are assumed continuously compounded.
    - Pricing/vega use `df.apply(axis=1)`, which can be slow for large datasets.
    - Put-call parity is checked using:
          P = C + K*exp(-rT) - S0*exp(-qT)

    Raises
    ------
    KeyError
        If any required column is missing.
    """

    # Actual time in datetime format
    t_val = pd.to_datetime(act_date)

    # Ensure that expiration dates are in datetime format
    # to compute the time to expiry as a year_frac
    # WARNING: 252 trading days are assumed by default.
    df['Exp Date'] = pd.to_datetime(df['Exp Date'], dayfirst=True, errors='coerce')
    df['To expiry'] = (df['Exp Date'] - t_val).dt.days / trading_days

    # Theoretical forward and consistency with 'ImplFwd'
    # and enforce implied forward consistency:
    df['CalcFwd'] = S0 * np.exp((df['Risk Free'] - df['Impl (Yld)']) * df['To expiry'])
    df['ImplFwd'] = df['CalcFwd']

    # Effective strike = forward * moneyness:
    df["Strike"] = df["ImplFwd"] * df["Moneyness"]

    # Basic filter: keep positive IV only.
    df = df[df["IV"] > 0]

    # Market prices (BS call & put prices) for each observation using quoted IV:
    df['Market call price'] = df.apply(
        lambda row: bs.price(
            iv=row['IV'],
            T=row['To expiry'],
            K=row['Strike'],
            option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
            option_type='call'
        ),
        axis=1
    )
    df['Market put price'] = df.apply(
        lambda row: bs.price(
            iv=row['IV'],
            T=row['To expiry'],
            K=row['Strike'],
            option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
            option_type='put'
        ),
        axis=1
    )

    # Redundant safety filter: strictly positive prices
    df = df[(df['Market put price'] > 0) & (df['Market call price'] > 0)]

    # Put/call parity check on market data:
    market_put_call_parity = (
        df['Market call price']
        + df['Strike'] * np.exp(-df['Risk Free'] * df['To expiry'])
        - S0 * np.exp(-df['Impl (Yld)'] * df['To expiry'])
    )
    
    put_call_parity_flag = np.max(df['Market put price'] - market_put_call_parity) > 1e-10
    if put_call_parity_flag:
        print("WARNING: put/call parity not satisfied in the market data. Continuing anyways.")

    # Liquidity filter by moneyness (reasonable trading zone)
    df = df[(df['Moneyness'] >= mn_low) & (df['Moneyness'] <= mn_high)]
    mn_low = df['Moneyness'].min()
    mn_high = df['Moneyness'].max()

    if vega:
        # Compute BS vega for weighting (prevents tiny OTM prices from dominating)
        df["Vega"] = df.apply(
            lambda row: bs.vega(
                iv=row['IV'],
                T=row['To expiry'],
                K=row['Strike'],
                option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
            ),
            axis=1
        )

    # Defining option types for a posible calibration in prices
    if market_quote == "call":
        df["Option type"] = "call"
        df["Market price"] = df["Market call price"]
    elif market_quote == "put":
        df["Option type"] = "put"
        df["Market price"] = df["Market put price"]
    elif market_quote == "otm":
        df["Option type"] = np.where(df["Moneyness"] >= 1.0, "call", "put")
        df["Market price"] = np.where(
            df["Option type"].eq("call"),
            df["Market call price"],
            df["Market put price"],
        )
    else:
        raise ValueError("market_quote must be one of: 'otm', 'call', 'put'.")

    # Clean reindex
    df = df.reset_index(drop=True)

    return df