"""
Main script — Heston model calibration
-------------------------------------

This script calibrates the Heston (1993) model to implied-volatility data
loaded from Excel/CSV. It wires the full flow from data import through
parameter estimation using global and local optimizers.

High-level flow
---------------
1) Data import
   - Load spot and valuation date (via 'adapt_excel' in convert_df).
   - Convert moneyness to effective strikes using the implied forward.
   - Compute market option prices (calls/puts) via Black-Scholes using the quoted IV.

2) Preprocessing
   - Filter for positive IV.
   - Check and enforce put/call parity in market prices.
   - Select a reasonable moneyness band (e.g., 80%-120%) and representative maturities.
   - Normalize indices.

3) Loss function
   - MSE in prices with vega weights: See http://dx.doi.org/10.22201/fca.24488410e.2022.2789
   - Vectorized by maturity (one CF for multiple strikes).

5) Optimization callbacks
   - Differential Evolution (global): iteration logging and early-stopping by no-improvement/time.
   - L-BFGS-B (local): track loss every iteration.

6) Parameter initialization
   - v0 ≈ (ATM IV of shortest tenor)^2.
   - theta = v0; kappa ≈ 3; sigma ≈ 0.5; rho ≈ -0.5; lambda = 0 (fixed).
   - Future: integrate last valid calibrations (< 1 month) as prior.

7) Sanity checks
   - Validate put/call parity under Heston with the initial seed.

8) Calibration
   - Global stage: differential_evolution.
   - Local refinement: L-BFGS-B.
   - Report optimal params, final loss, and Feller condition.

Dependencies
------------
- pandas, numpy, scipy, matplotlib, os
- Local modules:
  * convert_df: Excel/CSV reading and adaptation.
  * black_scholes_calculator: BS prices, vega, IV solver.
  * heston_vanilla_pricer: Heston pricer.

Inputs
------
- Excel/CSV with implied-vol data (e.g., SPX_YY_MM_DD.xlsx).

Outputs
-------
- Optimal Heston parameters (v0, kappa, theta, sigma, rho, lambda).
- Calibration and convergence logs.
- Validation (put/call parity; Feller condition).

Notes
-----
- Calibration is under the risk-neutral measure (lambda = 0).
- Callbacks give real-time traceability of the fit.
"""

# Standard/third-party libs
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (kept for potential 3D plots)
import matplotlib.ticker as mtick
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from datetime import datetime
from pathlib import Path
import os

# Project modules
from heston_model_calibration.utils import market_data_preprocessing as dp
from heston_model_calibration.pricing import black_scholes as bs
from heston_model_calibration.pricing import heston as hp


########################################################################
############################## FILE IMPORT #############################
# Volatility file & folder:
# - 'book_name' must exist inside 'folder_name'.
# - dp.adapt_excel returns: Spot (S0), valuation date (act_date), and adapted DataFrame.
# Ruta a este archivo
THIS_FILE = Path(__file__).resolve()

# Raíz del proyecto (repo)
PROJECT_ROOT = THIS_FILE.parents[3]

# Ruta al Excel
DATA_DIR = PROJECT_ROOT / "data" / "spx"
BOOK_NAME = 'SPX_17_10_25.xlsx'
SHEET_NAME = 'Mid'
folder_name = ''
ticker = BOOK_NAME.split('_')[0]
S0, act_date, df = dp.adapt_excel(DATA_DIR, BOOK_NAME, SHEET_NAME)
########################################################################


########################################################################
############################ SPOT AND RATES ############################
# Common option parameters:
# - S0: spot
# - r : risk-free rate (column 'Risk Free', continuous comp.)
# - q : dividend/repo yield (column 'Impl (Yld)', continuous comp.)
option_args = (S0, df['Risk Free'], df['Impl (Yld)'])
# Note: reading spot from the Excel avoids hardcoding externally.
########################################################################


########################################################################
########################### DATA TREATMENT #############################
# Time-to-expiry (years) from valuation date:
t_val = pd.to_datetime(act_date)
df['Exp Date'] = pd.to_datetime(df['Exp Date'], dayfirst=True, errors='coerce')
maturities = df['Exp Date'].unique()
df['To expiry'] = (df['Exp Date'] - t_val).dt.days / 365

# Theoretical forward and consistency with 'ImplFwd':
df['CalcFwd'] = S0 * np.exp((df['Risk Free'] - df['Impl (Yld)']) * df['To expiry'])
# Enforce implied forward consistency:
df['ImplFwd'] = df['CalcFwd']

# Effective strike = forward * moneyness:
df["Strike"] = df["ImplFwd"] * df["Moneyness"]

# Basic filter: keep positive IV only.
df = df[df["IV"] > 0]

# Market prices (BS) for each observation (call and put) using quoted IV:
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

# Put/call parity check on market data:
df['Market put price C/P'] = (
    df['Market call price']
    + df['Strike'] * np.exp(-df['Risk Free'] * df['To expiry'])
    - S0 * np.exp(-df['Impl (Yld)'] * df['To expiry'])
)
print("\nPut/Call parity check (market):",
      np.max(df['Market put price'] - df['Market put price C/P']) < 1e-10)

# Redundant safety filter: strictly positive prices
df = df[(df['Market put price'] > 0) & (df['Market call price'] > 0)]

# Liquidity filter by moneyness (reasonable trading zone)
mn_low = 0.8
mn_high = 1.2
df = df[(df['Moneyness'] >= mn_low) & (df['Moneyness'] <= mn_high)]
mn_low = df['Moneyness'].min()
mn_high = df['Moneyness'].max()

# Clean reindex
df = df.reset_index(drop=True)

print("\nData batch size: {} points\n".format(df.shape[0]))
########################################################################


########################################################################
######################### SIMULATION OPTIONS ###########################
# Choose option type per row by moneyness (simple OTM/ITM split)
# The procedure is independent of this choice due to the call/put parity
# and the functional form of the loss functions
df['Option type'] = np.where(df['Moneyness'] >= 1, 'call', 'put')
df['Market price'] = np.where(df['Moneyness'] >= 1, df['Market call price'], df['Market put price'])

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
########################################################################


########################################################################
########################### LOSS FUNCTION ##############################
def loss_function(heston_params, df, market_prices, vega_floor=1e-12):
    """
    Heston calibration loss:
      - Default: price error weighted by vega (prevents very cheap OTM options dominating).
      - Vectorized by maturity: reuse one CF per T for all strikes K at that T.
    """
    # Extract numpy arrays for speed
    T = df['To expiry'].to_numpy(dtype=float)
    K = df['Strike'].to_numpy(dtype=float)
    r = df['Risk Free'].to_numpy(dtype=float)
    q = df['Impl (Yld)'].to_numpy(dtype=float)
    market_prices = df['Market price'].to_numpy(dtype=float)
    vega = np.maximum(df["Vega"].to_numpy(), vega_floor)
    opt_type = df['Option type'].to_numpy()

    # Evaluate by unique maturities: one CF, multiple strikes
    model_prices = np.empty_like(market_prices)
    uniq_T = np.unique(T)
    for Tj in uniq_T:
        idx = np.where(T == Tj)[0]
        Kj = K[idx]
        rj = float(r[idx][0])
        qj = float(q[idx][0])
        optj = opt_type[idx]
        # Heston pricing in batch for all strikes at this T
        pj = hp.vanilla_price(
            T=Tj,
            K=Kj,
            option_params=(S0, rj, qj),
            heston_params=heston_params,
            option_type=optj,
            N=185,
        )
        model_prices[idx] = pj

    loss = np.mean(((market_prices - model_prices) / vega) ** 2)
    return loss
########################################################################


########################################################################
######################### OPTIMIZATION CALLBACKS #######################
# Early-stopping hyperparameters for Differential Evolution (DE)
TOL   = 1e-12
ATOL  = 1e-6
PATIENCE_CB     = 8         # number of consecutive DE iters without improvement
MIN_REL_IMPROV  = 1e-5      # minimum relative improvement to reset patience
MIN_ABS_IMPROV  = 1e-6      # minimum absolute improvement to reset patience
MAX_SECONDS_CB  = None      # optional time limit

# Global callback state (simple logging)
mkt_arr   = df['Market price'].to_numpy()
init_time = datetime.now()
state     = {"best": np.inf, "best_x": None, "iters": 0, "stale": 0}

def callback_ga(xk, convergence):
    """DE callback: log progress and stop on no-improvement/time."""
    cur = float(loss_function(xk, df, option_args, mkt_arr))
    state["iters"] += 1

    prev = state["best"]
    rel_impr = (prev - cur) / max(abs(prev), 1e-12)
    abs_impr = (prev - cur)
    improved = (cur < prev) and ((rel_impr > MIN_REL_IMPROV) or (abs_impr > MIN_ABS_IMPROV))

    if improved:
        state["best"]   = cur
        state["stale"]  = 0
    else:
        state["stale"] += 1

    # Progress logging
    elapsed = (datetime.now() - init_time).total_seconds()
    print(f"Iteration {state['iters']:>3} — Elapsed: {elapsed*1000:.2f} ms — "
          f"Loss: {cur:.6e} — "
          f"Stale: {state['stale']} — Params: {xk}")

    # Early-stopping criteria
    if (MAX_SECONDS_CB is not None) and (elapsed >= MAX_SECONDS_CB):
        print("↳ Early stop (callback): time budget reached")
        return True
    if PATIENCE_CB and state["stale"] >= PATIENCE_CB:
        print("↳ Early stop (callback): patience reached (no improvement)")
        return True
    return False

# L-BFGS-B callback: track loss history
max_iter = 100
loss_history = []
init_time = datetime.now()

def callback(params):
    current_loss = loss_function(params, df, option_args, df['Market price'])
    loss_history.append(current_loss)
    print(f"Iteration {len(loss_history)} — Elapsed: {(datetime.now()-init_time).total_seconds()*1000:.2f} ms — "
          f"Loss: {current_loss:.6e} — Params: {params}")
########################################################################


########################################################################
######################### INITIAL HESTON PARAMS ########################
# Heuristic seed:
# - v0  ~ (ATM IV of the closest-to-expiry slice)^2.
# - theta ~ v0 (neutral long-run variance).
# - kappa, sigma, rho in typical equity ranges.
# - lambda = 0 (risk-neutral pricing).
print(df['Strike'] - S0)
iv_atm_guess = float(df.loc[(df['Strike'] - S0).abs().idxmin(), 'IV'])
v0_init = max(1e-6, iv_atm_guess**2)
theta_init = v0_init
kappa_init = 3.0
sigma_init = 0.5
rho_init = -0.5
x0 = [v0_init, kappa_init, theta_init, sigma_init, rho_init, 0.0]  # lambda = 0 (risk-neutral)

# Parameter bounds
bounds = [
    (1e-4, 1),    # v0   >= 0
    (1e-4, 15),   # kappa >= 0
    (1e-4, 1),    # theta >= 0
    (1e-4, 2),    # sigma >= 0
    (-0.9, 0.1),  # rho   in (-1, 1)
    (-0.0, 0.0),  # lambda fixed to 0
]
########################################################################


########################################################################
############################## CHECKS ##################################
# Put/Call parity under Heston with the initial seed:
heston_calls = df.apply(
    lambda row: hp.vanilla_price(
        T=row['To expiry'],
        K=row['Strike'],
        option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
        heston_params=x0,
        option_type="call"
    ),
    axis=1
)

heston_puts = df.apply(
    lambda row: hp.vanilla_price(
        T=row['To expiry'],
        K=row['Strike'],
        option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
        heston_params=x0,
        option_type="put"
    ),
    axis=1
)

heston_puts_via_parity = (
    heston_calls
    + df['Strike'] * np.exp(-df['Risk Free'] * df['To expiry'])
    - S0 * np.exp(-df['Impl (Yld)'] * df['To expiry'])
)
diffs = heston_puts - heston_puts_via_parity
print("\nHeston — Put/Call parity (seed): {}.   Max|diff|: {}".format(
    np.max(diffs) < 1e-10, np.max(np.abs(diffs))
))
########################################################################


########################################################################
############################ CALIBRATION ###############################
print(f"\nInitial params: [v0, kappa, theta, sigma, rho, lambda]")
print(f"Initial params: {np.round(x0, 3)}")
print(f"\nStarting Differential Evolution (global)")
print(f"=================================================================")
result_de = differential_evolution(
    loss_function, bounds=bounds, args=(df, option_args, mkt_arr),
    strategy='best1bin', popsize=10, maxiter=100,
    tol=TOL, atol=ATOL,
    mutation=(0.3, 0.8), recombination=0.9,
    polish=False, seed=7, updating='immediate',
    workers=1, callback=callback_ga
)
print(f"\nGlobal method completed.")
x0 = result_de.x

print(f"\nStarting L-BFGS-B (local refinement)")
print(f"=================================================================")
result = minimize(
    loss_function, x0, method="L-BFGS-B",
    args=(df, option_args, mkt_arr),
    bounds=bounds,
    options={'maxiter': max_iter, 'ftol': 1e-12},
    callback=callback
)

optimal_heston = result.x
optimal_loss = result.fun
print("\nOptimized parameters:", optimal_heston)
print("Final loss:", optimal_loss)

# Feller condition (2*kappa*theta > sigma^2)
v0_opt, kappa_opt, theta_opt, sigma_opt, rho_opt, lambda_opt = optimal_heston
print("Feller condition:", bool(2*kappa_opt*theta_opt > sigma_opt**2))
########################################################################


########################################################################
######################### CALIBRATION ANALYSIS #########################
df['Heston price'] = df.apply(
    lambda row: hp.vanilla_price(
        T=row['To expiry'],
        K=row['Strike'],
        option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
        heston_params=optimal_heston,
        option_type=row['Option type'],
        N=128
    ),
    axis=1
)

df['Heston IV'] = df.apply(
    lambda row: bs.iv_solver(
        mkt_price=row['Heston price'],
        T=row['To expiry'],
        K=row['Strike'],
        option_params=(S0, row['Risk Free'], row['Impl (Yld)']),
        option_type=row['Option type']
    ),
    axis=1
)

# Bid/Ask sheets (if available)
_, _, df_bid = dp.adapt_excel(DATA_DIR, BOOK_NAME, 'Bid')
df_bid = df_bid[df_bid["IV"] > 0]

_, _, df_ask = dp.adapt_excel(DATA_DIR, BOOK_NAME, 'Ask')
df_ask = df_ask[df_ask["IV"] > 0]

# Align moneyness & expiries with main set
df_bid = df_bid[df_bid['Moneyness'].isin(df['Moneyness'])]
df_ask = df_ask[df_ask['Moneyness'].isin(df['Moneyness'])]
df_bid = df_bid[df_bid['Expiry'].isin(df['Expiry'])]
df_ask = df_ask[df_ask['Expiry'].isin(df['Expiry'])]

# Recompute time-to-expiry on bid/ask
df_bid['Exp Date'] = pd.to_datetime(df_bid['Exp Date'], dayfirst=True, errors='coerce')
maturities = df_bid['Exp Date'].unique()
df_bid['To expiry'] = (df_bid['Exp Date'] - t_val).dt.days / 365

df_ask['Exp Date'] = pd.to_datetime(df_ask['Exp Date'], dayfirst=True, errors='coerce')
maturities = df_ask['Exp Date'].unique()
df_ask['To expiry'] = (df_ask['Exp Date'] - t_val).dt.days / 365

# Clean reindex
df = df.reset_index(drop=True)

# Plot smiles per maturity
mn_grid = np.linspace(mn_low, mn_high, 300)
for Tj in sorted(df['To expiry'].unique()):
    # Market subsets
    aux = df[df['To expiry'] == Tj]
    aux_bid = df_bid[df_bid['To expiry'] == Tj]
    aux_ask = df_ask[df_ask['To expiry'] == Tj]

    fwdj = aux['ImplFwd'].iloc[0]
    K_grid = mn_grid * fwdj
    rj = aux['Risk Free'].iloc[0]
    qj = aux['Impl (Yld)'].iloc[0]
    opt_type_grid = np.where(mn_grid >= 1, "call", "put")

    # Model prices on strike grid
    prices_heston = [
        hp.vanilla_price(
            T=Tj, K=Kk,
            option_params=(S0, rj, qj),
            heston_params=optimal_heston,
            option_type=otype
        )
        for Kk, otype in zip(K_grid, opt_type_grid)
    ]

    # Convert model prices to IV on the same grid
    ivs_heston = []
    for Kk, Pk, otype in zip(K_grid, prices_heston, opt_type_grid):
        try:
            ivs_heston.append(
                bs.iv_solver(
                    mkt_price=Pk, T=Tj, K=Kk,
                    option_params=(S0, rj, qj),
                    option_type=otype
                )
            )
        except Exception:
            ivs_heston.append(np.nan)
    ivs_heston = np.array(ivs_heston)

    # Market data aligned in forward moneyness
    mn_mkt = aux['Strike'].to_numpy() / fwdj
    iv_mkt = aux['IV'].to_numpy()
    iv_mkt_bid = aux_bid['IV'].to_numpy()
    iv_mkt_ask = aux_ask['IV'].to_numpy()

    # Plot smile (IV)
    plt.figure()
    plt.title(f"IV Smile — {aux['Expiry'].iloc[0]}")
    plt.plot(100*mn_mkt, 100*iv_mkt_bid, 'o', color='blue', label='Market bid IV')
    plt.plot(100*mn_mkt, 100*iv_mkt_ask, 'o', color='red', label='Market ask IV')
    plt.plot(100*mn_grid, 100*ivs_heston, color='green', label='Heston IV')
    plt.xlabel("Moneyness K/F (%)")
    plt.ylabel("IV (%)")
    plt.legend()
    plt.tight_layout()

plt.show()