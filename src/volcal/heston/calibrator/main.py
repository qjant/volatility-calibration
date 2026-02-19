import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from datetime import datetime
from pathlib import Path

# Project modules
from volcal.market_data.preprocessing import DataLoader
from volcal.utils import black_scholes as bs
from volcal.heston.pricer.main import HestonPricer
from .refine_data import refine_data
from volcal.utils.params import vec_to_params, params_to_vec
from .loss import loss_function, LossConfig
from .callbacks import make_de_callback, make_lbfgs_callback, DEEarlyStop
from .reporting import print_calibration_summary
from .checks import (
    check_put_call_parity,
    add_model_columns,
    plot_iv_smiles_vs_bid_ask,
)


# Volatility file & folder:
# - 'book_name' must exist inside 'folder_name'.
# - preprocessing.load_iv_table returns: Spot (S0), valuation date (act_date), and adapted DataFrame.
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "spx"
BOOK_NAME = 'SPX_17_10_25.xlsx'

loader = DataLoader(DATA_DIR, BOOK_NAME)
S0, act_date, df = loader.load_iv_table('Mid')
option_args = (S0, df['Risk Free'], df['Impl (Yld)'])

# Refine input data
df = refine_data(S0, act_date, df, mn_low=0.8, mn_high=1.2, vega=True, trading_days=252, market_quote="otm")

# Build the pricer engine
pricer = HestonPricer("laguerre").config(N=185)

# Configure the loss function
PARAM_KEYS = ("v0", "kappa", "theta", "sigma", "rho")
cfg = LossConfig(
    param_keys=PARAM_KEYS,
    vega_floor=1e-12
)

# Heuristic seed (under risk-neutral measure):
# - v0  ~ (ATM IV of the closest-to-expiry slice)^2.
# - theta ~ v0 (neutral long-run variance).
# - kappa, sigma, rho in typical equity ranges.
iv_atm_guess = float(df.loc[(df['Strike'] - S0).abs().idxmin(), 'IV'])
v0_init = max(1e-6, iv_atm_guess**2)
theta_init = v0_init
kappa_init = 3.0
sigma_init = 0.5
rho_init = -0.5
x0_init = np.array([v0_init, kappa_init, theta_init, sigma_init, rho_init], dtype=float)
seed_params = vec_to_params(x0_init, cfg.param_keys)
check_put_call_parity(df=df, pricer=pricer, S0=float(S0), heston_params=seed_params, tol_abs=1e-10)

# Parameter bounds
bounds = [
    (1e-4, 1),    # v0   >= 0
    (1e-4, 15),   # kappa >= 0
    (1e-4, 1),    # theta >= 0
    (1e-4, 2),    # sigma >= 0
    (-0.9, 0.1),  # rho   in (-1, 1)
]


# Early-stopping hyperparameters for Differential Evolution (DE)
TOL = 1e-12
ATOL = 1e-6
PATIENCE_CB = 8 # consecutive DE iters without improvement
MIN_REL_IMPROV = 1e-5
MIN_ABS_IMPROV = 1e-6
MAX_SECONDS_CB = None # optional time 
max_iter = 100

x0 = x0_init
print(f"\nInitial seed: {np.round(x0, 3)}")
print(f"\nStarting Differential Evolution (global)")
print(f"==========================================================================================")
t0 = datetime.now()

obj = lambda x: loss_function(x, pricer=pricer, S0=float(S0), df=df, cfg=cfg)
v2p = lambda x: vec_to_params(x, cfg.param_keys)

cb_de = make_de_callback(
    obj,
    v2p,
    early=DEEarlyStop(
        patience=PATIENCE_CB,
        min_rel_improv=MIN_REL_IMPROV,
        min_abs_improv=MIN_ABS_IMPROV,
        max_seconds=MAX_SECONDS_CB,
    ),
    tag="DE",
)


loss_init = float(obj(x0))
result_de = differential_evolution(
    obj,
    bounds=bounds,
    strategy="best1bin",
    popsize=10,
    maxiter=100,
    tol=TOL,
    atol=ATOL,
    mutation=(0.3, 0.8),
    recombination=0.9,
    polish=False,
    seed=7,
    updating="immediate",
    workers=1,
    callback=cb_de,
)

print("\nGlobal method completed.")
x0 = result_de.x
loss_de = float(result_de.fun)

print("\nStarting L-BFGS-B (local refinement)")
print("==========================================================================================")

cb_lbfgs, loss_history = make_lbfgs_callback(obj, v2p, tag="LBFGS")

result = minimize(
    obj,
    x0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": max_iter, "ftol": 1e-12},
    callback=cb_lbfgs,
)

loss_opt = float(result.fun)
optimal_vec = result.x
optimal_params = vec_to_params(optimal_vec, cfg.param_keys)

print_calibration_summary(
    title="Heston calibration summary",
    method=f"{pricer.method} | DE -> L-BFGS-B",
    keys=cfg.param_keys,
    x_init=np.asarray(x0_init, dtype=float),
    x_de=np.asarray(result_de.x, dtype=float),
    x_opt=np.asarray(optimal_vec, dtype=float),
    params_opt=optimal_params,
    loss_init=loss_init,
    loss_de=loss_de,
    loss_opt=loss_opt,
    t0=t0,
    notes=None,
)


'''Checks'''
df_checked = add_model_columns(
    df=df,
    pricer=pricer,
    bs=bs,
    S0=float(S0),
    heston_params=optimal_params,
)

plot_iv_smiles_vs_bid_ask(
    df=df_checked,
    pricer=pricer,
    bs=bs,
    loader=loader,
    S0=float(S0),
    act_date=act_date,
    heston_params=optimal_params,
)