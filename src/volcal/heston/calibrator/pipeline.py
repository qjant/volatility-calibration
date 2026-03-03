# src/volcal/heston/calibrator/main.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

from volcal.utils import black_scholes as bs
from volcal.utils.params import vec_to_params
from volcal.heston.pricer.main import HestonPricer

from volcal.heston.calibrator.refine_data import refine_data
from volcal.heston.calibrator.loss import loss_function, LossConfig
from volcal.heston.calibrator.callbacks import make_de_callback, make_lbfgs_callback, DEEarlyStop
from volcal.heston.calibrator.reporting import print_calibration_summary
from volcal.heston.calibrator.checks import check_put_call_parity, add_model_columns


@dataclass(frozen=True)
class HestonCalibrationConfig:
    # data filters
    mn_low: float = 0.8
    mn_high: float = 1.2
    vega: bool = True
    trading_days: int = 252
    market_quote: str = "otm"

    # pricer
    pricer_method: str = "laguerre"
    pricer_N: int = 185

    # loss
    vega_floor: float = 1e-12
    param_keys: Tuple[str, ...] = ("v0", "kappa", "theta", "sigma", "rho")

    # DE
    de_popsize: int = 10
    de_maxiter: int = 100
    de_tol: float = 1e-12
    de_atol: float = 1e-6
    de_mutation: Tuple[float, float] = (0.3, 0.8)
    de_recombination: float = 0.9
    de_seed: int = 7
    de_polish: bool = False

    # early stop DE
    patience: int = 8
    min_rel_improv: float = 1e-5
    min_abs_improv: float = 1e-6
    max_seconds: Optional[float] = None

    # LBFGS
    lbfgs_maxiter: int = 100
    lbfgs_ftol: float = 1e-12

    # checks
    parity_tol_abs: float = 1e-10


DEFAULT_BOUNDS = {
    "v0": (1e-4, 1.0),
    "kappa": (1e-4, 15.0),
    "theta": (1e-4, 1.0),
    "sigma": (1e-4, 2.0),
    "rho": (-0.9, 0.1),
}


def _bounds_to_list(bounds: Dict[str, Tuple[float, float]], keys: Tuple[str, ...]) -> list[tuple[float, float]]:
    return [tuple(bounds[k]) for k in keys]


def _seed_from_atm_iv(S0: float, df: pd.DataFrame, keys: Tuple[str, ...]) -> np.ndarray:
    iv_atm_guess = float(df.loc[(df["Strike"] - S0).abs().idxmin(), "IV"])
    v0_init = max(1e-6, iv_atm_guess**2)
    theta_init = v0_init
    kappa_init = 3.0
    sigma_init = 0.5
    rho_init = -0.5
    x = np.array([v0_init, kappa_init, theta_init, sigma_init, rho_init], dtype=float)

    # in case you ever reorder keys:
    if keys != ("v0", "kappa", "theta", "sigma", "rho"):
        d = {"v0": x[0], "kappa": x[1], "theta": x[2], "sigma": x[3], "rho": x[4]}
        x = np.array([d[k] for k in keys], dtype=float)
    return x


def calibrate_heston(
    *,
    S0: float,
    act_date,
    df: pd.DataFrame,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    cfg: HestonCalibrationConfig = HestonCalibrationConfig(),
    x0: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calibra Heston sobre un DataFrame (una 'slice' o un conjunto completo).
    El usuario controla bounds y cfg.
    Devuelve params óptimos + losses + df con columnas del modelo (opcional).
    """
    t0 = datetime.now()

    # 1) refine
    df_ref = refine_data(
        S0, act_date, df,
        mn_low=cfg.mn_low, mn_high=cfg.mn_high,
        vega=cfg.vega, trading_days=cfg.trading_days,
        market_quote=cfg.market_quote,
    )

    # 2) engine + loss cfg
    pricer = HestonPricer(cfg.pricer_method).config(N=cfg.pricer_N)
    loss_cfg = LossConfig(param_keys=cfg.param_keys, vega_floor=cfg.vega_floor)

    # 3) seed
    if x0 is None:
        x0 = _seed_from_atm_iv(float(S0), df_ref, loss_cfg.param_keys)

    seed_params = vec_to_params(x0, loss_cfg.param_keys)
    check_put_call_parity(
        df=df_ref, pricer=pricer, S0=float(S0), heston_params=seed_params, tol_abs=cfg.parity_tol_abs
    )

    # 4) bounds
    b = DEFAULT_BOUNDS if bounds is None else bounds
    bounds_list = _bounds_to_list(b, loss_cfg.param_keys)

    obj = lambda x: loss_function(x, pricer=pricer, S0=float(S0), df=df_ref, cfg=loss_cfg)
    v2p = lambda x: vec_to_params(x, loss_cfg.param_keys)

    # 5) callbacks (optional)
    cb_de = make_de_callback(
        obj,
        v2p,
        early=DEEarlyStop(
            patience=cfg.patience,
            min_rel_improv=cfg.min_rel_improv,
            min_abs_improv=cfg.min_abs_improv,
            max_seconds=cfg.max_seconds,
        ),
        tag="DE",
    )
    cb_lbfgs, loss_history = make_lbfgs_callback(obj, v2p, tag="LBFGS")

    # 6) run DE -> LBFGS
    loss_init = float(obj(x0))
    if verbose:
        print(f"\nInitial seed: {np.round(x0, 6)}")
        print("\nStarting Differential Evolution (global)")
        print("=" * 90)

    result_de = differential_evolution(
        obj,
        bounds=bounds_list,
        strategy="best1bin",
        popsize=cfg.de_popsize,
        maxiter=cfg.de_maxiter,
        tol=cfg.de_tol,
        atol=cfg.de_atol,
        mutation=cfg.de_mutation,
        recombination=cfg.de_recombination,
        polish=cfg.de_polish,
        seed=cfg.de_seed,
        updating="immediate",
        workers=1,
        callback=cb_de,
    )

    x_de = np.asarray(result_de.x, dtype=float)
    loss_de = float(result_de.fun)

    if verbose:
        print("\nStarting L-BFGS-B (local refinement)")
        print("=" * 90)

    result = minimize(
        obj,
        x_de,
        method="L-BFGS-B",
        bounds=bounds_list,
        options={"maxiter": cfg.lbfgs_maxiter, "ftol": cfg.lbfgs_ftol},
        callback=cb_lbfgs,
    )

    x_opt = np.asarray(result.x, dtype=float)
    loss_opt = float(result.fun)
    params_opt = vec_to_params(x_opt, loss_cfg.param_keys)

    if verbose:
        print_calibration_summary(
            title="Heston calibration summary",
            method=f"{pricer.method} | DE -> L-BFGS-B",
            keys=loss_cfg.param_keys,
            x_init=np.asarray(x0, dtype=float),
            x_de=x_de,
            x_opt=x_opt,
            params_opt=params_opt,
            loss_init=loss_init,
            loss_de=loss_de,
            loss_opt=loss_opt,
            t0=t0,
            notes=None,
        )

    return {
        "params": params_opt,
        "x_init": np.asarray(x0, dtype=float),
        "x_de": x_de,
        "x_opt": x_opt,
        "loss_init": loss_init,
        "loss_de": loss_de,
        "loss_opt": loss_opt,
        "df_refined": df_ref,
        "pricer": pricer,
        "loss_history": loss_history,
        "runtime_seconds": (datetime.now() - t0).total_seconds(),
    }