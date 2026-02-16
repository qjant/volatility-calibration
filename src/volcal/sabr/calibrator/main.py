"""
Sliced SABR calibration (per-expiry) + diagnostics + parameter persistence.

Production notes:
- Pure functions where possible.
- Minimal, English comments only where they add clarity.
- Explicit configuration and logging.
- Avoid duplicated helpers and inconsistent argument order.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize, differential_evolution

# Project modules
import  as conv
import black_scholes_calculator as bs
import SABR_pricer as sp
import metrics as m


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================
@dataclass(frozen=True)
class CalibrationConfig:
    data_name: str = "SPX_03_10_25.xlsx"
    folder_name: str = "volas"
    mn_low: float = 0.8
    mn_high: float = 1.2
    allowed_expiries: Tuple[str, ...] = ("1M", "2M", "3M", "6M", "1Y", "18M", "2Y")

    method: str = "OTMs"          # "calls" | "puts" | "OTMs"
    loss_type: str = "iv"         # keep for metadata/reporting
    weights_setup: str = "Vegas"  # "ATMTriang" | "ATMExp" | "PerTenor" | "Vegas" | "ExpVegas"

    # Differential Evolution
    de_popsize: int = 10
    de_maxiter: int = 100
    de_tol: float = 1e-12
    de_atol: float = 1e-6
    de_mutation: Tuple[float, float] = (0.3, 0.8)
    de_recombination: float = 0.9
    de_seed: int = 7

    # L-BFGS-B
    lbfgs_maxiter: int = 100
    lbfgs_ftol: float = 1e-12

    # Optional regularization (off by default)
    lam_smooth: float = 0.0
    lambda_atm_override: Optional[float] = 0.0  # set None to use dynamic schedule

    # Output
    output_folder: str = "log"
    params_xlsx: str = "sabr_args_log.xlsx"
    sheet_name: str = "param_history"

    # Plotting
    plot_percent_axis: bool = True
    rel_error_plot: bool = False


# =============================================================================
# DATA PREP
# =============================================================================
def load_and_prepare_data(cfg: CalibrationConfig) -> Tuple[float, datetime, pd.DataFrame, str]:
    """Load Excel and create a cleaned dataset suitable for calibration."""
    ticker = cfg.data_name.split("_")[0]
    S0, act_date, df = conv.adapt_excel(cfg.data_name, cfg.folder_name)

    t0 = pd.to_datetime(act_date)

    df = df.copy()
    df["Exp Date"] = pd.to_datetime(df["Exp Date"], dayfirst=True, errors="coerce")
    df["To expiry"] = (df["Exp Date"] - t0).dt.days / 365.0

    # Enforce forward consistency with (S0, r, q)
    df["CalcFwd"] = S0 * np.exp((df["Risk Free"] - df["Impl (Yld)"]) * df["To expiry"])
    fwd_diff = (df["ImplFwd"] - df["CalcFwd"]).describe()
    logger.info("ImplFwd - CalcFwd stats:\n%s", fwd_diff.to_string())
    df["ImplFwd"] = df["CalcFwd"]

    # Effective strike from moneyness
    df["Strike"] = df["ImplFwd"] * df["Moneyness"]

    # Keep only positive IV quotes
    df = df[df["IV"] > 0].copy()

    # Market prices reconstructed with Black-Scholes
    df["Market call price"] = df.apply(
        lambda row: bs.price(
            iv=row["IV"],
            T=row["To expiry"],
            K=row["Strike"],
            option_params=(S0, row["Risk Free"], row["Impl (Yld)"]),
            option_type="call",
        ),
        axis=1,
    )
    df["Market put price"] = df.apply(
        lambda row: bs.price(
            iv=row["IV"],
            T=row["To expiry"],
            K=row["Strike"],
            option_params=(S0, row["Risk Free"], row["Impl (Yld)"]),
            option_type="put",
        ),
        axis=1,
    )

    # Put-call parity check (numerical)
    df["Market put price C/P"] = (
        df["Market call price"]
        + df["Strike"] * np.exp(-df["Risk Free"] * df["To expiry"])
        - S0 * np.exp(-df["Impl (Yld)"] * df["To expiry"])
    )
    parity_ok = np.max(np.abs(df["Market put price"] - df["Market put price C/P"])) < 1e-10
    logger.info("Put/Call parity check (market reconstruction): %s", parity_ok)

    # Filter invalid/degenerate prices
    df = df[(df["Market put price"] > 0) & (df["Market call price"] > 0)].copy()

    # Liquidity band by moneyness
    df = df[(df["Moneyness"] >= cfg.mn_low) & (df["Moneyness"] <= cfg.mn_high)].copy()

    # Tenor filter for stability
    df = df[df["Expiry"].isin(cfg.allowed_expiries)].copy()

    df = df.reset_index(drop=True)
    logger.info("Data batch: %d points", df.shape[0])

    return S0, pd.to_datetime(act_date), df, ticker


def assign_pricing_batch(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Select which options to value and store selected market price in df['Price']."""
    df = df.copy()
    if method == "calls":
        df["calc type"] = "call"
        df["Price"] = df["Market call price"]
        msg = "ITM + OTM calls"
    elif method == "puts":
        df["calc type"] = "put"
        df["Price"] = df["Market put price"]
        msg = "ITM + OTM puts"
    elif method == "OTMs":
        df["calc type"] = np.where(df["Moneyness"] >= 1, "call", "put")
        df["Price"] = np.where(df["Moneyness"] >= 1, df["Market call price"], df["Market put price"])
        msg = "OTM puts + OTM calls"
    else:
        df["calc type"] = "call"
        df["Price"] = df["Market call price"]
        msg = "Unknown method -> default calls"

    logger.info("Option batch: %s", msg)
    return df


def compute_vega(df: pd.DataFrame, S0: float) -> pd.DataFrame:
    """Compute Black-Scholes vega per row using market IV."""
    df = df.copy()
    df["Vega"] = df.apply(
        lambda row: bs.vega(
            iv=row["IV"],
            T=row["To expiry"],
            K=row["Strike"],
            option_params=(S0, row["Risk Free"], row["Impl (Yld)"]),
        ),
        axis=1,
    )
    return df


def get_slice_weights(aux_df: pd.DataFrame, weights_setup: str) -> np.ndarray:
    """Return per-slice weights as a numpy array."""
    if weights_setup == "ATMTriang":
        w = 1.0 - np.abs(aux_df["Moneyness"].to_numpy(dtype=float) - 1.0)
    elif weights_setup == "ATMExp":
        alpha = 5.0
        w = np.exp(-alpha * np.abs(aux_df["Moneyness"].to_numpy(dtype=float) - 1.0))
    elif weights_setup == "PerTenor":
        w = np.ones(len(aux_df), dtype=float)
    elif weights_setup == "Vegas":
        w = aux_df["Vega"].to_numpy(dtype=float)
    elif weights_setup == "ExpVegas":
        gamma = 2.0
        mness = aux_df["Moneyness"].to_numpy(dtype=float)
        vegas = aux_df["Vega"].to_numpy(dtype=float)
        w = vegas * np.exp(-gamma * (mness**2))
    else:
        w = np.ones(len(aux_df), dtype=float)

    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        w = np.ones(len(aux_df), dtype=float)
    return w


# =============================================================================
# SABR: INITIAL GUESS + BOUNDS
# =============================================================================
def initial_sabr_guess(df: pd.DataFrame, S0: float) -> np.ndarray:
    """Heuristic SABR seed [alpha, beta, rho, nu] from ATM."""
    idx_atm = (df["Strike"] - S0).abs().idxmin()
    f_atm = float(df.loc[idx_atm, "ImplFwd"])
    iv_atm = float(df.loc[idx_atm, "IV"])

    beta = 0.9
    alpha = max(1e-6, iv_atm * (f_atm ** (1.0 - beta)))
    rho = -0.3
    nu = 0.3
    return np.array([alpha, beta, rho, nu], dtype=float)


def sabr_bounds() -> list[tuple[float, float]]:
    return [
        (1e-6, 5.0),      # alpha
        (0.0, 1.0),       # beta
        (-0.99, 0.99),    # rho
        (1e-6, 10.0),     # nu
    ]


# =============================================================================
# LOSS (PER SLICE)
# =============================================================================
def loss_sabr_slice(
    sabr_args: np.ndarray,
    aux_df: pd.DataFrame,
    weights: np.ndarray,
    *,
    prev_params: Optional[np.ndarray] = None,
    lam_smooth: float = 0.0,
    lambda_atm_override: Optional[float] = 0.0,
) -> float:
    """
    Slice loss on IVs: weighted MSE across strikes + optional ATM emphasis + optional smoothing across tenors.
    """
    T = float(aux_df["To expiry"].iloc[0])
    K = aux_df["Strike"].to_numpy(dtype=float)
    f = float(aux_df["ImplFwd"].iloc[0])
    r = float(aux_df["Risk Free"].iloc[0])
    q = float(aux_df["Impl (Yld)"].iloc[0])
    iv_mkt = aux_df["IV"].to_numpy(dtype=float)

    iv_model = sp.SABR_implied_vol(
        T=T,
        K=K,
        underlying_args=(f, r, q),
        sabr_args=sabr_args,
    )

    err = (iv_mkt - iv_model) ** 2
    w = weights / max(weights.sum(), 1e-16)
    loss = float(np.sum(w * err))

    # Optional ATM emphasis (defaults off)
    if lambda_atm_override is None:
        # Dynamic schedule (edit if you want it back on)
        lambda_atm = 0.0
    else:
        lambda_atm = float(lambda_atm_override)

    if lambda_atm != 0.0:
        atm_idx = int(np.argmin(np.abs(K - f)))
        loss += lambda_atm * float(err[atm_idx])

    # Optional smoothness across tenor parameters (defaults off)
    if prev_params is not None and lam_smooth != 0.0:
        loss += float(lam_smooth * np.sum((sabr_args - prev_params) ** 2))

    if not np.isfinite(loss):
        return float("inf")
    return loss


# =============================================================================
# OPTIMIZATION
# =============================================================================
def calibrate_slice(
    aux_df: pd.DataFrame,
    *,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    cfg: CalibrationConfig,
    prev_params: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Calibrate one expiry slice with DE -> L-BFGS-B."""
    weights = get_slice_weights(aux_df, cfg.weights_setup)

    def _loss(x: np.ndarray) -> float:
        return loss_sabr_slice(
            x,
            aux_df,
            weights,
            prev_params=prev_params,
            lam_smooth=cfg.lam_smooth,
            lambda_atm_override=cfg.lambda_atm_override,
        )

    # Differential Evolution (global)
    result_de = differential_evolution(
        lambda x: _loss(np.asarray(x, dtype=float)),
        bounds=bounds,
        strategy="best1bin",
        popsize=cfg.de_popsize,
        maxiter=cfg.de_maxiter,
        tol=cfg.de_tol,
        atol=cfg.de_atol,
        mutation=cfg.de_mutation,
        recombination=cfg.de_recombination,
        polish=False,
        seed=cfg.de_seed,
        updating="immediate",
        workers=1,
    )
    x_de = np.asarray(result_de.x, dtype=float)

    # Local refinement
    result_lbfgs = minimize(
        lambda x: _loss(np.asarray(x, dtype=float)),
        x_de,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": cfg.lbfgs_maxiter, "ftol": cfg.lbfgs_ftol},
    )

    x_opt = np.asarray(result_lbfgs.x, dtype=float)
    loss_opt = float(result_lbfgs.fun)
    return x_opt, loss_opt


def calibrate_sliced_sabr(df: pd.DataFrame, S0: float, cfg: CalibrationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Run per-expiry calibration and return (to_expiries, params_by_expiry)."""
    to_exp = np.sort(df["To expiry"].unique().astype(float))
    params = np.zeros((len(to_exp), 4), dtype=float)

    x0 = initial_sabr_guess(df, S0)
    bounds = sabr_bounds()

    prev_params = None
    for i, T in enumerate(to_exp):
        aux_df = df[df["To expiry"] == T].copy()
        expiry_lbl = str(aux_df["Expiry"].iloc[0]) if len(aux_df) else f"T={T:.4f}"
        logger.info("Calibrating slice %s (T=%.6f). Initial x0=%s", expiry_lbl, T, np.round(x0, 4))

        x_opt, loss_opt = calibrate_slice(
            aux_df,
            x0=x0,
            bounds=bounds,
            cfg=cfg,
            prev_params=prev_params,
        )
        logger.info("Slice %s done. Loss=%.6e | params=%s", expiry_lbl, loss_opt, np.round(x_opt, 6))

        params[i] = x_opt
        prev_params = x_opt
        x0 = x_opt  # warm start next slice

    return to_exp, params


# =============================================================================
# PARAMETER PERSISTENCE
# =============================================================================
def ensure_cols_order(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Market date", "Underlying", "Method", "loss_type", "weights_setup", "rmse",
        "alpha", "beta", "rho", "nu", "Spot", "notes", "Date of calibration",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    df["Market date"] = pd.to_datetime(df["Market date"], errors="coerce").dt.date
    df["Date of calibration"] = pd.to_datetime(df["Date of calibration"], errors="coerce").dt.date
    return df[cols]


def save_sabr_args(
    xlsx_path: str,
    sheet_name: str,
    params: np.ndarray,
    as_of: datetime | str,
    *,
    ticker: str,
    method: str,
    loss_type: str,
    weights_setup: str,
    rmse: Optional[float] = None,
    spot: Optional[float] = None,
    notes: Optional[str] = None,
) -> None:
    """Append/update SABR parameters in an Excel sheet, deduped by (Market date, Underlying, Method)."""
    if isinstance(as_of, str):
        as_of = pd.to_datetime(as_of, dayfirst=True, errors="coerce")
    as_of_day = pd.to_datetime(as_of, errors="coerce").date()
    created_day = datetime.now().date()

    row = {
        "Market date": as_of_day,
        "Underlying": ticker,
        "Method": method,
        "loss_type": loss_type,
        "weights_setup": weights_setup,
        "rmse": rmse,
        "alpha": float(params[0]),
        "beta": float(params[1]),
        "rho": float(params[2]),
        "nu": float(params[3]),
        "Spot": spot,
        "notes": notes,
        "Date of calibration": created_day,
    }
    new_df = ensure_cols_order(pd.DataFrame([row]))

    if os.path.exists(xlsx_path):
        try:
            existing = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
            existing = ensure_cols_order(existing)
            out = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            out = new_df.copy()
    else:
        out = new_df.copy()

    out = ensure_cols_order(out)
    out = (
        out.sort_values(["Date of calibration"])
        .drop_duplicates(subset=["Market date", "Underlying", "Method"], keep="last")
        .sort_values(["Market date", "Underlying", "Method"])
    )

    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)

    mode = "a" if os.path.exists(xlsx_path) else "w"
    if mode == "a":
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            out.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as writer:
            out.to_excel(writer, sheet_name=sheet_name, index=False)


# =============================================================================
# DIAGNOSTICS / PLOTS
# =============================================================================
def build_param_dict(to_exp: np.ndarray, params: np.ndarray) -> Dict[float, np.ndarray]:
    return {float(T): params[i] for i, T in enumerate(to_exp)}


def add_model_columns(df: pd.DataFrame, sabr_params_by_T: Dict[float, np.ndarray]) -> pd.DataFrame:
    df = df.copy()

    df["Model price"] = df.apply(
        lambda row: sp.vanilla_price(
            T=row["To expiry"],
            K=row["Strike"],
            underlying_args=(row["ImplFwd"], row["Risk Free"], row["Impl (Yld)"]),
            sabr_args=sabr_params_by_T[float(row["To expiry"])],
            option_type=row["calc type"],
        ),
        axis=1,
    )

    df["Model IV"] = df.apply(
        lambda row: sp.SABR_implied_vol(
            T=row["To expiry"],
            K=row["Strike"],
            underlying_args=(row["ImplFwd"], row["Risk Free"], row["Impl (Yld)"]),
            sabr_args=sabr_params_by_T[float(row["To expiry"])],
        ),
        axis=1,
    )
    return df


def plot_params(to_exp: np.ndarray, params: np.ndarray, act_date: datetime, expiry_labels: Iterable[str]) -> None:
    plt.figure(figsize=(10, 6))
    plt.title(f"SABR optimal parameters - {act_date.strftime('%d/%m/%Y')}")
    plt.plot(to_exp, params[:, 0], "o-", label=r"$\alpha$")
    plt.plot(to_exp, params[:, 1], "o-", label=r"$\beta$")
    plt.plot(to_exp, params[:, 2], "o-", label=r"$\rho$")
    plt.plot(to_exp, params[:, 3], "o-", label=r"$\nu$")
    plt.xlabel("Tenor")
    plt.xticks(to_exp, list(expiry_labels), rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fit_errors_by_tenor(df: pd.DataFrame, *, cfg: CalibrationConfig, mn_low: float, mn_high: float) -> None:
    # Robust ATM selection (closest to 1.0) instead of exact equality
    df = df.copy()
    to_exp = np.sort(df["To expiry"].unique().astype(float))

    def _pick_by_moneyness(target: float) -> pd.DataFrame:
        grp = df.groupby("To expiry")
        rows = []
        for T, g in grp:
            idx = (g["Moneyness"] - target).abs().idxmin()
            rows.append(df.loc[idx])
        return pd.DataFrame(rows)

    atm_df = _pick_by_moneyness(1.0)
    put_df = _pick_by_moneyness(mn_low)
    call_df = _pick_by_moneyness(mn_high)

    err_atm = (atm_df["Model IV"] - atm_df["IV"]).to_numpy()
    err_put = (put_df["Model IV"] - put_df["IV"]).to_numpy()
    err_call = (call_df["Model IV"] - call_df["IV"]).to_numpy()

    plt.figure(figsize=(10, 6))
    if cfg.rel_error_plot:
        plt.plot(to_exp, np.abs(err_atm / atm_df["IV"].to_numpy()), "o-", label="ATM")
        plt.plot(to_exp, np.abs(err_put / put_df["IV"].to_numpy()), "o-", label="Put wing")
        plt.plot(to_exp, np.abs(err_call / call_df["IV"].to_numpy()), "o-", label="Call wing")
        plt.ylabel("|Relative error|")
    else:
        plt.plot(to_exp, np.abs(err_atm), "o-", label="ATM")
        plt.plot(to_exp, np.abs(err_put), "o-", label="Put wing")
        plt.plot(to_exp, np.abs(err_call), "o-", label="Call wing")
        plt.ylabel("|Absolute error|")

    plt.axhline(y=0.0, linestyle="--", linewidth=1)
    plt.title(f"Sliced SABR: fit error vs tenor (weights: {cfg.weights_setup})")
    plt.xlabel("Tenor")
    plt.xticks(to_exp, atm_df["Expiry"].tolist(), rotation=45, ha="right")

    if cfg.plot_percent_axis:
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_smiles(df: pd.DataFrame, to_exp: np.ndarray, params: np.ndarray, mn_low: float, mn_high: float) -> None:
    mness_grid = np.linspace(mn_low, mn_high, 200)

    for i, T in enumerate(to_exp):
        aux = df[df["To expiry"] == float(T)].copy()
        if aux.empty:
            continue

        f = float(aux["ImplFwd"].iloc[0])
        r = float(aux["Risk Free"].iloc[0])
        q = float(aux["Impl (Yld)"].iloc[0])
        K_grid = mness_grid * f

        iv_grid = sp.SABR_implied_vol(
            T=float(T),
            K=K_grid,
            underlying_args=(f, r, q),
            sabr_args=params[i],
        )

        plt.figure()
        plt.title(f"SABR smile calibration - Tenor: {aux['Expiry'].iloc[0]}")
        plt.plot(aux["Moneyness"] * 100, aux["IV"] * 100, "o", label="Market")
        plt.plot(mness_grid * 100, iv_grid * 100, label="SABR")
        plt.xlabel("Moneyness (K/F) [%]")
        plt.ylabel("IV [%]")
        plt.legend()
        plt.tight_layout()

    plt.show()


def plot_price_heatmap_abs_error_bps(df: pd.DataFrame) -> None:
    df = df.copy()
    df["abs_error_points"] = np.abs(df["Model price"] - df["Price"])
    df["abs_error_bps"] = 100.0 * df["abs_error_points"]

    pivot = df.pivot(index="To expiry", columns="Moneyness", values="abs_error_bps")
    pivot.index = pivot.index.astype(float)
    pivot.columns = pivot.columns.astype(float)
    pivot = pivot.sort_index().sort_index(axis=1)

    error_matrix = pivot.to_numpy(dtype=float)

    x_edges = np.linspace(pivot.columns.min(), pivot.columns.max(), pivot.shape[1] + 1)
    y_edges = np.linspace(pivot.index.min(), pivot.index.max(), pivot.shape[0] + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        error_matrix,
        origin="lower",
        aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="viridis",
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Absolute error (bps)")

    ax.set_xlabel("Moneyness (K/F)")
    ax.set_ylabel("Maturity (T)")
    ax.set_title("Absolute pricing error heatmap")

    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)
    ax.set_xticklabels([f"{100*x:.1f}" for x in pivot.columns], rotation=90, ha="center")

    # For the y-axis, show the expiry labels in the same order as pivot.index
    # Build mapping from T -> expiry label (closest match)
    T_to_expiry = (
        df.groupby("To expiry")["Expiry"]
        .first()
        .to_dict()
    )
    ax.set_yticklabels([T_to_expiry.get(float(T), f"{T:.3f}") for T in pivot.index])

    plt.tight_layout()
    plt.show()


def consistency_checks(df: pd.DataFrame, S0: float, sabr_params_by_T: Dict[float, np.ndarray]) -> None:
    """Spot-check: SABR direct price vs BS price using SABR IV."""
    row = df.iloc[int(df["abs_error_points"].idxmax())] if "abs_error_points" in df.columns else df.iloc[0]
    S = S0
    F = float(row["ImplFwd"])
    K = float(row["Strike"])
    T = float(row["To expiry"])
    r = float(row["Risk Free"])
    q = float(row["Impl (Yld)"])
    iv_mkt = float(row["IV"])

    price_bs_mkt = bs.price(iv=iv_mkt, T=T, K=K, option_params=(S, r, q), option_type=row["calc type"])
    iv_sabr = sp.SABR_implied_vol(T=T, K=K, underlying_args=(F, r, q), sabr_args=sabr_params_by_T[T])
    price_bs_sabr = bs.price(iv=iv_sabr, T=T, K=K, option_params=(S, r, q), option_type=row["calc type"])
    price_sabr_direct = sp.vanilla_price(T=T, K=K, underlying_args=(F, r, q), sabr_args=sabr_params_by_T[T], option_type=row["calc type"])

    logger.info("Check Δ1 = SABR_direct − BS(iv_sabr): %.6e", price_sabr_direct - price_bs_sabr)
    logger.info("Check Δ2 = BS(iv_mkt) − BS(iv_sabr):   %.6e", price_bs_mkt - price_bs_sabr)

    vega = bs.vega(iv=iv_mkt, T=T, K=K, option_params=(S, r, q))
    dC = price_bs_mkt - price_bs_sabr
    d_sigma_est = dC / max(vega, 1e-12)
    logger.info("Estimated IV gap: %.6e vol points (%.2f bps)", d_sigma_est, 1e4 * abs(d_sigma_est))


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    cfg = CalibrationConfig()

    S0, act_date, df, ticker = load_and_prepare_data(cfg)
    df = assign_pricing_batch(df, cfg.method)
    df = compute_vega(df, S0)

    # Optional: SABR put-call parity sanity (seed-based)
    x0 = initial_sabr_guess(df, S0)
    sabr_calls = df.apply(
        lambda row: sp.vanilla_price(
            T=row["To expiry"],
            K=row["Strike"],
            underlying_args=(row["ImplFwd"], row["Risk Free"], row["Impl (Yld)"]),
            sabr_args=x0,
            option_type="call",
        ),
        axis=1,
    )
    sabr_puts = df.apply(
        lambda row: sp.vanilla_price(
            T=row["To expiry"],
            K=row["Strike"],
            underlying_args=(row["ImplFwd"], row["Risk Free"], row["Impl (Yld)"]),
            sabr_args=x0,
            option_type="put",
        ),
        axis=1,
    )
    sabr_puts_parity = sabr_calls + df["Strike"] * np.exp(-df["Risk Free"] * df["To expiry"]) - S0 * np.exp(-df["Impl (Yld)"] * df["To expiry"])
    diffs = (sabr_puts - sabr_puts_parity).to_numpy()
    logger.info("SABR seed parity check: %s | max|diff|=%g", (np.max(np.abs(diffs)) < 1e-10), np.max(np.abs(diffs)))

    # Calibrate per slice
    to_exp, params = calibrate_sliced_sabr(df, S0, cfg)
    expiry_labels = [df[df["To expiry"] == T]["Expiry"].iloc[0] for T in to_exp]

    # Diagnostics and reporting
    plot_params(to_exp, params, act_date, expiry_labels)

    sabr_params_by_T = build_param_dict(to_exp, params)
    df = add_model_columns(df, sabr_params_by_T)

    per_tenor, per_moneyness, global_metrics = m.metrics_table(df)
    logger.info("Global metrics:\n%s", str(global_metrics))
    logger.info("Per tenor:\n%s", str(per_tenor))
    logger.info("Per moneyness:\n%s", str(per_moneyness))

    plot_fit_errors_by_tenor(df, cfg=cfg, mn_low=df["Moneyness"].min(), mn_high=df["Moneyness"].max())
    plot_smiles(df, to_exp, params, mn_low=df["Moneyness"].min(), mn_high=df["Moneyness"].max())
    plot_price_heatmap_abs_error_bps(df)

if __name__ == "__main__":
    main()
