from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd


# --------------------------- formatting ---------------------------

def _line(char: str = "=", n: int = 94) -> str:
    return char * n

def _fmt_sci(x: float) -> str:
    return f"{x: .3e}"

def _print_section(title: str) -> None:
    print()
    print(_line("="))
    print(title)
    print(_line("="))


# --------------------------- configs ---------------------------

@dataclass(frozen=True, slots=True)
class ParityCheckConfig:
    tol_abs: float = 1e-10


# --------------------------- checks ---------------------------

def check_put_call_parity(
    *,
    df: pd.DataFrame,
    pricer: Any,
    S0: float,
    heston_params: Mapping[str, float],
    tol_abs: float = 1e-10,
) -> dict[str, float | bool]:
    """
    Checks: P_put ?= P_call + K e^{-rT} - S0 e^{-qT}
    Uses row-wise pricing for clarity (fast enough for a check).
    """
    T = df["To expiry"].to_numpy(dtype=float)
    K = df["Strike"].to_numpy(dtype=float)
    r = df["Risk Free"].to_numpy(dtype=float)
    q = df["Impl (Yld)"].to_numpy(dtype=float)

    calls = np.empty_like(K, dtype=float)
    puts = np.empty_like(K, dtype=float)

    for i in range(len(df)):
        calls[i] = float(
            pricer.vanilla_price(
                T=float(T[i]),
                K=np.array([K[i]], dtype=float),
                option_params=(float(S0), float(r[i]), float(q[i])),
                heston_params=heston_params,
                option_type=np.array(["call"], dtype=str),
            )[0]
        )
        puts[i] = float(
            pricer.vanilla_price(
                T=float(T[i]),
                K=np.array([K[i]], dtype=float),
                option_params=(float(S0), float(r[i]), float(q[i])),
                heston_params=heston_params,
                option_type=np.array(["put"], dtype=str),
            )[0]
        )

    puts_parity = calls + K * np.exp(-r * T) - S0 * np.exp(-q * T)
    diffs = puts - puts_parity
    max_abs = float(np.max(np.abs(diffs)))
    ok = bool(max_abs < tol_abs)

    _print_section("Heston check — Put/Call parity")
    print(f"Status        : {'PASS' if ok else 'FAIL'}")
    print(f"tol_abs       : {tol_abs:g}")
    print(f"max|diff|     : {_fmt_sci(max_abs)}")
    print(f"mean(diff)    : {_fmt_sci(float(np.mean(diffs)))}")
    print(f"std(diff)     : {_fmt_sci(float(np.std(diffs)))}")

    return {"ok": ok, "max_abs": max_abs, "mean": float(np.mean(diffs)), "std": float(np.std(diffs))}


def add_model_columns(
    *,
    df: pd.DataFrame,
    pricer: Any,
    bs: Any,
    S0: float,
    heston_params: Mapping[str, float],
    price_col: str = "Heston price",
    iv_col: str = "Heston IV",
) -> pd.DataFrame:
    """
    Adds model price and implied vol columns to df (returns a copy).
    Expects `bs.iv_solver(...)` to exist.
    """
    out = df.copy()

    T = out["To expiry"].to_numpy(dtype=float)
    K = out["Strike"].to_numpy(dtype=float)
    r = out["Risk Free"].to_numpy(dtype=float)
    q = out["Impl (Yld)"].to_numpy(dtype=float)
    opt = out["Option type"].to_numpy(dtype=str)

    model_prices = np.empty(len(out), dtype=float)

    for i in range(len(out)):
        model_prices[i] = float(
            pricer.vanilla_price(
                T=float(T[i]),
                K=np.array([K[i]], dtype=float),
                option_params=(float(S0), float(r[i]), float(q[i])),
                heston_params=heston_params,
                option_type=np.array([opt[i]], dtype=str),
            )[0]
        )

    out[price_col] = model_prices

    # IV per row (robust to occasional failures)
    ivs = np.empty(len(out), dtype=float)
    ivs.fill(np.nan)

    for i in range(len(out)):
        try:
            ivs[i] = float(
                bs.iv_solver(
                    mkt_price=float(model_prices[i]),
                    T=float(T[i]),
                    K=float(K[i]),
                    option_params=(float(S0), float(r[i]), float(q[i])),
                    option_type=str(opt[i]),
                )
            )
        except Exception:
            ivs[i] = np.nan

    out[iv_col] = ivs

    _print_section("Model columns added")
    print(f"Added columns : {price_col!r}, {iv_col!r}")
    print(f"IV NaNs       : {int(np.isnan(ivs).sum())} / {len(ivs)}")

    return out


def plot_iv_smiles_vs_bid_ask(
    *,
    df: pd.DataFrame,
    pricer: Any,
    bs: Any,
    loader: Any,
    S0: float,
    act_date: str,
    trading_days: float = 252.0,
    heston_params: Mapping[str, float],
    mn_low: float = 0.8,
    mn_high: float = 1.2,
    n_grid: int = 100,
    title_prefix: str = "IV Smile",
) -> None:
    """
    For each maturity in df:
      - Plot market IV as mid points with vertical error bars spanning bid..ask
      - Plot model IV on a forward-moneyness grid

    Requirements:
      - df must contain: 'To expiry', 'Strike', 'Risk Free', 'Impl (Yld)', and 'ImplFwd'
      - Market IV mid can be taken from df['IV'] if available; otherwise computed as (bid+ask)/2
      - loader.load_iv_table('Bid'/'Ask') must return (_, _, df_bid/df_ask) with 'IV' and either:
          - 'Strike' or
          - 'Moneyness'
        plus an 'Exp Date' column (dayfirst) to recompute 'To expiry' consistently.
    """
    import matplotlib.pyplot as plt
    
    mn_grid = np.linspace(mn_low, mn_high, n_grid)

    print("\n" + "=" * 94)
    print("Plot — IV smiles vs bid/ask (mid ± bid/ask) + model IV")
    print("=" * 94)

    # Load bid/ask
    _, _, df_bid = loader.load_iv_table("Bid")
    _, _, df_ask = loader.load_iv_table("Ask")
    df_bid = df_bid[df_bid["IV"] > 0].copy()
    df_ask = df_ask[df_ask["IV"] > 0].copy()

    # Recompute time-to-expiry on bid/ask
    for dfx in (df_bid, df_ask):
        dfx["Exp Date"] = pd.to_datetime(dfx["Exp Date"], dayfirst=True, errors="coerce")
        dfx["To expiry"] = (dfx["Exp Date"] - pd.to_datetime(act_date)).dt.days / trading_days


    for Tj in sorted(df["To expiry"].unique()):
        aux = df[df["To expiry"] == Tj]
        aux_bid = df_bid[df_bid["To expiry"] == Tj]
        aux_ask = df_ask[df_ask["To expiry"] == Tj]

        if aux.empty:
            continue

        if "ImplFwd" not in aux.columns:
            raise KeyError("df must contain 'ImplFwd' per maturity to plot moneyness grid.")
        fwdj = float(aux["ImplFwd"].iloc[0])

        rj = float(aux["Risk Free"].iloc[0])
        qj = float(aux["Impl (Yld)"].iloc[0])

        # ---------------- Model IV on grid ----------------
        K_grid = mn_grid * fwdj
        opt_type_grid = np.where(mn_grid >= 1.0, "call", "put").astype(str)

        prices_grid = pricer.vanilla_price(
            T=float(Tj),
            K=np.asarray(K_grid, dtype=float),
            option_params=(float(S0), rj, qj),
            heston_params=heston_params,
            option_type=opt_type_grid,
        )

        ivs_grid = np.full_like(prices_grid, np.nan, dtype=float)
        for i, (Kk, Pk, otype) in enumerate(zip(K_grid, prices_grid, opt_type_grid)):
            try:
                ivs_grid[i] = float(
                    bs.iv_solver(
                        mkt_price=float(Pk),
                        T=float(Tj),
                        K=float(Kk),
                        option_params=(float(S0), rj, qj),
                        option_type=str(otype),
                    )
                )
            except Exception:
                ivs_grid[i] = np.nan

        # ---------------- Market error bars: mid ± (bid/ask) ----------------
        # Align bid/ask by Strike if available; otherwise by Moneyness
        if ("Strike" in aux_bid.columns) and ("Strike" in aux_ask.columns):
            key = "Strike"
            m_bid = aux_bid[[key, "IV"]].rename(columns={"IV": "IV_bid"})
            m_ask = aux_ask[[key, "IV"]].rename(columns={"IV": "IV_ask"})
            m = m_bid.merge(m_ask, on=key, how="inner")

            # mid source: df mid IV if possible, else computed as (bid+ask)/2
            if "IV" in aux.columns:
                m_mid = aux[[key, "IV"]].rename(columns={"IV": "IV_mid"})
                m = m.merge(m_mid, on=key, how="left")
            if ("IV_mid" not in m.columns) or m["IV_mid"].isna().all():
                m["IV_mid"] = 0.5 * (m["IV_bid"] + m["IV_ask"])

            mn_mkt = m[key].to_numpy(dtype=float) / fwdj

        elif ("Moneyness" in aux_bid.columns) and ("Moneyness" in aux_ask.columns):
            key = "Moneyness"
            m_bid = aux_bid[[key, "IV"]].rename(columns={"IV": "IV_bid"})
            m_ask = aux_ask[[key, "IV"]].rename(columns={"IV": "IV_ask"})
            m = m_bid.merge(m_ask, on=key, how="inner")

            if ("Moneyness" in aux.columns) and ("IV" in aux.columns):
                m_mid = aux[[key, "IV"]].rename(columns={"IV": "IV_mid"})
                m = m.merge(m_mid, on=key, how="left")
            if ("IV_mid" not in m.columns) or m["IV_mid"].isna().all():
                m["IV_mid"] = 0.5 * (m["IV_bid"] + m["IV_ask"])

            mn_mkt = m[key].to_numpy(dtype=float)

        else:
            raise KeyError("Bid/Ask data must include either 'Strike' or 'Moneyness' to draw error bars.")

        iv_mid = m["IV_mid"].to_numpy(dtype=float)
        iv_bid = m["IV_bid"].to_numpy(dtype=float)
        iv_ask = m["IV_ask"].to_numpy(dtype=float)

        # asymmetric error bars: [mid-bid, ask-mid]
        yerr = np.vstack([iv_mid - iv_bid, iv_ask - iv_mid])

        # ---------------- Plot ----------------
        plt.figure()
        exp_label = str(aux["Expiry"].iloc[0]) if "Expiry" in aux.columns else f"T={Tj:.6f}"
        plt.title(f"{title_prefix} — {exp_label}")

        plt.errorbar(
            100 * mn_mkt,
            100 * iv_mid,
            yerr=100 * yerr,
            fmt=".",
            capsize=3,
            elinewidth=1,
            label="Market IV (mid ± bid/ask)",
        )

        plt.plot(100 * mn_grid, 100 * ivs_grid, label="Heston IV")

        plt.xlabel("Moneyness K/F (%)")
        plt.ylabel("IV (%)")
        plt.legend()
        plt.tight_layout()

    plt.show()
    print('\n')