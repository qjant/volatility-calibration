from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import numpy as np


# --------------------------- formatting ---------------------------

def _fmt_elapsed(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:06.3f}"  # mm:ss.sss

def _fmt_sci(x: float) -> str:
    return f"{x: .3e}"

def _fmt_params(p: dict[str, float], *, max_items: int = 6) -> str:
    items = list(p.items())
    shown = items[:max_items]
    body = " ".join(f"{k}={v: .6g}" for k, v in shown)
    if len(items) > max_items:
        body += " …"
    return body


# --------------------------- config ---------------------------

@dataclass(frozen=True, slots=True)
class DEEarlyStop:
    patience: int = 8
    min_rel_improv: float = 1e-5
    min_abs_improv: float = 1e-6
    max_seconds: float | None = None


# --------------------------- factories ---------------------------

def make_de_callback(
    obj: Callable[[np.ndarray], float],
    vec_to_params: Callable[[np.ndarray], dict[str, float]],
    *,
    early: DEEarlyStop = DEEarlyStop(),
    tag: str = "DE",
    params_max_items: int = 6,
) -> Callable[[np.ndarray, float], bool]:
    """
    Returns a SciPy differential_evolution callback (xk, convergence)->bool.
    """
    t0 = datetime.now()
    state = {"best": np.inf, "iters": 0, "stale": 0}

    header = (
        f"{tag:<5} {'iter':>5}  {'time':>9}  {'loss':>12}  {'best':>12}  "
        f"{'Δabs':>11}  {'Δrel':>9}  {'stale':>5}   params"
    )
    print(header)
    print("-" * len(header))

    def callback(xk: np.ndarray, convergence: float) -> bool:
        cur = float(obj(xk))
        state["iters"] += 1

        prev_best = float(state["best"])
        delta_abs = prev_best - cur
        delta_rel = delta_abs / max(abs(prev_best), 1e-12)

        improved = (cur < prev_best) and ((delta_rel > early.min_rel_improv) or (delta_abs > early.min_abs_improv))
        if improved:
            state["best"] = cur
            state["stale"] = 0
        else:
            state["stale"] += 1

        elapsed = (datetime.now() - t0).total_seconds()
        p_str = _fmt_params(vec_to_params(xk), max_items=params_max_items)

        print(
            f"{tag:<5} {state['iters']:5d}  {_fmt_elapsed(elapsed):>9}  "
            f"{_fmt_sci(cur):>12}  {_fmt_sci(float(state['best'])):>12}  "
            f"{_fmt_sci(delta_abs):>11}  {delta_rel:9.2e}  {state['stale']:5d}   {p_str}"
        )

        if (early.max_seconds is not None) and (elapsed >= early.max_seconds):
            print(f"{tag:<5} ↳ early-stop: time budget reached ({early.max_seconds:.1f}s)")
            return True
        if early.patience and state["stale"] >= early.patience:
            print(f"{tag:<5} ↳ early-stop: patience reached ({early.patience} stale iters)")
            return True
        return False

    return callback


def make_lbfgs_callback(
    obj: Callable[[np.ndarray], float],
    vec_to_params: Callable[[np.ndarray], dict[str, float]],
    *,
    tag: str = "LBFGS",
    params_max_items: int = 6,
) -> tuple[Callable[[np.ndarray], None], list[float]]:
    """
    Returns (callback, loss_history) for scipy.optimize.minimize.
    """
    t0 = datetime.now()
    loss_history: list[float] = []

    header = (
        f"{tag:<5} {'iter':>5}  {'time':>9}  {'loss':>12}   params"
    )
    print(header)
    print("-" * len(header))

    def callback(xk: np.ndarray) -> None:
        cur = float(obj(xk))
        loss_history.append(cur)

        elapsed = (datetime.now() - t0).total_seconds()
        p_str = _fmt_params(vec_to_params(xk), max_items=params_max_items)

        print(
            f"{tag:<5} {len(loss_history):5d}  {_fmt_elapsed(elapsed):>9}  "
            f"{_fmt_sci(cur):>12}   {p_str}"
        )

    return callback, loss_history
