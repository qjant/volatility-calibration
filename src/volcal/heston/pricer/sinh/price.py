import time
import numpy as np
from scipy.special import lambertw

# Formulas are referred to the paper:
# BOYARCHENKO, S, LEVENDORSKIĬ, S. "SINH-ACCELERATION: EFFICIENT EVALUATION OF PROBABILITY DISTRIBUTIONS,
# OPTION PRICING, AND MONTE CARLO SIMULATIONS". IJTAF 2019; 22(03):1950011.


# ======================================================================================
# Lambert solver for (3.26): A*x + ln(x) = C  (A>0, x>0)
# ======================================================================================
def solve_lambda1_lambert(A: np.ndarray,
                          C: np.ndarray,
                          optional_newton_refinement: bool = False,
                          newton_steps: int = 2
                          ) -> np.ndarray:
    """
    Solve A*x + ln(x) = C for x>0 using the principal LambertW branch:
      x = W(A*exp(C)) / A
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    if np.any(A <= 0.0):
        raise ValueError("Lambert solver requires A>0 elementwise.")

    # exp(C) overflows around ~709 in float64
    C_clip = np.clip(C, -700.0, 700.0)
    z = A * np.exp(C_clip)

    W = lambertw(z)  # principal branch W0
    x = np.real(W) / A

    if optional_newton_refinement:
        newton_steps = int(max(newton_steps, 0))
        x = np.maximum(x, 1e-14)
        for _ in range(newton_steps):
            Fx  = A * x + np.log(x) - C
            Fpx = A + 1 / x
            x = np.maximum(x - Fx / Fpx, 1e-14)

    return x


def heston_B0_C0(tau: float, xi: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    sigma = float(params["sigma"])
    rho = float(params["rho"])

    xi = np.asarray(xi, dtype=np.complex128)
    i = 1j

    R2 = (
        kappa**2
        + (sigma**2 - 2 * rho * kappa * sigma) * i * xi
        + (sigma**2) * (1 - rho**2) * (xi**2)
    )
    R = np.sqrt(R2)
    R = np.where(np.real(R) < 0.0, -R, R)

    num = rho * sigma * i * xi - kappa + R
    den = rho * sigma * i * xi - kappa - R
    D = num / den
    D1 = D * (kappa + R) / (kappa - R)

    e_mtauR = np.exp(-tau * R)
    log_ratio = np.log(1 - D * e_mtauR) - np.log(1 - D)

    B0 = (kappa - R) * (1 - D1 * e_mtauR) / (1 - D * e_mtauR)
    C0 = (kappa * theta) * ((kappa - R) * tau - 2 * log_ratio)
    return B0, C0


# ======================================================================================
# Sinh parameters (paper Section 3.3), with LambertW for Lambda1
# ======================================================================================
def sinh_trap_params_heston(
    tau: float,
    strikes: np.ndarray,
    spot: float,
    r: float,
    q: float,
    params: dict,
    eps: float = 1e-12,
    safety: float = 0.95,
) -> tuple[dict, dict]:
    strikes = np.asarray(strikes, dtype=float)

    v0 = float(params["v0"])
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    sigma = float(params["sigma"])
    rho = float(params["rho"])

    # Known analytic strip of the CF for covered calls
    mu_p, mu_m = 0.0, -1.0

    mu0 = r - q - (kappa * theta * rho / sigma)
    zt = np.log(spot / strikes) - (rho / sigma) * v0 + mu0 * tau

    vbar = v0 + kappa * theta * tau
    cinf0 = vbar * sigma * np.sqrt(1 - rho**2)

    phi0 = -np.arctan(zt / cinf0)
    gamma_m = np.where(zt > 0.0, -np.pi / 2 - phi0, -np.pi / 2)
    gamma_p = np.where(zt > 0.0, np.pi / 2, np.pi / 2 - phi0)

    w = 0.5 * (gamma_m + gamma_p)
    d0 = 0.5 * (gamma_p - gamma_m)

    a_m = -np.sin(gamma_m)
    a_p = np.sin(gamma_p)

    denom = a_p + a_m
    denom = np.where(np.abs(denom) < 1e-14, np.sign(denom) * 1e-14, denom)

    w1 = (mu_p * a_m + mu_m * a_p) / denom
    b0 = (mu_p - mu_m) / denom

    d = safety * d0
    b = safety * b0

    E = np.log(1.0 / eps)
    zeta = 2.0 * np.pi * d / E

    # log(|Cinf|) to avoid overflow
    log_abs_cinf = (
        np.log(strikes)
        - r * tau
        + kappa * np.log(np.maximum(vbar, 1e-16))
        + (vbar * kappa)
        - np.log(2 * np.pi)
        - (kappa * theta) * np.log(4 * (1 - rho**2))
    )
    rhs = log_abs_cinf + np.log(1.0 / eps)

    Acoef = zt * np.sin(w) + cinf0 * np.cos(w)
    if np.any(Acoef <= 0.0):
        raise ValueError(
            "Contour decay coefficient (zt*sin(w) + c_inf*cos(w)) not positive for some strikes."
        )

    lambda1 = solve_lambda1_lambert(Acoef, rhs)
    Lambda = np.log(2.0 * lambda1 / b)

    sinh_cfg = {"w1": w1, "w": w, "b": b, "zt": zt}
    trap_cfg = {"zeta": zeta, "Lambda": Lambda}
    return sinh_cfg, trap_cfg


# ======================================================================================
# Pricing via sinh + symmetric trapezoid (covered call integral)
# ======================================================================================
def compute_price_heston_sinh(
    tau: float,
    strikes: float | np.ndarray,
    spot: float,
    r: float,
    q: float,
    params: dict,
    sinh_cfg: dict,
    trap_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))

    w1 = np.atleast_1d(np.asarray(sinh_cfg["w1"], dtype=float))
    w = np.atleast_1d(np.asarray(sinh_cfg["w"], dtype=float))
    b = np.atleast_1d(np.asarray(sinh_cfg["b"], dtype=float))
    zt = np.atleast_1d(np.asarray(sinh_cfg["zt"], dtype=float))

    zeta = np.atleast_1d(np.asarray(trap_cfg["zeta"], dtype=float))
    Lambda = np.atleast_1d(np.asarray(trap_cfg["Lambda"], dtype=float))

    if strikes.shape != w1.shape:
        print(strikes)
        print(w1)

        raise ValueError("Shapes mismatch: provide configs computed for the same strike grid.")

    Nk = np.ceil(Lambda / zeta).astype(int)
    N = int(np.max(Nk))

    j = np.arange(-N, N + 1, dtype=int)
    Y = j[None, :] * zeta[:, None]

    # Trapezoid weights
    W = np.ones((strikes.size, 2 * N + 1), dtype=float)
    W[:, 0] *= 0.5
    W[:, -1] *= 0.5

    iw_y = 1j * w[:, None] + Y
    xi = 1j * w1[:, None] + b[:, None] * np.sinh(iw_y)
    chi_prime = b[:, None] * np.cosh(iw_y)

    sigma = float(params["sigma"])
    v0 = float(params["v0"])

    B0, C0 = heston_B0_C0(tau, xi.ravel(), params)
    B0 = B0.reshape(xi.shape)
    C0 = C0.reshape(xi.shape)

    log_core = (v0 * B0 + C0) / (sigma**2)
    logI = 1j * xi * zt[:, None] + log_core
    I = np.exp(logI) * (chi_prime / (xi * (xi + 1j)))

    # Symmetrized sum for better cancellation
    mid = N
    I0 = I[:, mid] * W[:, mid]
    Ip = I[:, mid + 1 :] * W[:, mid + 1 :]
    Im = (I[:, :mid] * W[:, :mid])[:, ::-1]
    s = I0 + np.sum(Ip + Im, axis=1)

    covered_call = (strikes * np.exp(-r * tau) / (2.0 * np.pi)) * zeta * np.real(s)

    call = spot * np.exp(-q * tau) - covered_call
    put = strikes * np.exp(-r * tau) - covered_call

    # No-arbitrage clamps
    discS = spot * np.exp(-q * tau)
    discK = strikes * np.exp(-r * tau)
    intrC = np.maximum(discS - discK, 0.0)
    intrP = np.maximum(discK - discS, 0.0)

    call = np.minimum(np.maximum(call, intrC), discS)
    put = np.minimum(np.maximum(put, intrP), discK)

    # Wipe tiny noise
    tol0 = 1e-12
    call[np.abs(call) < tol0] = 0.0
    put[np.abs(put) < tol0] = 0.0

    return call, put, covered_call


def vanilla_price(T: float, K: np.ndarray, option_params: tuple, heston_params: dict, option_type: np.ndarray = "call"):
    """Convenience wrapper: returns (calls, puts, covered_call) for given strike grid."""
    spot, r, q = option_params
    sinh_cfg, trap_cfg = sinh_trap_params_heston(T, K, spot, r, q, heston_params, eps=1e-12, safety=0.95)

    calls, puts, _ = compute_price_heston_sinh(T, K, spot, r, q, heston_params, sinh_cfg, trap_cfg)
    return np.where(option_type == "call", calls, puts)


# ======================================================================================
# Example usage + profiling
# ======================================================================================
if __name__ == "__main__":
    heston_params = {"kappa": 2, "theta": 0.0314, "sigma": 1.2, "v0": 0.04125, "rho": -0.73}

    spot = 100.0
    tau = 30.0 / 365.0
    r = 0.03
    q = 0.01
    option_params = (spot, r, q)

    fwd = spot * np.exp((r - q) * tau)
    K = np.linspace(0.9 * fwd, 1.1 * fwd, 100)
    mness = K/fwd
    option_type = np.where(mness < 1, "put", "call")

    nruns = 1
    n_opts = K.size

    t0 = time.perf_counter()
    for _ in range(nruns):
        prices = vanilla_price(tau, K, option_params, heston_params, option_type)
    t1 = time.perf_counter()

    dt = t1 - t0
    mean_per_run = dt / nruns
    mean_per_option = mean_per_run / n_opts

    print(f"Total time: {dt:.6f} s for {nruns} runs")
    print(f"Mean time per run: {mean_per_run*1e3:.2f} ms")
    print(f"Mean time per option: {mean_per_option*1e6:.3f} µs/option  (n_opts={n_opts})")

    import matplotlib.pyplot as plt
    option_type = np.asarray(option_type)
    mask_call = option_type == "call"
    mask_put  = option_type == "put"
    plt.figure(figsize=(7, 4))
    plt.plot(mness[mask_call],prices[mask_call],"o",color="tab:blue",label="Call")
    plt.plot(mness[mask_put],prices[mask_put],"o",color="tab:orange",label="Put")
    plt.xlabel("Moneyness [K/F]")
    plt.ylabel("Option Price")
    plt.title("European Option Prices - Heston Model - Sinh-acceleration pricing")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

