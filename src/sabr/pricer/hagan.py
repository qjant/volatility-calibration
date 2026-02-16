import numpy as np
from scipy.stats import norm
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float

    def validate(self) -> None:
        if self.alpha <= 0 or self.nu <= 0:
            raise ValueError("alpha and nu must be > 0")
        if not (0 <= self.beta <= 1):
            raise ValueError("beta must be in [0,1]")
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in (-1, 1)")
        

def sabr_iv(T: float | np.ndarray,
            K: float | np.ndarray,
            F: float | np.ndarray,
            params: SABRParams,
            atm_eps: float = 1e-12
            ) -> float | np.ndarray:
    
    alpha, beta, rho, nu = params["alpha"], params["beta"], params["rho"], params["nu"]
    K = np.asarray(K, dtype=float)
    F = np.asarray(F, dtype=float)
    K, F = np.broadcast_arrays(K, F)

    atm_mask = np.abs(K - F) < atm_eps
    otm_mask = ~atm_mask

    iv = np.empty_like(K)
    # --- ATM formula (Hagan expansion) ---
    if np.any(atm_mask):
        F_atm = F[atm_mask]
        iv[atm_mask] = (alpha / F_atm**(1-beta)) * (
            1 + ((1-beta)**2/24 * (alpha / F_atm**(1-beta))**2
                + 1/4 * rho * beta * nu * alpha / F_atm**(1-beta)
                + (2-3*rho**2)/24 * nu**2) * T)

    # --- OTM formula ---
    if np.any(otm_mask):
        K_otm = K[otm_mask]
        F_otm = F[otm_mask]

        log_fk = np.log(F_otm / K_otm)
        z = (nu/alpha) * (F_otm*K_otm)**((1-beta)/2) * log_fk
        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1-rho))

        denom = (F_otm*K_otm)**((1-beta)/2) * (1 + (1-beta)**2/24*log_fk**2 + (1-beta)**4/1920*log_fk**4)

        iv[otm_mask] = (alpha / denom) * (z/x) * (1 + ((1-beta)**2/24 * (alpha**2)/(F_otm*K_otm)**(1-beta)
                                                    + 1/4 * rho*beta*nu*alpha/(F_otm*K_otm)**((1-beta)/2)
                                                    + (2-3*rho**2)/24 * nu**2) * T)

    return iv



def sabr_price(T: float | np.ndarray,
               K: float | np.ndarray,
               F: float | np.ndarray,
               r: float,
               params: SABRParams,
               option_type: str | np.ndarray ="call"
               ) -> float | np.ndarray:
    
    option_type = np.asarray(option_type)
    iv = sabr_iv(T, K, F, params)
    d1 = (np.log(F / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    prices = np.where(option_type == "call",
                      np.exp(-r*T) * (F * norm.cdf(d1) - K * norm.cdf(d2)),
                      np.exp(-r*T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))
    return prices





if __name__ == "__main__":
    # ---- Forward and risk-free rate ----
    F = 102
    r = 0.03

    # ---- Option configuration ----
    K = np.linspace(80.0,120.0) # strike
    T = 1.0 # maturity (years)

    # ---- SABR parameters ----
    params = {
        "alpha": 0.3, # ATM vol
        "beta": 0.5, # CEV parameter
        "nu": 0.6, # Vol-of-vol
        "rho": -0.2 # Correlation
    }

    init_time = time.perf_counter()
    price = sabr_price(T, K, F, r, params)
    fin_time = time.perf_counter()

    print(f"Prices: {price}")
    print(f"Computation time: {1e4*(fin_time - init_time):.2f} ms")