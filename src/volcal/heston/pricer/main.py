from __future__ import annotations

from typing import Literal, Mapping, Any
import numpy as np
from numpy.typing import NDArray

from volcal.heston.pricer.laguerre.price import vanilla_price as price_laguerre
from volcal.heston.pricer.sinh.price import vanilla_price as price_sinh

Method = Literal["laguerre", "sinh"]


class HestonPricer:
    def __init__(self, method: Method = "laguerre") -> None:
        self.method: Method = method
        self._engine = price_laguerre if method == "laguerre" else price_sinh

        # defaults
        self.N: int = 185          # laguerre
        self.eps: float = 1e-12    # sinh
        self.safety: float = 0.95  # sinh

    def config(self, **kwargs: Any) -> "HestonPricer":
        """
        Configure pricer depending on selected method.
        - laguerre: N (int)
        - sinh: eps (float), safety (float)
        """
        if self.method == "laguerre":
            allowed = {"N"}
            extra = set(kwargs) - allowed
            if extra:
                raise TypeError(f"laguerre config does not accept: {sorted(extra)}")
            if "N" in kwargs:
                self.N = int(kwargs["N"])

        else:  # "sinh"
            allowed = {"eps", "safety"}
            extra = set(kwargs) - allowed
            if extra:
                raise TypeError(f"sinh config does not accept: {sorted(extra)}")
            if "eps" in kwargs:
                self.eps = float(kwargs["eps"])
            if "safety" in kwargs:
                self.safety = float(kwargs["safety"])

        return self

    def set_method(self, method: Method) -> "HestonPricer":
        self.method = method
        self._engine = price_laguerre if method == "laguerre" else price_sinh
        return self

    def vanilla_price(
        self,
        *,
        T: float,
        K: NDArray[np.float64],
        option_params: tuple[float, float, float],   # (S0, r, q)
        heston_params: Mapping[str, float],
        option_type: NDArray[np.str_],
    ) -> NDArray[np.float64]:
        K = np.asarray(K, dtype=np.float64)
        option_type = np.asarray(option_type, dtype=str)
        option_params = tuple(map(float, option_params))
        heston_params = dict(heston_params)

        if self.method == "laguerre":
            out = self._engine(
                T=float(T),
                K=K,
                option_params=option_params,
                heston_params=heston_params,
                option_type=option_type,
                N=int(self.N),
            )
        else:
            out = self._engine(
                T=float(T),
                K=K,
                option_params=option_params,
                heston_params=heston_params,
                option_type=option_type,
                eps=float(self.eps),
                safety=float(self.safety),
            )

        return np.asarray(out, dtype=np.float64)




# ======================================================================================
# Example usage
# ======================================================================================
if __name__ == "__main__":
    import numpy as np

    # Dummy inputs
    T = 0.5
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)
    option_params = (100.0, 0.02, 0.01)  # (S0, r, q)
    heston_params = {"v0": 0.04, "kappa": 1.5, "theta": 0.04, "sigma": 0.6, "rho": -0.7}
    option_type = np.array(["call"] * K.size, dtype=str)

    # --- Laguerre ---
    pricer_l = HestonPricer("laguerre").config(N=185)
    prices_l = pricer_l.vanilla_price(
        T=T,
        K=K,
        option_params=option_params,
        heston_params=heston_params,
        option_type=option_type,
    )
    print("\n[LAGUERRE]")
    print("config:", {"N": pricer_l.N})
    print("prices:", np.round(prices_l, 6))

    # --- Sinh ---
    pricer_s = HestonPricer("sinh").config(eps=1e-12, safety=0.95)
    prices_s = pricer_s.vanilla_price(
        T=T,
        K=K,
        option_params=option_params,
        heston_params=heston_params,
        option_type=option_type,
    )
    print("\n[SINH]")
    print("config:", {"eps": pricer_s.eps, "safety": pricer_s.safety})
    print("prices:", np.round(prices_s, 6))

    print("\n\nLaguerre engine:", pricer_l._engine)
    print("Sinh engine:", pricer_s._engine)
    print("Laguerre call target:", getattr(pricer_l._engine, "price", pricer_l._engine))
    print("Sinh call target:", getattr(pricer_s._engine, "price", pricer_s._engine))
    diff = np.max(np.abs(prices_l - prices_s))
    print("max|laguerre-sinh| =", diff)




