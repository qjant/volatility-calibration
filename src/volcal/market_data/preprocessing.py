import numpy as np
import pandas as pd
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoader:
    DATA_DIR: str
    BOOK_NAME: str

    def load_iv_table(self, SHEET_NAME: str = 'Mid'):
        # Load Excel file
        data_route = os.path.join(self.DATA_DIR, self.BOOK_NAME)

        df = pd.read_excel(data_route, sheet_name=SHEET_NAME)

        # Identify moneyness columns
        strike_cols = [c for c in df.columns if "%" in str(c)]

        spots = df['Spot'].unique()
        spots = spots[~np.isnan(spots)]
        dates = df['Act Date'].unique()
        dates = dates[~np.isnan(dates)]
        if len(spots) != 1 and len(dates) != 1:
            print("Error: There must be a single spot price and date in the input DataFrame.")
            print("Setting S0=NaN by default.")
            S0 = np.nan
            act_date = np.nan
        else:
            S0 = spots[0]
            act_date = dates[0]

        # Reshape to long format: one volatility per row
        df_long = df.melt(
            id_vars=["Expiry", "Risk Free", "Exp Date", "ImplFwd", "Impl (Yld)"],
            value_vars=strike_cols,
            var_name="Moneyness",
            value_name="IV"
        )

        # Convert string percentages to floats in [0, 1]
        df_long["Moneyness"] = df_long["Moneyness"].str.replace("%", "").astype(float) / 100
        df_long["Risk Free"] = df_long["Risk Free"].astype(float) / 100
        df_long["Impl (Yld)"] = df_long["Impl (Yld)"].astype(float) / 100
        df_long["IV"] = df_long["IV"].astype(float) / 100

        # Reorder columns (for readability only)
        df_long = df_long[["Expiry", "Exp Date", "Risk Free", "ImplFwd", "Impl (Yld)", "Moneyness", "IV"]]

        return S0, act_date, df_long
