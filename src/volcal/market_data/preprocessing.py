import numpy as np
import pandas as pd
import os


"""
Excel adapter for Bloomberg-style implied volatility tables.

Expected input:
---------------
The Excel file must include the following columns (standard BLB format):

- 'Act Date' (str/datetime): Date of data snapshot.
- 'Spot' (float): Underlying spot price at snapshot time.
- 'Expiry' (str): Option tenor (e.g. '1M', '2M', '3M', '6M', ...).
- 'Exp Date' (str/datetime): Option expiration date (consistent with 'Expiry').
- 'Risk Free' (float, %): Implied risk-free rate in percentage.
- 'ImplFwd' (float, %): Implied forward level in percentage.
- One or more moneyness columns named as percentages (e.g. '30%', '40%', ..., '100%', ..., '300%')
  representing implied volatilities, also in percentage units.

Data requirements:
------------------
1. Columns 'Risk Free', 'ImplFwd', and each moneyness must be given in PERCENTAGE units.
2. Only moneyness columns should include a '%' symbol in their name.
3. The Excel file and this script should be located in the same directory.

This script reshapes the data into a long format, converts percentages into decimals,
and returns the spot value, snapshot date, and a normalized DataFrame.
"""


def load_iv_table(folder, file, sheet, save=False, absolute_route=False):
    """
    Load and reformat a Bloomberg-style implied volatility table.

    Parameters
    ----------
    folder : str
        Subfolder (relative to this script) containing the Excel file.
    file : str
        Excel filename.
    sheet : str or int
        Sheet name or index to read.
    save : bool, optional
        If True, exports the reshaped DataFrame to an Excel file (default: False).
    absolute_route : bool, optional
        Currently unused (reserved for future updates).

    Returns
    -------
    S0 : float or np.nan
        Unique spot price found in the input file (NaN if multiple values found).
    act_date : any
        Unique snapshot date from 'Act Date' (NaN if multiple values found).
    df_long : pandas.DataFrame
        Reshaped long-format DataFrame with columns:
        ['Expiry', 'Exp Date', 'Risk Free', 'ImplFwd', 'Impl (Yld)', 'Moneyness', 'IV'].
        All percentage fields are normalized to decimals.

    Notes
    -----
    - Moneyness columns are detected automatically by the presence of '%' in their name.
    - All percentage values ('Risk Free', 'Impl (Yld)', 'IV', and moneyness) are converted to decimals.
    - No consistency checks are performed between 'Expiry' and 'Exp Date'.
    """
    # Load Excel file
    script_route = os.path.dirname(os.path.abspath(__file__))
    data_route = os.path.join(script_route, folder, file)

    df = pd.read_excel(data_route, sheet_name=sheet)

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

    # Optionally export result
    if save:
        output_name = 'dep_' + file
        output_route = os.path.join(script_route, folder, output_name)
        df_long.to_excel(output_route, index=False)

    return S0, act_date, df_long
