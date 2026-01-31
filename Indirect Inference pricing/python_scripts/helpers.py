import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable
import os
from pathlib import Path


def apply_daily(S: pd.Series, func: Callable[[np.ndarray], float], name: str, min_ticks: int = 3) -> pd.Series:
    """
    Apply a function to intraday prices on a daily basis using resampling.
    Returns a pd.Series with daily results indexed by date.
    """

    if not isinstance(S.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    def _apply(x: pd.Series):
        prices = x.to_numpy(dtype=float)
        if prices.size < min_ticks:
            return np.nan
        return func(prices)

    daily_result = S.resample("1D").apply(_apply)

    # Set the name of the resulting Series
    daily_result.name = name

    return daily_result.dropna()


# Create a set of all stocks
def all_stocks(years: list[str], months: list[str], root_folder) -> list[str]:
    stocks = set()
    for year in years:
        for month in months:
            month_path = os.path.join(root_folder, year, month)
            if os.path.exists(month_path):
                for file in os.listdir(month_path):
                    if file.endswith(".csv"):
                        stock = file.replace(".csv", "")
                        stocks.add(stock)
    return sorted(list(stocks))


# Merge all data of the same stock and save it as parquet
def merge_and_convert_to_parquet(years: list[str], months: list[str], stocks: list, root_folder: str | Path, output_folder: str | Path) -> None:

    root_folder = Path(root_folder)
    output_folder = Path(output_folder)

    for stock in tqdm(stocks):
        dfs = []
        for year in years:
            for month in months:
                file_path = root_folder / year / month / f"{stock}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    dfs.append(df)

        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            full_df = full_df.sort_values(by="DateTime")
            full_df.to_parquet(output_folder / f"{stock}.parquet")


def set_datetime_index_and_solve_duplicates(df: pd.DataFrame, stock: str) -> pd.Series:
    '''
    Returns a pandas Series with tick by tick prices and DateTime as index
    '''

    # Set index and order
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)

    # Take median for duplicates
    df[stock] = df.groupby(level=0)[stock].transform("median")
    df = df[~df.index.duplicated(keep='first')]

    return df[stock]