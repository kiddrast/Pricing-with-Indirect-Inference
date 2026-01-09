import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np



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



def merge_and_convert(years: list[str], months: list[str], stocks: list, root_folder: str | Path, output_folder: str | Path) -> None:

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



def clean_data(df: pd.DataFrame, stock: str) -> pd.DataFrame:

    # Set index and order
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)

    # Take median for duplicates
    df[stock] = df.groupby(level=0)[stock].transform("median")
    df = df[~df.index.duplicated(keep='first')]

    return df



def realized_volatility(df: pd.DataFrame, stock: str) -> pd.Series:

    df = df.copy()
    df['logp'] = np.log(df[stock])
    df['rt'] = df['logp'].diff()

    rv = (df['rt'].pow(2).resample('1D').sum().pipe(np.sqrt).rename(f"{stock}_rv").dropna())

    return rv


def closing_price(df: pd.DataFrame, stock: str) -> pd.Series:   
    return df.resample('1D')[stock].last().rename(f"{stock}_close").dropna()



def build_rv_closing_per_stock(stocks: list[str],root_folder: str | Path,output_folder: str | Path) -> None:

    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for stock in tqdm(stocks):

        file_path = root_folder / f"{stock}.parquet"
        if not file_path.exists():
            continue

        # load & clean
        df = pd.read_parquet(file_path)
        df = clean_data(df, stock)

        # features
        rv = realized_volatility(df, stock)
        close = closing_price(df, stock)

        # merge (aligned by date)
        out_df = pd.concat([rv, close], axis=1, join="inner")

        # save
        out_df.to_csv(output_folder / f"{stock}.csv")









### Testing and Debugging ###
if __name__ == '__main__':

    stock = 'BCU_MI'

    df = pd.read_parquet(r"D:\stockData\FTSE_MIB_stocks_merged\BCU_MI.parquet")

    d = clean_data(df, stock)
    rv = realized_volatility(d, stock)
    cp = closing_price(d, stock)

    build_rv_closing_per_stock(['BCU_MI'],r'D:\stockData\FTSE_MIB_stocks_merged', r"D:\stockData\FTSE_MIB_rv_and_closing")

    print(cp)

