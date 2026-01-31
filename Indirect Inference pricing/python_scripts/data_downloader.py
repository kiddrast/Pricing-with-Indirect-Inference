from ib_insync import *
from datetime import date, datetime
import pandas as pd
import time
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 1

folder = Path(f"Indirect Inference pricing/option_data")
folder.mkdir(parents=True, exist_ok=True)

bar_size = "1 day"
duration = "10 D"          # chunk size per request
data_type = "MIDPOINT"     # for options MIDPOINT is often more robust than TRADES
useRTH = True

start_date = date(2016, 6, 1)   # how far back you want to try
end_date = date.today()

# Underlying universe (your banks) - adapt ticker formatting for IB
bank_underlyings = ["UCG", "ISP", "BAMI", "BMED", "BCU", "FBK", "BMPS"]

# How many strikes around ATM to download
n_strikes_each_side = 5

# Limit expirations to a manageable number (most recent expired ones)
max_expirations = 4


# -----------------------------
# HELPERS
# -----------------------------
def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Save dataframe to CSV without overwriting existing files."""
    out = path
    i = 1
    while out.exists():
        out = path.with_name(f"{path.stem}_{i}{path.suffix}")
        i += 1
    df.to_csv(out, index=False)
    return out


def fetch_historical_in_chunks(ib: IB, contract, start_date: date, end_date: date,
                               bar_size: str, duration: str, what_to_show: str,
                               useRTH: bool) -> pd.DataFrame:
    """
    Fetch historical data going backwards in time in fixed chunks,
    then concatenate and return a single dataframe.
    """
    all_bars = []
    req_end = end_date

    while True:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=req_end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=useRTH,
            formatDate=1
        )

        if not bars:
            break

        df = util.df(bars)
        all_bars.append(df)

        # Move end time backwards using the oldest returned bar
        req_end = bars[0].date

        # Stop if we are past the desired start_date
        if req_end.date() <= start_date:
            break

        # Small pacing to be nice with IB rate limits
        ib.sleep(0.2)

    if not all_bars:
        return pd.DataFrame()

    final_df = pd.concat(reversed(all_bars), ignore_index=True)
    return final_df


def get_underlying_contract(ib: IB, symbol: str):
    """Qualify the underlying stock contract for Italian equities."""
    # Many Italian stocks work with SMART + primaryExchange=BVME
    stk = Stock(symbol, "SMART", "EUR", primaryExchange="BVME")
    ib.qualifyContracts(stk)
    return stk


def pick_expired_expirations(expirations, today_yyyymmdd: str, max_exp: int):
    """Pick the most recent expired expirations (<= today)."""
    expired = sorted([e for e in expirations if e <= today_yyyymmdd], reverse=True)
    return sorted(expired[:max_exp])  # return ascending for nicer loops


def pick_atm_strikes(strikes, spot: float, n_each_side: int):
    """Pick strikes around the closest-to-spot strike."""
    strikes_sorted = sorted(strikes)
    if not strikes_sorted:
        return []

    # Find closest strike to spot
    closest_idx = min(range(len(strikes_sorted)), key=lambda i: abs(strikes_sorted[i] - spot))
    lo = max(0, closest_idx - n_each_side)
    hi = min(len(strikes_sorted), closest_idx + n_each_side + 1)
    return strikes_sorted[lo:hi]


# -----------------------------
# MAIN
# -----------------------------
ib = IB()
ib.connect(HOST, PORT, clientId=CLIENT_ID)

t0 = time.time()

for sym in bank_underlyings:
    try:
        underlying = get_underlying_contract(ib, sym)

        # Get spot (last) for ATM selection
        ticker = ib.reqMktData(underlying, "", False, False)
        ib.sleep(1.0)  # allow data to populate
        spot = ticker.marketPrice()
        if spot is None or spot != spot:  # NaN check
            print(f"[{sym}] Could not get spot price, skipping.")
            continue

        # Request option chain parameters
        chains = ib.reqSecDefOptParams(
            underlying.symbol,
            "",
            underlying.secType,
            underlying.conId
        )
        if not chains:
            print(f"[{sym}] No option chains found, skipping.")
            continue

        # Pick the chain that looks like IDEM if available, else first
        chain = None
        for ch in chains:
            if (ch.exchange or "").upper() in ("IDEM", "BVME", "SMART"):
                chain = ch
                break
        if chain is None:
            chain = chains[0]

        today_yyyymmdd = datetime.now().strftime("%Y%m%d")
        expirations = pick_expired_expirations(chain.expirations, today_yyyymmdd, max_expirations)
        if not expirations:
            print(f"[{sym}] No expired expirations found (or none returned), skipping.")
            continue

        strikes = pick_atm_strikes(chain.strikes, spot, n_strikes_each_side)
        if not strikes:
            print(f"[{sym}] No strikes returned, skipping.")
            continue

        print(f"\n[{sym}] spot={spot:.4f} | expirations={expirations} | strikes(sample)={strikes}")

        # Download calls (you can also do puts by looping right in ['C','P'])
        for expiry in expirations:
            for strike in strikes:
                opt = Option(sym, expiry, float(strike), "C", chain.exchange or "IDEM", currency="EUR")
                ib.qualifyContracts(opt)

                df_opt = fetch_historical_in_chunks(
                    ib=ib,
                    contract=opt,
                    start_date=start_date,
                    end_date=end_date,
                    bar_size=bar_size,
                    duration=duration,
                    what_to_show=data_type,
                    useRTH=useRTH
                )

                if df_opt.empty:
                    print(f"[{sym}] {expiry} {strike}C -> no data (IB may not have history).")
                    continue

                # Add contract metadata columns
                df_opt["underlying"] = sym
                df_opt["expiry"] = expiry
                df_opt["strike"] = float(strike)
                df_opt["right"] = "C"
                df_opt["exchange"] = chain.exchange or "IDEM"

                filename = f"{sym}_{expiry}_{strike}C_{bar_size.replace(' ', '')}.csv"
                outpath = safe_to_csv(df_opt, folder / filename)

                print(f"[{sym}] Saved {outpath} rows={len(df_opt)}")

                # Pacing
                ib.sleep(0.25)

    except Exception as e:
        print(f"[{sym}] ERROR: {e}")

total_time = round(time.time() - t0, 2)
print(f"\nDone. Total time: {total_time} seconds")

ib.disconnect()
