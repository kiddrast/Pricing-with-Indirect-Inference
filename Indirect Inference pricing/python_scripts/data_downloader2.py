from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd


class HistDataApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

        # Will be set by nextValidId
        self._next_id = None

        # Storage for bars keyed by reqId
        self._bars = {}

        # Flags to know when a request is done
        self._done = {}

    def nextValidId(self, orderId: int):
        # This is the first callback after connect
        self._next_id = orderId

    def get_next_req_id(self) -> int:
        # Generate unique request IDs
        self._next_id += 1
        return self._next_id

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"Error. reqId={reqId}, code={errorCode}, msg={errorString}")

    def historicalData(self, reqId, bar):
        # Store bars in a list of dicts
        if reqId not in self._bars:
            self._bars[reqId] = []

        self._bars[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "barCount": bar.barCount,
            "average": bar.average,
        })

    def historicalDataEnd(self, reqId, start, end):
        # Mark request as finished
        self._done[reqId] = True
        self.cancelHistoricalData(reqId)


def contract_stock(symbol: str, currency: str = "USD", exchange: str = "SMART") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.currency = currency
    c.exchange = exchange
    return c


def contract_option(symbol: str, lastTradeDateOrContractMonth: str, strike: float, right: str,
                    exchange: str = "IDEM", currency: str = "EUR") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "OPT"
    c.currency = currency
    c.exchange = exchange
    c.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth  # e.g. "20251219"
    c.strike = strike
    c.right = right  # "C" or "P"
    c.multiplier = "100"
    # If you want expired options to qualify, IB usually needs this:
    c.includeExpired = True
    return c


def run_historical_to_df(app: HistDataApp, contract: Contract,
                         endDateTime: str,
                         durationStr: str,
                         barSizeSetting: str,
                         whatToShow: str,
                         useRTH: int = 1) -> pd.DataFrame:
    """
    Request historical data and return it as a DataFrame.
    This is synchronous (waits until historicalDataEnd arrives).
    """

    reqId = app.get_next_req_id()
    app._done[reqId] = False
    app._bars[reqId] = []

    app.reqHistoricalData(
        reqId=reqId,
        contract=contract,
        endDateTime=endDateTime,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
        whatToShow=whatToShow,
        useRTH=useRTH,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # Wait until the request is completed
    while not app._done.get(reqId, False):
        time.sleep(0.05)

    df = pd.DataFrame(app._bars[reqId])

    # Convert the "date" field to datetime when possible
    # For daily bars, IB often returns 'YYYYMMDD'
    # For intraday, it can return 'YYYYMMDD  HH:MM:SS'
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df













if __name__ == "__main__":
    port = 7497  # 7496 live
    app = HistDataApp()
    app.connect("127.0.0.1", port, clientId=103)

    # Run the IB event loop in a separate thread (blocking otherwise)
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    # Wait until nextValidId arrives
    while app._next_id is None:
        time.sleep(0.1)

    # Example: QQQ stock
    c = contract_stock("QQQ", currency="USD", exchange="SMART")

    df = run_historical_to_df(
        app=app,
        contract=c,
        endDateTime="20250108 16:00:00 US/Eastern",
        durationStr="10 Y",
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        useRTH=1
    )

    print(df.head())
    print(df.tail())

    app.disconnect()
