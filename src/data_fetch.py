"""
data_fetch.py
-------------
Fetches and caches all data required for the inflation-market study.

Data Sources:
    - FRED API : macroeconomic / inflation series
    - yfinance  : asset price series (ETFs used as proxies)

Usage:
    from src.data_fetch import DataFetcher
    df = DataFetcher(fred_api_key="YOUR_KEY")
    macro  = df.get_macro()
    assets = df.get_assets()
    sectors = df.get_sectors()

Notes:
    - All data is cached to data/raw/ as CSV on first fetch.
    - Subsequent calls load from cache unless force_refresh=True.
    - Monthly frequency is used throughout for consistency.
    - Prices are converted to total-return monthly % changes.
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── FRED Series Catalogue ─────────────────────────────────────────────────────
FRED_SERIES = {
    "cpi_all"       : "CPIAUCSL",
    "cpi_core"      : "CPILFESL",
    "cpi_food"      : "CPIUFDSL",
    "cpi_energy"    : "CPIENGSL",
    "cpi_shelter"   : "CUSR0000SAH1",
    "cpi_medical"   : "CPIMEDSL",
    "pce"           : "PCEPI",
    "pce_core"      : "PCEPILFE",
    "fed_funds"     : "FEDFUNDS",
    "t10y"          : "GS10",
    "t2y"           : "GS2",
    "tips_10y"      : "DFII10",
    "breakeven_10y" : "T10YIE",
    "unrate"        : "UNRATE",
    "gdp"           : "GDP",
    "m2"            : "M2SL",
    "oil_wti"       : "DCOILWTICO",
}

ASSET_TICKERS = {
    "SPY"  : ("US Equities (S&P 500)",         "1993-01"),
    "IEF"  : ("US Treasuries 7-10yr",          "2002-07"),
    "TIP"  : ("TIPS (inflation-linked bonds)", "2003-12"),
    "GLD"  : ("Gold",                          "2004-11"),
    "VNQ"  : ("REITs (US Real Estate)",        "2004-09"),
    "DJP"  : ("Commodities",                   "2006-10"),
    "BIL"  : ("Cash / T-Bills",                "2007-05"),
    "SHY"  : ("Short Treasuries 1-3yr",        "2002-07"),
}

SECTOR_TICKERS = {
    "XLE"  : "Energy",
    "XLF"  : "Financials",
    "XLU"  : "Utilities",
    "XLI"  : "Industrials",
    "XLB"  : "Materials",
    "XLP"  : "Consumer Staples",
    "XLY"  : "Consumer Discretionary",
    "XLK"  : "Technology",
    "XLV"  : "Health Care",
    "XLRE" : "Real Estate",
}


class DataFetcher:
    """
    Central data-access object.

    Parameters
    ----------
    fred_api_key : str
    start_date   : str  (YYYY-MM-DD)
    end_date     : str  (YYYY-MM-DD)
    force_refresh: bool
    """

    def __init__(
        self,
        fred_api_key: str,
        start_date: str = "1960-01-01",
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ):
        self.fred_api_key  = fred_api_key
        self.start_date    = pd.Timestamp(start_date)
        self.end_date      = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
        self.force_refresh = force_refresh
        self._fred         = None

    @property
    def fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_api_key)
                log.info("FRED API connection established.")
            except ImportError:
                raise ImportError("Install fredapi:  pip install fredapi")
        return self._fred

    def _cache_path(self, name: str) -> Path:
        return DATA_RAW / f"{name}.csv"

    def _load_cache(self, name: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(name)
        if p.exists() and not self.force_refresh:
            log.info(f"  Loading from cache: {p.name}")
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df
        return None

    def _save_cache(self, df: pd.DataFrame, name: str):
        df.to_csv(self._cache_path(name))
        log.info(f"  Saved to cache: {self._cache_path(name).name}")

    @staticmethod
    def _to_monthly(series: pd.Series) -> pd.Series:
        return series.resample("ME").last()

    @staticmethod
    def _yoy_pct(series: pd.Series) -> pd.Series:
        return series.pct_change(12) * 100

    @staticmethod
    def _mom_pct(series: pd.Series) -> pd.Series:
        return series.pct_change(1) * 100

    def _fetch_fred_series(self, fred_id: str, name: str) -> pd.Series:
        raw = self.fred.get_series(
            fred_id,
            observation_start=self.start_date,
            observation_end=self.end_date,
        )
        raw.name = name
        if raw.index.freq is None or raw.index.inferred_freq in ("D", "B"):
            monthly = raw.resample("ME").mean()
        else:
            monthly = self._to_monthly(raw)
        return monthly

    def get_macro(self) -> pd.DataFrame:
        cache_name = "macro"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        log.info("Fetching macro data from FRED...")
        frames = {}
        for name, fred_id in FRED_SERIES.items():
            try:
                frames[name] = self._fetch_fred_series(fred_id, name)
                log.info(f"  ✓ {name} ({fred_id})")
            except Exception as e:
                log.warning(f"  ✗ {name} ({fred_id}): {e}")

        df = pd.DataFrame(frames)
        df = df.loc[self.start_date : self.end_date]

        df["cpi_yoy"]              = self._yoy_pct(df["cpi_all"])
        df["cpi_mom"]              = self._mom_pct(df["cpi_all"])
        df["core_cpi_yoy"]         = self._yoy_pct(df["cpi_core"])
        df["pce_yoy"]              = self._yoy_pct(df["pce"])
        df["core_pce_yoy"]         = self._yoy_pct(df["pce_core"])
        df["core_headline_spread"] = df["core_cpi_yoy"] - df["cpi_yoy"]
        df["yield_curve_slope"]    = df["t10y"] - df["t2y"]
        df["real_fed_funds"]       = df["fed_funds"] - df["cpi_yoy"]
        df["cpi_3m_ann"]           = ((df["cpi_all"] / df["cpi_all"].shift(3)) ** 4 - 1) * 100
        df["inflation_surprise"]   = df["cpi_3m_ann"] - df["cpi_yoy"]
        df["gdp"]                  = df["gdp"].ffill()
        df["m2_yoy"]               = self._yoy_pct(df["m2"])

        df.index.name = "date"
        df = df.sort_index()

        self._save_cache(df, cache_name)
        log.info(f"Macro data ready: {df.shape[0]} months, {df.shape[1]} columns.")
        return df

    def _fetch_yfinance(self, tickers: list, name: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Install yfinance:  pip install yfinance")

        log.info(f"Fetching {name} prices from yfinance: {tickers}")
        raw = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        monthly = prices.resample("ME").last()
        return monthly

    def _prices_to_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change() * 100

    def get_assets(self) -> dict:
        cache_prices  = self._load_cache("asset_prices")
        cache_returns = self._load_cache("asset_returns")

        if cache_prices is not None and cache_returns is not None:
            return {"prices": cache_prices, "returns": cache_returns}

        tickers = list(ASSET_TICKERS.keys())
        prices  = self._fetch_yfinance(tickers, "assets")
        returns = self._prices_to_returns(prices)

        self._save_cache(prices,  "asset_prices")
        self._save_cache(returns, "asset_returns")
        return {"prices": prices, "returns": returns}

    def get_sectors(self) -> dict:
        cache_prices  = self._load_cache("sector_prices")
        cache_returns = self._load_cache("sector_returns")

        if cache_prices is not None and cache_returns is not None:
            return {"prices": cache_prices, "returns": cache_returns}

        tickers = list(SECTOR_TICKERS.keys())
        prices  = self._fetch_yfinance(tickers, "sectors")
        returns = self._prices_to_returns(prices)

        self._save_cache(prices,  "sector_prices")
        self._save_cache(returns, "sector_returns")
        return {"prices": prices, "returns": returns}

    def get_master(self) -> pd.DataFrame:
        cache_name = "master"
        cached = self._load_cache(cache_name)
        if cached is not None:
            return cached

        macro   = self.get_macro()
        assets  = self.get_assets()["returns"].add_suffix("_ret")
        sectors = self.get_sectors()["returns"].add_suffix("_sec_ret")

        master = macro.join(assets, how="left").join(sectors, how="left")
        master = master.sort_index()
        master.index.name = "date"

        self._save_cache(master, "master")
        return master

    @staticmethod
    def compute_real_returns(
        returns: pd.DataFrame,
        cpi_mom: pd.Series,
    ) -> pd.DataFrame:
        r_nom = returns / 100
        r_inf = cpi_mom / 100
        real  = (1 + r_nom).divide(1 + r_inf, axis=0) - 1
        return real * 100

    def data_summary(self) -> pd.DataFrame:
        rows = []
        for csv in sorted(DATA_RAW.glob("*.csv")):
            df = pd.read_csv(csv, index_col=0, parse_dates=True)
            rows.append({
                "file"      : csv.name,
                "start"     : df.index.min().date(),
                "end"       : df.index.max().date(),
                "rows"      : len(df),
                "nulls_pct" : f"{df.isnull().mean().mean()*100:.1f}%",
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FRED_API_KEY", "")
    if not key:
        print("Usage:  python data_fetch.py <FRED_API_KEY>")
        sys.exit(1)

    fetcher = DataFetcher(fred_api_key=key, start_date="1960-01-01")
    master  = fetcher.get_master()
    print(master.tail(3).to_string())
    print(fetcher.data_summary().to_string(index=False))
