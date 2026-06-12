from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


class MarketConnector:
    """Fetch public market prices using yfinance, with Stooq as a lightweight fallback."""

    source_names = {"market", "market_basket"}

    def __init__(self, indicators: dict[str, dict[str, Any]], data_dir: Path, timeout: int = 20) -> None:
        self.indicators = indicators
        self.data_dir = data_dir
        self.timeout = timeout
        self.raw: pd.DataFrame = pd.DataFrame()
        self.processed: pd.DataFrame = pd.DataFrame()

    def _items(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (indicator_id, meta)
            for indicator_id, meta in self.indicators.items()
            if meta.get("source") in self.source_names
        ]

    @staticmethod
    def _split_tickers(value: str) -> list[str]:
        return [part.strip() for part in str(value).split(",") if part.strip()]

    def _all_needed_tickers(self) -> list[str]:
        tickers: set[str] = set()
        for _, meta in self._items():
            tickers.update(self._split_tickers(str(meta.get("ticker_or_series_id", ""))))
            benchmark = meta.get("benchmark_ticker")
            if benchmark:
                tickers.add(str(benchmark))
        return sorted(tickers)

    def _download_yfinance(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("yfinance unavailable; falling back to Stooq where possible: %s", exc)
            return pd.DataFrame()

        try:
            downloaded = yf.download(
                tickers=tickers,
                start=start_date,
                end=(pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("yfinance download failed; falling back to Stooq where possible: %s", exc)
            return pd.DataFrame()

        if downloaded.empty:
            return pd.DataFrame()

        if isinstance(downloaded.columns, pd.MultiIndex):
            if "Close" in downloaded.columns.get_level_values(0):
                prices = downloaded["Close"]
            elif "Adj Close" in downloaded.columns.get_level_values(0):
                prices = downloaded["Adj Close"]
            else:
                return pd.DataFrame()
        else:
            close_col = "Close" if "Close" in downloaded.columns else "Adj Close"
            prices = downloaded[[close_col]].rename(columns={close_col: tickers[0]})

        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices.columns = [str(col) for col in prices.columns]
        return prices

    @staticmethod
    def _stooq_symbol(ticker: str) -> str:
        if ticker.startswith("^"):
            return ticker.lower()
        return f"{ticker.lower()}.us"

    def _download_stooq_one(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        symbol = self._stooq_symbol(ticker)
        d1 = pd.to_datetime(start_date).strftime("%Y%m%d")
        d2 = pd.to_datetime(end_date).strftime("%Y%m%d")
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Stooq fetch failed for %s: %s", ticker, exc)
            return pd.Series(dtype="float64", name=ticker)

        if frame.empty or "Date" not in frame.columns or "Close" not in frame.columns:
            LOGGER.warning("Stooq returned no usable close data for %s", ticker)
            return pd.Series(dtype="float64", name=ticker)

        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
        series = frame.dropna(subset=["Date"]).set_index("Date")["Close"].sort_index()
        series.name = ticker
        return series

    def _download_prices(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        prices = self._download_yfinance(tickers, start_date, end_date)
        missing = [ticker for ticker in tickers if ticker not in prices.columns or prices[ticker].dropna().empty]
        if missing:
            fallback = [self._download_stooq_one(ticker, start_date, end_date) for ticker in missing]
            fallback_df = pd.concat(fallback, axis=1) if fallback else pd.DataFrame()
            prices = prices.combine_first(fallback_df) if not prices.empty else fallback_df
        return prices.sort_index()

    @staticmethod
    def _equal_weight_basket(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
        available = prices[[ticker for ticker in tickers if ticker in prices.columns]].dropna(how="all")
        if available.empty:
            return pd.Series(dtype="float64")
        normalized = available.divide(available.ffill().bfill().iloc[0]).replace([pd.NA], pd.NA)
        basket = normalized.mean(axis=1, skipna=True) * 100.0
        basket.name = ",".join(tickers)
        return basket

    def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        tickers = self._all_needed_tickers()
        if not tickers:
            self.raw = pd.DataFrame()
            return self.raw

        prices = self._download_prices(tickers, start_date, end_date)
        if prices.empty:
            LOGGER.warning("No market prices were available")
            self.raw = pd.DataFrame()
            return self.raw

        frames: list[pd.DataFrame] = []
        for indicator_id, meta in self._items():
            source = str(meta.get("source"))
            tickers_for_indicator = self._split_tickers(str(meta.get("ticker_or_series_id", "")))
            if not tickers_for_indicator:
                LOGGER.warning("Market indicator %s has no ticker", indicator_id)
                continue

            if source == "market_basket":
                series = self._equal_weight_basket(prices, tickers_for_indicator)
                source_series_id = ",".join(tickers_for_indicator)
            else:
                ticker = tickers_for_indicator[0]
                if ticker not in prices.columns:
                    LOGGER.warning("No market data available for %s (%s)", indicator_id, ticker)
                    continue
                series = prices[ticker].copy()
                benchmark = meta.get("benchmark_ticker")
                if benchmark:
                    benchmark = str(benchmark)
                    if benchmark not in prices.columns:
                        LOGGER.warning("No benchmark data available for %s benchmark %s", indicator_id, benchmark)
                        continue
                    series = series / prices[benchmark]
                    source_series_id = f"{ticker}/{benchmark}"
                else:
                    source_series_id = ticker

            frame = series.rename("raw_value").reset_index().rename(columns={"index": "date", "Date": "date"})
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"])
            if frame.empty:
                LOGGER.warning("Market indicator %s produced no usable rows", indicator_id)
                continue
            last_valid = frame.dropna(subset=["raw_value"])["date"].max()
            frame["indicator_id"] = indicator_id
            frame["source"] = source
            frame["source_series_id"] = source_series_id
            frame["last_updated_date"] = last_valid
            frames.append(frame[["date", "raw_value", "indicator_id", "source", "source_series_id", "last_updated_date"]])

        self.raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self.raw

    def clean(self) -> pd.DataFrame:
        self.processed = self.raw.copy()
        return self.processed

    def save_raw(self) -> Path | None:
        if self.raw.empty:
            return None
        path = self.data_dir / "raw" / "market_raw.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(path, index=False)
        return path

    def save_processed(self) -> Path | None:
        if self.processed.empty:
            return None
        path = self.data_dir / "processed" / "market_processed.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.processed.to_csv(path, index=False)
        return path
