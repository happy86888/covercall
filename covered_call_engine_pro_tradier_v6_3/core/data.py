from __future__ import annotations

from datetime import date
from typing import Optional, Any
import os

import numpy as np
import pandas as pd


DATA_SOURCE_LABEL = "Tradier API（若未設定 token 則自動 fallback yfinance）"


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(" ", "")


class MarketDataError(RuntimeError):
    pass


def _provider() -> str:
    return os.getenv("MARKET_DATA_PROVIDER", "auto").strip().lower()


def _tradier_token() -> str:
    return os.getenv("TRADIER_ACCESS_TOKEN", os.getenv("TRADIER_API_KEY", "")).strip()


def _tradier_base_url() -> str:
    return os.getenv("TRADIER_BASE_URL", "https://api.tradier.com/v1").rstrip("/")


def _use_tradier() -> bool:
    return _provider() in {"auto", "tradier"} and bool(_tradier_token())


def _tradier_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        import requests
    except Exception as exc:
        raise MarketDataError("使用 Tradier 需要 requests，請先安裝 requirements.txt") from exc
    url = f"{_tradier_base_url()}{path}"
    headers = {"Authorization": f"Bearer {_tradier_token()}", "Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    if resp.status_code >= 400:
        raise MarketDataError(f"Tradier API 錯誤 {resp.status_code}: {resp.text[:240]}")
    return resp.json()


def load_yfinance():
    try:
        import yfinance as yf
        return yf
    except Exception as exc:
        raise MarketDataError("尚未安裝 yfinance，請先執行 pip install -r requirements.txt") from exc


def get_ticker(ticker: str):
    yf = load_yfinance()
    return yf.Ticker(normalize_ticker(ticker))


def _yfinance_price(ticker: str) -> float:
    tk = get_ticker(ticker)
    fast = getattr(tk, "fast_info", {}) or {}
    for key in ["last_price", "regularMarketPrice", "previous_close", "lastPrice"]:
        try:
            value = fast.get(key) if hasattr(fast, "get") else fast[key]
            if value and value > 0:
                return float(value)
        except Exception:
            pass
    hist = tk.history(period="5d", interval="1d", auto_adjust=False)
    if hist.empty:
        raise MarketDataError(f"抓不到 {ticker} 股價資料")
    return float(hist["Close"].dropna().iloc[-1])


def get_price(ticker: str) -> float:
    ticker = normalize_ticker(ticker)
    if _use_tradier():
        try:
            data = _tradier_get("/markets/quotes", {"symbols": ticker})
            quote = (data.get("quotes") or {}).get("quote")
            if isinstance(quote, list):
                quote = quote[0] if quote else None
            if quote:
                for key in ["last", "close", "bid", "ask", "prevclose"]:
                    value = quote.get(key)
                    if value is not None and float(value) > 0:
                        return float(value)
        except Exception:
            if _provider() == "tradier":
                raise
    return _yfinance_price(ticker)


def get_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    tk = get_ticker(ticker)
    hist = tk.history(period=period, interval="1d", auto_adjust=True)
    if hist.empty:
        raise MarketDataError(f"抓不到 {ticker} 歷史資料")
    return hist


def realized_volatility(ticker: str, lookback: int = 20) -> float:
    hist = get_history(ticker, period="6mo")
    returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    if len(returns) < max(10, lookback // 2):
        return 0.30
    return float(returns.tail(lookback).std() * np.sqrt(252))


def trend_score(ticker: str) -> tuple[float, list[str]]:
    hist = get_history(ticker, period="1y")
    close = hist["Close"].dropna()
    notes: list[str] = []
    if len(close) < 60:
        return 55.0, ["歷史資料不足，技術分數採中性"]
    last = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    high_60 = float(close.tail(60).max())
    ret_20 = float(last / close.iloc[-21] - 1) if len(close) > 21 else 0.0
    score = 60.0
    if last > ma20 > ma50:
        score += 10
        notes.append("股價位於 20MA 與 50MA 之上，趨勢偏強")
    elif last < ma20 < ma50:
        score -= 15
        notes.append("股價低於 20MA 與 50MA，covered call 可能只是補跌，不是好收租")
    if ret_20 > 0.12:
        score -= 12
        notes.append("近 20 個交易日漲幅偏大，賣 call 可能過早截斷上漲")
    if last > high_60 * 0.97:
        score -= 8
        notes.append("現價接近 60 日高點，若仍想持有，建議降低 Delta")
    if ret_20 < -0.10:
        score -= 10
        notes.append("近期跌幅偏大，權利金可能不足以補償下跌風險")
    return max(0.0, min(100.0, score)), notes


def _yfinance_expirations(ticker: str) -> list[str]:
    tk = get_ticker(ticker)
    expirations = list(getattr(tk, "options", []) or [])
    if not expirations:
        raise MarketDataError(f"抓不到 {ticker} 選擇權到期日；可能此標的無選擇權或資料源暫時失效")
    return expirations


def get_expirations(ticker: str) -> list[str]:
    ticker = normalize_ticker(ticker)
    if _use_tradier():
        try:
            data = _tradier_get("/markets/options/expirations", {"symbol": ticker, "includeAllRoots": "false", "strikes": "false"})
            dates = ((data.get("expirations") or {}).get("date") or [])
            if isinstance(dates, str):
                dates = [dates]
            if dates:
                return list(dates)
        except Exception:
            if _provider() == "tradier":
                raise
    return _yfinance_expirations(ticker)


def _yfinance_calls(ticker: str, expiry: str) -> pd.DataFrame:
    tk = get_ticker(ticker)
    chain = tk.option_chain(expiry)
    calls = chain.calls.copy()
    if calls.empty:
        raise MarketDataError(f"{ticker} {expiry} 沒有 call option 資料")
    return calls


def _tradier_calls_to_frame(options: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for opt in options:
        if opt.get("option_type") != "call":
            continue
        greeks = opt.get("greeks") or {}
        iv = greeks.get("mid_iv") or greeks.get("smv_vol") or opt.get("implied_volatility")
        rows.append({
            "strike": opt.get("strike", 0),
            "bid": opt.get("bid", 0),
            "ask": opt.get("ask", 0),
            "lastPrice": opt.get("last", opt.get("close", 0)),
            "impliedVolatility": iv,
            "openInterest": opt.get("open_interest", 0),
            "volume": opt.get("volume", 0),
        })
    return pd.DataFrame(rows)


def get_calls(ticker: str, expiry: str) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    if _use_tradier():
        try:
            data = _tradier_get("/markets/options/chains", {"symbol": ticker, "expiration": expiry, "greeks": "true"})
            options = ((data.get("options") or {}).get("option") or [])
            if isinstance(options, dict):
                options = [options]
            calls = _tradier_calls_to_frame(options)
            if not calls.empty:
                return calls
        except Exception:
            if _provider() == "tradier":
                raise
    return _yfinance_calls(ticker, expiry)


def next_earnings_date(ticker: str) -> Optional[date]:
    tk = get_ticker(ticker)
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("EarningsDate")
        else:
            raw = None
            if hasattr(cal, "loc") and "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"].iloc[0]
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if raw is None or pd.isna(raw):
            return None
        return pd.to_datetime(raw).date()
    except Exception:
        return None


def next_ex_dividend_info(ticker: str) -> tuple[Optional[date], float]:
    """Best-effort next ex-dividend date and dividend amount.

    Many free data sources do not provide reliable future ex-dividend dates.
    This function uses yfinance metadata when available and falls back to None.
    The amount is an estimate based on the most recent regular dividend.
    """
    tk = get_ticker(ticker)
    ex_date: Optional[date] = None
    amount = 0.0
    try:
        cal = tk.calendar
        raw = None
        if isinstance(cal, dict):
            raw = cal.get("Ex-Dividend Date") or cal.get("ExDividendDate") or cal.get("exDividendDate")
        elif hasattr(cal, "loc"):
            for key in ["Ex-Dividend Date", "ExDividendDate", "exDividendDate"]:
                if key in cal.index:
                    raw = cal.loc[key].iloc[0]
                    break
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if raw is not None and not pd.isna(raw):
            ex_date = pd.to_datetime(raw, unit="s", errors="coerce").date() if isinstance(raw, (int, float)) else pd.to_datetime(raw, errors="coerce").date()
    except Exception:
        ex_date = None
    try:
        info = getattr(tk, "info", {}) or {}
        raw_ts = info.get("exDividendDate")
        if ex_date is None and raw_ts:
            ex_date = pd.to_datetime(raw_ts, unit="s", errors="coerce").date()
        div_rate = info.get("dividendRate") or 0
        if div_rate and float(div_rate) > 0:
            amount = float(div_rate) / 4.0
    except Exception:
        pass
    try:
        divs = tk.dividends
        if amount <= 0 and divs is not None and len(divs) > 0:
            amount = float(divs.dropna().iloc[-1])
    except Exception:
        pass
    if ex_date is not None and ex_date < date.today():
        # yfinance sometimes returns the latest past ex-date; avoid treating it as an upcoming event.
        ex_date = None
    return ex_date, float(max(amount, 0.0))


def sample_watchlist() -> list[str]:
    return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META"]
