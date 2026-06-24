from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .data import get_history, normalize_ticker
from .pricing import RISK_FREE_RATE

OBJECTIVE_BACKTEST_RULES = {
    "保守收租": {"dte": 30, "otm": 0.05, "vol_mult": 0.85, "name": "30 天 / 約 5% OTM"},
    "平衡收租": {"dte": 21, "otm": 0.035, "vol_mult": 0.90, "name": "21 天 / 約 3.5% OTM"},
    "提高現金流": {"dte": 14, "otm": 0.025, "vol_mult": 0.95, "name": "14 天 / 約 2.5% OTM"},
    "本來就想賣出": {"dte": 21, "otm": 0.015, "vol_mult": 0.90, "name": "21 天 / 約 1.5% OTM"},
}

@dataclass
class BacktestSummary:
    ticker: str
    objective: str
    model_rule: str
    start: str
    end: str
    trades: int
    assignment_rate: float
    premium_collected_pct: float
    covered_call_return: float
    buy_hold_return: float
    excess_return: float
    max_drawdown: float
    annualized_return: float
    notes: list[str]
    equity_curve: pd.DataFrame
    trades_frame: pd.DataFrame


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, dte: int, iv: float, r: float = RISK_FREE_RATE) -> float:
    spot = max(float(spot), 0.01)
    strike = max(float(strike), 0.01)
    t = max(int(dte), 1) / 365.0
    sigma = max(float(iv), 0.05)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return max(0.0, spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2))


def _max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    dd = values / running_max - 1.0
    return float(dd.min())


def _annualized_return(total_return: float, days: int) -> float:
    if days <= 0 or total_return <= -1:
        return 0.0
    return float((1.0 + total_return) ** (365.0 / days) - 1.0)


def model_backtest_covered_call(ticker: str, objective: str = "平衡收租", period: str = "2y", shares: int = 100, dte_override: int | None = None) -> BacktestSummary:
    """Approximate a covered-call program using historical stock prices.

    Not a true historical options backtest. It estimates option credit using Black-Scholes
    with realized volatility at each entry date, then caps stock upside at the selected strike.
    Use it to compare behavior/risk, not to claim exact past fills.
    """
    ticker = normalize_ticker(ticker)
    base_rule = OBJECTIVE_BACKTEST_RULES.get(objective, OBJECTIVE_BACKTEST_RULES["平衡收租"])
    rule = dict(base_rule)
    if dte_override is not None:
        dte_override = int(dte_override)
        if dte_override <= 7:
            rule.update({"dte": 7, "otm": 0.018 if objective != "保守收租" else 0.030, "name": "7 天週期 / 短收息模型"})
        elif dte_override <= 14:
            rule.update({"dte": 14, "otm": 0.025 if objective != "保守收租" else 0.040, "name": "14 天週期 / 雙週收息模型"})
        else:
            rule.update({"dte": 30, "otm": 0.040 if objective != "本來就想賣出" else 0.025, "name": "30 天週期 / 月收息模型"})
    dte = int(rule["dte"])
    otm = float(rule["otm"])
    vol_mult = float(rule["vol_mult"])
    hist = get_history(ticker, period=period)
    close = hist["Close"].dropna().copy()
    if len(close) < max(90, dte * 5):
        raise ValueError(f"{ticker} 歷史資料不足，無法做模型回測")

    returns = np.log(close / close.shift(1)).dropna()
    rv = returns.rolling(20).std() * math.sqrt(252)
    initial_stock = float(close.iloc[20])
    initial_value = initial_stock * shares
    strategy_value = initial_value
    account_values = []
    trade_rows = []
    assignments = 0
    trades = 0
    premium_collected = 0.0

    i = 20
    while i + dte < len(close):
        entry_date = close.index[i]
        exit_date = close.index[i + dte]
        entry = float(close.iloc[i])
        exit_px = float(close.iloc[i + dte])
        vol = float(rv.iloc[i]) if np.isfinite(rv.iloc[i]) and rv.iloc[i] > 0 else 0.30
        iv = max(0.12, min(1.20, vol * vol_mult))
        strike = round(entry * (1.0 + otm), 2)
        premium_per_share = black_scholes_call(entry, strike, dte, iv)
        premium = premium_per_share * shares
        premium_collected += premium
        assigned = exit_px > strike
        assignments += int(assigned)
        trades += 1
        if assigned:
            cycle_stock_pnl = (strike - entry) * shares
        else:
            cycle_stock_pnl = (exit_px - entry) * shares
        strategy_value += premium + cycle_stock_pnl
        buy_hold_value = initial_value * (exit_px / initial_stock)
        account_values.append({"date": exit_date, "account_value": strategy_value, "buy_hold_value": buy_hold_value})
        trade_rows.append({
            "entry_date": entry_date.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "entry": round(entry, 2),
            "exit": round(exit_px, 2),
            "strike": round(strike, 2),
            "premium_per_share": round(premium_per_share, 2),
            "premium_total": round(premium, 0),
            "assigned": "是" if assigned else "否",
            "iv_model": f"{iv:.1%}",
        })
        i += dte

    eq = pd.DataFrame(account_values)
    if eq.empty:
        raise ValueError(f"{ticker} 無法產生回測交易")
    days = (pd.to_datetime(eq["date"].iloc[-1]) - pd.to_datetime(eq["date"].iloc[0])).days
    cc_return = float(eq["account_value"].iloc[-1] / initial_value - 1.0)
    bh_return = float(eq["buy_hold_value"].iloc[-1] / initial_value - 1.0)
    notes = [
        "這是模型回測：用歷史股價與 Black-Scholes 估算權利金，不是真實歷史 option bid/ask 成交回測。",
        "用途是判斷 covered call 相對買進持有的行為特徵：收權利金、犧牲上漲、降低部分波動。",
    ]
    return BacktestSummary(
        ticker=ticker,
        objective=objective,
        model_rule=str(rule["name"]),
        start=pd.to_datetime(eq["date"].iloc[0]).date().isoformat(),
        end=pd.to_datetime(eq["date"].iloc[-1]).date().isoformat(),
        trades=trades,
        assignment_rate=assignments / max(trades, 1),
        premium_collected_pct=premium_collected / initial_value,
        covered_call_return=cc_return,
        buy_hold_return=bh_return,
        excess_return=cc_return - bh_return,
        max_drawdown=_max_drawdown(eq["account_value"]),
        annualized_return=_annualized_return(cc_return, days),
        notes=notes,
        equity_curve=eq,
        trades_frame=pd.DataFrame(trade_rows),
    )


def summarize_backtests(tickers: list[str], objective: str = "平衡收租", period: str = "2y", limit: int = 10, dte_override: int | None = None) -> pd.DataFrame:
    rows = []
    for ticker in tickers[:limit]:
        try:
            bt = model_backtest_covered_call(ticker, objective=objective, period=period, dte_override=dte_override)
            rows.append({
                "Ticker": bt.ticker,
                "模型規則": bt.model_rule,
                "交易次數": bt.trades,
                "履約率": f"{bt.assignment_rate:.1%}",
                "權利金/本金": f"{bt.premium_collected_pct:.1%}",
                "Covered Call 報酬": f"{bt.covered_call_return:.1%}",
                "買進持有報酬": f"{bt.buy_hold_return:.1%}",
                "差異": f"{bt.excess_return:.1%}",
                "最大回撤": f"{bt.max_drawdown:.1%}",
                "年化報酬": f"{bt.annualized_return:.1%}",
            })
        except Exception as exc:
            rows.append({"Ticker": ticker, "模型規則": "失敗", "交易次數": 0, "履約率": "-", "權利金/本金": "-", "Covered Call 報酬": "-", "買進持有報酬": "-", "差異": str(exc)[:80], "最大回撤": "-", "年化報酬": "-"})
    return pd.DataFrame(rows)
