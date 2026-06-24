from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from .data import (
    MarketDataError,
    get_calls,
    get_expirations,
    get_price,
    next_earnings_date,
    next_ex_dividend_info,
    normalize_ticker,
    realized_volatility,
    trend_score,
)
from .models import Candidate, Holding
from .pricing import call_delta, call_price, call_theta_per_day

PROFILE_RULES = {
    "保守收租": {"delta": (0.10, 0.22), "dte": (21, 45), "min_yield": 0.0025, "assignment_penalty": 1.20},
    "平衡收租": {"delta": (0.16, 0.30), "dte": (14, 35), "min_yield": 0.0035, "assignment_penalty": 1.00},
    "提高現金流": {"delta": (0.22, 0.42), "dte": (7, 28), "min_yield": 0.0050, "assignment_penalty": 0.75},
    "本來就想賣出": {"delta": (0.28, 0.55), "dte": (7, 35), "min_yield": 0.0030, "assignment_penalty": 0.50},
}

CORE_ETFS = {"SPY", "VOO", "IVV", "QQQ", "DIA", "IWM", "VTI", "VT", "SCHD"}
QUALITY_BUCKET = {
    "SPY": 95, "VOO": 95, "IVV": 95, "VTI": 93, "VT": 90, "QQQ": 90, "DIA": 86, "IWM": 78,
    "AAPL": 88, "MSFT": 90, "GOOGL": 84, "GOOG": 84, "AMZN": 82, "META": 82, "NVDA": 76, "AMD": 70,
    "TSLA": 62, "COIN": 55, "PLTR": 58, "SOFI": 50,
}


def _to_float(value, default: float = 0.0) -> float:
    """Convert yfinance/CSV values safely. yfinance often returns NaN for OI/volume/IV."""
    try:
        if value is None:
            return default
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        result = _to_float(value, float(default))
        return int(max(result, 0))
    except Exception:
        return default


def _score_range(value: float, low: float, high: float, ideal: float | None = None) -> float:
    if ideal is None:
        ideal = (low + high) / 2
    if low <= value <= high:
        return 100.0 - min(abs(value - ideal) / max(ideal - low, high - ideal, 1e-9), 1.0) * 15
    if value < low:
        return max(0.0, 100.0 - (low - value) / max(low, 0.01) * 90)
    return max(0.0, 100.0 - (value - high) / max(high, 0.01) * 90)


def _liquidity_score(open_interest: int, volume: int, spread_pct: float) -> float:
    # yfinance/Tradier sometimes returns zero OI during stale sessions.
    # Covered call execution should still be guided with limit orders instead of hard-failing.
    oi_score = min(100.0, np.log1p(max(open_interest, 0)) / np.log1p(1000) * 100)
    vol_score = min(100.0, np.log1p(max(volume, 0)) / np.log1p(300) * 100)
    activity_score = max(oi_score, vol_score * 0.85)
    if spread_pct <= 0.04:
        spread_score = 100
    elif spread_pct <= 0.10:
        spread_score = 78
    elif spread_pct <= 0.20:
        spread_score = 55
    elif spread_pct <= 0.35:
        spread_score = 32
    else:
        spread_score = 18
    return float(0.60 * activity_score + 0.40 * spread_score)


def _premium_score(premium_yield: float, annualized_yield: float, min_yield: float) -> float:
    base = min(100.0, premium_yield / max(min_yield, 0.001) * 80)
    # Do not reject high annualized yields automatically; flag them in warnings instead.
    if annualized_yield > 1.20:
        base -= 10
    return float(max(0.0, min(100.0, base)))


def _event_score(expiry: date, earnings: date | None) -> tuple[float, list[str]]:
    if earnings is None:
        return 80.0, ["未取得下一次財報日，事件風險採中性偏保守"]
    today = date.today()
    days_to_earnings = (earnings - today).days
    if days_to_earnings < 0:
        return 85.0, ["近期未偵測到即將到來的財報風險"]
    if days_to_earnings <= 7:
        return 10.0, [f"財報距今 {days_to_earnings} 天，建議等財報後再開倉"]
    if today <= earnings <= expiry:
        return 20.0, [f"到期日前會經過財報日 {earnings.isoformat()}，不建議賣一般 covered call"]
    if days_to_earnings <= 14:
        return 45.0, [f"財報日 {earnings.isoformat()} 接近，建議降低 Delta 或暫停"]
    return 90.0, [f"下一次財報日 {earnings.isoformat()} 未落在主要持倉期間內"]


def _dividend_risk_score(expiry: date, ex_dividend: date | None, dividend_amount: float, spot: float, strike: float, mid: float) -> tuple[float, list[str], str]:
    """Score early assignment risk around ex-dividend dates.

    For American-style equity calls, early assignment risk rises when a short call is ITM
    near ex-dividend date and remaining extrinsic value is less than the dividend.
    """
    if ex_dividend is None or dividend_amount <= 0:
        return 85.0, ["未偵測到即將到來的除息日，除息風險採中性"], "未偵測"
    today = date.today()
    if ex_dividend < today:
        return 85.0, ["未偵測到即將到來的除息日，除息風險採中性"], "未偵測"
    if ex_dividend > expiry:
        return 90.0, [f"下一次除息日 {ex_dividend.isoformat()} 不在本次合約期間內"], "低"
    intrinsic = max(0.0, spot - strike)
    extrinsic = max(0.0, mid - intrinsic)
    if strike <= spot and extrinsic < dividend_amount:
        return 15.0, [f"除息日 {ex_dividend.isoformat()} 落在合約期間內，且 call 價外時間價值可能低於股息，早期履約風險高"], "高"
    if ex_dividend <= expiry:
        return 55.0, [f"除息日 {ex_dividend.isoformat()} 落在合約期間內；若 call 變價內，要在除息日前檢查是否平倉或 roll"], "中"
    return 85.0, ["除息風險不明顯"], "低"


def _management_plan(c: Candidate) -> list[str]:
    close_profit_low = c.mid * 0.30
    close_profit_high = c.mid * 0.50
    roll_watch_price = c.strike - max(c.mid * 0.25, c.last_price * 0.003)
    return [
        f"50%–70% 獲利了結：若 call 價格從 ${c.mid:.2f} 跌到約 ${close_profit_high:.2f}–${close_profit_low:.2f}，可考慮 Buy to Close 收工。",
        f"接近履約價：若股價逼近 ${roll_watch_price:.2f}–${c.strike:.2f}，不要等到最後一天才處理，先決定接受履約或往後/往上 roll。",
        "到期前 3 天：若仍接近價內，請檢查是否要平倉、roll，或讓股票被賣出。",
        "不追單：掛 limit 在 mid 附近，成交不了再小幅讓價；spread 太大時寧可放棄。",
    ]


def _assignment_score(spot: float, strike: float, holding: Holding, delta: float) -> tuple[float, list[str]]:
    notes = []
    score = 80.0
    if holding.minimum_sell_price and strike < holding.minimum_sell_price:
        gap = holding.minimum_sell_price - strike
        score -= min(60, gap / holding.minimum_sell_price * 300)
        notes.append(f"Strike 低於你設定的最低願意賣出價 {holding.minimum_sell_price:.2f}")
    if holding.cost_basis and strike < holding.cost_basis:
        score -= 30
        notes.append("Strike 低於成本，若被履約可能不符合你的心理賣出價格")
    if delta > 0.40 and holding.objective in ["保守收租", "平衡收租"]:
        score -= 20
        notes.append("Delta 偏高，被履約機率較高")
    if strike <= spot:
        score -= 50
        notes.append("Strike 未高於現價，不適合一般 covered call 收租")
    return max(0.0, min(100.0, score)), notes


def _cycle_bucket(dte: int) -> str:
    if dte <= 10:
        return "7天週期"
    if dte <= 24:
        return "14天週期"
    return "30天週期"


def _action_cycle(candidate: Candidate, holding: Holding) -> tuple[str, str]:
    """Return a user-facing cadence recommendation such as 7/14/30 days.

    The recommendation is deliberately practical: it maps the actual option DTE to
    the closest management cycle, then explains why that cycle fits or does not fit
    the selected objective.
    """
    dte = candidate.dte
    bucket = _cycle_bucket(dte)
    if candidate.event_score < 35:
        return "暫停", "財報或事件風險太近，先不要為了收權利金硬開倉"

    if holding.objective == "保守收租":
        if dte <= 10:
            return "不優先 7 天", "保守收租不建議常做 7 天，Gamma 變化快，除非只是小部位試單"
        if dte <= 24:
            return "14 天可做但偏短", "可以做，但保守者更適合 21–45 天，避免太頻繁管理"
        return "30 天尤佳", "保守收租優先選約 25–45 天，權利金、時間價值與管理壓力較平衡"

    if holding.objective == "平衡收租":
        if dte <= 10:
            return "7 天可做", "適合短週期收息，但要更常檢查，且不能用市價單"
        if dte <= 24:
            return "14 天尤佳", "平衡收租通常優先選 14–24 天，權利金與上漲空間比較均衡"
        return "30 天可做", "適合不想頻繁管理的人，但資金與股票會被綁比較久"

    if holding.objective == "提高現金流":
        if dte <= 10:
            return "7 天尤佳", "提高現金流可優先看 7–10 天，但要接受較高 Gamma 與履約風險"
        if dte <= 24:
            return "14 天可做", "14 天可提高權利金穩定度，比每週硬做更容易管理"
        return "30 天不優先", "30 天也能做，但現金流週轉較慢，除非權利金明顯更好"

    if holding.objective == "本來就想賣出":
        if dte <= 10:
            return "7 天可做", "你本來就想賣出時，短週期接近目標價的 call 可當成計畫內出場"
        if dte <= 24:
            return "14 天尤佳", "14 天通常能兼顧權利金與被履約機率，適合計畫性賣出"
        return "30 天可做", "若目標是用較高價掛賣股票，一個月期可以換到較多權利金"

    return bucket, "依目前 DTE 對應的實務管理週期"


def _cadence(candidate: Candidate, holding: Holding) -> str:
    if candidate.event_score < 35:
        return "暫停；等財報或事件後再評估"
    if holding.objective == "保守收租":
        if candidate.dte < 14:
            return "這是短週期試單；保守者平常以 21–45 天為主"
        return "每 3–4 週檢查一次，符合條件才開倉"
    if holding.objective == "平衡收租":
        if candidate.dte <= 10:
            return "第 7–10 天可做週收息，但要接受較高 Gamma 風險"
        if candidate.dte <= 24:
            return "第 14–24 天尤佳；每 2 週檢查一次"
        return "第 25–35 天較穩定；每 3–4 週檢查一次"
    if holding.objective == "提高現金流":
        if candidate.dte <= 14:
            return "第 7–14 天尤佳；每週檢查，不要每週硬做"
        return "第 14–28 天可做；每 1–2 週檢查"
    if holding.objective == "本來就想賣出":
        return "第 7–35 天皆可；被履約是計畫內賣出"
    return "每 2–4 週檢查一次"


def _decision(score: float, event_score: float, liquidity_score: float, premium_yield: float, min_yield: float) -> str:
    if event_score < 35:
        return "暫停"
    if premium_yield < min_yield * 0.35:
        return "權利金偏低"
    if score >= 70 and liquidity_score >= 38:
        return "可做"
    if score >= 56:
        return "可做，但限價單"
    if score >= 44:
        return "觀察"
    return "暫不操作"


def _build_candidate(row: pd.Series, ticker: str, expiry: date, dte: int, spot: float, rv20: float, tech: float, tech_notes: list[str], event: float, event_notes: list[str], holding: Holding, ex_dividend: date | None = None, dividend_amount: float = 0.0) -> Candidate | None:
    strike = _to_float(row.get("strike", 0), 0)
    bid = _to_float(row.get("bid", 0), 0)
    ask = _to_float(row.get("ask", 0), 0)
    last = _to_float(row.get("lastPrice", 0), 0)
    if strike <= spot or strike <= 0:
        return None
    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    if mid <= 0:
        return None
    iv = _to_float(row.get("impliedVolatility", np.nan), np.nan)
    # Some providers return unusably tiny IV values (for example 0.004 = 0.4%).
    # For listed US equity options, that is usually stale/bad data, so fall back to RV.
    if not np.isfinite(iv) or iv <= 0 or iv < 0.08:
        iv = max(rv20, 0.25)
    if iv > 3.0:
        iv = iv / 100.0
    delta = call_delta(spot, strike, dte, iv)
    theta = call_theta_per_day(spot, strike, dte, iv)
    oi = _to_int(row.get("openInterest", 0), 0)
    volume = _to_int(row.get("volume", 0), 0)
    spread_pct = float((ask - bid) / mid) if bid > 0 and ask > 0 and mid > 0 else 0.50
    premium_yield = float(mid / spot)
    annualized_yield = float(premium_yield * 365 / max(dte, 1))
    upside = float((strike / spot) - 1)
    rules = PROFILE_RULES[holding.objective]
    d_low, d_high = rules["delta"]
    min_yield = rules["min_yield"]
    quality = float(QUALITY_BUCKET.get(ticker, 65 if ticker not in CORE_ETFS else 90))
    liq = _liquidity_score(oi, volume, spread_pct)
    prem = _premium_score(premium_yield, annualized_yield, min_yield)
    delta_fit = _score_range(delta, d_low, d_high)
    dte_low, dte_high = rules["dte"]
    dte_fit = _score_range(dte, dte_low, dte_high)
    assign, assign_notes = _assignment_score(spot, strike, holding, delta)
    div_score, div_notes, early_assignment_risk = _dividend_risk_score(expiry, ex_dividend, dividend_amount, spot, strike, mid)
    iv_edge = float(iv / max(rv20, 0.05))
    iv_score = 70 if iv_edge >= 1.0 else 45
    if iv_edge >= 1.25:
        iv_score = 88
    if iv_edge >= 1.70:
        iv_score = 78
    premium_component = 0.75 * prem + 0.25 * iv_score
    total = (
        0.16 * quality + 0.16 * liq + 0.18 * premium_component + 0.12 * tech +
        0.12 * event + 0.10 * div_score + 0.10 * assign + 0.06 * ((delta_fit + dte_fit) / 2)
    )
    total = max(0.0, min(100.0, float(total)))
    decision = _decision(total, event, liq, premium_yield, min_yield)
    rationale = []
    if premium_yield >= min_yield:
        rationale.append(f"單次權利金約 {premium_yield:.2%}，達到此目標的最低門檻")
    else:
        rationale.append(f"單次權利金約 {premium_yield:.2%}，低於此目標的理想門檻")
    rationale.append(f"Delta 約 {delta:.2f}，目標區間為 {d_low:.2f}–{d_high:.2f}")
    rationale.append(f"DTE {dte} 天，目標週期為 {dte_low}–{dte_high} 天")
    rationale.extend(tech_notes[:2])
    rationale.extend(event_notes[:1])
    rationale.extend(div_notes[:1])
    warnings = []
    if spread_pct > 0.15:
        warnings.append("bid-ask spread 偏大：不要市價單，請用接近 mid 的 limit order 試單")
    if oi < 100 and volume < 50:
        warnings.append("OI / volume 偏低：可以學習判斷，但不適合固定機械化操作")
    if annualized_yield > 0.60:
        warnings.append("年化權利金很高，通常代表股價風險也很高，不要只看年化數字")
    if early_assignment_risk in {"中", "高"}:
        warnings.extend(div_notes[:1])
    warnings.extend(assign_notes)
    c = Candidate(
        ticker=ticker, expiry=expiry, dte=dte, strike=strike, last_price=spot, bid=bid, ask=ask, mid=mid,
        premium=mid * 100, premium_yield=premium_yield, annualized_yield=annualized_yield,
        upside_to_strike=upside, implied_volatility=iv, delta=delta, theta=theta, open_interest=oi,
        volume=volume, spread_pct=spread_pct, iv_edge=iv_edge, probability_assignment=max(0.0, min(1.0, delta)),
        breakeven_price=spot - mid, max_profit_if_called=(strike - spot + mid) * 100,
        max_profit_pct_if_called=(strike - spot + mid) / spot, ex_dividend_date=ex_dividend,
        dividend_amount=dividend_amount, early_assignment_risk=early_assignment_risk,
        quality_score=quality, liquidity_score=liq, premium_score=premium_component, technical_score=tech,
        event_score=event, assignment_score=assign, total_score=total, decision=decision, cadence="", action_cycle="",
        action_cycle_reason="", rationale=rationale, warnings=warnings
    )
    c.cadence = _cadence(c, holding)
    c.action_cycle, c.action_cycle_reason = _action_cycle(c, holding)
    return c


def analyze_ticker(holding: Holding, max_expiries: int = 8) -> list[Candidate]:
    ticker = normalize_ticker(holding.ticker)
    holding.ticker = ticker
    spot = _to_float(get_price(ticker), 0)
    if spot <= 0:
        raise MarketDataError(f"抓不到 {ticker} 可用股價資料")
    rv20 = max(_to_float(realized_volatility(ticker, 20), 0.30), 0.05)
    tech, tech_notes = trend_score(ticker)
    earnings = next_earnings_date(ticker)
    ex_dividend, dividend_amount = next_ex_dividend_info(ticker)
    candidates: list[Candidate] = []
    today = date.today()
    expirations = get_expirations(ticker)
    rules = PROFILE_RULES[holding.objective]
    broad_low = max(5, rules["dte"][0] - 7)
    broad_high = min(60, rules["dte"][1] + 14)
    useful_expiries = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if broad_low <= dte <= broad_high:
            useful_expiries.append((exp, exp_date, dte))
    for exp, exp_date, dte in useful_expiries[:max_expiries]:
        event, event_notes = _event_score(exp_date, earnings)
        try:
            calls = get_calls(ticker, exp)
        except Exception:
            continue
        calls = calls[calls["strike"] > spot].copy()
        if calls.empty:
            continue
        for _, row in calls.iterrows():
            candidate = _build_candidate(row, ticker, exp_date, dte, spot, rv20, tech, tech_notes, event, event_notes, holding, ex_dividend, dividend_amount)
            if candidate is not None:
                candidates.append(candidate)
    def _rank(c: Candidate):
        rules = PROFILE_RULES[holding.objective]
        d_low, d_high = rules["delta"]
        t_low, t_high = rules["dte"]
        in_delta = d_low <= c.delta <= d_high
        in_dte = t_low <= c.dte <= t_high
        spread_ok = c.spread_pct <= 0.18
        activity_ok = (c.open_interest >= 50) or (c.volume >= 50)
        actionable = c.decision in {"可做", "可做，但限價單"}
        # Prefer a practical management cycle by objective instead of only mathematical score.
        cycle = _cycle_bucket(c.dte)
        preferred_cycle = {
            "保守收租": "30天週期",
            "平衡收租": "14天週期",
            "提高現金流": "7天週期",
            "本來就想賣出": "14天週期",
        }.get(holding.objective, "14天週期")
        cycle_fit = cycle == preferred_cycle or (holding.objective == "提高現金流" and cycle == "14天週期")
        # Covered call is an execution tool: prefer a clean bid/ask and visible activity over a mathematically high score.
        return (actionable, spread_ok, in_delta, cycle_fit, in_dte, activity_ok, c.total_score, c.premium_yield)
    candidates.sort(key=_rank, reverse=True)
    return candidates



def estimate_ticker_candidates(holding: Holding) -> list[Candidate]:
    """Create teaching/backup candidates when live option chains are unavailable.

    These are NOT executable quotes. They use current price + historical volatility +
    Black-Scholes to answer the user's practical question: should I look at 7, 14,
    or 30 day covered calls for this ticker? Formal execution still requires broker
    bid/ask, OI, volume and a real listed expiration.
    """
    ticker = normalize_ticker(holding.ticker)
    holding.ticker = ticker
    spot = _to_float(get_price(ticker), 0)
    if spot <= 0:
        raise MarketDataError(f"抓不到 {ticker} 可用股價資料，無法產生模型估算")
    rv20 = max(_to_float(realized_volatility(ticker, 20), 0.30), 0.12)
    iv = min(max(rv20 * 1.10, 0.18), 1.20)
    tech, tech_notes = trend_score(ticker)
    earnings = next_earnings_date(ticker)
    ex_dividend, dividend_amount = next_ex_dividend_info(ticker)
    rules = PROFILE_RULES[holding.objective]
    d_low, d_high = rules["delta"]
    if holding.objective == "保守收租":
        target_delta = min(max((d_low + d_high) / 2, 0.14), 0.20)
    elif holding.objective == "提高現金流":
        target_delta = min(max((d_low + d_high) / 2, 0.28), 0.38)
    elif holding.objective == "本來就想賣出":
        target_delta = min(max((d_low + d_high) / 2, 0.32), 0.48)
    else:
        target_delta = min(max((d_low + d_high) / 2, 0.18), 0.28)

    # Pick realistic strike increments.
    if spot < 25:
        step = 0.5
    elif spot < 100:
        step = 1.0
    elif spot < 250:
        step = 2.5
    else:
        step = 5.0

    candidates: list[Candidate] = []
    today = date.today()
    for dte in [7, 14, 30]:
        expiry = today + timedelta(days=dte)
        event, event_notes = _event_score(expiry, earnings)
        # Scan OTM strikes from about 0.5% to 15% above spot.
        best_strike = None
        best_gap = 999.0
        seen = set()
        for pct in np.linspace(0.005, 0.15, 90):
            raw = spot * (1 + float(pct))
            strike = round(round(raw / step) * step, 2)
            if strike <= spot or strike in seen:
                continue
            seen.add(strike)
            delta = call_delta(spot, strike, dte, iv)
            gap = abs(delta - target_delta)
            if gap < best_gap:
                best_gap = gap
                best_strike = strike
        if best_strike is None:
            continue
        mid = max(call_price(spot, best_strike, dte, iv), 0.01)
        # Model bid/ask with a moderate spread so users know this is not a quote.
        bid = max(mid * 0.94, 0.01)
        ask = max(mid * 1.06, bid + 0.01)
        row = pd.Series({
            "strike": best_strike,
            "bid": bid,
            "ask": ask,
            "lastPrice": mid,
            "impliedVolatility": iv,
            "openInterest": 0,
            "volume": 0,
        })
        c = _build_candidate(row, ticker, expiry, dte, spot, rv20, tech, tech_notes, event, event_notes, holding, ex_dividend, dividend_amount)
        if c is None:
            continue
        c.decision = "估算參考"
        c.total_score = max(min(c.total_score, 62.0), 56.0)
        c.warnings.insert(0, "這是模型估算，不是券商即時 option chain。正式下單前必須用券商確認到期日、strike、bid/ask、OI、volume。")
        c.rationale.insert(0, f"資料源未取得可用 option chain，系統用現價與 20 日波動率估算 {dte} 天週期，幫你判斷應該先研究 7 / 14 / 30 天哪個方向。")
        candidates.append(c)
    candidates.sort(key=lambda c: (c.action_cycle == "14 天尤佳", c.total_score, c.premium_yield), reverse=True)
    return candidates


def candidates_to_frame(candidates: Iterable[Candidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append({
            "Ticker": c.ticker,
            "決策": c.decision,
            "分數": round(c.total_score, 1),
            "建議週期": c.action_cycle,
            "建議賣出": f"第 {c.dte} 天到期 / {c.expiry.isoformat()} / ${c.strike:.2f} Call",
            "操作": f"Sell to Open {c.ticker} {c.expiry.isoformat()} ${c.strike:.2f} Call，限價約 ${c.mid:.2f}",
            "週期說明": c.action_cycle_reason,
            "管理提醒": c.cadence,
            "到期日": c.expiry.isoformat(),
            "DTE": c.dte,
            "現價": round(c.last_price, 2),
            "Strike": round(c.strike, 2),
            "Delta": round(c.delta, 2),
            "粗估履約機率": f"{c.probability_assignment:.0%}",
            "Mid": round(c.mid, 2),
            "每張權利金": round(c.premium, 0),
            "打平價": round(c.breakeven_price, 2),
            "被履約最大利潤/張": round(c.max_profit_if_called, 0),
            "單次權利金%": f"{c.premium_yield:.2%}",
            "年化%": f"{c.annualized_yield:.1%}",
            "上漲空間%": f"{c.upside_to_strike:.1%}",
            "IV": f"{c.implied_volatility:.1%}",
            "IV/HV": round(c.iv_edge, 2),
            "OI": c.open_interest,
            "Volume": c.volume,
            "Spread%": f"{c.spread_pct:.1%}",
            "除息風險": c.early_assignment_risk,
        })
    return pd.DataFrame(rows)
