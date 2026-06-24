from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Holding:
    ticker: str
    shares: int = 100
    cost_basis: Optional[float] = None
    minimum_sell_price: Optional[float] = None
    objective: str = "平衡收租"


@dataclass
class Candidate:
    ticker: str
    expiry: date
    dte: int
    strike: float
    last_price: float
    bid: float
    ask: float
    mid: float
    premium: float
    premium_yield: float
    annualized_yield: float
    upside_to_strike: float
    implied_volatility: float
    delta: float
    theta: float
    open_interest: int
    volume: int
    spread_pct: float
    iv_edge: float
    probability_assignment: float
    breakeven_price: float
    max_profit_if_called: float
    max_profit_pct_if_called: float
    quality_score: float
    liquidity_score: float
    premium_score: float
    technical_score: float
    event_score: float
    assignment_score: float
    total_score: float
    decision: str
    cadence: str
    action_cycle: str
    action_cycle_reason: str
    rationale: list[str]
    warnings: list[str]
    ex_dividend_date: Optional[date] = None
    dividend_amount: float = 0.0
    early_assignment_risk: str = ""
