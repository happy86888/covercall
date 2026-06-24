from __future__ import annotations

import math

RISK_FREE_RATE = 0.045
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _safe(value: float, default: float) -> float:
    try:
        if value is None or math.isnan(value) or math.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


def _norm_cdf(x: float) -> float:
    """Standard normal CDF without scipy, for easier Streamlit deployment."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF without scipy, for easier Streamlit deployment."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(spot: float, strike: float, dte: int, iv: float, r: float = RISK_FREE_RATE) -> tuple[float, float, float, float]:
    spot = max(_safe(spot, 0.0), 0.01)
    strike = max(_safe(strike, spot), 0.01)
    t = max(int(_safe(dte, 1)), 1) / 365.0
    sigma = max(_safe(iv, 0.30), 0.01)
    root_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    return d1, d2, t, sigma


def call_delta(spot: float, strike: float, dte: int, iv: float, r: float = RISK_FREE_RATE) -> float:
    d1, _, _, _ = _d1_d2(spot, strike, dte, iv, r)
    return float(_norm_cdf(d1))


def call_price(spot: float, strike: float, dte: int, iv: float, r: float = RISK_FREE_RATE) -> float:
    """Black-Scholes call price for fallback/teaching estimates. Not a live quote."""
    spot = max(_safe(spot, 0.0), 0.01)
    strike = max(_safe(strike, spot), 0.01)
    d1, d2, t, _ = _d1_d2(spot, strike, dte, iv, r)
    return float(spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2))


def call_theta_per_day(spot: float, strike: float, dte: int, iv: float, r: float = RISK_FREE_RATE) -> float:
    spot = max(_safe(spot, 0.0), 0.01)
    strike = max(_safe(strike, spot), 0.01)
    d1, d2, t, sigma = _d1_d2(spot, strike, dte, iv, r)
    theta_year = -spot * _norm_pdf(d1) * sigma / (2 * math.sqrt(t)) - r * strike * math.exp(-r * t) * _norm_cdf(d2)
    return float(theta_year / 365.0)
