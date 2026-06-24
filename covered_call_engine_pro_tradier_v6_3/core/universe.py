from __future__ import annotations

from dataclasses import dataclass

from .web_universe import web_universe_tickers


@dataclass(frozen=True)
class UniversePreset:
    name: str
    description: str
    tickers: list[str]


CORE_ETF_TICKERS = [
    "SPY","QQQ","DIA","IWM","VTI","VOO","IVV","SCHD","XLK","XLF","XLV","XLE","XLY","XLP","XLI","XLB","XLU","XLRE","XLC",
    "SMH","SOXX","ARKK","HYG","LQD","TLT","IEF","GLD","SLV","USO","XBI","KRE","IYT","ITA","XME","XRT","VNQ","EFA","EEM","FXI",
]

MEGA_LARGE_CAP_TICKERS = [
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","NVDA","AVGO","TSLA","BRK-B","JPM","V","MA","UNH","LLY","XOM","COST","WMT","HD","PG","NFLX","JNJ","ABBV","KO","PEP","BAC","CRM","ORCL","MRK","CVX","AMD","ADBE","MCD","CSCO","WFC","QCOM","TMO","INTU","AMAT","IBM","GE","AXP","DIS","NOW","CAT","TXN","UBER","PM","VZ","MS","NEE","RTX","GS","LOW","PFE","HON","UNP","SPGI","BLK","C","ISRG","BKNG","LRCX","SCHW","DE","SYK","MDT","ADI","TJX","AMGN","VRTX","GILD","REGN","PANW","PLD","MMC","CB","ADP","FI","ETN","MU","KLAC","BSX","SO","ELV","ICE","DUK","PGR","NKE","ZTS","EQIX","SHW","APH","CL","CME","MO","WM","USB","MCO","HCA","PH","BDX","AON","MAR","FDX","GM","F","T","ABT","COP","SLB","OXY","EOG","MPC","PSX",
]

LIQUID_GROWTH_TICKERS = [
    "AMD","NVDA","TSLA","NFLX","PANW","NOW","SHOP","UBER","ABNB","CRWD","SNOW","INTC","MU","QCOM","ADBE","CRM","AVGO","ARM","MRVL","DDOG","NET","MDB","ZS","TEAM","ROKU","SQ","PYPL","MELI","SE","BABA","PDD","JD","BIDU","LI","NIO","XPEV",
]

HIGH_PREMIUM_WATCH_TICKERS = [
    "TSLA","NVDA","AMD","PLTR","COIN","MSTR","SOFI","RIVN","HOOD","MARA","RIOT","SMCI","ARM","GME","AMC","CVNA","UPST","AFRM","AI","IONQ","RBLX","DKNG","LCID","NIO","BILI","SOUN","WULF","HIMS","CELH","ENVX",
]

DIVIDEND_VALUE_TICKERS = [
    "KO","PEP","PG","JNJ","ABBV","MRK","CVX","XOM","MCD","WMT","COST","HD","LOW","PM","MO","T","VZ","IBM","INTC","PFE","BMY","GILD","AMGN","O","SPG","DUK","SO","NEE","DOW","MMM","CAT","DE","UPS","RTX","LMT","NOC","USB","JPM","BAC","C","WFC","SCHW",
]

SECTOR_ROTATION_TICKERS = [
    "JPM","BAC","WFC","C","GS","MS","AXP","SCHW","BLK","BX","XOM","CVX","COP","SLB","OXY","EOG","MPC","PSX","LNG","FCX","NEM","AA","CLF","NUE","LIN","APD","SHW","BA","CAT","DE","GE","HON","RTX","LMT","NOC","UPS","FDX","UNP","DAL","UAL","AAL","CCL","RCL","MAR","HLT","BKNG","ABNB","DIS","CMCSA","TMUS","VZ","T",
]

PRESETS: dict[str, UniversePreset] = {
    "保守核心池": UniversePreset(
        "保守核心池",
        "大型 ETF 與成熟大型股，適合先找流動性好、比較不投機的 covered call 標的。",
        CORE_ETF_TICKERS[:20] + ["AAPL","MSFT","GOOGL","AMZN","META","JPM","V","MA","UNH","COST","WMT","PG","KO","PEP","MCD","HD","XOM","CVX"],
    ),
    "ETF 多元池": UniversePreset(
        "ETF 多元池",
        "跨指數、產業、債券、商品與海外 ETF；可降低單一公司財報跳空風險，但仍需注意 ETF 本身波動。",
        CORE_ETF_TICKERS,
    ),
    "大型股 100 池": UniversePreset(
        "大型股 100 池",
        "以美股大型高流動性公司為主，比原本自動池更多元，但掃描時間會較長。",
        MEGA_LARGE_CAP_TICKERS,
    ),
    "科技成長池": UniversePreset(
        "科技成長池",
        "科技與成長股為主，權利金通常較高，但股價波動與被履約風險也較高。",
        LIQUID_GROWTH_TICKERS,
    ),
    "股息價值池": UniversePreset(
        "股息價值池",
        "偏股息、價值、成熟產業標的；covered call 前仍要檢查除息日前提前履約風險。",
        DIVIDEND_VALUE_TICKERS,
    ),
    "產業輪動池": UniversePreset(
        "產業輪動池",
        "金融、能源、工業、旅遊、材料與通訊等產業，避免自動查找只集中在科技股。",
        SECTOR_ROTATION_TICKERS,
    ),
    "高權利金觀察池": UniversePreset(
        "高權利金觀察池",
        "探索高 IV / 高權利金標的。不代表推薦買進，必須更重視風險提醒與是否願意持有正股。",
        HIGH_PREMIUM_WATCH_TICKERS,
    ),
    "網路熱門期權池": UniversePreset(
        "網路熱門期權池",
        "根據公開的 most-active options / option-volume 觀察方向整理的高流動性候選清單。這不是即時推薦，而是避免 yfinance 批次抓 option chain 失敗時的穩定起點。",
        web_universe_tickers(),
    ),
    "綜合自動池": UniversePreset(
        "綜合自動池",
        "混合 ETF、大型股、流動成長股、價值股、產業股與少量高權利金標的，用來快速掃出今天值得研究的候選。",
        CORE_ETF_TICKERS + MEGA_LARGE_CAP_TICKERS + LIQUID_GROWTH_TICKERS + DIVIDEND_VALUE_TICKERS + SECTOR_ROTATION_TICKERS + HIGH_PREMIUM_WATCH_TICKERS,
    ),
}


def dedupe_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in tickers:
        ticker = raw.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            result.append(ticker)
    return result


def get_universe(preset_name: str, limit: int | None = None) -> list[str]:
    preset = PRESETS.get(preset_name, PRESETS["綜合自動池"])
    tickers = dedupe_tickers(preset.tickers)
    if limit is not None:
        return tickers[:limit]
    return tickers


def auto_preset_for_objective(objective: str) -> str:
    if objective == "保守收租":
        return "保守核心池"
    if objective == "提高現金流":
        return "網路熱門期權池"
    if objective == "本來就想賣出":
        return "科技成長池"
    return "網路熱門期權池"
