from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class WebUniverseItem:
    ticker: str
    name: str
    category: str
    role: str
    preferred_cycle: str
    why: str
    cautions: str


# Curated from public option-volume / most-active references and common liquid-option covered-call underlyings.
# This is a starting universe, not a buy list. Users should still verify quote, option chain, ex-dividend,
# earnings, bid/ask, OI and volume in their broker.
WEB_COVERED_CALL_UNIVERSE: list[WebUniverseItem] = [
    WebUniverseItem("SPY", "SPDR S&P 500 ETF", "ETF核心", "核心收租/市場曝險", "14天或30天", "選擇權流動性極佳，適合作為 covered call 教學與核心池起點。", "ETF仍有市場下跌風險；注意除息與早期履約。"),
    WebUniverseItem("QQQ", "Invesco QQQ Trust", "ETF核心", "科技指數收租", "14天", "流動性佳、權利金通常高於SPY，適合平衡收租。", "科技權重高，漲勢中容易被履約。"),
    WebUniverseItem("IWM", "Russell 2000 ETF", "ETF核心", "小型股波動收租", "14天", "波動度較高，權利金通常比大型指數ETF更有感。", "小型股波動大，回撤可能比權利金大。"),
    WebUniverseItem("DIA", "Dow Jones ETF", "ETF核心", "成熟藍籌收租", "30天", "成分股較成熟，適合保守/平衡池比較。", "權利金可能不如QQQ/IWM。"),
    WebUniverseItem("VTI", "Total US Market ETF", "ETF核心", "全市場核心", "30天", "適合長期核心持股做保守 covered call。", "選擇權流動性通常弱於SPY。"),
    WebUniverseItem("VOO", "Vanguard S&P 500 ETF", "ETF核心", "S&P 500核心", "30天", "核心資產屬性強，可做低頻保守收租研究。", "流動性通常弱於SPY；若要交易常用SPY替代觀察。"),
    WebUniverseItem("XLK", "Technology Select Sector SPDR", "產業ETF", "科技產業", "14天", "科技產業ETF，避免單一公司財報風險。", "仍受科技類股波動影響。"),
    WebUniverseItem("SMH", "VanEck Semiconductor ETF", "產業ETF", "半導體產業", "14天", "半導體波動高，權利金較有感。", "波動與缺口風險高，適合較積極使用者。"),
    WebUniverseItem("SOXX", "iShares Semiconductor ETF", "產業ETF", "半導體產業", "14天", "與SMH類似，可用於半導體covered call候選。", "注意spread與流動性。"),
    WebUniverseItem("XLF", "Financial Select Sector SPDR", "產業ETF", "金融產業", "30天", "金融產業ETF，權利金較溫和，適合分散池。", "受利率與金融事件影響。"),
    WebUniverseItem("XLE", "Energy Select Sector SPDR", "產業ETF", "能源產業", "14天或30天", "能源波動高於防禦股，權利金較有感。", "油價/地緣政治會造成跳空。"),
    WebUniverseItem("XLV", "Health Care Select Sector SPDR", "產業ETF", "醫療產業", "30天", "相對防禦，適合保守池分散。", "權利金可能較低。"),
    WebUniverseItem("XLP", "Consumer Staples SPDR", "產業ETF", "民生消費", "30天", "低波動防禦產業，可搭配保守收租。", "權利金偏低是常態。"),
    WebUniverseItem("XLY", "Consumer Discretionary SPDR", "產業ETF", "非必需消費", "14天或30天", "波動比XLP高，適合平衡池。", "受景氣與龍頭股影響。"),
    WebUniverseItem("TLT", "20+ Year Treasury Bond ETF", "債券ETF", "利率曝險收租", "30天", "有週期性波動，可做非股票類曝險觀察。", "利率波動可能造成長債大幅變動。"),
    WebUniverseItem("GLD", "Gold ETF", "商品ETF", "黃金曝險", "30天", "商品曝險與股票相關性不同，可作候選池分散。", "商品ETF權利金與流動性要逐筆確認。"),

    WebUniverseItem("AAPL", "Apple", "大型科技", "核心個股收租", "14天", "大型股、期權活躍，適合示範7/14/30比較。", "財報前不建議硬賣；大漲時容易被履約。"),
    WebUniverseItem("MSFT", "Microsoft", "大型科技", "核心個股收租", "14天或30天", "大型高品質公司，covered call常見候選。", "權利金通常不如高波動股；注意財報。"),
    WebUniverseItem("NVDA", "NVIDIA", "大型科技", "高權利金/高波動", "7天或14天", "期權活躍且權利金高，適合積極現金流觀察。", "波動很大，可能快速漲破strike或下跌。"),
    WebUniverseItem("TSLA", "Tesla", "大型科技", "高權利金/高波動", "7天或14天", "高IV與活躍期權，適合高現金流策略研究。", "不是保守收租標的；股價跳動可能遠超權利金。"),
    WebUniverseItem("AMD", "Advanced Micro Devices", "半導體", "高權利金/成長", "7天或14天", "半導體高流動個股，權利金通常有感。", "波動與新聞風險高。"),
    WebUniverseItem("AVGO", "Broadcom", "半導體", "大型半導體", "14天", "大型半導體權利金較有感，可搭配NVDA/AMD比較。", "股價高，合約名目金額大。"),
    WebUniverseItem("AMZN", "Amazon", "大型科技", "成長核心", "14天", "大型高流動個股，適合平衡收租。", "財報與雲端/消費數據影響大。"),
    WebUniverseItem("META", "Meta Platforms", "大型科技", "成長核心", "14天", "選擇權活躍，權利金較大型防禦股有感。", "財報跳空與監管風險。"),
    WebUniverseItem("GOOGL", "Alphabet Class A", "大型科技", "成長核心", "14天或30天", "大型高流動個股，適合平衡池。", "AI/廣告/反壟斷新聞會影響波動。"),
    WebUniverseItem("NFLX", "Netflix", "大型科技", "高波動成長", "7天或14天", "期權活躍、權利金有感。", "財報與訂戶數據可能跳空。"),
    WebUniverseItem("PLTR", "Palantir", "高權利金", "高IV觀察", "7天", "期權活躍，權利金高。", "投機屬性強，不適合只為權利金買進正股。"),
    WebUniverseItem("COIN", "Coinbase", "高權利金", "加密曝險", "7天", "高波動、高權利金，適合積極策略研究。", "受加密貨幣劇烈波動影響。"),
    WebUniverseItem("MSTR", "MicroStrategy", "高權利金", "Bitcoin proxy", "7天", "權利金極高，期權活躍。", "風險極高，covered call無法保護大跌。"),
    WebUniverseItem("HOOD", "Robinhood", "高權利金", "金融科技", "7天或14天", "波動與期權活躍度較高。", "個股風險高，適合觀察不適合盲目持有。"),

    WebUniverseItem("JPM", "JPMorgan Chase", "金融/價值", "成熟大型股", "30天", "大型金融股，適合價值/股息池。", "利率、信用風險、財報需注意。"),
    WebUniverseItem("BAC", "Bank of America", "金融/價值", "大型銀行", "14天或30天", "期權流動性通常不差，權利金較溫和。", "利率與金融風險。"),
    WebUniverseItem("XOM", "Exxon Mobil", "能源/價值", "股息+能源", "14天或30天", "成熟能源股，波動比防禦股高。", "除息與油價風險。"),
    WebUniverseItem("CVX", "Chevron", "能源/價值", "股息+能源", "30天", "大型能源股，適合平衡/保守比較。", "油價與除息早履約風險。"),
    WebUniverseItem("KO", "Coca-Cola", "防禦/股息", "保守收租", "30天", "低波動成熟股，可做防禦型covered call教學。", "權利金偏低；除息日前檢查早履約。"),
    WebUniverseItem("PEP", "PepsiCo", "防禦/股息", "保守收租", "30天", "成熟防禦股，適合保守池。", "權利金偏低；除息前風險。"),
    WebUniverseItem("PG", "Procter & Gamble", "防禦/股息", "保守收租", "30天", "低波動股息股，適合保守收租比較。", "權利金低，交易前看spread。"),
    WebUniverseItem("JNJ", "Johnson & Johnson", "防禦/股息", "保守收租", "30天", "成熟醫療股息股。", "權利金通常不高；注意公司事件。"),
    WebUniverseItem("MCD", "McDonald's", "防禦/消費", "平衡/保守", "30天", "成熟大型消費股，可做低頻covered call。", "股價高、權利金需逐筆確認。"),
    WebUniverseItem("WMT", "Walmart", "防禦/消費", "保守收租", "30天", "低波動成熟大型股。", "權利金可能偏低。"),
    WebUniverseItem("COST", "Costco", "防禦/消費", "高品質大型股", "30天", "高品質但股價高，適合小心操作。", "名目金額高；財報前注意。"),
    WebUniverseItem("HD", "Home Depot", "消費/價值", "平衡收租", "30天", "成熟大型股，適合低頻平衡池。", "受房市/利率影響。"),
]


def web_universe_frame() -> pd.DataFrame:
    return pd.DataFrame([item.__dict__ for item in WEB_COVERED_CALL_UNIVERSE])


def web_universe_tickers(category: str | None = None, limit: int | None = None) -> list[str]:
    items = WEB_COVERED_CALL_UNIVERSE
    if category and category != "全部":
        items = [x for x in items if x.category == category]
    tickers = [x.ticker for x in items]
    if limit is not None:
        return tickers[:limit]
    return tickers


def web_categories() -> list[str]:
    cats = []
    for item in WEB_COVERED_CALL_UNIVERSE:
        if item.category not in cats:
            cats.append(item.category)
    return ["全部"] + cats
