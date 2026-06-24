from __future__ import annotations

import io
from dataclasses import asdict

import pandas as pd
import streamlit as st

from core.data import MarketDataError, sample_watchlist
try:
    from core.data import DATA_SOURCE_LABEL
except Exception:
    DATA_SOURCE_LABEL = "yfinance"
from core.engine import PROFILE_RULES, analyze_ticker, estimate_ticker_candidates, candidates_to_frame
from core.backtest import model_backtest_covered_call, summarize_backtests
from core.models import Holding
from core.universe import PRESETS, auto_preset_for_objective, get_universe
from core.web_universe import web_universe_frame, web_categories, web_universe_tickers

st.set_page_config(
    page_title="Covered Call Decision Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root { --gold:#A87932; --ink:#27231F; --muted:#766D63; --paper:#FFFDF8; --bg:#F8F3EA; --line:#E8DCC8; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #fffaf1; border-right: 1px solid var(--line); }
h1, h2, h3 { letter-spacing: -0.02em; color: var(--ink); }
.hero { padding: 26px 30px; border-radius: 24px; border:1px solid var(--line); background: linear-gradient(135deg,#fffdf8,#f2e4cd); margin-bottom: 18px; }
.hero h1 { margin: 0 0 8px 0; font-size: 2.1rem; }
.hero p { color: var(--muted); font-size: 1.02rem; margin:0; line-height:1.65; }
.card { border:1px solid var(--line); border-radius:22px; background: var(--paper); padding:22px; box-shadow:0 8px 28px rgba(60,42,20,.06); margin: 12px 0; }
.card-title { font-size: 1.25rem; font-weight: 750; margin-bottom: 6px; }
.badge { display:inline-block; padding:6px 11px; border-radius:999px; font-size:.86rem; font-weight:700; margin-right:6px; border:1px solid var(--line); }
.badge-good { background:#eef7ee; color:#266334; }
.badge-mid { background:#fff4d7; color:#7b5713; }
.badge-bad { background:#fdeaea; color:#8c2c2c; }
.metric-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin-top:14px; }
.metric-tile { background:#fffaf1; border:1px solid var(--line); border-radius:16px; padding:14px; }
.metric-label { color:var(--muted); font-size:.82rem; }
.metric-value { font-size:1.25rem; font-weight:800; margin-top:4px; }
.small { color:var(--muted); font-size:.9rem; line-height:1.6; }
.warning { border-left:4px solid #b84d3b; background:#fff6f4; padding:12px 14px; border-radius:12px; margin-top:10px; }
.reason { border-left:4px solid var(--gold); background:#fffaf1; padding:12px 14px; border-radius:12px; margin-top:10px; }
.action-plan { border-left:5px solid #1f7a4f; background:#eef8f0; padding:14px 16px; border-radius:14px; margin:14px 0; font-size:1rem; line-height:1.65; }
.operation { border-left:5px solid #51728d; background:#f1f7fb; padding:14px 16px; border-radius:14px; margin:14px 0; font-size:1rem; line-height:1.65; }
.cycle-box { border-left:5px solid #a87932; background:#fffaf1; padding:14px 16px; border-radius:14px; margin:14px 0; font-size:1rem; line-height:1.65; }
.footer-note { color:var(--muted); font-size:.86rem; line-height:1.6; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def decision_badge(decision: str) -> str:
    if decision == "可做":
        cls = "badge-good"
    elif decision in ["可做，但限價單", "觀察", "權利金偏低", "估算參考"]:
        cls = "badge-mid"
    else:
        cls = "badge-bad"
    return f'<span class="badge {cls}">{decision}</span>'


def render_candidate_card(c, title="最佳候選", holding=None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">{title}：{c.ticker} {c.expiry.isoformat()} ${c.strike:.2f} Call</div>', unsafe_allow_html=True)
    st.markdown(decision_badge(c.decision) + f'<span class="badge">分數 {c.total_score:.1f}/100</span><span class="badge">建議週期：{c.action_cycle}</span>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-tile"><div class="metric-label">現價</div><div class="metric-value">${c.last_price:.2f}</div></div>
            <div class="metric-tile"><div class="metric-label">Delta</div><div class="metric-value">{c.delta:.2f}</div></div>
            <div class="metric-tile"><div class="metric-label">單次權利金</div><div class="metric-value">{c.premium_yield:.2%}</div></div>
            <div class="metric-tile"><div class="metric-label">每張約收</div><div class="metric-value">${c.premium:,.0f}</div></div>
            <div class="metric-tile"><div class="metric-label">DTE</div><div class="metric-value">{c.dte} 天</div></div>
            <div class="metric-tile"><div class="metric-label">上漲空間</div><div class="metric-value">{c.upside_to_strike:.1%}</div></div>
            <div class="metric-tile"><div class="metric-label">Spread</div><div class="metric-value">{c.spread_pct:.1%}</div></div>
            <div class="metric-tile"><div class="metric-label">OI / Vol</div><div class="metric-value">{c.open_interest:,} / {c.volume:,}</div></div>
            <div class="metric-tile"><div class="metric-label">粗估履約機率</div><div class="metric-value">{c.probability_assignment:.0%}</div></div>
            <div class="metric-tile"><div class="metric-label">打平價</div><div class="metric-value">${c.breakeven_price:.2f}</div></div>
            <div class="metric-tile"><div class="metric-label">被履約最大利潤/張</div><div class="metric-value">${c.max_profit_if_called:,.0f}</div></div>
            <div class="metric-tile"><div class="metric-label">除息早履約風險</div><div class="metric-value">{c.early_assignment_risk}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    shares_owned = int(holding.shares if holding else 100)
    contracts = int(shares_owned // 100)
    if contracts <= 0:
        st.markdown('<div class="warning"><b>持股不足：</b><br>Covered call 需要每 100 股對應 1 張 call。你目前輸入的股數不足 100 股，系統只做學習判斷，不建議直接下單。</div>', unsafe_allow_html=True)
        contracts = 1
    total_premium = c.premium * contracts
    bid_credit = c.bid * 100 * contracts if c.bid > 0 else total_premium
    ask_credit = c.ask * 100 * contracts if c.ask > 0 else total_premium
    effective_sale = c.strike + c.mid
    st.markdown(
        '<div class="cycle-box"><b>週期判斷：</b><br>'
        + f'我會把這筆歸類為 <b>{c.action_cycle}</b>。{c.action_cycle_reason}<br>'
        + f'實際合約是第 {c.dte} 天到期；不是每週硬做，而是有符合條件才開倉。'
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="action-plan"><b>本次建議：</b><br>'
        + f'賣出「第 {c.dte} 天到期」的 {c.ticker} ${c.strike:.2f} Call，到期日 {c.expiry.isoformat()}。<br>'
        + f'若你有 {shares_owned:,} 股，建議賣 {int(shares_owned // 100)} 張；限價先掛 ${c.mid:.2f} 附近。<br>'
        + f'預估可收約 ${total_premium:,.0f}（合理區間約 ${bid_credit:,.0f}–${ask_credit:,.0f}）。若被履約，等同約 ${effective_sale:.2f} 賣出正股。'
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="operation"><b>實際操作教學：</b><br>'
        + f'• 下單方向：Sell to Open / 賣出開倉 {contracts} 張 {c.ticker} {c.expiry.isoformat()} ${c.strike:.2f} Call。<br>'
        + f'• 價格：不要用市價單；先掛 limit ${c.mid:.2f}，成交不了再往 bid 小幅讓價。<br>'
        + f'• 你的收入來自權利金，不是股息；這筆權利金會立刻入帳，但代價是 ${c.strike:.2f} 以上的上漲空間被賣掉。<br>'
        + '• 管理：若權利金先賺到 50%–70%，可考慮 Buy to Close；若股價接近 strike 或 Delta 升到 0.50 以上，要決定接受履約或往後/往上 roll。'
        + '</div>',
        unsafe_allow_html=True,
    )
    ex_div_text = c.ex_dividend_date.isoformat() if c.ex_dividend_date else "未偵測"
    st.markdown(
        '<div class="operation"><b>出場 / Roll / 除息檢查：</b><br>'
        + f'• 獲利了結：若 call 從 ${c.mid:.2f} 跌到約 ${c.mid * 0.50:.2f} 以下，可考慮 Buy to Close。<br>'
        + f'• 接近履約：若股價接近 ${c.strike:.2f} 或 Delta 升到 0.50 以上，請決定接受履約或 roll out/up。<br>'
        + f'• 除息檢查：下一次除息日 {ex_div_text}，預估股息 ${c.dividend_amount:.2f}；若短 call 變價內且時間價值低於股息，早期履約風險上升。<br>'
        + '• 稅務 / 成本：被履約可能實現股票資本利得或損失；手續費、稅務與匯率請用券商資料確認。'
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="reason"><b>為什麼：</b><br>' + '<br>'.join([f"• {x}" for x in c.rationale[:5]]) + '</div>', unsafe_allow_html=True)
    if c.warnings:
        st.markdown('<div class="warning"><b>風險提醒：</b><br>' + '<br>'.join([f"• {x}" for x in c.warnings[:5]]) + '</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def run_analysis(holding: Holding, limit: int = 50):
    with st.spinner(f"正在分析 {holding.ticker.upper()} 的股價、到期日與 call option chain..."):
        candidates = analyze_ticker(holding)
    if not candidates:
        st.warning("沒有抓到券商 / yfinance 的即時 option chain。以下改用模型估算回答 7 / 14 / 30 天哪個週期值得研究；正式下單仍要回券商確認 bid/ask、OI、volume。")
        try:
            candidates = estimate_ticker_candidates(holding)
        except Exception as exc:
            st.error(f"連模型估算也失敗：{exc}")
            return []
    return candidates[:limit]


def cycle_summary(candidates):
    actionable = [c for c in candidates if c.decision in ["可做", "可做，但限價單", "估算參考"]]
    buckets = {"7天可做": [], "14天可做": [], "30天可做": [], "其他/暫停": []}
    for c in actionable:
        label = c.action_cycle
        if "7" in label:
            buckets["7天可做"].append(c)
        elif "14" in label:
            buckets["14天可做"].append(c)
        elif "30" in label:
            buckets["30天可做"].append(c)
        else:
            buckets["其他/暫停"].append(c)
    return buckets

def render_cycle_metrics(candidates):
    buckets = cycle_summary(candidates)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("7 天可做", len(buckets["7天可做"]))
    c2.metric("14 天可做", len(buckets["14天可做"]))
    c3.metric("30 天可做", len(buckets["30天可做"]))
    c4.metric("其他 / 暫停", len(buckets["其他/暫停"]))
    notes = []
    for name in ["7天可做", "14天可做", "30天可做"]:
        if buckets[name]:
            best = sorted(buckets[name], key=lambda x: x.total_score, reverse=True)[0]
            notes.append(f"{name}首選：{best.ticker} 第 {best.dte} 天 ${best.strike:.2f} Call，限價約 ${best.mid:.2f}")
    if notes:
        st.info("；".join(notes))


def render_cycle_comparison(candidates):
    """Show the best candidate for each practical cycle: 7/14/30 days."""
    if not candidates:
        return
    buckets = {"7天": [], "14天": [], "30天": []}
    for c in candidates:
        label = c.action_cycle
        if "7" in label:
            buckets["7天"].append(c)
        elif "14" in label:
            buckets["14天"].append(c)
        elif "30" in label:
            buckets["30天"].append(c)
    rows = []
    for name, items in buckets.items():
        if not items:
            rows.append({"週期": name, "系統判斷": "沒有合適候選", "建議合約": "-", "限價": "-", "權利金/張": "-", "重點": "目前到期日或資料中沒有符合此週期的 OTM call"})
            continue
        best = sorted(items, key=lambda x: (x.decision in ["可做", "可做，但限價單"], x.total_score), reverse=True)[0]
        rows.append({
            "週期": name,
            "系統判斷": best.action_cycle,
            "建議合約": f"第 {best.dte} 天 / {best.expiry.isoformat()} / ${best.strike:.2f} Call",
            "限價": f"${best.mid:.2f}",
            "權利金/張": f"${best.premium:,.0f}",
            "重點": best.action_cycle_reason,
        })
    st.markdown("### 7 / 14 / 30 天週期比較")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


with st.sidebar:
    st.markdown("### Covered Call Decision Engine")
    st.caption("版本：Pro v6.3 / Web Universe + Risk Guardrails｜資料源：" + DATA_SOURCE_LABEL)
    st.caption("不是追高權利金工具，而是判斷：現在該不該賣 call、賣哪個週期、用什麼 Delta。")
    st.divider()
    st.markdown("#### 操作目標")
    objective = st.selectbox(
        "你這次賣 covered call 的目的",
        list(PROFILE_RULES.keys()),
        index=1,
        help="不同目標會改變 Delta、DTE、權利金門檻與被履約容忍度。",
    )
    st.markdown("#### 全域設定")
    max_rows = st.slider("結果顯示筆數", 5, 100, 25, 5)
    st.divider()
    st.markdown('<div class="footer-note">資料源使用 yfinance，適合研究與 MVP。若要更穩定，建議升級串接 Tradier / Polygon / IBKR。正式交易前請用券商報價確認 bid/ask、成交量、財報、除息與稅務影響。</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>Covered Call Decision Engine</h1>
        <p>輸入持股、觀察名單，或讓系統自動掃描候選池。系統會直接告訴你：建議賣第幾天到期、哪個 strike、賣幾張、限價掛多少、預估收多少權利金，以及之後該 Buy to Close、Roll，還是接受履約。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_tab, list_tab, auto_tab, guide_tab, scan_tab, backtest_tab, manual_tab, rules_tab = st.tabs(["我已有持股", "網路名單", "自動查找", "如何操作", "找適合標的", "策略回測", "手動分析", "規則說明"])

with main_tab:
    st.subheader("我已持有股票，想判斷現在該不該賣 call")
    col1, col2, col3, col4 = st.columns([1.1, 1, 1, 1])
    with col1:
        ticker = st.text_input("股票代號", value="AAPL", placeholder="例如 AAPL, SPY, QQQ")
    with col2:
        shares = st.number_input("持股股數", min_value=0, value=100, step=100)
    with col3:
        cost_basis = st.number_input("平均成本，可留 0", min_value=0.0, value=0.0, step=1.0)
    with col4:
        min_sell = st.number_input("最低願意賣出價，可留 0", min_value=0.0, value=0.0, step=1.0)
    if st.button("產生 Covered Call 建議", type="primary", use_container_width=True):
        holding = Holding(
            ticker=ticker,
            shares=int(shares),
            cost_basis=cost_basis or None,
            minimum_sell_price=min_sell or None,
            objective=objective,
        )
        try:
            candidates = run_analysis(holding, max_rows)
            if candidates:
                render_candidate_card(candidates[0], "系統首選", holding)
                render_cycle_comparison(candidates)
                render_cycle_metrics(candidates)
                st.markdown("### 其他候選合約")
                df = candidates_to_frame(candidates)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "下載 CSV",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{ticker.upper()}_covered_call_candidates.csv",
                    mime="text/csv",
                )
        except MarketDataError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"分析失敗：{e}")


with list_tab:
    st.subheader("網路熱門期權名單：先從高流動性候選池開始")
    st.info("這裡不是即時報價掃描，而是根據公開的 most-active options / option-volume 方向整理的 covered call 候選宇宙。用途是：先有一份穩定名單，再挑 5–15 檔進券商或 Tradier 做真實 option chain 檢查，避免 yfinance 批次掃描一直被 rate limit。")
    df_web = web_universe_frame()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        category = st.selectbox("分類", web_categories(), index=0)
    with col2:
        cycle_filter = st.selectbox("優先週期", ["全部", "7天", "14天", "30天"], index=0)
    with col3:
        max_list_rows = st.slider("最多顯示", 10, 80, 45, 5)

    view = df_web.copy()
    if category != "全部":
        view = view[view["category"] == category]
    if cycle_filter != "全部":
        view = view[view["preferred_cycle"].str.contains(cycle_filter, regex=False)]
    view = view.head(max_list_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("名單檔數", len(view))
    c2.metric("ETF / 產業ETF", int(view["category"].str.contains("ETF", regex=False).sum()))
    c3.metric("大型/防禦/價值", int(view["category"].str.contains("大型|防禦|價值|金融|能源", regex=True).sum()))
    c4.metric("高權利金觀察", int(view["category"].str.contains("高權利金", regex=False).sum()))

    st.markdown("### 建議操作邏輯")
    st.markdown("""
- **不是看到名單就買股票**：covered call 第一原則是你本來就願意持有正股，且願意在 strike 價被賣掉。
- **7 天**：適合 NVDA、TSLA、PLTR、COIN 這類高波動標的，但要更常管理。
- **14 天**：適合 AAPL、QQQ、AMD、META 這類平衡收租；多數人可先從 14 天開始練。
- **30 天**：適合 SPY、DIA、KO、PG、JNJ、MCD 這類核心/防禦/股息標的。
- **正式下單前**：一定回券商確認 bid/ask、volume、open interest、財報日、除息日。
""")
    st.dataframe(
        view.rename(columns={
            "ticker": "Ticker", "name": "名稱", "category": "分類", "role": "用途",
            "preferred_cycle": "優先週期", "why": "為什麼列入", "cautions": "注意事項"
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "下載網路熱門期權候選 CSV",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="covered_call_web_universe.csv",
        mime="text/csv",
    )
    st.markdown("### 下一步怎麼用")
    st.markdown("""
1. 從上表挑 5–15 檔你真的願意持有的標的。
2. 貼到「找適合標的」或券商選擇權鏈。
3. 優先看系統建議週期：7 / 14 / 30 天。
4. 選 OTM Call，不要賣價內；Limit 價先抓 Bid/Ask 中間。
5. 若你的工具還沒接 Tradier token，少做大批量即時掃描，避免免費資料源 rate limit。
""")


with auto_tab:
    st.subheader("自動查找：系統主動掃描適合 covered call 的標的")
    suggested_preset = auto_preset_for_objective(objective)
    preset_names = list(PRESETS.keys())
    preset_index = preset_names.index(suggested_preset) if suggested_preset in preset_names else 0
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        preset_name = st.selectbox("自動候選池", preset_names, index=preset_index)
    with col2:
        universe_limit = st.slider("最多掃描檔數", 5, 200, 30, 5, help="yfinance 免費資料源較慢；第一次建議 20–50 檔，確認正常後再擴大。")
    with col3:
        min_auto_score = st.slider("最低顯示分數", 0, 90, 55, 5)
    preset = PRESETS[preset_name]
    tickers = get_universe(preset_name, universe_limit)
    preview = ", ".join(tickers[:35]) + (" ..." if len(tickers) > 35 else "")
    st.info(f"{preset.description} 目前將掃描 {len(tickers)} 檔：" + preview)
    st.caption("自動查找會逐檔抓 option chain，免費 yfinance 很容易 rate limit。若只是要先找名單，請先看「網路名單」頁；若要穩定自動掃描，請設定 Tradier token。")

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        auto_per_ticker_limit = st.slider("每檔保留候選", 1, 3, 1)
    with col_b:
        only_actionable = st.checkbox("只顯示可做 / 限價單", value=False)
    with col_c:
        include_watchlist = st.checkbox("合併我的觀察名單 / 自訂股票池", value=False)
    run_auto_backtest = st.checkbox("同時對可做清單做模型回測", value=False, help="會用歷史股價和 Black-Scholes 做近似模型回測；不是實際歷史 option 成交回測。")
    use_model_fallback = st.checkbox("資料源抓不到 option chain 時，用模型估算 7/14/30 天候選", value=True, help="估算結果不能直接下單，只用來判斷應該先研究哪個週期。正式交易請用券商 / Tradier 報價確認。")
    bt_top_n = st.slider("回測前幾檔", 3, 20, 8, 1) if run_auto_backtest else 0

    extra_text = ""
    if include_watchlist:
        extra_text = st.text_area("額外觀察名單或自訂股票池", value=", ".join(sample_watchlist()), height=100, help="可以貼上你自己的持股、學員名單、或券商匯出的候選股票代號。")

    if st.button("開始自動查找", type="primary", use_container_width=True):
        scan_tickers = list(tickers)
        if include_watchlist and extra_text.strip():
            raw_extra = extra_text.replace("\n", ",").split(",")
            for x in raw_extra:
                t = x.strip().upper()
                if t and t not in scan_tickers:
                    scan_tickers.append(t)
        all_candidates = []
        skipped = []
        progress = st.progress(0)
        status = st.empty()
        for i, t in enumerate(scan_tickers):
            status.write(f"自動查找中：{t}")
            try:
                holding = Holding(ticker=t, objective=objective)
                cs = analyze_ticker(holding)
                if not cs and use_model_fallback:
                    cs = estimate_ticker_candidates(holding)
                    skipped.append(f"{t}: 即時 option chain 無候選，已改用模型估算")
                kept = cs[:auto_per_ticker_limit]
                all_candidates.extend(kept)
            except Exception as exc:
                if use_model_fallback:
                    try:
                        holding = Holding(ticker=t, objective=objective)
                        cs = estimate_ticker_candidates(holding)
                        all_candidates.extend(cs[:auto_per_ticker_limit])
                        skipped.append(f"{t}: 即時資料失敗，已改用模型估算；原錯誤：{exc}")
                    except Exception as exc2:
                        skipped.append(f"{t}: {exc}; 模型估算也失敗：{exc2}")
                else:
                    skipped.append(f"{t}: {exc}")
            progress.progress((i + 1) / max(len(scan_tickers), 1))
        status.empty()

        if all_candidates:
            all_candidates.sort(key=lambda c: c.total_score, reverse=True)
            filtered = [c for c in all_candidates if c.total_score >= min_auto_score]
            if only_actionable:
                filtered = [c for c in filtered if c.decision in ["可做", "可做，但限價單"]]
            if filtered:
                # Keep one best contract per ticker for the headline list.
                best_by_ticker = {}
                for c in filtered:
                    if c.ticker not in best_by_ticker:
                        best_by_ticker[c.ticker] = c
                actionable = [c for c in best_by_ticker.values() if c.decision in ["可做", "可做，但限價單"]]
                watchable = [c for c in best_by_ticker.values() if c.decision in ["觀察", "權利金偏低"]]
                colm1, colm2, colm3, colm4 = st.columns(4)
                colm1.metric("本次掃描標的", len(scan_tickers))
                colm2.metric("有候選合約", len(best_by_ticker))
                colm3.metric("可做 / 限價單", len(actionable))
                colm4.metric("觀察清單", len(watchable))
                st.success(f"本次自動查找找到 {len(actionable)} 檔可做或可用限價單試單的 covered call 標的。")
                render_cycle_metrics(list(best_by_ticker.values()))

                render_candidate_card(actionable[0] if actionable else filtered[0], "自動查找首選")
                st.markdown("### 今日可做清單：每檔只列最佳一張")
                headline = sorted(best_by_ticker.values(), key=lambda c: (c.decision in ["可做", "可做，但限價單"], c.total_score), reverse=True)
                headline_df = candidates_to_frame(headline[:max_rows])
                st.dataframe(headline_df, use_container_width=True, hide_index=True)

                st.markdown("### 全部候選合約")
                df = candidates_to_frame(filtered[:max_rows])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "下載自動查找結果 CSV",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="covered_call_auto_finder.csv",
                    mime="text/csv",
                )
                if run_auto_backtest and actionable:
                    st.markdown("### 可做清單模型回測")
                    bt_tickers = [c.ticker for c in actionable[:bt_top_n]]
                    with st.spinner("正在做模型回測：" + ", ".join(bt_tickers)):
                        bt_df = summarize_backtests(bt_tickers, objective=objective, period="2y", limit=bt_top_n)
                    st.dataframe(bt_df, use_container_width=True, hide_index=True)
                    st.caption("回測說明：這是用歷史股價 + Black-Scholes 估算權利金的模型回測，不是真實歷史選擇權成交價。它適合判斷策略行為，不適合拿來宣稱實際績效。")
                st.markdown("### 如何解讀")
                st.markdown(
                    "自動查找只代表這些標的在目前資料下較值得研究，不代表應該直接買進正股。Covered call 的第一原則仍然是：你必須願意持有這檔股票，且願意在 strike 價被賣掉。"
                )
            else:
                st.warning("有抓到候選，但沒有通過目前的分數 / 決策篩選。可以降低最低分數，或改用較寬鬆的操作目標。")
        else:
            st.warning("沒有產生候選結果。這通常是資料源問題，不代表市場完全沒有 covered call 機會。請先設定 Tradier token，或勾選模型估算模式。")
        if skipped:
            st.markdown("### 資料源 / 估算紀錄")
            st.caption("如果大量標的都顯示 yfinance 抓不到 option chain，代表免費資料源不穩；正式自動查找建議使用 Tradier API。")
            with st.expander(f"查看掃描紀錄，共 {len(skipped)} 筆", expanded=True):
                st.write("\n".join(skipped[:200]))

with guide_tab:
    st.subheader("如何實際操作 Sell Covered Call")
    st.info("先確認你已經持有每 100 股正股，才賣 1 張 call。這個工具會告訴你研究哪個週期、哪個 strike、限價掛多少；最後下單仍以券商畫面的 bid/ask、成交量、持倉量為準。")
    st.markdown("""
### 1. 先選你的目的
- **保守收租**：不太想被履約，通常看 21–45 天、Delta 約 0.10–0.22。
- **平衡收租**：想收權利金，也保留上漲空間，通常看 14–35 天、Delta 約 0.16–0.30。
- **提高現金流**：想提高收入，通常看 7–28 天、Delta 約 0.22–0.42，但更容易被履約。
- **本來就想賣出**：你願意在 strike 價賣掉股票，Delta 可以更高。

### 2. 下單方式
在券商選到對應股票的 **CALLS**，找系統建議的到期日與 strike，方向選：

`Sell to Open / 賣出開倉`

張數：每 100 股賣 1 張。例如你有 300 股，就最多賣 3 張。

### 3. 不要用市價單
用 **Limit Order**。限價先掛在 bid/ask 的中間價附近：

`Limit ≈ (Bid + Ask) / 2`

成交不了再小幅往 bid 讓價；spread 太大就放棄。

### 4. 收到的是權利金，不是股息
你立刻收到權利金，但代價是：如果到期股價高於 strike，你的股票可能被用 strike 價賣掉。

### 5. 出場規則
- Call 價格跌到原本賣出價的 30%–50%：可考慮 **Buy to Close** 提前收工。
- 股價接近 strike 或 Delta 超過 0.50：決定接受履約，或 **Roll out/up**。
- 除息日前：如果 short call 已價內且時間價值低於股息，早期履約風險升高。

### 6. 7 / 14 / 30 天怎麼選
- **7 天**：收入週轉快，但 Gamma 風險大，要常看盤。
- **14 天**：多數人比較平衡，權利金與管理頻率較合理。
- **30 天**：比較穩，不用頻繁管理，但股票被綁較久。
""")
    st.warning("如果自動查找沒有結果，通常不是市場沒有機會，而是免費資料源 yfinance 暫時抓不到完整 option chain。要讓自動掃描穩定，建議設定 Tradier token；沒有 token 時可以先看單檔分析或模型估算。")


with scan_tab:
    st.subheader("掃描觀察名單，找比較適合 covered call 的標的")
    default_watchlist = ", ".join(sample_watchlist())
    watchlist_text = st.text_area("觀察名單，請用逗號或換行分隔", value=default_watchlist, height=110)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        per_ticker_limit = st.slider("每檔取前幾名", 1, 5, 1)
    with col_b:
        st.info("掃描多檔會逐檔抓 option chain，yfinance 很容易 Too Many Requests。建議先從「網路名單」挑 5–15 檔，再用券商/Tradier確認；不要一次掃太多。")
    if st.button("開始掃描", type="primary", use_container_width=True):
        raw = watchlist_text.replace("\n", ",").split(",")
        tickers = [x.strip().upper() for x in raw if x.strip()]
        all_candidates = []
        progress = st.progress(0)
        status = st.empty()
        for i, t in enumerate(tickers):
            status.write(f"分析中：{t}")
            try:
                holding = Holding(ticker=t, objective=objective)
                cs = analyze_ticker(holding)
                all_candidates.extend(cs[:per_ticker_limit])
            except Exception as exc:
                st.warning(f"{t} 跳過：{exc}")
            progress.progress((i + 1) / max(len(tickers), 1))
        status.empty()
        if all_candidates:
            all_candidates.sort(key=lambda c: c.total_score, reverse=True)
            best_by_ticker = {}
            for c in all_candidates:
                if c.ticker not in best_by_ticker:
                    best_by_ticker[c.ticker] = c
            actionable = [c for c in best_by_ticker.values() if c.decision in ["可做", "可做，但限價單"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("掃描檔數", len(tickers))
            c2.metric("有候選合約", len(best_by_ticker))
            c3.metric("可做 / 限價單", len(actionable))
            render_cycle_metrics(list(best_by_ticker.values()))
            render_candidate_card(actionable[0] if actionable else all_candidates[0], "全名單首選")
            df = candidates_to_frame(list(best_by_ticker.values())[:max_rows])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "下載掃描結果 CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="covered_call_watchlist_scan.csv",
                mime="text/csv",
            )
        else:
            st.warning("沒有產生任何候選結果。")


with backtest_tab:
    st.subheader("策略回測：先看這個 covered call 週期是否適合")
    st.info("此回測使用歷史股價 + Black-Scholes 模型估算權利金，不是真實歷史 option bid/ask 成交價。用途是比較行為特徵：收權利金、被履約率、犧牲上漲、最大回撤。")
    col1, col2, col3, col4 = st.columns([1.4, .8, .8, .8])
    with col1:
        bt_tickers_text = st.text_input("回測股票代號，可輸入多檔", value="AAPL, KO, SPY, QQQ", help="用逗號分隔。")
    with col2:
        bt_period = st.selectbox("歷史區間", ["1y", "2y", "5y"], index=1)
    with col3:
        bt_cycle = st.selectbox("回測週期", ["依操作目標", "7天", "14天", "30天"], index=0)
    with col4:
        bt_limit = st.slider("最多回測檔數", 1, 20, 8, 1)
    bt_dte_override = None if bt_cycle == "依操作目標" else int(bt_cycle.replace("天", ""))
    if st.button("開始模型回測", type="primary", use_container_width=True):
        raw = bt_tickers_text.replace("\n", ",").split(",")
        bt_tickers = [x.strip().upper() for x in raw if x.strip()][:bt_limit]
        if not bt_tickers:
            st.warning("請至少輸入一個股票代號。")
        else:
            with st.spinner("正在回測：" + ", ".join(bt_tickers)):
                summary_df = summarize_backtests(bt_tickers, objective=objective, period=bt_period, limit=bt_limit, dte_override=bt_dte_override)
            st.markdown("### 多檔回測摘要")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            first = bt_tickers[0]
            try:
                bt = model_backtest_covered_call(first, objective=objective, period=bt_period, dte_override=bt_dte_override)
                st.markdown(f"### {first} 詳細回測")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("交易次數", bt.trades)
                m2.metric("履約率", f"{bt.assignment_rate:.1%}")
                m3.metric("Covered Call 報酬", f"{bt.covered_call_return:.1%}")
                m4.metric("買進持有報酬", f"{bt.buy_hold_return:.1%}")
                chart_df = bt.equity_curve.set_index("date")[["account_value", "buy_hold_value"]]
                st.line_chart(chart_df)
                st.markdown("### 交易明細")
                st.dataframe(bt.trades_frame.tail(30), use_container_width=True, hide_index=True)
                for note in bt.notes:
                    st.caption(note)
            except Exception as exc:
                st.warning(f"{first} 詳細回測失敗：{exc}")


with manual_tab:
    st.subheader("手動分析：貼上或匯入你自己的 option chain")
    st.markdown("目前手動分析提供欄位檢查與簡易排序。正式版可改成直接串接券商 API。")
    sample = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "spot": [200, 200],
        "expiry": ["2026-07-17", "2026-07-17"],
        "strike": [210, 215],
        "bid": [2.1, 1.2],
        "ask": [2.3, 1.35],
        "impliedVolatility": [0.28, 0.30],
        "openInterest": [1200, 800],
        "volume": [500, 200],
    })
    uploaded = st.file_uploader("上傳 CSV", type=["csv"])
    if uploaded:
        df_in = pd.read_csv(uploaded)
    else:
        df_in = sample
    edited = st.data_editor(df_in, use_container_width=True, num_rows="dynamic")
    required = {"ticker", "spot", "expiry", "strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"}
    missing = required - set(edited.columns)
    if missing:
        st.error("缺少欄位：" + ", ".join(sorted(missing)))
    else:
        df = edited.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["premium_yield"] = df["mid"] / df["spot"]
        df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
        df["upside_to_strike"] = df["strike"] / df["spot"] - 1
        df["rough_score"] = (
            df["premium_yield"].clip(0, 0.03) / 0.03 * 35 +
            (1 - df["spread_pct"].clip(0, 0.30) / 0.30) * 30 +
            (df["openInterest"].clip(0, 2000) / 2000) * 20 +
            (df["upside_to_strike"].clip(0, 0.15) / 0.15) * 15
        )
        st.dataframe(df.sort_values("rough_score", ascending=False), use_container_width=True, hide_index=True)

with rules_tab:
    st.subheader("這個工具的決策邏輯")
    st.markdown(
        """
        ### 不是只看年化權利金
        Covered call 最大的陷阱，是看到很高的年化權利金就以為是收租。實際上，高權利金常常代表高波動、高事件風險，或市場正在替你定價一個很大的下跌/上漲不確定性。

        ### 自動查找
        自動查找會從內建候選池掃描 ETF、大型股、流動成長股與高權利金觀察標的，再用同一套 covered call 決策引擎排序。它不是「推薦買進股票」，而是幫你把今天比較值得研究的 covered call 候選先篩出來。

        ### 評分模組
        - **標的品質**：ETF 與大型核心持股分數較高，高投機股較低。
        - **流動性**：open interest、volume、bid-ask spread。
        - **權利金吸引力**：單次權利金、年化權利金、IV / 20 日實現波動率。
        - **技術位置**：是否剛大漲、是否接近高點、是否跌勢中。
        - **事件風險**：財報日前後會大幅降低分數。
        - **被履約可接受度**：Strike 是否高於你的成本與最低願意賣出價。

        ### 週期建議
        - **保守收租**：通常 3–4 週檢查一次，Delta 約 0.12–0.25。
        - **平衡收租**：通常 2–4 週，Delta 約 0.18–0.32。
        - **提高現金流**：通常 1–2 週檢查一次，Delta 約 0.25–0.45，但不代表每週硬做。
        - **本來就想賣出**：可以接受較高 Delta，重點是 Strike 是否接近你的理想賣價。

        ### v6 新增的實務風控
        - **除息 / 早期履約檢查**：如果 ex-dividend date 落在持倉期間，系統會提醒短 call 變價內時的早期履約風險。
        - **打平價與被履約最大利潤**：不只看權利金，也會算如果被履約，等同用多少價格賣出股票。
        - **7 / 14 / 30 天週期比較**：單檔查詢會同時列出三個週期的最佳候選，而不是只給一張合約。
        - **50%–70% 獲利管理**：提醒 Buy to Close、Roll、接受履約三種管理路徑。

        ### 重要限制
        yfinance 的 option data 適合做 MVP 和研究，不適合直接作為下單依據。正式交易請用券商即時報價確認。這個工具不提供投資建議，只提供決策輔助與風險檢查。
        """
    )
