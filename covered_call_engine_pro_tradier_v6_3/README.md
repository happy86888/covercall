# Covered Call Decision Engine - Pro v6.3

這是 covered call 決策工具的 Pro v6.3 版，支援：

- 我已有持股：輸入股票、股數、成本、最低願意賣出價，產生 covered call 操作建議。
- 自動查找：掃描候選池，統計幾檔可做 / 限價單 / 觀察。
- 7 / 14 / 30 天週期建議：直接判斷本次較適合短週期、雙週期或一個月期。
- 如何操作：內建 Sell to Open、Limit Order、Buy to Close、Roll、除息檢查 SOP。
- 策略回測：用歷史股價 + Black-Scholes 做模型回測。
- 資料源 fallback：有 Tradier token 時優先使用 Tradier；沒有 token 時使用 yfinance。
- v6.3 新增：若資料源抓不到 option chain，可用模型估算 7 / 14 / 30 天候選，避免自動查找空白。

## 重要限制

模型估算不是即時券商報價，不能直接拿來下單。正式交易前請務必用券商畫面確認：

- 到期日
- strike
- bid / ask
- spread
- open interest
- volume
- 除息日與財報日

如果你要穩定自動查找，建議設定 Tradier API token。

## Streamlit Cloud 部署

Main file path 若你的 repo 結構是：

```text
covered_call_decision_engine_deployable/app.py
```

就填：

```text
covered_call_decision_engine_deployable/app.py
```

如果 app.py 在 repo 最外層，就填：

```text
app.py
```

## Tradier Secrets

Streamlit Cloud > Manage app > Settings > Secrets：

```toml
MARKET_DATA_PROVIDER = "auto"
TRADIER_ACCESS_TOKEN = "你的 Tradier token"
TRADIER_BASE_URL = "https://api.tradier.com/v1"
```


## v6.3 新增
- 新增「網路名單」頁籤：根據公開 most-active options / option-volume 方向整理高流動性 covered call 候選池。
- 新增「網路熱門期權池」自動候選池，避免只靠使用者手動輸入 AAPL / KO。
- 自動掃描文案改為 rate-limit friendly：先看名單，再挑 5–15 檔精準掃描。
- 仍保留 Tradier / yfinance / 模型估算架構；正式批次掃描建議使用 Tradier token。
