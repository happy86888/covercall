# 部署說明

## 覆蓋更新

你目前若已經有 GitHub repo + Streamlit App，不需要重建 App。

1. 解壓縮新版 ZIP。
2. 進入 `covered_call_engine_pro_tradier/`。
3. 把裡面的所有檔案覆蓋到 GitHub 原本的專案資料夾，例如：

```text
covercall / covered_call_decision_engine_deployable /
```

4. Commit changes。
5. Streamlit 會自動重新部署。

## Main file path

如果你的 app.py 位於：

```text
covered_call_decision_engine_deployable/app.py
```

Streamlit Main file path 維持：

```text
covered_call_decision_engine_deployable/app.py
```

## 沒產生自動查找結果怎麼辦

這通常是資料源問題，不一定代表市場沒有 covered call 機會。

免費 yfinance 有時會抓不到 option chain，尤其是批次掃描很多股票時。v6.3 新增「模型估算」fallback，可以先回答 7 / 14 / 30 天哪個週期值得研究；但正式下單仍需券商或 Tradier 報價確認。

如果你要穩定做自動查找，建議設定 Tradier token。


## v6.3 新增
- 新增「網路名單」頁籤：根據公開 most-active options / option-volume 方向整理高流動性 covered call 候選池。
- 新增「網路熱門期權池」自動候選池，避免只靠使用者手動輸入 AAPL / KO。
- 自動掃描文案改為 rate-limit friendly：先看名單，再挑 5–15 檔精準掃描。
- 仍保留 Tradier / yfinance / 模型估算架構；正式批次掃描建議使用 Tradier token。
