Ben 財務與投資模擬器 v7.3 修正說明

本次重點：
1. 修正 drawdown 邏輯
- 不再只看「年末資產 vs 歷史高點」
- 新增「年內估計低點」概念
- 依持股加權波動 + 風險群組相關性，估算 portfolio_vol_estimate_pct
- drawdown 改為：peak_assets_before_year vs min(end_total_assets, intrayear_trough_assets_twd)
- 因此在固定報酬情境下，就算年末仍上漲，也能反映年內可能回撤，不再常見 0.0%

2. 修正 Custom 預設配置
- Custom 不再預設等於 Recommended
- 改為 defensive / slower but steadier 的自訂底稿
- 目的是讓 Recommended 與 Custom 一開始就有明確差異，避免結果過於接近而誤判模型失效

3. 補強 Recommended / Custom 過近檢查
- 新增權重距離 portfolio_weight_distance_pct
- 若配置差很多，但最終終值差不到 0.5%，在驗證頁顯示 WARNING
- 若幾乎完全相同，顯示 FAIL

4. 在 tab 5 / 6 頂部印出「目前實際吃進模擬的配置 / 情境」
- 顯示 Current / Recommended / Custom 的來源
- 顯示現金%、加權報酬%、加權波動%、前五大持股
- 顯示目前情境、情境來源、mode override、market shift、AI excess、max drawdown hint

注意：
- 這版是 best-effort 的可操作修正，不是假裝精準的量化研究系統。
- 若未來要更嚴謹，下一步建議接入真實歷史價格，並把 hist_10y / vol_3y / vol_5y 真的接上。
