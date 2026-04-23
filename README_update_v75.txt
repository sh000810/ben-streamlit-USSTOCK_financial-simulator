Ben 財務與投資模擬器 v7.5 更新摘要

本版已完成：
1. Cash bucket 修正
- 現金 / Cash / Cash & Cash Investments 一律固定：
  - expected_return_pct = 1.5
  - volatility_pct = 0.0
- Assumption Audit 也同步改為 cash_fixed_1p5_0vol / cash_fixed_zero_vol，避免被 general bucket 覆蓋。

2. 單一情境模擬頁新增「組合基礎加權報酬 / 波動 與情境額外加成」
- 顯示三組：Current / Recommended / Custom
- 欄位包含：
  - 基礎加權報酬率%
  - 基礎加權波動率%
  - AI 直接曝險權重%
  - market_return_shift_pct
  - weighted_ai_excess_shift_pct
  - scenario_shift_pct
  - effective_portfolio_return_pct
  - vol_multiplier
  - effective_portfolio_vol_pct

3. 單一路徑與 Monte Carlo 明確分離
- Tab 3：單一路徑 deterministic path
- Tab 4：Monte Carlo 分布
- Monte Carlo 顯示：
  - P50 / P25 / P75 / P5 Worst Case
  - 資產耗盡機率
  - 最大回撤中位數
  - 終值分布圖
  - 最大回撤分布圖

4. 年度 LOG 增加可驗算欄位
- base_portfolio_return_pct
- market_return_shift_pct
- weighted_ai_excess_shift_pct
- scenario_shift_pct
- effective_return_pct
- portfolio_return_pct（實際年度結果）

5. 假設透明化 / LOG 與 驗證 / 除錯 頁頂部
- 新增 portfolio_effect_df 表格
- 可以直接看到目前情境與實際吃進模擬的有效報酬/波動。

本版尚未完成：
- 歷史價格資料接入（hist_10y_cagr_pct / vol_3y_weekly_pct / vol_5y_weekly_pct 仍以後續補齊為主）
- Excel 預測資料欄位自動診斷的進一步增強
