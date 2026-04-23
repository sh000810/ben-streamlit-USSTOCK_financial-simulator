Ben 財務與投資模擬器 v7.8

本版新增：
1. 歷史假設資料 CSV 匯入（選填）
   - 可上傳 ticker / hist_10y_cagr_pct / vol_3y_weekly_pct / vol_5y_weekly_pct 等欄位
   - Assumption Audit 會優先吃匯入值，再 fallback 到 bucket default
   - 側欄提供範本下載

2. Assumption Audit 覆蓋率摘要
   - Current / Recommended / Custom 會顯示 hist_10y / vol_3y / vol_5y 覆蓋率
   - 可快速看出目前是欄位完成，還是資料真的補齊

3. AI 驗證 Prompt 自動產生
   - 可單獨下載 AI_VALIDATION_PROMPT.txt
   - AI 驗證包 ZIP 內也會自動附上同一份 prompt

4. 驗證頁可直接看到 WARNING / FAIL 明細
   - 不只顯示數量，也能直接看到是哪幾項在警告

5. 原始資料與下載頁也可直接下載 AI 驗證包 ZIP / AI 驗證 Prompt

目的：
- 讓後續「補歷史資料」可以不用改 code，先用 CSV 匯入
- 讓 AI / 人工驗證時少打一輪說明文字
