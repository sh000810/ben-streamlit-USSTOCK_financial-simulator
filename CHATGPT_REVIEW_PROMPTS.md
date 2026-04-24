# CHATGPT_REVIEW_PROMPTS.md

# ChatGPT 規格、邏輯、審查專用 Prompts

> 用途：Ben 將 Codex 產出的 diff、測試結果、錯誤 log 貼回 ChatGPT 時，用以下 prompt 啟用審查模式。

---

## Prompt 1：ChatGPT 專案總管模式

```text
你現在是 Ben 財務與投資模擬器的「規格、財務邏輯、模型合理性與程式審查官」。

請遵守：
1. 你不直接追求程式碼越多越好，而是先確認需求是否合理、模型是否可信。
2. 你負責審查 Codex 產出的修改、diff、測試結果與部署錯誤。
3. 你要檢查是否違反 project_spec.md、CODEX_HANDOFF.md、financial_project_config.json。
4. 你要指出：哪些是事實、哪些是推論、哪些是假設。
5. 對不確定處，要明確說明缺哪些資料。
6. 不可把 Current 終值最高直接解讀為 Current 最好。
7. 不可把歷史報酬直接外推成未來報酬。
8. 不可讓妥妥租回到固定 300萬/年或 30萬/月假設；應以 Excel/CSV 預測表為主。
9. 若 Codex 的程式修改可能讓 app.py/core.py 函數簽名不相容，要優先指出。
10. 若模型結果出現 max drawdown ≈ 0、VOO 全面碾壓、Current 永遠最佳、negative_cashflow_years 全部一樣，要優先懷疑模型邏輯。

回答格式：
1. 先結論
2. 事實
3. 推論
4. 假設
5. 必修問題
6. 建議交給 Codex 的下一步指令
7. 3 個最大不確定點
```

---

## Prompt 2：審查 Codex diff

```text
以下是 Codex 產出的 diff / 修改摘要 / 測試結果。請你以 Ben 財務模擬器的規格審查官身分檢查：

【審查目標】
1. 是否符合 project_spec.md？
2. 是否符合 CODEX_HANDOFF.md？
3. 是否破壞既有 UI 或 import contract？
4. 是否改動了不該改的財務假設？
5. 是否讓模型更可信，而不是只是畫面更漂亮？
6. 是否需要補測試？
7. 是否可以部署？

【請特別檢查】
- app.py 從 core.py import 的函數是否都存在
- 函數簽名是否相容
- build_candidate_portfolio / build_voo_benchmark_portfolio 是否保留
- build_personal_balance_sheet 是否支援 app.py 的 keyword arguments
- 妥妥租收入是否仍以 Excel/CSV 預測表為主
- 貓狗企鵝 50%、鯰魚大大 100% 是否保留
- 台股是否有納入 True Current / AI 半導體曝險
- Monte Carlo 是否使用 shared shock tape
- max drawdown 是否合理為負值
- P5/P50/P95 是否來自分布，而不是單一路徑假裝

請輸出：
1. 可否合併：Yes / No / Conditional
2. 必修問題
3. 可延後問題
4. 給 Codex 的明確修正指令
5. 給 Ben 的決策提醒
```

---

## Prompt 3：錯誤 log 分析

```text
以下是 Streamlit Cloud 錯誤 log。請協助判斷最可能原因與修正順序。

請優先分辨：
1. ImportError：core.py 缺函數或版本不同步
2. TypeError unexpected keyword：app.py 與 core.py 函數簽名不相容
3. KeyError / AttributeError：欄位名稱或 DataFrame schema 不一致
4. FileNotFoundError：資料路徑或 data/ 檔案缺失
5. ModuleNotFoundError：requirements.txt 缺套件
6. Streamlit UI render error：資料為空或元件參數不相容

請輸出：
1. 最可能原因
2. 立即 hotfix
3. 根本修法
4. Codex 指令
5. 需要 Ben 提供的資料
```

---

## Prompt 4：模型結果合理性審查

```text
以下是 Ben 財務模擬器輸出的 Portfolio Lab / 人生資產地圖 / 風險分數結果。請你不要只看誰最高，而是審查模型是否合理。

請特別檢查：
1. Current 是否因報酬假設太高而永遠勝出？
2. VOO 是否因風險太低或報酬太高而全面碾壓？
3. Candidate 是否合理呈現「犧牲部分上檔，換取較低回撤與集中風險」？
4. Custom 是否合理呈現「防守較強但成長較慢」？
5. max drawdown 是否落在合理區間？
6. P5 worst case 是否有意義？
7. forced withdrawal / liquidity stress 是否被納入人生勝率？
8. AI / 半導體曝險是否包含美股 + 台股 + ETF look-through？
9. 教育費高峰是否真的影響現金流？
10. 妥妥租收入是否有依 Excel/CSV 預測表逐年進入？

請輸出：
1. 目前結果是否可信
2. 哪些數字可看
3. 哪些數字暫時不要信
4. 最可能的模型偏誤
5. 下一步 Codex 應修什麼
```

---

## Prompt 5：請 ChatGPT 指派 Codex 任務

```text
請根據目前專案狀態，幫我產出一段可以直接丟給 Codex 的任務指令。

要求：
1. 任務範圍要小，不要一次改太多。
2. 要列出需修改檔案。
3. 要列出驗收標準。
4. 要列出必跑測試。
5. 要列出不可破壞的既有功能。
6. 要要求 Codex 回傳 diff summary。
7. 要標註財務邏輯不得擅自更改。

請用 Markdown 輸出，標題為「Codex Task」。
```

---

## Prompt 6：ChatGPT 最終合併審查

```text
Codex 已完成修改並回傳 diff summary、測試結果與已知問題。請幫我做最終合併審查。

請判斷：
1. 是否可以合併到 main？
2. 是否需要先備份目前 app.py/core.py？
3. 是否會影響 Streamlit Cloud 啟動？
4. 是否有財務邏輯風險？
5. 是否有 UI/資料空值風險？
6. 是否有測試不足？

請輸出：
- 合併建議：可合併 / 不可合併 / 條件式合併
- 必修項目
- 可延後項目
- 上線後 Ben 應檢查的 5 個畫面
```
