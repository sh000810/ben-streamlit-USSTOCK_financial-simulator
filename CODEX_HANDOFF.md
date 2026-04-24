# CODEX_HANDOFF.md

# Ben Financial Simulator｜Codex 接手手冊

> Version: 2026-04-24 v1.0  
> Owner / PM: Ben  
> Spec / Logic / Review: ChatGPT  
> Implementation / Tests / Diff: Codex

---

## 0. 給 Codex 的一句話

這不是單純股票回測工具，而是 Ben 的「個人財務狀況追蹤 + 通往致富的人生地圖 + 股票配置決策系統」。

請不要只追求 UI 好看或終值最高。系統核心目標是：

1. 長期勝率，而不是短期紙上報酬。
2. 避免教育費高峰、事業收入衰退、股市大跌時被迫賣股。
3. 讓每個計算都有來源、假設、LOG、驗證與可追溯性。
4. 協助 Ben 判斷 Current / Candidate / VOO / Custom 等配置，哪一種更適合他的人生。

---

## 1. 工作分工規則

### 1.1 Ben 的角色

Ben 是最終決策者與需求提出者。

Ben 會決定：

- 哪些人生目標重要。
- 哪些現金流假設要採用。
- 是否接受某個投資配置方向。
- 是否要把某次 Codex diff 合併上線。

### 1.2 ChatGPT 的角色

ChatGPT 負責：

- 規格整理。
- 財務邏輯設計。
- 假設審查。
- 模型合理性檢查。
- 判斷結果是否有「看起來怪」的地方。
- 指派 Codex 具體任務。
- 審查 Codex 產出的 diff、測試結果、部署結果。

ChatGPT 不應直接把「未經測試的大段程式」當作最終答案。ChatGPT 應優先輸出：規格、檢查清單、驗收標準、風險說明。

### 1.3 Codex 的角色

Codex 負責：

- 修改 `core.py` / `app.py` / config / tests。
- 跑測試。
- 整理 diff。
- 確認 import 不破。
- 確認 Streamlit 主要頁面不 crash。
- 回報有哪些地方改了、哪些地方未完成、哪些地方需要 ChatGPT/Ben 做邏輯決策。

Codex 不應擅自更改財務邏輯假設。若發現邏輯衝突，先提出問題，不要自行「合理化」。

---

## 2. 專案檔案結構建議

```text
ben-streamlit-USSTOCK_financial-simulator/
├── app.py
├── core.py
├── project_spec.md
├── CODEX_HANDOFF.md
├── CHATGPT_REVIEW_PROMPTS.md
├── PROJECT_WORKFLOW.md
├── financial_project_config.json
├── financial_project_config.py
├── requirements.txt
├── data/
│   ├── Individual-Positions-2026-04-22-184253(1).csv
│   ├── 妥妥租_資料庫 - [預測]未來十年稅後淨利表.csv
│   ├── Moneybook_帳戶_20260424_1.csv
│   ├── Moneybook_台股證券手動新增庫存_20260424_1.csv
│   ├── Moneybook_明細_20260424_1.csv
│   ├── Moneybook_帳單_20260424_1.csv
│   └── Moneybook_保單_20260424_1.csv
└── tests/
    ├── test_import_contract.py
    ├── test_balance_sheet.py
    ├── test_tuotuozu_projection.py
    ├── test_portfolio_builders.py
    └── test_risk_engine.py
```

目前若 repo 尚未有 `tests/`，請 Codex 優先建立最小測試。

---

## 3. 必讀規格來源

Codex 每次開始前，請依序閱讀：

1. `project_spec.md`
2. `CODEX_HANDOFF.md`
3. `financial_project_config.json`
4. `financial_project_config.py`
5. `core.py`
6. `app.py`
7. Streamlit Cloud 最新錯誤 log，若有

若 `project_spec.md` 與程式碼衝突，請回報，不要自行假設。

---

## 4. Ben 的最新財務基準

### 4.1 資產口徑

系統必須同時追蹤：

```text
總資產 = 股票 + 現金 + 房產市值
淨資產 = 總資產 - 房貸 - 其他負債
流動資產 = 美股 + 台股 + 現金
可投資現金 = 現金 - 硬安全水位
```

### 4.2 最新基準值

| 項目 | 目前基準 |
|---|---:|
| 美股 | 約 1,070 萬 TWD |
| 台股 | 約 276 萬 TWD |
| 台幣 / 外幣現金 | 約 400～428 萬 TWD |
| 房產市值 | 約 850 萬 TWD |
| 房貸 | 約 -900 萬 TWD |
| 硬安全水位 | 150 萬 TWD |

### 4.3 台股明細

| 標的 | 金額 |
|---|---:|
| 2330 台積電 | 約 172 萬 |
| 0050 | 約 69 萬 |
| 6752 叡揚資訊 | 約 34 萬 |

台股必須納入整體 AI / 半導體曝險，不可只分析美股。

---

## 5. 妥妥租收入規則

### 5.1 最新原則

妥妥租收入模式一律以 **Excel / CSV 預測表** 為主。

請忽略舊口徑：

- 年淨利 300 萬。
- 月現金流 30 萬。

### 5.2 團隊認列規則

| team / group | Ben 可認列比例 |
|---|---:|
| 貓狗企鵝 | 50% |
| 鯰魚大大 | 100% |
| 其他 / 無法辨識 | 預設 100%，但需在 audit 表標示 |

`load_business_income_projection()` 或等效函式應輸出：年度原始淨利、Ben 可認列淨利、team recognition audit、無法辨識 team 的 warning。

---

## 6. 支出與教育費規則

目前 200 萬 / 年視為「現階段低消」或基礎家庭生活費，不包含完整教育費高峰。Moneybook 有公司支出與個人支出混雜問題，暫時不能直接把 Moneybook 原始總支出視為個人支出。

2034–2038 是教育費高峰。建議保留三層情境：

| 情境 | 兩孩教育費 / 年 |
|---|---:|
| 保守 | 60 萬 |
| 基準 | 90 萬 |
| 壓力 | 120 萬 |

此數字不可寫死為單一真理，應在 UI 中可調。

---

## 7. 房貸與槓桿規則

Ben 希望房貸長期採 interest-only：

```text
房貸利息 = 房貸本金 × 年利率
900 萬 × 2.2% = 19.8 萬 / 年，約 1.65 萬 / 月
```

系統仍應保留壓力情境：成功展延、轉貸利率上升、被迫本息攤還。

---

## 8. Portfolio Lab 比較組合

至少必須支援：

1. `Current`：真實美股組合。
2. `True Current`：美股 + 台股 + 現金。
3. `Recommended`：成長整理版。
4. `Custom`：防守版。
5. `Candidate`：Ben 成長防守平衡候選版。
6. `100% VOO Benchmark`：大盤基準。
7. `VOO + BRK.B Benchmark`：穩健核心基準。

若 UI 暫時只顯示部分組合，請不要刪除 builder 函式。

---

## 9. 風險引擎設計原則

任何配置比較都不可只用「最終資產」判斷。必須至少同時看：

- P50 終值。
- P5 worst case。
- P75 / P95 upside。
- 最大回撤。
- 現金流缺口年份。
- forced withdrawal / liquidity stress。
- AI / 半導體總曝險。
- 人生勝率分數。

不得直接將過去 10 年高報酬外推到未來。

正確邏輯：

```text
model_return_pct = 歷史參考 × bucket 折現規則 + 人工保守上限
model_vol_pct = 0.6 × 5Y weekly annualized vol + 0.4 × 3Y weekly annualized vol
```

若無資料，使用 bucket default，並標示 `confidence = low`。

v11 系列需保留：market regime、shared shock tape、correlation groups、AI/semi 在 bear/crash 的額外估值壓縮懲罰。但懲罰不可過度，否則會變成 VOO 永遠勝出。

---

## 10. Codex 每次工作流程

### Step 1：同步規格

閱讀：`project_spec.md`、`CODEX_HANDOFF.md`、`financial_project_config.json`。

### Step 2：確認目前錯誤

若 Ben 提供 Streamlit log，先修 crash，不要先做新功能。

### Step 3：跑最小測試

```bash
python -m py_compile app.py core.py financial_project_config.py
python - <<'PYTEST'
import core
required = [
    'build_candidate_portfolio',
    'build_voo_benchmark_portfolio',
    'build_personal_balance_sheet',
    'load_moneybook_accounts',
    'load_tw_stock_positions',
    'clean_moneybook_transactions',
    'summarize_monthly_spending',
    'build_dynamic_dca_plan',
    'build_life_stage_targets',
    'combine_us_tw_current_portfolio',
]
missing = [name for name in required if not hasattr(core, name)]
print('missing:', missing)
assert not missing
PYTEST
```

### Step 4：小步修改

一次只做一個主題。不要一次大改 UI + core + 風險引擎。

### Step 5：產出 diff summary

每次交付請回報：

```md
## Changed Files
## What Changed
## Tests Run
## Known Risks
## Questions for ChatGPT/Ben
```

---

## 11. 最小驗收標準

每次 merge 前必須滿足：

1. `from core import (...)` 不可壞。
2. Streamlit app 至少能啟動到首頁。
3. 個人財務追蹤頁不可 crash。
4. Portfolio Lab 不可因缺欄位 crash。
5. 妥妥租 Excel / CSV 無資料時要 graceful fallback。
6. Moneybook 檔案未上傳時要可正常顯示空表。
7. `core.py` helper 函數要與 `app.py` call signature 相容。
8. 任何模型數字若只是代理值，UI 必須標示。

---

## 12. 目前最重要的待辦

### P0：穩定性

- 補 tests。
- 固定 import contract。
- 讓 `app.py` 和 `core.py` 的函數簽名不再漂移。

### P1：決策可信度

- Portfolio Lab 加入 Monte Carlo P5/P50/P95。
- 目標達成率改用分布，而不是單一路徑。
- forced withdrawal / liquidity stress 進 UI。

### P2：真實 Current

- 台股正式併入 True Current Portfolio。
- 0050 look-through 中台積電權重應可設定。
- 美股 + 台股 + ETF look-through 合併計算 AI / 半導體曝險。

### P3：Moneybook 清理

- 建立可編輯分類規則表。
- 清理個人 / 公司支出混雜。
- 輸出可信的月低消與年度支出基準。

---

## 13. 禁止事項

Codex 不得：

1. 擅自刪除舊 builder 函數。
2. 擅自改財務假設。
3. 用固定 300 萬 / 年或 30 萬 / 月取代妥妥租 Excel。
4. 把 2034 年 1.45 億重新設為硬目標。
5. 把 Current 終值最高直接解讀為 Current 最好。
6. 只為了讓圖好看而壓低風險。
7. 把沒有資料的數字包裝成精準值。
8. 在沒跑 import contract test 前交付。

---

## 14. 給 Codex 的第一個任務建議

```text
建立 tests/test_import_contract.py，固定 app.py 目前需要從 core.py import 的所有函數。
修正目前 core.py / app.py 的函數簽名不相容問題。
確保 Streamlit Cloud 不再因 ImportError / TypeError 啟動失敗。
```

完成後再做 Portfolio Lab 的功能強化。
