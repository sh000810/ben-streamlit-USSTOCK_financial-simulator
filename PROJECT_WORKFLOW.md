# PROJECT_WORKFLOW.md

# Ben 財務模擬器｜ChatGPT + Codex 協作流程

## 1. 核心運作方式

本專案採三方協作：

```text
Ben = PM / 最終決策者
ChatGPT = 規格、財務邏輯、模型合理性、審查
Codex = 程式修改、測試、diff、修 bug
```

## 2. 標準流程

### 2.1 Ben 提出需求

Ben 應描述：想解決的問題、目前看到的異常、希望新增的畫面或指標、是否已有錯誤 log。

### 2.2 ChatGPT 轉成規格

ChatGPT 應輸出：需求是否合理、財務邏輯定義、資料欄位與函數需求、驗收標準、給 Codex 的 task。

### 2.3 Codex 實作

Codex 應：修改程式、跑測試、回傳 diff summary、回報風險與未完成。

### 2.4 ChatGPT 審查

ChatGPT 應：審查是否符合規格、判斷模型結果是否合理、指出是否可以合併、如不通過產出下一輪 Codex 修正指令。

---

## 3. 任務狀態分類

| 狀態 | 說明 |
|---|---|
| SPEC | ChatGPT 正在定義規格 |
| CODEX_TASK | 已交給 Codex |
| IMPLEMENTED | Codex 已改程式 |
| REVIEW | ChatGPT 正在審查 |
| NEEDS_FIX | 需 Codex 修正 |
| ACCEPTED | Ben 可合併或已上線 |

---

## 4. 每次 Codex PR / diff 必須回報

```md
# Diff Summary

## Files Changed
## Purpose
## What Changed
## Tests Run
## Screens / Outputs Checked
## Known Risks
## Follow-up Questions
```

---

## 5. 最小測試清單

```bash
python -m py_compile app.py core.py financial_project_config.py
```

若有 pytest：

```bash
pytest -q
```

Import contract：

```bash
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
missing = [x for x in required if not hasattr(core, x)]
assert not missing, missing
print('import contract OK')
PYTEST
```

---

## 6. 上線後 Ben 應檢查畫面

1. App 是否成功啟動。
2. 資料讀取狀態：美股、台股、現金、房產、房貸是否接近基準值。
3. 個人財務追蹤頁是否不 crash。
4. Portfolio Lab 是否顯示 Current / Candidate / VOO / Custom。
5. 最大回撤是否合理為負值，而不是長期 0%。
6. 妥妥租 Excel / CSV 認列結果是否顯示貓狗企鵝 50%、鯰魚大大 100%。
7. AI / 半導體曝險是否包含台積電。
8. 目標達成率 / 人生勝率是否有標示其資料來源。

---

## 7. 專案治理原則

1. 先穩定，再擴充。
2. 先修 crash，再加圖。
3. 先固定 contract，再改引擎。
4. 任何結果都要能追溯到 assumptions。
5. 不要讓 UI 的漂亮圖掩蓋模型不穩。
6. 不要把規則代理值假裝成金融真理。
