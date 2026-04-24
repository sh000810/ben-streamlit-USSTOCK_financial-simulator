# CODEX_FIRST_TASK.md

# Codex Task：建立 import contract tests 並穩定 app/core 介面

## 背景

目前 Ben 財務模擬器曾多次因 `app.py` 與 `core.py` 不同步而發生：

- `ImportError: cannot import name ... from core`
- `TypeError: got an unexpected keyword argument ...`

請優先建立測試與 contract，避免未來每次改風險引擎或 UI 都造成 Streamlit Cloud 啟動失敗。

## 任務範圍

只做穩定性，不做新 UI。

## 需修改 / 新增檔案

- `tests/test_import_contract.py`
- 必要時修改 `core.py`
- 必要時修改 `app.py`

## 必須測試的 core 函數

```python
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
```

## 驗收標準

1. `python -m py_compile app.py core.py financial_project_config.py` 通過。
2. `pytest -q` 通過，若 repo 尚無 pytest，請新增最小 tests 並更新 requirements。
3. `app.py` 所有 `from core import (...)` 名稱都存在。
4. `build_personal_balance_sheet()` 支援 app.py 現有 keyword arguments。
5. Moneybook 檔案缺失或空資料時，app 不應 crash。
6. 妥妥租預測表缺失時，app 應 graceful fallback，而不是 crash。

## 禁止事項

- 不要改財務假設。
- 不要改妥妥租認列規則。
- 不要改 Candidate / VOO / Custom 的投資邏輯。
- 不要大幅重構 UI。

## 回報格式

```md
## Changed Files
## What Changed
## Tests Run
## Known Risks
## Questions for Ben / ChatGPT
```
