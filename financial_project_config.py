"""Ben 財務與投資模擬器：系統可讀設定檔

用法：
    from financial_project_config import PROJECT_CONFIG, get_config

此檔是 project_spec.md 的 Python config 版本，供 Streamlit / core.py 直接讀取。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

PROJECT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "Ben 財務與投資模擬器",
        "mission": "個人財務狀況追蹤 + 通往致富的人生地圖 + 股票配置可視化與驗證",
        "decision_principle": "不是追求最高紙上報酬，而是在安全合理範圍內提高 long-term win rate，降低中途出局機率。",
        "current_year": 2026,
        "current_age": 44,
        "updated_at": "2026-04-24",
    },
    "asset_policy": {
        "total_assets_formula": "stocks + cash + real_estate_market_value",
        "net_worth_formula": "total_assets - liabilities",
        "liquid_assets_formula": "us_stocks + tw_stocks + cash",
        "cash_floor_twd": 1_500_000,
        "cash_floor_policy": "硬安全水位，不投入市場。超過安全水位的現金可用 dynamic DCA 分批投入。",
        "baseline": {
            "us_portfolio_twd": 10_700_000,
            "tw_stocks_twd": 2_761_390,
            "cash_twd": 4_000_000,
            "real_estate_market_value_twd": 8_500_000,
            "mortgage_balance_twd": 9_000_000,
            "mortgage_rate_pct": 2.2,
            "mortgage_mode": "interest_only_preferred",
        },
    },
    "tuotuozu_income_policy": {
        "mode": "excel_projection_only",
        "ignore_fixed_income_assumptions": True,
        "forbidden_assumptions": ["300萬年淨利固定值", "30萬月現金流固定值"],
        "source_priority": [
            "uploaded_excel",
            "data/妥妥租_預測.xlsx",
            "data/妥妥租_資料庫 - [預測]未來十年稅後淨利表.csv",
            "repo_root_csv",
        ],
        "recognition_rules": [
            {"team_keyword": "貓狗企鵝", "recognition_rate": 0.5, "description": "Ben 僅認列 50% 稅後淨利"},
            {"team_keyword": "鯰魚大大", "recognition_rate": 1.0, "description": "Ben 可全額認列稅後淨利"},
        ],
        "default_recognition_rate_for_unknown_team": 0.0,
        "fallback_after_projection_end": "continue_decay_from_last_value",
        "audit_required": True,
        "audit_columns": ["year", "team", "raw_net_profit_twd", "recognition_rate", "recognized_net_profit_twd"],
    },
    "income_policy": {
        "salary_annual_twd": 1_200_000,
        "salary_growth_pct": 2.0,
        "retirement_age": 65,
        "current_rent_income_monthly_twd": 35_000,
        "inheritance_rent_monthly_twd": 60_000,
        "inheritance_age": 62,
    },
    "expense_policy": {
        "base_living_expense_annual_twd": 2_000_000,
        "base_living_expense_note": "目前低消，不含 2034-2038 教育費高峰完整壓力。Moneybook 導入自 2025/7 起，且公司/個人支出混雜，需清理後才可直接用。",
        "inflation_pct": 2.5,
        "education_peak_years": [2034, 2035, 2036, 2037, 2038],
        "education_cost_scenarios_twd_per_year": {
            "conservative": 600_000,
            "base": 900_000,
            "stress": 1_200_000,
        },
        "education_assumption": "台北兩個小孩，公立學校但有補習/安親/才藝/升學補強。未來應改為逐年逐孩模型。",
    },
    "moneybook_policy": {
        "purpose": "作為個人資產追蹤與支出估算 base，但不可直接把原始支出視為個人生活費。",
        "known_issues": [
            "真正導入記帳約從 2025/7 開始",
            "公司消費與個人消費混雜，會高估個人支出",
            "信用卡繳款、帳戶互轉、投資移轉需排除，避免重複計算",
            "目前低消約 15萬/月，系統 base 先用 200萬/年",
        ],
        "exclude_keywords": ["信用卡費", "繳信用卡", "轉帳", "跨行", "投資", "買股", "證券", "妥妥租", "QLAB", "學問", "公司", "租客", "屋主", "修繕", "代墊"],
        "future_feature": "可編輯分類規則表 keyword -> rule_type -> include_in_personal_expense",
    },
    "dca_policy": {
        "mode": "dynamic",
        "fixed_dca_is_only_floor": True,
        "monthly_dca_floor_twd": 100_000,
        "cash_floor_twd": 1_500_000,
        "formula": "max(0, monthly_income - monthly_expense - next_12_month_special_reserve - cash_floor_replenishment)",
        "suggested_dca_rate_of_surplus": 0.8,
        "aggressive_dca_rate_of_surplus": 1.0,
        "excess_cash_deployment_months": [12, 24],
    },
    "portfolio_policy": {
        "required_portfolios": ["Current", "Recommended", "Custom", "Candidate", "100% VOO Benchmark"],
        "current_must_include": ["US portfolio", "TW stocks", "cash"],
        "benchmarks": {
            "voo_benchmark": {"CASH": 14.0, "VOO": 86.0},
            "candidate": {"CASH": 14.0, "VOO": 25.0, "BRK.B": 15.0, "NVDA": 7.0, "MSFT": 7.0, "GOOG": 5.0, "AMZN": 5.0, "ASML": 5.0, "AVGO": 3.0, "ANET": 3.0, "NOW": 3.0, "EQIX": 2.0, "MA": 2.0, "ETN": 2.0, "LMT": 1.0, "BTI": 1.0},
        },
        "risk_rules": [
            "AI/半導體曝險必須合併美股、台股台積電、ETF look-through",
            "不要把規則引擎當真理",
            "不要把歷史高報酬直接外推未來",
            "個股若 base return 接近 VOO 且波動較高，必須用 bull/bear 分布證明其存在價值",
        ],
        "ticker_canonicalization": {"BRK/B": "BRK.B", "CASH & CASH INVESTMENTS": "CASH", "Cash": "CASH"},
    },
    "simulation_policy": {
        "required_modes": ["deterministic", "monte_carlo"],
        "scenario_required": ["Base", "Bull", "Bear", "AI super bull", "valuation reset", "income stress"],
        "metrics_required": ["P50", "P25", "P75", "P5 worst case", "ruin_probability", "max_drawdown", "cashflow_gap", "goal_achievement_probability"],
        "monte_carlo_rule": "所有 portfolio 必須共用同一組 shock tape，才能公平比較勝率。",
        "drawdown_rule": "drawdown_pct 必須 <= 0；若年度報酬有波動但 drawdown 長期為 0，視為模型警訊。",
    },
    "goals": {
        "removed_goal": "2034 年 1.45 億不再作為硬目標",
        "life_stage_targets_enabled": True,
        "goal_bands": {
            "age_45": {"total_assets_twd": [28_000_000, 35_000_000], "net_worth_twd": [19_000_000, 26_000_000]},
            "age_50": {"total_assets_twd": [45_000_000, 60_000_000], "net_worth_twd": [35_000_000, 50_000_000]},
            "age_55": {"total_assets_twd": [60_000_000, 90_000_000], "net_worth_twd": [50_000_000, 80_000_000]},
            "age_60": {"total_assets_twd": [80_000_000, 120_000_000], "net_worth_twd": [70_000_000, 110_000_000]},
            "age_65": {"total_assets_twd": [100_000_000, 160_000_000], "net_worth_twd": [90_000_000, 150_000_000]},
        },
    },
    "development_priority": [
        "Fix ImportError and ensure app/core function sync",
        "Add config loader reading financial_project_config.json",
        "Add Candidate and 100% VOO Benchmark into Portfolio Lab",
        "Merge TW stock positions into True Current Portfolio",
        "Create child-by-child education schedule",
        "Implement shared Monte Carlo shock tape",
        "Add editable Moneybook classification rules",
    ],
}


def get_config() -> Dict[str, Any]:
    """Return a deep copy so callers can mutate safely."""
    return deepcopy(PROJECT_CONFIG)


def get(path: str, default: Any = None) -> Any:
    """Read nested config by dot path, e.g. get('asset_policy.cash_floor_twd')."""
    cur: Any = PROJECT_CONFIG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
