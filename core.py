from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
CURRENT_AGE = 44

RECOMMENDED_WEIGHTS = {
    "Cash": 8.0,
    "VOO": 18.0,
    "BRK.B": 14.0,
    "NVDA": 8.0,
    "MSFT": 8.0,
    "ASML": 8.0,
    "AMZN": 6.0,
    "GOOG": 6.0,
    "NOW": 5.0,
    "ANET": 5.0,
    "AVGO": 4.0,
    "EQIX": 3.0,
    "MA": 3.0,
    "ETN": 2.0,
    "ORCL": 2.0,
}

DEFENSIVE_CUSTOM_WEIGHTS = {
    "Cash": 20.0,
    "VOO": 22.0,
    "BRK.B": 20.0,
    "MSFT": 8.0,
    "GOOG": 6.0,
    "AMZN": 5.0,
    "EQIX": 5.0,
    "MA": 4.0,
    "ETN": 4.0,
    "BTI": 3.0,
    "LMT": 2.0,
    "ORCL": 1.0,
}

ETF_LOOKTHROUGH = {
    "QQQ": {
        "MSFT": 8.8,
        "NVDA": 8.5,
        "AAPL": 7.4,
        "AMZN": 5.7,
        "AVGO": 4.7,
        "META": 3.7,
        "GOOG": 3.2,
        "GOOGL": 2.8,
        "COST": 2.6,
        "TSLA": 2.4,
        "NFLX": 2.0,
        "ASML": 1.8,
        "AMD": 1.6,
    },
    "VOO": {
        "MSFT": 6.4,
        "NVDA": 6.1,
        "AAPL": 5.8,
        "AMZN": 3.8,
        "META": 2.7,
        "AVGO": 2.2,
        "GOOGL": 1.9,
        "GOOG": 1.7,
        "BRK.B": 1.6,
        "TSLA": 1.4,
        "JPM": 1.3,
        "LLY": 1.3,
        "V": 1.1,
        "MA": 0.9,
        "COST": 0.9,
        "ASML": 0.0,
    },
}

RISK_GROUP_CORR = {
    "cash": {"cash": 1.00, "market_core": 0.10, "ai_core": 0.05, "semi": 0.05, "cloud": 0.05, "software": 0.05, "infra": 0.05, "compounders": 0.10, "industrial": 0.10, "defensive": 0.10, "other": 0.10},
    "market_core": {"cash": 0.10, "market_core": 1.00, "ai_core": 0.85, "semi": 0.80, "cloud": 0.82, "software": 0.78, "infra": 0.72, "compounders": 0.70, "industrial": 0.65, "defensive": 0.55, "other": 0.60},
    "ai_core": {"cash": 0.05, "market_core": 0.85, "ai_core": 1.00, "semi": 0.88, "cloud": 0.86, "software": 0.82, "infra": 0.78, "compounders": 0.62, "industrial": 0.50, "defensive": 0.35, "other": 0.55},
    "semi": {"cash": 0.05, "market_core": 0.80, "ai_core": 0.88, "semi": 1.00, "cloud": 0.72, "software": 0.62, "infra": 0.68, "compounders": 0.55, "industrial": 0.50, "defensive": 0.30, "other": 0.50},
    "cloud": {"cash": 0.05, "market_core": 0.82, "ai_core": 0.86, "semi": 0.72, "cloud": 1.00, "software": 0.82, "infra": 0.76, "compounders": 0.64, "industrial": 0.48, "defensive": 0.28, "other": 0.50},
    "software": {"cash": 0.05, "market_core": 0.78, "ai_core": 0.82, "semi": 0.62, "cloud": 0.82, "software": 1.00, "infra": 0.72, "compounders": 0.66, "industrial": 0.44, "defensive": 0.26, "other": 0.50},
    "infra": {"cash": 0.05, "market_core": 0.72, "ai_core": 0.78, "semi": 0.68, "cloud": 0.76, "software": 0.72, "infra": 1.00, "compounders": 0.60, "industrial": 0.56, "defensive": 0.30, "other": 0.48},
    "compounders": {"cash": 0.10, "market_core": 0.70, "ai_core": 0.62, "semi": 0.55, "cloud": 0.64, "software": 0.66, "infra": 0.60, "compounders": 1.00, "industrial": 0.58, "defensive": 0.40, "other": 0.52},
    "industrial": {"cash": 0.10, "market_core": 0.65, "ai_core": 0.50, "semi": 0.50, "cloud": 0.48, "software": 0.44, "infra": 0.56, "compounders": 0.58, "industrial": 1.00, "defensive": 0.46, "other": 0.52},
    "defensive": {"cash": 0.10, "market_core": 0.55, "ai_core": 0.35, "semi": 0.30, "cloud": 0.28, "software": 0.26, "infra": 0.30, "compounders": 0.40, "industrial": 0.46, "defensive": 1.00, "other": 0.45},
    "other": {"cash": 0.10, "market_core": 0.60, "ai_core": 0.55, "semi": 0.50, "cloud": 0.50, "software": 0.50, "infra": 0.48, "compounders": 0.52, "industrial": 0.52, "defensive": 0.45, "other": 1.00},
}

SCENARIO_DEFAULTS = [
    {
        "scenario_name": "AI 超級大牛市",
        "mode": "fixed",
        "market_return_shift_pct": 3.0,
        "ai_excess_return_pct": 7.0,
        "vol_multiplier": 0.85,
        "early_negative_years": 0,
        "early_negative_return_pct": -15.0,
        "early_ai_penalty_pct": -5.0,
        "recovery_years": 0,
        "recovery_boost_pct": 0.0,
        "business_decay_pct": 5.0,
        "inflation_pct": 2.5,
        "education_multiplier": 1.0,
        "max_drawdown_hint_pct": -28.0,
    },
    {
        "scenario_name": "中性成長 / 估值正常化",
        "mode": "fixed",
        "market_return_shift_pct": 0.0,
        "ai_excess_return_pct": 2.0,
        "vol_multiplier": 1.00,
        "early_negative_years": 0,
        "early_negative_return_pct": -12.0,
        "early_ai_penalty_pct": -4.0,
        "recovery_years": 0,
        "recovery_boost_pct": 0.0,
        "business_decay_pct": 10.0,
        "inflation_pct": 2.5,
        "education_multiplier": 1.0,
        "max_drawdown_hint_pct": -35.0,
    },
    {
        "scenario_name": "先殺估值，再恢復",
        "mode": "path",
        "market_return_shift_pct": 0.0,
        "ai_excess_return_pct": 2.0,
        "vol_multiplier": 1.20,
        "early_negative_years": 2,
        "early_negative_return_pct": -18.0,
        "early_ai_penalty_pct": -8.0,
        "recovery_years": 3,
        "recovery_boost_pct": 7.0,
        "business_decay_pct": 10.0,
        "inflation_pct": 2.5,
        "education_multiplier": 1.0,
        "max_drawdown_hint_pct": -48.0,
    },
    {
        "scenario_name": "AI 成長是真的，但股價表現不如預期",
        "mode": "fixed",
        "market_return_shift_pct": -2.0,
        "ai_excess_return_pct": -1.0,
        "vol_multiplier": 1.05,
        "early_negative_years": 0,
        "early_negative_return_pct": -12.0,
        "early_ai_penalty_pct": -4.0,
        "recovery_years": 0,
        "recovery_boost_pct": 0.0,
        "business_decay_pct": 10.0,
        "inflation_pct": 2.5,
        "education_multiplier": 1.0,
        "max_drawdown_hint_pct": -38.0,
    },
    {
        "scenario_name": "景氣壓力 / 收入衰退 / 家庭支出壓力疊加",
        "mode": "monte_carlo",
        "market_return_shift_pct": -4.0,
        "ai_excess_return_pct": -1.0,
        "vol_multiplier": 1.25,
        "early_negative_years": 1,
        "early_negative_return_pct": -16.0,
        "early_ai_penalty_pct": -6.0,
        "recovery_years": 2,
        "recovery_boost_pct": 3.0,
        "business_decay_pct": 15.0,
        "inflation_pct": 4.0,
        "education_multiplier": 1.1,
        "max_drawdown_hint_pct": -45.0,
    },
    {
        "scenario_name": "自訂情境",
        "mode": "monte_carlo",
        "market_return_shift_pct": 0.0,
        "ai_excess_return_pct": 0.0,
        "vol_multiplier": 1.00,
        "early_negative_years": 0,
        "early_negative_return_pct": -10.0,
        "early_ai_penalty_pct": -4.0,
        "recovery_years": 0,
        "recovery_boost_pct": 0.0,
        "business_decay_pct": 10.0,
        "inflation_pct": 2.5,
        "education_multiplier": 1.0,
        "max_drawdown_hint_pct": -35.0,
    },
]

CLASSIFICATION_OVERRIDES = {
    "Cash": {"asset_class": "Cash", "theme": "Cash", "role_bucket": "安全墊", "risk_group": "cash", "ai_direct": False, "expected_return_pct": 1.5, "volatility_pct": 0.0, "sell_priority": 0},
    "Cash & Cash Investments": {"asset_class": "Cash", "theme": "Cash", "role_bucket": "安全墊", "risk_group": "cash", "ai_direct": False, "expected_return_pct": 1.5, "volatility_pct": 0.0, "sell_priority": 0},
    "VOO": {"asset_class": "ETF", "theme": "US Core ETF", "role_bucket": "核心底盤", "risk_group": "market_core", "ai_direct": False, "expected_return_pct": 8.0, "volatility_pct": 16.0, "sell_priority": 1},
    "QQQ": {"asset_class": "ETF", "theme": "Tech ETF", "role_bucket": "科技ETF", "risk_group": "market_core", "ai_direct": True, "expected_return_pct": 9.0, "volatility_pct": 20.0, "sell_priority": 1},
    "BRK.B": {"asset_class": "Equity", "theme": "Compounder", "role_bucket": "核心底盤", "risk_group": "compounders", "ai_direct": False, "expected_return_pct": 9.0, "volatility_pct": 18.0, "sell_priority": 3},
    "NVDA": {"asset_class": "Equity", "theme": "AI Compute", "role_bucket": "AI 核心", "risk_group": "ai_core", "ai_direct": True, "expected_return_pct": 14.0, "volatility_pct": 34.0, "sell_priority": 7},
    "MSFT": {"asset_class": "Equity", "theme": "Cloud/AI Platform", "role_bucket": "AI 核心", "risk_group": "cloud", "ai_direct": True, "expected_return_pct": 11.0, "volatility_pct": 24.0, "sell_priority": 6},
    "ASML": {"asset_class": "Equity", "theme": "Semi Upstream", "role_bucket": "半導體上游", "risk_group": "semi", "ai_direct": True, "expected_return_pct": 11.0, "volatility_pct": 28.0, "sell_priority": 6},
    "AMZN": {"asset_class": "Equity", "theme": "Cloud/AI Platform", "role_bucket": "AI 核心", "risk_group": "cloud", "ai_direct": True, "expected_return_pct": 10.5, "volatility_pct": 27.0, "sell_priority": 6},
    "GOOG": {"asset_class": "Equity", "theme": "Cloud/AI Platform", "role_bucket": "AI 核心", "risk_group": "cloud", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 24.0, "sell_priority": 6},
    "GOOGL": {"asset_class": "Equity", "theme": "Cloud/AI Platform", "role_bucket": "AI 核心", "risk_group": "cloud", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 24.0, "sell_priority": 6},
    "NOW": {"asset_class": "Equity", "theme": "Enterprise AI Workflow", "role_bucket": "企業 AI workflow", "risk_group": "software", "ai_direct": True, "expected_return_pct": 10.5, "volatility_pct": 26.0, "sell_priority": 5},
    "ANET": {"asset_class": "Equity", "theme": "AI Network Infra", "role_bucket": "網通基建", "risk_group": "infra", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 28.0, "sell_priority": 5},
    "AVGO": {"asset_class": "Equity", "theme": "Semi/Infra", "role_bucket": "半導體上游", "risk_group": "semi", "ai_direct": True, "expected_return_pct": 10.5, "volatility_pct": 27.0, "sell_priority": 5},
    "EQIX": {"asset_class": "REIT/Infra", "theme": "Data Center", "role_bucket": "數位基建", "risk_group": "infra", "ai_direct": True, "expected_return_pct": 8.5, "volatility_pct": 18.0, "sell_priority": 2},
    "MA": {"asset_class": "Equity", "theme": "Payment Compounder", "role_bucket": "非 AI 複利資產", "risk_group": "compounders", "ai_direct": False, "expected_return_pct": 9.0, "volatility_pct": 20.0, "sell_priority": 3},
    "ETN": {"asset_class": "Equity", "theme": "Industrial Power", "role_bucket": "工業電氣基建", "risk_group": "industrial", "ai_direct": False, "expected_return_pct": 8.5, "volatility_pct": 18.0, "sell_priority": 3},
    "ORCL": {"asset_class": "Equity", "theme": "Enterprise AI Workflow", "role_bucket": "企業 AI workflow", "risk_group": "software", "ai_direct": True, "expected_return_pct": 8.5, "volatility_pct": 21.0, "sell_priority": 4},
    "AAPL": {"asset_class": "Equity", "theme": "Mega Cap Tech", "role_bucket": "科技平台", "risk_group": "market_core", "ai_direct": False, "expected_return_pct": 8.0, "volatility_pct": 22.0, "sell_priority": 4},
    "AMD": {"asset_class": "Equity", "theme": "AI Compute", "role_bucket": "AI 衛星", "risk_group": "ai_core", "ai_direct": True, "expected_return_pct": 11.0, "volatility_pct": 36.0, "sell_priority": 6},
    "MRVL": {"asset_class": "Equity", "theme": "Semi/Infra", "role_bucket": "AI 衛星", "risk_group": "semi", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 34.0, "sell_priority": 5},
    "META": {"asset_class": "Equity", "theme": "AI Platform", "role_bucket": "科技平台", "risk_group": "ai_core", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 29.0, "sell_priority": 5},
    "VRT": {"asset_class": "Equity", "theme": "Data Center Power", "role_bucket": "AI 衛星", "risk_group": "infra", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 30.0, "sell_priority": 5},
    "BTI": {"asset_class": "Equity", "theme": "Defensive Yield", "role_bucket": "防守", "risk_group": "defensive", "ai_direct": False, "expected_return_pct": 6.5, "volatility_pct": 18.0, "sell_priority": 2},
    "DDOG": {"asset_class": "Equity", "theme": "Cloud Software", "role_bucket": "AI 衛星", "risk_group": "software", "ai_direct": True, "expected_return_pct": 10.0, "volatility_pct": 33.0, "sell_priority": 5},
    "INTC": {"asset_class": "Equity", "theme": "Semi", "role_bucket": "半導體", "risk_group": "semi", "ai_direct": False, "expected_return_pct": 6.5, "volatility_pct": 30.0, "sell_priority": 4},
    "NFLX": {"asset_class": "Equity", "theme": "Consumer Platform", "role_bucket": "其他成長", "risk_group": "other", "ai_direct": False, "expected_return_pct": 8.5, "volatility_pct": 28.0, "sell_priority": 4},
    "CUK": {"asset_class": "Equity", "theme": "Travel", "role_bucket": "景氣循環", "risk_group": "other", "ai_direct": False, "expected_return_pct": 7.0, "volatility_pct": 30.0, "sell_priority": 2},
    "DIS": {"asset_class": "Equity", "theme": "Media", "role_bucket": "其他", "risk_group": "other", "ai_direct": False, "expected_return_pct": 7.0, "volatility_pct": 24.0, "sell_priority": 2},
    "LMT": {"asset_class": "Equity", "theme": "Defense", "role_bucket": "防守", "risk_group": "defensive", "ai_direct": False, "expected_return_pct": 7.0, "volatility_pct": 16.0, "sell_priority": 2},
    "MU": {"asset_class": "Equity", "theme": "Semi", "role_bucket": "半導體", "risk_group": "semi", "ai_direct": True, "expected_return_pct": 8.5, "volatility_pct": 32.0, "sell_priority": 4},
    "DT": {"asset_class": "Equity", "theme": "Software", "role_bucket": "其他", "risk_group": "software", "ai_direct": False, "expected_return_pct": 7.5, "volatility_pct": 24.0, "sell_priority": 2},
    "NNDM": {"asset_class": "Equity", "theme": "Speculative", "role_bucket": "高風險衛星", "risk_group": "other", "ai_direct": False, "expected_return_pct": 5.0, "volatility_pct": 45.0, "sell_priority": 7},
}


def _clean_numeric(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text in {"", "--", "N/A", "nan", "None"}:
        return 0.0
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = text.replace("(", "-").replace(")", "")
    try:
        return float(text)
    except ValueError:
        return 0.0


def normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    mapping = {}
    for col in columns:
        low = col.lower()
        if "symbol" in low and "ticker" not in mapping:
            mapping["ticker"] = col
        elif "description" in low and "name" not in mapping:
            mapping["name"] = col
        elif "qty" in low or "quantity" in low:
            mapping["quantity"] = col
        elif "mkt val" in low or "market value" in low:
            mapping["market_value"] = col
        elif low.strip() == "cost basis" or "cost basis" in low:
            mapping["cost_basis"] = col
        elif "cost/share" in low or "cost per share" in low:
            mapping["cost_per_share"] = col
        elif low.strip() == "price" and "price" not in mapping:
            mapping["price"] = col
        elif "asset type" in low:
            mapping["asset_type"] = col
    return mapping


def classify_ticker(ticker: str, asset_type: str = "") -> Dict[str, object]:
    ticker = (ticker or "").strip()
    if ticker in CLASSIFICATION_OVERRIDES:
        return CLASSIFICATION_OVERRIDES[ticker].copy()

    asset_type_low = (asset_type or "").lower()
    if "cash" in asset_type_low:
        return CLASSIFICATION_OVERRIDES["Cash"].copy()
    if "etf" in asset_type_low or "fund" in asset_type_low:
        return {
            "asset_class": "ETF",
            "theme": "Other ETF",
            "role_bucket": "ETF",
            "risk_group": "market_core",
            "ai_direct": False,
            "expected_return_pct": 8.0,
            "volatility_pct": 18.0,
            "sell_priority": 1,
        }
    return {
        "asset_class": "Equity",
        "theme": "Other Equity",
        "role_bucket": "其他",
        "risk_group": "other",
        "ai_direct": False,
        "expected_return_pct": 8.0,
        "volatility_pct": 24.0,
        "sell_priority": 4,
    }


def load_positions(source: BytesIO | StringIO | str | Path) -> Tuple[pd.DataFrame, List[str]]:
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, skiprows=1)
    else:
        df = pd.read_csv(source, skiprows=1)

    mapping = normalize_columns(df.columns)
    required = ["ticker", "name", "quantity", "market_value", "cost_basis", "asset_type"]
    missing = [item for item in required if item not in mapping]

    result = pd.DataFrame()
    result["ticker"] = df[mapping.get("ticker", df.columns[0])].fillna("").astype(str).str.strip()
    result["name"] = df[mapping.get("name", df.columns[0])].fillna("").astype(str).str.strip()
    result["quantity"] = df[mapping.get("quantity", df.columns[0])].map(_clean_numeric)
    result["market_value_usd"] = df[mapping.get("market_value", df.columns[0])].map(_clean_numeric)
    result["cost_basis_usd"] = df[mapping.get("cost_basis", df.columns[0])].map(_clean_numeric)
    result["price_usd"] = df[mapping.get("price", df.columns[0])].map(_clean_numeric) if "price" in mapping else 0.0
    result["cost_per_share_usd"] = df[mapping.get("cost_per_share", df.columns[0])].map(_clean_numeric) if "cost_per_share" in mapping else 0.0
    result["asset_type_raw"] = df[mapping.get("asset_type", df.columns[0])].fillna("").astype(str).str.strip()

    result = result[result["ticker"].ne("")].copy()
    result = result[~result["ticker"].str.contains("Positions Total", case=False, na=False)].copy()

    classifications = result.apply(lambda row: classify_ticker(row["ticker"], row["asset_type_raw"]), axis=1, result_type="expand")
    result = pd.concat([result.reset_index(drop=True), classifications.reset_index(drop=True)], axis=1)

    total_mv = result["market_value_usd"].sum()
    result["weight_pct"] = np.where(total_mv > 0, result["market_value_usd"] / total_mv * 100, 0.0)
    result["unrealized_gain_usd"] = result["market_value_usd"] - result["cost_basis_usd"]
    result["gain_pct"] = np.where(result["cost_basis_usd"] > 0, result["unrealized_gain_usd"] / result["cost_basis_usd"] * 100, 0.0)
    result["include"] = True
    return result.reset_index(drop=True), missing


def build_recommended_portfolio() -> pd.DataFrame:
    rows = []
    for ticker, weight in RECOMMENDED_WEIGHTS.items():
        cls = classify_ticker(ticker)
        rows.append(
            {
                "ticker": ticker,
                "name": ticker,
                "weight_pct": float(weight),
                "asset_class": cls["asset_class"],
                "theme": cls["theme"],
                "role_bucket": cls["role_bucket"],
                "risk_group": cls["risk_group"],
                "ai_direct": bool(cls["ai_direct"]),
                "expected_return_pct": float(cls["expected_return_pct"]),
                "volatility_pct": float(cls["volatility_pct"]),
                "sell_priority": int(cls["sell_priority"]),
                "include": True,
            }
        )
    return pd.DataFrame(rows)


def build_defensive_portfolio() -> pd.DataFrame:
    rows = []
    for ticker, weight in DEFENSIVE_CUSTOM_WEIGHTS.items():
        cls = classify_ticker(ticker)
        rows.append(
            {
                "ticker": ticker,
                "name": ticker,
                "weight_pct": float(weight),
                "asset_class": cls["asset_class"],
                "theme": cls["theme"],
                "role_bucket": cls["role_bucket"],
                "risk_group": cls["risk_group"],
                "ai_direct": bool(cls["ai_direct"]),
                "expected_return_pct": float(cls["expected_return_pct"]),
                "volatility_pct": float(cls["volatility_pct"]),
                "sell_priority": int(cls["sell_priority"]),
                "include": True,
            }
        )
    return pd.DataFrame(rows)


def normalize_weights(df: pd.DataFrame, weight_col: str = "weight_pct") -> pd.DataFrame:
    df = df.copy()
    included = df["include"] if "include" in df.columns else pd.Series(True, index=df.index)
    total = df.loc[included, weight_col].clip(lower=0).sum()
    if total <= 0:
        return df
    df.loc[included, weight_col] = df.loc[included, weight_col].clip(lower=0) / total * 100
    return df




def apply_cash_reserve_target(df: pd.DataFrame, cash_reserve_target_pct: float, weight_col: str = "weight_pct") -> pd.DataFrame:
    """Force the included portfolio weights to keep a target cash weight.

    Rules:
    - Detect cash rows by ticker or asset_class.
    - If no cash row exists and target > 0, create one.
    - Adjust only included rows.
    - Preserve relative weights among non-cash holdings.
    """
    work = df.copy()
    if work.empty or weight_col not in work.columns:
        return work

    if "include" not in work.columns:
        work["include"] = True

    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    included = work["include"].fillna(False).astype(bool)
    if not included.any():
        return work

    target = max(0.0, min(float(cash_reserve_target_pct), 100.0))

    ticker = work.get("ticker", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().str.upper()
    asset_class = work.get("asset_class", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
    cash_mask = included & (
        ticker.isin(["CASH", "CASH & CASH INVESTMENTS"])
        | asset_class.str.contains("cash", na=False)
    )

    if target > 0 and not cash_mask.any():
        cls = classify_ticker("Cash", "Cash")
        new_row = {c: pd.NA for c in work.columns}
        new_row.update({
            "ticker": "Cash",
            "name": "Cash",
            weight_col: 0.0,
            "asset_class": cls["asset_class"],
            "theme": cls["theme"],
            "role_bucket": cls["role_bucket"],
            "risk_group": cls["risk_group"],
            "ai_direct": bool(cls["ai_direct"]),
            "expected_return_pct": float(cls["expected_return_pct"]),
            "volatility_pct": float(cls["volatility_pct"]),
            "sell_priority": int(cls["sell_priority"]),
            "include": True,
        })
        work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)
        included = work["include"].fillna(False).astype(bool)
        ticker = work.get("ticker", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().str.upper()
        asset_class = work.get("asset_class", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
        cash_mask = included & (
            ticker.isin(["CASH", "CASH & CASH INVESTMENTS"])
            | asset_class.str.contains("cash", na=False)
        )

    total_included = float(work.loc[included, weight_col].clip(lower=0).sum())
    if total_included <= 0:
        return work

    work.loc[included, weight_col] = work.loc[included, weight_col].clip(lower=0) / total_included * 100.0
    total_included = 100.0

    current_cash = float(work.loc[cash_mask, weight_col].sum()) if cash_mask.any() else 0.0
    non_cash_mask = included & ~cash_mask
    current_non_cash = float(work.loc[non_cash_mask, weight_col].sum()) if non_cash_mask.any() else 0.0

    if abs(current_cash - target) < 1e-9:
        return work

    # Set final cash allocation.
    if cash_mask.any():
        cash_indices = list(work.index[cash_mask])
        work.loc[cash_indices, weight_col] = 0.0
        work.loc[cash_indices[0], weight_col] = target

    target_non_cash = max(0.0, total_included - target)
    if non_cash_mask.any():
        if current_non_cash > 0:
            scale = target_non_cash / current_non_cash
            work.loc[non_cash_mask, weight_col] = work.loc[non_cash_mask, weight_col] * scale
        else:
            # No non-cash holdings: assign all included weight to the cash row.
            first_cash_idx = work.index[cash_mask][0] if cash_mask.any() else None
            if first_cash_idx is not None:
                work.loc[included, weight_col] = 0.0
                work.loc[first_cash_idx, weight_col] = 100.0
                return work

    work = normalize_weights(work, weight_col=weight_col)
    return work

def portfolio_to_sim_input(df: pd.DataFrame, start_assets_twd: float) -> pd.DataFrame:
    work = df.copy()
    if "include" not in work.columns:
        work["include"] = True
    work = work[work["include"]].copy()
    work = normalize_weights(work)
    work["weight"] = work["weight_pct"] / 100.0
    work["value_twd"] = start_assets_twd * work["weight"]
    work["expected_return"] = work["expected_return_pct"] / 100.0
    work["volatility"] = work["volatility_pct"] / 100.0
    return work.reset_index(drop=True)


def build_comparison(current_df: pd.DataFrame, rec_df: pd.DataFrame) -> pd.DataFrame:
    cur = current_df[["ticker", "weight_pct", "role_bucket", "theme", "ai_direct", "risk_group"]].copy()
    cur = cur.rename(columns={"weight_pct": "current_weight_pct"})
    rec = rec_df[["ticker", "weight_pct", "role_bucket", "theme", "ai_direct", "risk_group"]].copy()
    rec = rec.rename(columns={"weight_pct": "recommended_weight_pct"})
    comp = pd.merge(cur, rec, on="ticker", how="outer", suffixes=("_current", "_recommended"))
    comp["current_weight_pct"] = comp["current_weight_pct"].fillna(0.0)
    comp["recommended_weight_pct"] = comp["recommended_weight_pct"].fillna(0.0)
    comp["delta_weight_pct"] = comp["recommended_weight_pct"] - comp["current_weight_pct"]

    def action(row: pd.Series) -> str:
        if row["current_weight_pct"] > 0 and row["recommended_weight_pct"] == 0:
            return "EXIT"
        if row["current_weight_pct"] == 0 and row["recommended_weight_pct"] > 0:
            return "ADD"
        if row["delta_weight_pct"] >= 1.0:
            return "ADD"
        if row["delta_weight_pct"] <= -1.0:
            return "TRIM"
        return "KEEP"

    comp["action_bucket"] = comp.apply(action, axis=1)
    for col in ["role_bucket_current", "role_bucket_recommended", "theme_current", "theme_recommended", "risk_group_current", "risk_group_recommended"]:
        if col not in comp.columns:
            comp[col] = ""
    comp["role_bucket"] = comp["role_bucket_current"].fillna("").replace("", pd.NA).fillna(comp["role_bucket_recommended"])
    comp["theme"] = comp["theme_current"].fillna("").replace("", pd.NA).fillna(comp["theme_recommended"])
    comp["ai_direct"] = comp["ai_direct_current"].fillna(False) | comp["ai_direct_recommended"].fillna(False)
    comp["risk_group"] = comp["risk_group_current"].fillna("").replace("", pd.NA).fillna(comp["risk_group_recommended"])
    return comp[["ticker", "current_weight_pct", "recommended_weight_pct", "delta_weight_pct", "action_bucket", "role_bucket", "theme", "ai_direct", "risk_group"]].sort_values(["action_bucket", "delta_weight_pct"], ascending=[True, False]).reset_index(drop=True)


def bucket_exposure(df: pd.DataFrame, column: str) -> pd.DataFrame:
    work = df[df.get("include", True)].copy()
    if work.empty:
        return pd.DataFrame(columns=[column, "weight_pct"])
    return (
        work.groupby(column, dropna=False)["weight_pct"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def _effective_count(weights: np.ndarray) -> float:
    weights = weights[weights > 0]
    if len(weights) == 0:
        return 0.0
    hhi = np.sum(np.square(weights))
    return 1.0 / hhi if hhi > 0 else 0.0


def weighted_corr_score(group_weights: Dict[str, float]) -> float:
    groups = list(group_weights.keys())
    score = 0.0
    for g1 in groups:
        for g2 in groups:
            w1 = group_weights[g1]
            w2 = group_weights[g2]
            corr = RISK_GROUP_CORR.get(g1, {}).get(g2, 0.5)
            score += w1 * w2 * corr
    return score


def compute_risk_duplicate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    work = normalize_weights(df[df.get("include", True)].copy())
    if work.empty:
        return {}
    weights = work["weight_pct"].to_numpy() / 100.0
    ai_exposure = work.loc[work["ai_direct"], "weight_pct"].sum()
    theme_weights = (work.groupby("theme")["weight_pct"].sum() / 100.0).to_dict()
    group_weights = (work.groupby("risk_group")["weight_pct"].sum() / 100.0).to_dict()
    category_weights = (work.groupby("asset_class")["weight_pct"].sum() / 100.0).to_dict()
    single_name_max = work["weight_pct"].max()
    effective_names = _effective_count(weights)
    effective_categories = _effective_count(np.array(list(category_weights.values())))
    theme_hhi = np.sum(np.square(np.array(list(theme_weights.values()))))
    same_drop = weighted_corr_score(group_weights) * 100.0
    overlap = compute_etf_overlap(df)
    return {
        "直接 AI 曝險 %": float(ai_exposure),
        "ETF 重疊曝險 %": float(overlap["overlap_exposure_pct"].sum()),
        "單一風險主題集中度 HHI": float(theme_hhi),
        "類別分散度（有效類別數）": float(effective_categories),
        "單一持股過重程度（最大權重%）": float(single_name_max),
        "同跌風險程度（0-100）": float(same_drop),
        "有效持股數": float(effective_names),
    }


def compute_etf_overlap(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_weights(df[df.get("include", True)].copy())
    if work.empty:
        return pd.DataFrame(columns=["etf", "constituent", "portfolio_direct_weight_pct", "etf_weight_pct", "lookthrough_constituent_pct", "overlap_exposure_pct"])
    direct = work.set_index("ticker")["weight_pct"].to_dict()
    rows = []
    for etf, constituents in ETF_LOOKTHROUGH.items():
        etf_weight = direct.get(etf, 0.0)
        if etf_weight <= 0:
            continue
        for constituent, constituent_pct in constituents.items():
            direct_weight = direct.get(constituent, 0.0)
            overlap_exposure = etf_weight * constituent_pct / 100.0 if direct_weight > 0 else 0.0
            rows.append(
                {
                    "etf": etf,
                    "constituent": constituent,
                    "portfolio_direct_weight_pct": direct_weight,
                    "etf_weight_pct": etf_weight,
                    "lookthrough_constituent_pct": constituent_pct,
                    "overlap_exposure_pct": overlap_exposure,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["overlap_flag"] = result["portfolio_direct_weight_pct"] > 0
    return result.sort_values(["etf", "overlap_exposure_pct", "portfolio_direct_weight_pct"], ascending=[True, False, False]).reset_index(drop=True)


def scenario_table() -> pd.DataFrame:
    return pd.DataFrame(SCENARIO_DEFAULTS)


def get_scenario_row(scenarios: pd.DataFrame, scenario_name: str) -> pd.Series:
    row = scenarios.loc[scenarios["scenario_name"] == scenario_name]
    if row.empty:
        return scenarios.iloc[0]
    return row.iloc[0]


def estimate_portfolio_volatility_pct(holdings: pd.DataFrame) -> float:
    if holdings.empty:
        return 0.0
    weights = pd.to_numeric(holdings.get("weight", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vols = pd.to_numeric(holdings.get("volatility", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    groups = holdings.get("risk_group", pd.Series("other", index=holdings.index)).fillna("other").astype(str).tolist()
    if len(weights) == 0:
        return 0.0
    cov = np.zeros((len(weights), len(weights)), dtype=float)
    for i in range(len(weights)):
        for j in range(len(weights)):
            corr = RISK_GROUP_CORR.get(groups[i], {}).get(groups[j], 0.5)
            cov[i, j] = vols[i] * vols[j] * corr
    port_var = float(weights @ cov @ weights.T)
    if port_var <= 0:
        return 0.0
    return float(np.sqrt(port_var) * 100.0)


def _calc_return_for_row(row: pd.Series, scenario: pd.Series, year_index: int, mode_override: Optional[str], rng: np.random.Generator, market_z: Optional[float], group_zs: Dict[str, float]) -> float:
    mode = (mode_override or scenario.get("mode", "fixed") or "fixed").strip()
    mean = float(row["expected_return"]) + float(scenario.get("market_return_shift_pct", 0.0)) / 100.0
    if bool(row.get("ai_direct", False)):
        mean += float(scenario.get("ai_excess_return_pct", 0.0)) / 100.0
    vol = max(0.0, float(row["volatility"]) * float(scenario.get("vol_multiplier", 1.0)))

    if mode == "path":
        early_years = int(scenario.get("early_negative_years", 0))
        recovery_years = int(scenario.get("recovery_years", 0))
        if year_index <= early_years:
            base = float(scenario.get("early_negative_return_pct", -10.0)) / 100.0
            if bool(row.get("ai_direct", False)):
                base += float(scenario.get("early_ai_penalty_pct", 0.0)) / 100.0
            return max(-0.95, base)
        if year_index <= early_years + recovery_years:
            recovery = mean + float(scenario.get("recovery_boost_pct", 0.0)) / 100.0
            return max(-0.95, recovery)
        return max(-0.95, mean)

    if mode == "monte_carlo":
        market_component = (market_z or 0.0) * 0.55
        group_component = group_zs.get(str(row.get("risk_group", "other")), 0.0) * 0.30
        idio_component = rng.standard_normal() * 0.15
        shock = market_component + group_component + idio_component
        value = mean + vol * shock
        return float(np.clip(value, -0.95, 2.0))

    return max(-0.95, mean)


def _apply_withdrawal_strategy(holdings: pd.DataFrame, amount_needed: float, strategy: str) -> Tuple[pd.DataFrame, float]:
    work = holdings.copy()
    if amount_needed <= 0:
        return work, 0.0

    def draw_from_index(order_idx: List[int], need: float) -> float:
        for idx in order_idx:
            if need <= 0:
                break
            available = float(work.at[idx, "value_twd"])
            if available <= 0:
                continue
            sold = min(available, need)
            work.at[idx, "value_twd"] = available - sold
            need -= sold
        return need

    cash_idx = work.index[work["asset_class"].eq("Cash")].tolist()
    etf_idx = work.index[work["asset_class"].eq("ETF")].tolist()
    non_cash_idx = work.index[~work["asset_class"].eq("Cash")].tolist()

    remaining = amount_needed
    if strategy == "比例賣出":
        total_non_cash = work.loc[non_cash_idx, "value_twd"].sum()
        total_cash = work.loc[cash_idx, "value_twd"].sum()
        if total_cash > 0:
            remaining = draw_from_index(cash_idx, remaining)
        if remaining > 0 and total_non_cash > 0:
            available_df = work.loc[non_cash_idx].copy()
            total = available_df["value_twd"].sum()
            if total > 0:
                for idx, row in available_df.iterrows():
                    if remaining <= 0:
                        break
                    proportional = amount_needed * (row["value_twd"] / total)
                    sold = min(work.at[idx, "value_twd"], proportional)
                    work.at[idx, "value_twd"] -= sold
                    remaining -= sold
                if remaining > 1e-6:
                    remaining = draw_from_index(non_cash_idx, remaining)
    elif strategy == "先賣現金 / ETF":
        remaining = draw_from_index(cash_idx, remaining)
        remaining = draw_from_index(etf_idx, remaining)
        other_idx = [idx for idx in work.index.tolist() if idx not in cash_idx + etf_idx]
        remaining = draw_from_index(other_idx, remaining)
    else:  # 先賣波動低資產
        order_df = work.sort_values(["volatility", "sell_priority", "value_twd"], ascending=[True, True, False])
        remaining = draw_from_index(order_df.index.tolist(), remaining)
    return work, remaining


def maybe_rebalance(holdings: pd.DataFrame, frequency_years: int, current_year_idx: int) -> pd.DataFrame:
    if frequency_years <= 0:
        return holdings
    if current_year_idx % frequency_years != 0:
        return holdings
    work = holdings.copy()
    total = work["value_twd"].sum()
    if total <= 0:
        return work
    weights = work["weight"].to_numpy()
    work["value_twd"] = total * weights
    return work


def education_cost_for_year(calendar_year: int, phase_1_annual: float, phase_2_annual: float) -> float:
    if 2026 <= calendar_year <= 2033:
        return phase_1_annual
    if 2034 <= calendar_year <= 2038:
        return phase_2_annual
    return 0.0



def portfolio_effective_stats(holdings: pd.DataFrame, scenario: pd.Series, year_index: int, mode_override: Optional[str]) -> Dict[str, float]:
    start_total_assets = float(holdings["value_twd"].sum()) if not holdings.empty else 0.0
    if start_total_assets <= 0:
        return {
            "base_portfolio_return_pct": 0.0,
            "market_shift_pct": 0.0,
            "weighted_ai_excess_shift_pct": 0.0,
            "scenario_shift_pct": 0.0,
            "effective_expected_return_pct": 0.0,
            "effective_portfolio_vol_pct": 0.0,
            "ai_direct_weight_pct": 0.0,
            "base_portfolio_vol_pct": 0.0,
        }
    weights = holdings["value_twd"] / start_total_assets
    base_portfolio_return_pct = float((holdings["expected_return"] * weights).sum() * 100.0)
    base_portfolio_vol_pct = estimate_portfolio_volatility_pct(holdings)
    ai_direct_weight_pct = float(weights[holdings["ai_direct"].fillna(False)].sum() * 100.0)
    market_shift_pct = float(scenario.get("market_return_shift_pct", 0.0))
    weighted_ai_excess_shift_pct = ai_direct_weight_pct * float(scenario.get("ai_excess_return_pct", 0.0)) / 100.0
    vol_multiplier = float(scenario.get("vol_multiplier", 1.0))
    mode = (mode_override or scenario.get("mode", "fixed") or "fixed").strip()
    effective_expected_return_pct = base_portfolio_return_pct + market_shift_pct + weighted_ai_excess_shift_pct
    if mode == "path":
        early_years = int(scenario.get("early_negative_years", 0))
        recovery_years = int(scenario.get("recovery_years", 0))
        if year_index <= early_years:
            effective_expected_return_pct = float(scenario.get("early_negative_return_pct", -10.0)) + ai_direct_weight_pct * float(scenario.get("early_ai_penalty_pct", 0.0)) / 100.0
        elif year_index <= early_years + recovery_years:
            effective_expected_return_pct = base_portfolio_return_pct + market_shift_pct + weighted_ai_excess_shift_pct + float(scenario.get("recovery_boost_pct", 0.0))
    scenario_shift_pct = effective_expected_return_pct - base_portfolio_return_pct
    effective_portfolio_vol_pct = base_portfolio_vol_pct * vol_multiplier
    return {
        "base_portfolio_return_pct": float(base_portfolio_return_pct),
        "market_shift_pct": float(market_shift_pct),
        "weighted_ai_excess_shift_pct": float(weighted_ai_excess_shift_pct),
        "scenario_shift_pct": float(scenario_shift_pct),
        "effective_expected_return_pct": float(effective_expected_return_pct),
        "effective_portfolio_vol_pct": float(effective_portfolio_vol_pct),
        "ai_direct_weight_pct": float(ai_direct_weight_pct),
        "base_portfolio_vol_pct": float(base_portfolio_vol_pct),
    }

def simulate_portfolio(
    portfolio_df: pd.DataFrame,
    scenario: pd.Series,
    years: int,
    start_assets_twd: float,
    start_age: int,
    current_year: int,
    business_profit_annual: float,
    business_decay_pct: float,
    living_expense_annual: float,
    inflation_pct: float,
    edu_phase1_annual: float,
    edu_phase2_annual: float,
    mortgage_annual: float,
    inheritance_age: int,
    inherited_rent_monthly: float,
    withdrawal_strategy: str,
    rebalance_frequency_years: int,
    mode_override: Optional[str],
    seed: int = 42,
) -> pd.DataFrame:
    holdings = portfolio_to_sim_input(portfolio_df, start_assets_twd)
    if holdings.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    peak_assets = start_assets_twd
    ruin_year = None
    records = []

    for year_idx in range(1, years + 1):
        cal_year = current_year + year_idx - 1
        age = start_age + year_idx - 1
        start_total_assets = float(holdings["value_twd"].sum())
        start_cash = float(holdings.loc[holdings["asset_class"] == "Cash", "value_twd"].sum())

        market_z = rng.standard_normal()
        group_zs = {group: rng.standard_normal() for group in set(holdings["risk_group"].tolist())}
        returns = []
        for _, row in holdings.iterrows():
            returns.append(_calc_return_for_row(row, scenario, year_idx, mode_override, rng, market_z, group_zs))
        holdings["annual_return"] = returns
        holdings["value_twd"] = holdings["value_twd"] * (1.0 + holdings["annual_return"])

        end_before_cashflow = float(holdings["value_twd"].sum())
        portfolio_return_pct = (end_before_cashflow / start_total_assets - 1.0) * 100 if start_total_assets > 0 else 0.0

        effective_business_decay = float(scenario.get("business_decay_pct", business_decay_pct)) / 100.0
        business_income = max(0.0, business_profit_annual * ((1.0 - effective_business_decay) ** (year_idx - 1)))
        effective_inflation = float(scenario.get("inflation_pct", inflation_pct)) / 100.0
        living_expense = living_expense_annual * ((1.0 + effective_inflation) ** (year_idx - 1))
        education = education_cost_for_year(cal_year, edu_phase1_annual, edu_phase2_annual) * float(scenario.get("education_multiplier", 1.0))
        inheritance_income = inherited_rent_monthly * 12.0 if age >= inheritance_age else 0.0
        total_income = business_income + inheritance_income
        total_expense = living_expense + education + mortgage_annual
        net_cashflow = total_income - total_expense
        withdrawal = 0.0
        leftover_deficit = 0.0

        if net_cashflow >= 0:
            cash_mask = holdings["asset_class"] == "Cash"
            if cash_mask.any():
                first_cash_idx = holdings.index[cash_mask][0]
                holdings.at[first_cash_idx, "value_twd"] += net_cashflow
            else:
                cash_row = pd.DataFrame(
                    [{
                        "ticker": "Cash",
                        "name": "Cash",
                        "asset_class": "Cash",
                        "theme": "Cash",
                        "role_bucket": "安全墊",
                        "risk_group": "cash",
                        "ai_direct": False,
                        "expected_return": 0.015,
                        "volatility": 0.0,
                        "weight": 0.0,
                        "value_twd": net_cashflow,
                        "sell_priority": 0,
                        "annual_return": 0.0,
                    }]
                )
                holdings = pd.concat([holdings, cash_row], ignore_index=True)
        else:
            withdrawal = -net_cashflow
            holdings, leftover_deficit = _apply_withdrawal_strategy(holdings, withdrawal, withdrawal_strategy)

        holdings = maybe_rebalance(holdings, rebalance_frequency_years, year_idx)
        end_total_assets = float(max(0.0, holdings["value_twd"].sum()))
        end_cash = float(holdings.loc[holdings["asset_class"] == "Cash", "value_twd"].sum())
        peak_assets = max(peak_assets, end_total_assets)
        drawdown_pct = (end_total_assets / peak_assets - 1.0) * 100 if peak_assets > 0 else 0.0
        min_cash_buffer_months = end_cash / (total_expense / 12.0) if total_expense > 0 else 0.0

        if ruin_year is None and (end_total_assets <= 0 or leftover_deficit > 0):
            ruin_year = cal_year

        records.append(
            {
                "year_index": year_idx,
                "calendar_year": cal_year,
                "age": age,
                "start_assets_twd": start_total_assets,
                "portfolio_return_pct": portfolio_return_pct,
                "end_before_cashflow_twd": end_before_cashflow,
                "business_income_twd": business_income,
                "inheritance_income_twd": inheritance_income,
                "living_expense_twd": living_expense,
                "education_expense_twd": education,
                "mortgage_expense_twd": mortgage_annual,
                "total_income_twd": total_income,
                "total_expense_twd": total_expense,
                "net_cashflow_twd": net_cashflow,
                "withdrawal_twd": withdrawal,
                "uncovered_deficit_twd": leftover_deficit,
                "end_cash_twd": end_cash,
                "end_assets_twd": end_total_assets,
                "drawdown_pct": drawdown_pct,
                "cash_buffer_months": min_cash_buffer_months,
                "ruin_flag": 1 if ruin_year is not None else 0,
            }
        )

    result = pd.DataFrame(records)
    result["ruin_year"] = ruin_year
    return result


def summarize_simulation(result: pd.DataFrame, start_assets_twd: float) -> Dict[str, float]:
    if result.empty:
        return {}
    final_assets = float(result.iloc[-1]["end_assets_twd"])
    ruin_year = result.iloc[-1]["ruin_year"]
    max_drawdown = float(result["drawdown_pct"].min())
    worst_year = float(result["portfolio_return_pct"].min())
    ten_year_assets = float(result.loc[result["year_index"] == 10, "end_assets_twd"].iloc[0]) if (result["year_index"] == 10).any() else np.nan
    twenty_year_assets = float(result.loc[result["year_index"] == 20, "end_assets_twd"].iloc[0]) if (result["year_index"] == 20).any() else np.nan
    deficit_years = int((result["net_cashflow_twd"] < 0).sum())
    min_cash_buffer = float(result["cash_buffer_months"].min())

    ruin_component = 1.0 if pd.isna(ruin_year) else max(0.0, 1.0 - ((ruin_year - CURRENT_YEAR + 1) / max(1, result["year_index"].max())))
    drawdown_component = max(0.0, 1.0 - abs(max_drawdown) / 60.0)
    growth_component = min(1.0, final_assets / max(start_assets_twd, 1.0) / 2.0)
    buffer_component = min(1.0, min_cash_buffer / 12.0)
    cashflow_component = max(0.0, 1.0 - deficit_years / max(1, result.shape[0]))
    life_fit_score = round(100 * (0.35 * ruin_component + 0.20 * drawdown_component + 0.20 * growth_component + 0.15 * buffer_component + 0.10 * cashflow_component), 1)

    return {
        "最終資產終值": final_assets,
        "資產耗盡年份": ruin_year if not pd.isna(ruin_year) else None,
        "最大回撤 %": max_drawdown,
        "年度虧損最深值 %": worst_year,
        "10 年後淨資產": ten_year_assets,
        "20 年後淨資產": twenty_year_assets,
        "最低現金緩衝（月）": min_cash_buffer,
        "人生適配分數": life_fit_score,
    }


def run_monte_carlo_compare(
    current_portfolio: pd.DataFrame,
    recommended_portfolio: pd.DataFrame,
    scenario: pd.Series,
    years: int,
    start_assets_twd: float,
    start_age: int,
    current_year: int,
    business_profit_annual: float,
    business_decay_pct: float,
    living_expense_annual: float,
    inflation_pct: float,
    edu_phase1_annual: float,
    edu_phase2_annual: float,
    mortgage_annual: float,
    inheritance_age: int,
    inherited_rent_monthly: float,
    withdrawal_strategy: str,
    rebalance_frequency_years: int,
    mode_override: Optional[str],
    simulations: int = 300,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for i in range(simulations):
        current_result = simulate_portfolio(
            current_portfolio,
            scenario,
            years,
            start_assets_twd,
            start_age,
            current_year,
            business_profit_annual,
            business_decay_pct,
            living_expense_annual,
            inflation_pct,
            edu_phase1_annual,
            edu_phase2_annual,
            mortgage_annual,
            inheritance_age,
            inherited_rent_monthly,
            withdrawal_strategy,
            rebalance_frequency_years,
            mode_override,
            seed + i,
        )
        rec_result = simulate_portfolio(
            recommended_portfolio,
            scenario,
            years,
            start_assets_twd,
            start_age,
            current_year,
            business_profit_annual,
            business_decay_pct,
            living_expense_annual,
            inflation_pct,
            edu_phase1_annual,
            edu_phase2_annual,
            mortgage_annual,
            inheritance_age,
            inherited_rent_monthly,
            withdrawal_strategy,
            rebalance_frequency_years,
            mode_override,
            seed + i,
        )
        cur_summary = summarize_simulation(current_result, start_assets_twd)
        rec_summary = summarize_simulation(rec_result, start_assets_twd)
        rows.append(
            {
                "simulation": i + 1,
                "current_final_assets_twd": cur_summary.get("最終資產終值", np.nan),
                "recommended_final_assets_twd": rec_summary.get("最終資產終值", np.nan),
                "current_ruin": 0 if cur_summary.get("資產耗盡年份") is None else 1,
                "recommended_ruin": 0 if rec_summary.get("資產耗盡年份") is None else 1,
                "current_max_drawdown_pct": cur_summary.get("最大回撤 %", np.nan),
                "recommended_max_drawdown_pct": rec_summary.get("最大回撤 %", np.nan),
            }
        )
    sims = pd.DataFrame(rows)
    metrics = {
        "A 方案勝過 B 方案的機率": float((sims["current_final_assets_twd"] > sims["recommended_final_assets_twd"]).mean() * 100.0),
        "B 方案勝過 A 方案的機率": float((sims["recommended_final_assets_twd"] > sims["current_final_assets_twd"]).mean() * 100.0),
        "Current 資產耗盡機率": float((sims["current_ruin"] == 1).mean() * 100.0),
        "Recommended 資產耗盡機率": float((sims["recommended_ruin"] == 1).mean() * 100.0),
        "Current 終值中位數": float(sims["current_final_assets_twd"].median()),
        "Recommended 終值中位數": float(sims["recommended_final_assets_twd"].median()),
    }
    return sims, metrics


def default_control_values(uploaded_total_usd: Optional[float], fx_rate: float) -> Dict[str, float]:
    start_assets = uploaded_total_usd * fx_rate if uploaded_total_usd else 16_500_000.0
    return {
        "start_assets_twd": float(start_assets),
        "fx_rate": fx_rate,
        "business_profit_annual": 3_000_000.0,
        "business_decay_pct": 10.0,
        "living_expense_annual": 1_650_000.0,
        "inflation_pct": 2.5,
        "edu_phase1_annual": 750_000.0,
        "edu_phase2_annual": 1_500_000.0,
        "mortgage_annual": 270_000.0,
        "inheritance_age": 62,
        "inherited_rent_monthly": 60_000.0,
        "simulation_years": 20,
        "rebalance_frequency_years": 1,
        "cash_reserve_target_pct": 8.0,
    }


def metric_table(metrics: Dict[str, float], label: str) -> pd.DataFrame:
    rows = []
    for k, v in metrics.items():
        rows.append({"portfolio": label, "metric": k, "value": v})
    return pd.DataFrame(rows)


# ===== v5 overrides: income engines + Excel projection =====

def load_business_income_projection(source: BytesIO | str | Path) -> Tuple[List[float], pd.DataFrame]:
    """
    Read 妥妥租 business income projection from xlsx.
    Expected sheet name: 預測未來十年稅後淨利表
    Returns (annual net profit list, preview dataframe)
    """
    try:
        xl = pd.ExcelFile(source)
    except Exception:
        return [], pd.DataFrame(columns=["year", "net_profit_twd"])

    target_sheet = None
    preferred = ["預測未來十年稅後淨利表", "預測未來十年稅後淨利", "稅後淨利", "Sheet1"]
    for cand in preferred:
        if cand in xl.sheet_names:
            target_sheet = cand
            break
    if target_sheet is None:
        target_sheet = xl.sheet_names[0]

    try:
        raw = xl.parse(target_sheet)
    except Exception:
        return [], pd.DataFrame(columns=["year", "net_profit_twd"])

    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    year_col, profit_col = None, None
    for c in raw.columns:
        low = c.lower()
        if year_col is None and ("年" in c or "year" in low):
            year_col = c
        if profit_col is None and (("稅後" in c and "利" in c) or "net profit" in low or ("profit" in low and "gross" not in low)):
            profit_col = c
    if year_col is None or profit_col is None:
        if raw.shape[1] >= 2:
            year_col = raw.columns[0]
            profit_col = raw.columns[1]
        else:
            return [], pd.DataFrame(columns=["year", "net_profit_twd"])

    df = pd.DataFrame({
        "year": pd.to_numeric(raw[year_col], errors="coerce"),
        "net_profit_twd": raw[profit_col].map(_clean_numeric),
    }).dropna(subset=["year"])
    if df.empty:
        return [], pd.DataFrame(columns=["year", "net_profit_twd"])
    df["year"] = df["year"].astype(int)
    df = df.groupby("year", as_index=False)["net_profit_twd"].sum().sort_values("year").reset_index(drop=True)
    return df["net_profit_twd"].tolist(), df


def default_control_values(uploaded_total_usd: Optional[float], fx_rate: float) -> Dict[str, float]:
    start_assets = uploaded_total_usd * fx_rate if uploaded_total_usd else 16_500_000.0
    return {
        "start_assets_twd": float(start_assets),
        "fx_rate": fx_rate,
        "salary_annual": 1_200_000.0,
        "salary_growth_pct": 2.0,
        "retirement_age": 65,
        "tuotuozu_base_annual": 3_000_000.0,
        "tuotuozu_decay_pct": 10.0,
        "living_expense_annual": 1_650_000.0,
        "inflation_pct": 2.5,
        "edu_phase1_annual": 750_000.0,
        "edu_phase2_annual": 1_500_000.0,
        "mortgage_annual": 270_000.0,
        "inheritance_age": 62,
        "inherited_rent_monthly": 60_000.0,
        "simulation_years": 20,
        "rebalance_frequency_years": 1,
        "cash_reserve_target_pct": 8.0,
        "monte_carlo_sims": 500,
    }


def _salary_income_for_year(salary_annual: float, growth_pct: float, start_age: int, retirement_age: int, year_idx: int) -> float:
    age = start_age + year_idx - 1
    if age >= retirement_age:
        return 0.0
    growth = growth_pct / 100.0
    return max(0.0, salary_annual * ((1.0 + growth) ** (year_idx - 1)))


def _tuotuozu_income_for_year(
    year_idx: int,
    base_annual: float,
    decay_pct: float,
    mode: str,
    projection_list: Optional[List[float]] = None,
    fallback_mode: str = "continue_decay_from_last_value",
) -> float:
    projection_list = projection_list or []
    if mode == "Excel 預測":
        if year_idx <= len(projection_list):
            return max(0.0, float(projection_list[year_idx - 1]))
        if fallback_mode == "zero_after_list_end":
            return 0.0
        if len(projection_list) == 0:
            decay = decay_pct / 100.0
            return max(0.0, base_annual * ((1.0 - decay) ** (year_idx - 1)))
        last_value = float(projection_list[-1])
        decay = decay_pct / 100.0
        extra_years = year_idx - len(projection_list)
        return max(0.0, last_value * ((1.0 - decay) ** extra_years))
    decay = decay_pct / 100.0
    return max(0.0, base_annual * ((1.0 - decay) ** (year_idx - 1)))


def simulate_portfolio(
    portfolio_df: pd.DataFrame,
    scenario: pd.Series,
    years: int,
    start_assets_twd: float,
    start_age: int,
    current_year: int,
    salary_annual: float,
    salary_growth_pct: float,
    retirement_age: int,
    tuotuozu_mode: str,
    tuotuozu_base_annual: float,
    tuotuozu_decay_pct: float,
    tuotuozu_projection_list: Optional[List[float]],
    tuotuozu_fallback_mode: str,
    living_expense_annual: float,
    inflation_pct: float,
    edu_phase1_annual: float,
    edu_phase2_annual: float,
    mortgage_annual: float,
    inheritance_age: int,
    inherited_rent_monthly: float,
    withdrawal_strategy: str,
    rebalance_frequency_years: int,
    mode_override: Optional[str],
    seed: int = 42,
) -> pd.DataFrame:
    holdings = portfolio_to_sim_input(portfolio_df, start_assets_twd)
    if holdings.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    peak_assets = start_assets_twd
    ruin_year = None
    records = []

    for year_idx in range(1, years + 1):
        cal_year = current_year + year_idx - 1
        age = start_age + year_idx - 1
        start_total_assets = float(holdings["value_twd"].sum())

        effect_stats = portfolio_effective_stats(holdings, scenario, year_idx, mode_override)
        market_z = rng.standard_normal()
        group_zs = {group: rng.standard_normal() for group in set(holdings["risk_group"].tolist())}
        returns = []
        for _, row in holdings.iterrows():
            returns.append(_calc_return_for_row(row, scenario, year_idx, mode_override, rng, market_z, group_zs))
        holdings["annual_return"] = returns
        holdings["value_twd"] = holdings["value_twd"] * (1.0 + holdings["annual_return"])

        end_before_cashflow = float(holdings["value_twd"].sum())
        portfolio_return_pct = (end_before_cashflow / start_total_assets - 1.0) * 100 if start_total_assets > 0 else 0.0
        portfolio_vol_estimate_pct = estimate_portfolio_volatility_pct(holdings)
        scenario_drawdown_hint_pct = abs(float(scenario.get("max_drawdown_hint_pct", 0.0)))
        mode_name = (mode_override or scenario.get("mode", "fixed") or "fixed").strip()
        base_stress_factor = 0.60 if mode_name in {"fixed", "path"} else 0.35
        stress_factor = min(0.95, base_stress_factor + scenario_drawdown_hint_pct / 200.0)
        intrayear_trough_return_pct = min(
            portfolio_return_pct,
            portfolio_return_pct - portfolio_vol_estimate_pct * stress_factor,
        )
        intrayear_trough_assets_twd = max(0.0, start_total_assets * (1.0 + intrayear_trough_return_pct / 100.0))

        salary_income = _salary_income_for_year(salary_annual, salary_growth_pct, start_age, retirement_age, year_idx)
        effective_decay = float(scenario.get("business_decay_pct", tuotuozu_decay_pct))
        tuotuozu_income = _tuotuozu_income_for_year(
            year_idx,
            tuotuozu_base_annual,
            effective_decay,
            tuotuozu_mode,
            tuotuozu_projection_list,
            tuotuozu_fallback_mode,
        )
        effective_inflation = float(scenario.get("inflation_pct", inflation_pct)) / 100.0
        living_expense = living_expense_annual * ((1.0 + effective_inflation) ** (year_idx - 1))
        education = education_cost_for_year(cal_year, edu_phase1_annual, edu_phase2_annual) * float(scenario.get("education_multiplier", 1.0))
        inheritance_income = inherited_rent_monthly * 12.0 if age >= inheritance_age else 0.0

        total_income = salary_income + tuotuozu_income + inheritance_income
        total_expense = living_expense + education + mortgage_annual
        net_cashflow = total_income - total_expense
        withdrawal = 0.0
        leftover_deficit = 0.0

        if net_cashflow >= 0:
            cash_mask = holdings["asset_class"] == "Cash"
            if cash_mask.any():
                first_cash_idx = holdings.index[cash_mask][0]
                holdings.at[first_cash_idx, "value_twd"] += net_cashflow
            else:
                cash_row = pd.DataFrame(
                    [{
                        "ticker": "Cash",
                        "name": "Cash",
                        "asset_class": "Cash",
                        "theme": "Cash",
                        "role_bucket": "安全墊",
                        "risk_group": "cash",
                        "ai_direct": False,
                        "expected_return": 0.015,
                        "volatility": 0.0,
                        "weight": 0.0,
                        "value_twd": net_cashflow,
                        "sell_priority": 0,
                        "annual_return": 0.0,
                    }]
                )
                holdings = pd.concat([holdings, cash_row], ignore_index=True)
        else:
            withdrawal = -net_cashflow
            holdings, leftover_deficit = _apply_withdrawal_strategy(holdings, withdrawal, withdrawal_strategy)

        holdings = maybe_rebalance(holdings, rebalance_frequency_years, year_idx)
        end_total_assets = float(max(0.0, holdings["value_twd"].sum()))
        end_cash = float(holdings.loc[holdings["asset_class"] == "Cash", "value_twd"].sum())
        peak_assets_before_year = max(peak_assets, start_total_assets)
        drawdown_base_assets_twd = min(end_total_assets, intrayear_trough_assets_twd)
        drawdown_pct = (drawdown_base_assets_twd / peak_assets_before_year - 1.0) * 100 if peak_assets_before_year > 0 else 0.0
        peak_assets = max(peak_assets, end_total_assets)
        cash_buffer_months = end_cash / (total_expense / 12.0) if total_expense > 0 else 0.0

        if ruin_year is None and (end_total_assets <= 0 or leftover_deficit > 0):
            ruin_year = cal_year

        records.append(
            {
                "year_index": year_idx,
                "calendar_year": cal_year,
                "age": age,
                "start_assets_twd": start_total_assets,
                "base_portfolio_return_pct": effect_stats["base_portfolio_return_pct"],
                "market_return_shift_pct": effect_stats["market_shift_pct"],
                "weighted_ai_excess_shift_pct": effect_stats["weighted_ai_excess_shift_pct"],
                "scenario_shift_pct": effect_stats["scenario_shift_pct"],
                "effective_return_pct": effect_stats["effective_expected_return_pct"],
                "effective_portfolio_return_pct": effect_stats["effective_expected_return_pct"],
                "base_portfolio_vol_pct": effect_stats["base_portfolio_vol_pct"],
                "effective_portfolio_vol_pct": effect_stats["effective_portfolio_vol_pct"],
                "ai_direct_weight_pct": effect_stats["ai_direct_weight_pct"],
                "portfolio_return_pct": portfolio_return_pct,
                "portfolio_vol_estimate_pct": portfolio_vol_estimate_pct,
                "intrayear_trough_assets_twd": intrayear_trough_assets_twd,
                "drawdown_method": "peak_assets_before_year vs min(end_total_assets, intrayear_trough_assets_twd)",
                "end_before_cashflow_twd": end_before_cashflow,
                "salary_income_twd": salary_income,
                "tuotuozu_income_twd": tuotuozu_income,
                "business_income_twd": tuotuozu_income,
                "inheritance_income_twd": inheritance_income,
                "living_expense_twd": living_expense,
                "education_expense_twd": education,
                "mortgage_expense_twd": mortgage_annual,
                "total_income_twd": total_income,
                "total_expense_twd": total_expense,
                "net_cashflow_twd": net_cashflow,
                "withdrawal_twd": withdrawal,
                "uncovered_deficit_twd": leftover_deficit,
                "end_cash_twd": end_cash,
                "end_assets_twd": end_total_assets,
                "drawdown_pct": drawdown_pct,
                "cash_buffer_months": cash_buffer_months,
                "ruin_flag": 1 if ruin_year is not None else 0,
            }
        )

    result = pd.DataFrame(records)
    result["ruin_year"] = ruin_year
    return result


def run_monte_carlo_compare(
    current_portfolio: pd.DataFrame,
    recommended_portfolio: pd.DataFrame,
    custom_portfolio: Optional[pd.DataFrame],
    scenario: pd.Series,
    years: int,
    start_assets_twd: float,
    start_age: int,
    current_year: int,
    salary_annual: float,
    salary_growth_pct: float,
    retirement_age: int,
    tuotuozu_mode: str,
    tuotuozu_base_annual: float,
    tuotuozu_decay_pct: float,
    tuotuozu_projection_list: Optional[List[float]],
    tuotuozu_fallback_mode: str,
    living_expense_annual: float,
    inflation_pct: float,
    edu_phase1_annual: float,
    edu_phase2_annual: float,
    mortgage_annual: float,
    inheritance_age: int,
    inherited_rent_monthly: float,
    withdrawal_strategy: str,
    rebalance_frequency_years: int,
    mode_override: Optional[str],
    simulations: int = 300,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    portfolio_inputs = {"Current": current_portfolio, "Recommended": recommended_portfolio}
    if custom_portfolio is not None:
        portfolio_inputs["Custom"] = custom_portfolio

    rows = []
    for i in range(simulations):
        sim_row = {"simulation": i + 1}
        finals = {}
        seed_bump = 1
        for name, portfolio in portfolio_inputs.items():
            result = simulate_portfolio(
                portfolio, scenario, years, start_assets_twd, start_age, current_year,
                salary_annual, salary_growth_pct, retirement_age,
                tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, tuotuozu_projection_list, tuotuozu_fallback_mode,
                living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual,
                inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override,
                seed=seed + i * 10 + seed_bump,
            )
            final_assets = float(result.iloc[-1]["end_assets_twd"]) if not result.empty else 0.0
            ruin_flag = int((result["uncovered_deficit_twd"] > 0).any() or (result["end_assets_twd"] <= 0).any()) if not result.empty else 1
            max_drawdown = float(result["drawdown_pct"].min()) if not result.empty else -100.0
            key = name.lower()
            sim_row[f"{key}_final_assets_twd"] = final_assets
            sim_row[f"{key}_ruin"] = ruin_flag
            sim_row[f"{key}_max_drawdown_pct"] = max_drawdown
            finals[name] = final_assets
            seed_bump += 1
        sim_row["recommended_beats_current"] = int(finals.get("Recommended", 0.0) > finals.get("Current", 0.0))
        if "Custom" in finals:
            sim_row["recommended_beats_custom"] = int(finals.get("Recommended", 0.0) > finals.get("Custom", 0.0))
            sim_row["current_beats_custom"] = int(finals.get("Current", 0.0) > finals.get("Custom", 0.0))
        rows.append(sim_row)
    sims = pd.DataFrame(rows)
    metrics: Dict[str, float] = {}
    for name in portfolio_inputs:
        key = name.lower()
        final_col = f"{key}_final_assets_twd"
        ruin_col = f"{key}_ruin"
        drawdown_col = f"{key}_max_drawdown_pct"
        metrics[f"{name} P50 終值"] = float(sims[final_col].median())
        metrics[f"{name} P25 終值"] = float(sims[final_col].quantile(0.25))
        metrics[f"{name} P75 終值"] = float(sims[final_col].quantile(0.75))
        metrics[f"{name} P5 Worst Case 終值"] = float(sims[final_col].quantile(0.05))
        metrics[f"{name} 資產耗盡機率"] = float(sims[ruin_col].mean() * 100.0)
        metrics[f"{name} 最大回撤中位數"] = float(sims[drawdown_col].median())
    metrics["Recommended 勝過 Current 機率"] = float(sims["recommended_beats_current"].mean() * 100.0)
    if "recommended_beats_custom" in sims.columns:
        metrics["Recommended 勝過 Custom 機率"] = float(sims["recommended_beats_custom"].mean() * 100.0)
    if "current_beats_custom" in sims.columns:
        metrics["Current 勝過 Custom 機率"] = float(sims["current_beats_custom"].mean() * 100.0)
    return sims, metrics
