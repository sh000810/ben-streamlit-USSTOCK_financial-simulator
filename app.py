
from __future__ import annotations

import json
import importlib.util
import zipfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (
    CURRENT_AGE,
    CURRENT_YEAR,
    apply_cash_reserve_target,
    build_comparison,
    build_defensive_portfolio,
    build_recommended_portfolio,
    bucket_exposure,
    compute_etf_overlap,
    compute_risk_duplicate_metrics,
    default_control_values,
    get_scenario_row,
    load_business_income_projection,
    load_positions,
    normalize_weights,
    run_monte_carlo_compare,
    scenario_table,
    simulate_portfolio,
    summarize_simulation,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV = DATA_DIR / "Individual-Positions-2026-04-22-184253(1).csv"
SAMPLE_BIZ_XLSX = DATA_DIR / "妥妥租_預測.xlsx"
SETTINGS_PATH = DATA_DIR / "user_saved_settings.json"

st.set_page_config(page_title="Ben 財務與投資模擬器", layout="wide")
st.title("Ben 財務與投資模擬器 · 透明化驗算版 v7.7")
st.caption("加入：Cash bucket 修正、組合加權報酬/波動、情境額外加成揭露、單一路徑與 Monte Carlo 明確分離、年度生效報酬 LOG、AI 驗證包 ZIP 一鍵下載。")

st.markdown("""
<style>
button[data-baseweb="tab"] {
    background-color: #1f2937 !important;
    color: #d1d5db !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #374151 !important;
    padding: 10px 18px !important;
    margin-right: 6px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease-in-out !important;
}
button[data-baseweb="tab"]:hover {
    background-color: #334155 !important;
    color: #ffffff !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    background-color: #dc2626 !important;
    color: white !important;
    border-bottom: 2px solid #fca5a5 !important;
}
div[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #374151;
    padding-bottom: 2px;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    padding: 0.6rem 0.8rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)


def fmt_num(x: Any, digits: int = 0) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "—"


def fmt_pct(x: Any, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):,.{digits}f}%"
    except Exception:
        return "—"


def fmt_human(x: Any, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "—"
        x = float(x)
        abs_x = abs(x)
        if abs_x >= 1_000_000_000:
            return f"{x/1_000_000_000:.{digits}f}B"
        if abs_x >= 1_000_000:
            return f"{x/1_000_000:.{digits}f}M"
        if abs_x >= 1_000:
            return f"{x/1_000:.{digits}f}K"
        return f"{x:,.{digits}f}" if digits > 0 else f"{x:,.0f}"
    except Exception:
        return "—"


def sidebar_number_input(label: str, *, value: float, min_value: float = 0.0, max_value: float | None = None, step: float = 1.0, digits: int = 1, help: str | None = None, key: str | None = None) -> float:
    kwargs = {
        "label": label,
        "min_value": min_value,
        "value": float(value),
        "step": step,
        "format": f"%.{digits}f",
        "help": help,
    }
    if max_value is not None:
        kwargs["max_value"] = max_value
    if key is not None:
        kwargs["key"] = key
    result = st.number_input(**kwargs)
    st.caption(f"目前值：{fmt_num(result, digits)}")
    return float(result)


SUPER_WINNER_AI = {"NVDA", "AMD", "ANET", "AVGO", "ASML", "MRVL", "MU"}
MEGA_CAP_AI = {"MSFT", "GOOG", "GOOGL", "AMZN", "META", "AAPL"}
WORKFLOW_SET = {"NOW", "ORCL", "DDOG", "DT"}
CORE_COMPOUNDER_SET = {"BRK.B", "MA", "ETN", "EQIX", "QQQ", "VOO"}
DEFENSIVE_SET = {"BTI", "LMT", "DIS"}
SPECULATIVE_SET = {"CUK", "INTC", "NNDM"}


def detect_bucket(row: pd.Series) -> str:
    ticker = str(row.get("ticker", "")).upper().strip()
    asset_class = str(row.get("asset_class", "")).lower()
    role_bucket = str(row.get("role_bucket", "")).lower()
    if ticker in {"CASH", "CASH & CASH INVESTMENTS"} or "cash" in asset_class:
        return "cash"
    if ticker in SUPER_WINNER_AI:
        return "ai_super_winner"
    if ticker in MEGA_CAP_AI:
        return "mega_cap_ai"
    if ticker in WORKFLOW_SET:
        return "workflow"
    if ticker in CORE_COMPOUNDER_SET or "etf" in asset_class or "核心" in role_bucket:
        return "core_compounder"
    if ticker in DEFENSIVE_SET or "防守" in role_bucket:
        return "defensive"
    if ticker in SPECULATIVE_SET or "高風險" in role_bucket:
        return "speculative"
    return "general_equity"


def assumption_from_bucket(row: pd.Series) -> Dict[str, Any]:
    bucket = detect_bucket(row)
    hist = row.get("hist_10y_cagr_pct", None)
    hist_ok = hist is not None and pd.notna(hist)
    legacy_return = float(row.get("expected_return_pct", 0) or 0)
    legacy_vol = float(row.get("volatility_pct", 0) or 0)

    if bucket == "cash":
        rule_return, rule_vol = 1.5, 0.0
        note = "現金 / 類現金，固定以 1.5% 報酬、0% 波動處理。"
    elif bucket == "ai_super_winner":
        rule_return, rule_vol = 10.0, max(32.0, legacy_vol or 0)
        note = "AI爆發股，模型值採保守折現；若未接真實10年價格資料，先用規則預設。"
    elif bucket == "mega_cap_ai":
        rule_return, rule_vol = 9.0, max(24.0, legacy_vol or 0)
        note = "Mega Cap AI 平台，保留成長但不直接照搬過去10年。"
    elif bucket == "workflow":
        rule_return, rule_vol = 8.5, max(24.0, legacy_vol or 0)
        note = "高品質軟體 / workflow，若缺10年資料，先用類別基準。"
    elif bucket == "core_compounder":
        rule_return, rule_vol = 7.5, max(18.0, legacy_vol or 0)
        note = "核心底盤 / 複利股 / ETF，報酬假設偏保守，重視可持續性。"
    elif bucket == "defensive":
        rule_return, rule_vol = 6.5, max(16.0, legacy_vol or 0)
        note = "防守 / 低成長，預期報酬較低，但波動也應較低。"
    elif bucket == "speculative":
        rule_return, rule_vol = 4.5, max(35.0, legacy_vol or 0)
        note = "高風險 / speculative，低報酬高波動處理。"
    else:
        rule_return, rule_vol = max(6.5, legacy_return or 8.0), max(20.0, legacy_vol or 22.0)
        note = "一般股票，暫以 bucket default 處理。"

    if hist_ok:
        if bucket == "ai_super_winner":
            model_return = min(float(hist) * 0.35, 12.0)
            method = "min(hist_10y_cagr_pct * 0.35, 12.0)"
        elif bucket == "mega_cap_ai":
            model_return = min(float(hist) * 0.45, 10.0)
            method = "min(hist_10y_cagr_pct * 0.45, 10.0)"
        elif bucket == "workflow":
            model_return = min(float(hist) * 0.45, 10.0)
            method = "min(hist_10y_cagr_pct * 0.45, 10.0)"
        elif bucket == "core_compounder":
            model_return = min(float(hist) * 0.60, 9.0)
            method = "min(hist_10y_cagr_pct * 0.60, 9.0)"
        elif bucket == "defensive":
            model_return = min(max(float(hist) * 0.50, 5.5), 7.5)
            method = "defensive_hist_discount"
        elif bucket == "speculative":
            model_return = min(max(float(hist) * 0.25, 3.0), 6.0)
            method = "speculative_hist_discount"
        elif bucket == "cash":
            model_return = rule_return
            method = "cash_fixed_1p5_0vol"
        else:
            model_return = rule_return
            method = f"bucket_default_{bucket}"
        confidence = "medium"
        hist_source = row.get("hist_10y_source", "manual")
    else:
        model_return = rule_return
        method = "cash_fixed_1p5_0vol" if bucket == "cash" else f"bucket_default_{bucket}"
        confidence = "high" if bucket == "cash" else "low"
        hist_source = row.get("hist_10y_source", "rule_based_cash") if bucket == "cash" else row.get("hist_10y_source", "unavailable")

    vol_5y = row.get("vol_5y_weekly_pct", None)
    vol_3y = row.get("vol_3y_weekly_pct", None)
    if bucket == "cash":
        model_vol = 0.0
        vol_method = "cash_fixed_zero_vol"
        confidence = "high"
    elif pd.notna(vol_5y) and pd.notna(vol_3y):
        model_vol = 0.6 * float(vol_5y) + 0.4 * float(vol_3y)
        vol_method = "0.6 * vol_5y_weekly_pct + 0.4 * vol_3y_weekly_pct"
        confidence = "high" if confidence != "low" else "medium"
    else:
        model_vol = rule_vol
        vol_method = f"bucket_default_vol_{bucket}"

    return {
        "classification_bucket": bucket,
        "hist_10y_cagr_pct": row.get("hist_10y_cagr_pct", pd.NA),
        "hist_10y_source": hist_source,
        "model_return_pct": row.get("model_return_pct", model_return) if pd.notna(row.get("model_return_pct", pd.NA)) else model_return,
        "model_return_method": row.get("model_return_method", method) or method,
        "vol_5y_weekly_pct": row.get("vol_5y_weekly_pct", pd.NA),
        "vol_3y_weekly_pct": row.get("vol_3y_weekly_pct", pd.NA),
        "model_vol_pct": row.get("model_vol_pct", model_vol) if pd.notna(row.get("model_vol_pct", pd.NA)) else model_vol,
        "model_vol_method": row.get("model_vol_method", vol_method) or vol_method,
        "confidence": row.get("confidence", confidence) or confidence,
        "notes": row.get("notes", note) or note,
        "updated_at": row.get("updated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    }


def enrich_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col, default in {
        "hist_10y_cagr_pct": pd.NA,
        "hist_10y_source": "unavailable",
        "model_return_pct": pd.NA,
        "model_return_method": "",
        "vol_5y_weekly_pct": pd.NA,
        "vol_3y_weekly_pct": pd.NA,
        "model_vol_pct": pd.NA,
        "model_vol_method": "",
        "confidence": "low",
        "notes": "",
        "updated_at": "",
        "classification_bucket": "",
    }.items():
        if col not in work.columns:
            work[col] = default
    assumption_rows = [assumption_from_bucket(work.iloc[i]) for i in range(len(work))]
    assump_df = pd.DataFrame(assumption_rows, index=work.index)
    for col in assump_df.columns:
        work[col] = assump_df[col]
    return work


def prepare_simulation_df(df: pd.DataFrame) -> pd.DataFrame:
    work = enrich_assumptions(df)
    work["expected_return_pct"] = pd.to_numeric(work["model_return_pct"], errors="coerce").fillna(pd.to_numeric(work.get("expected_return_pct"), errors="coerce")).fillna(0.0)
    work["volatility_pct"] = pd.to_numeric(work["model_vol_pct"], errors="coerce").fillna(pd.to_numeric(work.get("volatility_pct"), errors="coerce")).fillna(0.0)
    return work


def format_table_df(df: pd.DataFrame, pct_cols: List[str] | None = None, human_cols: List[str] | None = None, keep_cols: List[str] | None = None) -> pd.DataFrame:
    pct_cols = pct_cols or []
    human_cols = human_cols or []
    work = df.copy()
    if keep_cols is not None:
        keep = [c for c in keep_cols if c in work.columns]
        work = work[keep]
    for col in work.columns:
        if col in pct_cols:
            work[col] = work[col].map(lambda x: fmt_pct(x, 1))
        elif col in human_cols:
            work[col] = work[col].map(lambda x: fmt_num(x, 0))
    return work


def portfolio_weight_distance_pct(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    a = normalize_weights(df_a[df_a.get("include", True)].copy()) if not df_a.empty else pd.DataFrame(columns=["ticker", "weight_pct"])
    b = normalize_weights(df_b[df_b.get("include", True)].copy()) if not df_b.empty else pd.DataFrame(columns=["ticker", "weight_pct"])
    a_map = a.set_index("ticker")["weight_pct"].to_dict() if not a.empty else {}
    b_map = b.set_index("ticker")["weight_pct"].to_dict() if not b.empty else {}
    tickers = sorted(set(a_map) | set(b_map))
    if not tickers:
        return 0.0
    return float(sum(abs(a_map.get(t, 0.0) - b_map.get(t, 0.0)) for t in tickers) / 2.0)


def portfolio_runtime_label(editor_key: str, default_label: str) -> str:
    if editor_key in st.session_state:
        return f"目前頁面編輯值（{editor_key}）"
    return default_label


def portfolio_context_rows(current_df: pd.DataFrame, recommended_df: pd.DataFrame, custom_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df, label in [
        ("Current", current_df, portfolio_runtime_label("current_editor", "fallback: saved_current / current_default")),
        ("Recommended", recommended_df, portfolio_runtime_label("recommended_editor", "fallback: saved_recommended / build_recommended_portfolio")),
        ("Custom", custom_df, portfolio_runtime_label("custom_editor", "fallback: saved_custom / build_defensive_portfolio")),
    ]:
        included = df[df.get("include", True)].copy()
        included = normalize_weights(included) if not included.empty else included
        cash_pct = float(included.loc[included["asset_class"].astype(str).str.lower().eq("cash"), "weight_pct"].sum()) if not included.empty and "asset_class" in included.columns else 0.0
        weighted_return = float((pd.to_numeric(included.get("expected_return_pct", 0), errors="coerce").fillna(0.0) * pd.to_numeric(included.get("weight_pct", 0), errors="coerce").fillna(0.0)).sum() / 100.0) if not included.empty else 0.0
        weighted_vol = float((pd.to_numeric(included.get("volatility_pct", 0), errors="coerce").fillna(0.0) * pd.to_numeric(included.get("weight_pct", 0), errors="coerce").fillna(0.0)).sum() / 100.0) if not included.empty else 0.0
        top5 = ", ".join((included.sort_values("weight_pct", ascending=False).head(5)["ticker"].astype(str).tolist())) if not included.empty else "—"
        rows.append({
            "方案": name,
            "目前使用來源": label,
            "納入筆數": int(included.shape[0]),
            "現金%": cash_pct,
            "加權報酬%": weighted_return,
            "加權波動%": weighted_vol,
            "前五大持股": top5,
        })
    return pd.DataFrame(rows)


def scenario_context_df(scenario_row: pd.Series, scenario_name: str, mode_override_value: str | None) -> pd.DataFrame:
    source_label = "目前頁面編輯值（scenario_editor）" if "scenario_editor" in st.session_state else "fallback: saved_scenarios / scenario_table"
    return pd.DataFrame([
        {
            "目前情境": scenario_name,
            "情境來源": source_label,
            "情境模式": scenario_row.get("mode", "—"),
            "mode_override": mode_override_value or "未覆蓋",
            "market_shift%": float(scenario_row.get("market_return_shift_pct", 0.0)),
            "ai_excess%": float(scenario_row.get("ai_excess_return_pct", 0.0)),
            "vol_multiplier": float(scenario_row.get("vol_multiplier", 1.0)),
            "max_drawdown_hint%": float(scenario_row.get("max_drawdown_hint_pct", 0.0)),
        }
    ])




def portfolio_scenario_effect_rows(current_df: pd.DataFrame, recommended_df: pd.DataFrame, custom_df: pd.DataFrame, scenario_row: pd.Series, mode_override_value: str | None) -> pd.DataFrame:
    rows = []
    mode_name = (mode_override_value or scenario_row.get("mode", "fixed") or "fixed")
    market_shift_pct = float(scenario_row.get("market_return_shift_pct", 0.0))
    ai_excess_pct = float(scenario_row.get("ai_excess_return_pct", 0.0))
    vol_multiplier = float(scenario_row.get("vol_multiplier", 1.0))
    recovery_boost_pct = float(scenario_row.get("recovery_boost_pct", 0.0))
    early_negative_return_pct = float(scenario_row.get("early_negative_return_pct", 0.0))
    early_ai_penalty_pct = float(scenario_row.get("early_ai_penalty_pct", 0.0))
    early_negative_years = int(scenario_row.get("early_negative_years", 0))
    recovery_years = int(scenario_row.get("recovery_years", 0))

    for name, df in [("Current", current_df), ("Recommended", recommended_df), ("Custom", custom_df)]:
        included = df[df.get("include", True)].copy()
        included = normalize_weights(included) if not included.empty else included
        weights = pd.to_numeric(included.get("weight_pct", 0), errors="coerce").fillna(0.0)
        returns = pd.to_numeric(included.get("expected_return_pct", 0), errors="coerce").fillna(0.0)
        vols = pd.to_numeric(included.get("volatility_pct", 0), errors="coerce").fillna(0.0)
        ai_flags = included.get("ai_direct", pd.Series(False, index=included.index)).fillna(False).astype(bool)
        base_return_pct = float((returns * weights).sum() / 100.0) if not included.empty else 0.0
        base_vol_pct = float((vols * weights).sum() / 100.0) if not included.empty else 0.0
        ai_weight_pct = float(weights[ai_flags].sum()) if not included.empty else 0.0
        weighted_ai_excess_shift_pct = ai_weight_pct * ai_excess_pct / 100.0
        scenario_shift_pct = market_shift_pct + weighted_ai_excess_shift_pct
        effective_portfolio_return_pct = base_return_pct + scenario_shift_pct
        if mode_name == "path":
            if early_negative_years > 0:
                effective_portfolio_return_pct = early_negative_return_pct + ai_weight_pct * early_ai_penalty_pct / 100.0
                scenario_shift_pct = effective_portfolio_return_pct - base_return_pct
            elif recovery_years > 0:
                effective_portfolio_return_pct = base_return_pct + market_shift_pct + weighted_ai_excess_shift_pct + recovery_boost_pct
                scenario_shift_pct = effective_portfolio_return_pct - base_return_pct
        rows.append({
            "方案": name,
            "基礎加權報酬率%": base_return_pct,
            "基礎加權波動率%": base_vol_pct,
            "AI 直接曝險權重%": ai_weight_pct,
            "market_return_shift_pct": market_shift_pct,
            "weighted_ai_excess_shift_pct": weighted_ai_excess_shift_pct,
            "scenario_shift_pct": scenario_shift_pct,
            "effective_portfolio_return_pct": effective_portfolio_return_pct,
            "vol_multiplier": vol_multiplier,
            "effective_portfolio_vol_pct": base_vol_pct * vol_multiplier,
            "模擬模式": mode_name,
        })
    return pd.DataFrame(rows)

def build_validation_checks(current_df: pd.DataFrame, recommended_df: pd.DataFrame, custom_df: pd.DataFrame, rec_res: pd.DataFrame | None, cur_res: pd.DataFrame | None, cus_res: pd.DataFrame | None, tuotuozu_mode: str, biz_projection_list: List[float], retirement_age: int, edu_phase2_annual: float, scenario_name: str) -> pd.DataFrame:
    checks = []
    for name, df in [("Current", current_df), ("Recommended", recommended_df), ("Custom", custom_df)]:
        weight_sum = float(df.loc[df["include"], "weight_pct"].sum()) if not df.empty else 0.0
        checks.append({"檢查項目": f"{name} 權重總和", "結果": "PASS" if abs(weight_sum - 100) <= 0.5 else "WARNING", "說明": f"目前合計 {weight_sum:.2f}%（模擬前會自動正規化）"})
        has_nan = df.replace([float("inf"), float("-inf")], pd.NA).isna().any().any()
        checks.append({"檢查項目": f"{name} NaN / null / inf", "結果": "WARNING" if has_nan else "PASS", "說明": "若有缺值，應回頭檢查 CSV 或假設欄位。"})
    if rec_res is None or rec_res.empty:
        checks.append({"檢查項目": "最小檢查集狀態", "結果": "INFO", "說明": "尚未產生完整模擬結果，但權重與缺值檢查已完成。"})
        return pd.DataFrame(checks)

    year_seq_ok = rec_res["calendar_year"].is_monotonic_increasing and rec_res["calendar_year"].nunique() == rec_res.shape[0]
    checks.append({"檢查項目": "年度序列", "結果": "PASS" if year_seq_ok else "FAIL", "說明": "年份應連續且不重複"})
    salary_ok = True
    post_ret = rec_res.loc[rec_res["age"] >= retirement_age, "salary_income_twd"]
    if not post_ret.empty:
        salary_ok = float(post_ret.max()) == 0.0
    checks.append({"檢查項目": "本業收入退休後歸零", "結果": "PASS" if salary_ok else "FAIL", "說明": "退休後薪資應歸零"})
    if tuotuozu_mode == "Excel 預測":
        checks.append({"檢查項目": "妥妥租 Excel 載入", "結果": "PASS" if len(biz_projection_list) > 0 else "FAIL", "說明": f"目前讀到 {len(biz_projection_list)} 年"})
    edu_peak_ok = rec_res.loc[rec_res["calendar_year"].between(2034, 2038), "education_expense_twd"].max() >= edu_phase2_annual * 0.9
    checks.append({"檢查項目": "教育費高峰", "結果": "PASS" if edu_peak_ok else "WARNING", "說明": "2034-2038 應明顯高於前段"})
    drawdown_zero = bool((rec_res["drawdown_pct"].abs().max() == 0) and (rec_res["portfolio_return_pct"].abs().max() > 0))
    checks.append({"檢查項目": "Drawdown 非常態 0%", "結果": "FAIL" if drawdown_zero else "PASS", "說明": "若有波動但最大回撤長期為 0，通常代表邏輯異常。"})
    different_weights = abs(float(recommended_df.loc[recommended_df["include"], "weight_pct"].sum()) - float(custom_df.loc[custom_df["include"], "weight_pct"].sum())) >= 0 # placeholder
    same_results = False
    near_same_results = False
    weight_distance_pct = portfolio_weight_distance_pct(recommended_df, custom_df)
    if cus_res is not None and not cus_res.empty:
        rec_final = float(rec_res["end_assets_twd"].iloc[-1])
        cus_final = float(cus_res["end_assets_twd"].iloc[-1])
        final_gap_pct = abs(rec_final - cus_final) / max(abs(rec_final), abs(cus_final), 1.0) * 100.0
        same_results = final_gap_pct <= 0.01 and weight_distance_pct >= 1.0
        near_same_results = final_gap_pct <= 0.50 and weight_distance_pct >= 5.0
    result_flag = "FAIL" if same_results else ("WARNING" if near_same_results else "PASS")
    checks.append({
        "檢查項目": "Custom / Recommended 結果是否異常過近",
        "結果": result_flag,
        "說明": f"情境：{scenario_name} / 權重距離 {weight_distance_pct:.1f}% / 若配置差很多但終值差不到 0.5%，代表配置可能沒真正進模擬或模型敏感度不足。",
    })
    ruin_missing = bool((rec_res["end_assets_twd"] < 0).any() and not (rec_res["ruin_flag"] == 1).any())
    checks.append({"檢查項目": "負資產耗盡邏輯", "結果": "FAIL" if ruin_missing else "PASS", "說明": "若資產為負，應標示耗盡年份。"})
    return pd.DataFrame(checks)


def build_diagnostic_summary(scenario_name: str, global_settings: Dict[str, Any], current_df: pd.DataFrame, recommended_df: pd.DataFrame, custom_df: pd.DataFrame, current_metrics: Dict[str, Any], cur_sum: Dict[str, Any], rec_sum: Dict[str, Any], cus_sum: Dict[str, Any], validation_df: pd.DataFrame) -> str:
    warnings = validation_df.loc[validation_df["結果"].isin(["WARNING", "FAIL"]), "檢查項目"].tolist()
    top_holdings = current_df.sort_values("weight_pct", ascending=False).head(10)[["ticker", "weight_pct"]]
    top_text = "\n".join([f"- {r.ticker}: {fmt_pct(r.weight_pct,1)}" for r in top_holdings.itertuples()])
    return f"""版本時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
全域設定摘要:
- 匯率: {global_settings['fx_rate']}
- 起始流動資產: {fmt_num(global_settings['start_assets_twd'],0)}
- 模擬年數: {global_settings['simulation_years']}
- Monte Carlo 次數: {global_settings['monte_carlo_sims']}
- 本業年薪: {fmt_num(global_settings['salary_annual'],0)}
- 妥妥租模式: {global_settings['tuotuozu_mode']}
- 妥妥租年度淨利: {fmt_num(global_settings['tuotuozu_base_annual'],0)}
- 通膨率: {fmt_pct(global_settings['inflation_pct'],1)}

當前選擇情境:
- {scenario_name}

結果摘要:
- Current 最終資產: {fmt_human(cur_sum.get('最終資產終值'),1)} / 最大回撤: {fmt_pct(cur_sum.get('最大回撤 %'),1)}
- Recommended 最終資產: {fmt_human(rec_sum.get('最終資產終值'),1)} / 最大回撤: {fmt_pct(rec_sum.get('最大回撤 %'),1)}
- Custom 最終資產: {fmt_human(cus_sum.get('最終資產終值'),1)} / 最大回撤: {fmt_pct(cus_sum.get('最大回撤 %'),1)}

前10大持股與權重:
{top_text}

Current 風險摘要:
- AI 直接曝險: {fmt_pct(current_metrics.get('直接 AI 曝險 %'),1)}
- ETF 重疊曝險: {fmt_pct(current_metrics.get('ETF 重疊曝險 %'),1)}
- 同跌風險程度: {fmt_num(current_metrics.get('同跌風險程度（0-100）'),0)}

驗證警告:
{chr(10).join(['- ' + w for w in warnings]) if warnings else '- 無重大警告'}
"""




@st.cache_data(show_spinner=False)
def build_mc_path_log(
    scenario_dict: Dict[str, Any],
    current_df_records: List[Dict[str, Any]],
    recommended_df_records: List[Dict[str, Any]],
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
    tuotuozu_projection_list: List[float],
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
    mode_override: str | None,
    simulations: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    scenario = pd.Series(scenario_dict)
    current_df = pd.DataFrame(current_df_records)
    recommended_df = pd.DataFrame(recommended_df_records)
    logs: List[Dict[str, Any]] = []
    for i in range(simulations):
        for label, port_df, seed_offset in [("Current", current_df, 1), ("Recommended", recommended_df, 2)]:
            result = simulate_portfolio(
                port_df, scenario, years, start_assets_twd, start_age, current_year,
                salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct,
                tuotuozu_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual,
                edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy,
                rebalance_frequency_years, mode_override, seed=seed + i * 2 + seed_offset
            )
            if result.empty:
                continue
            temp = result.copy()
            temp["portfolio"] = label
            temp["simulation_id"] = i + 1
            temp["random_seed"] = seed + i * 2 + seed_offset
            temp["peak_assets_twd"] = temp["end_assets_twd"].cummax()
            temp["investment_gain_twd"] = temp["end_before_cashflow_twd"] - temp["start_assets_twd"]
            rename_map = {
                "start_assets_twd": "begin_assets_twd",
                "mortgage_expense_twd": "mortgage_twd",
                "education_expense_twd": "edu_expense_twd",
                "business_income_twd": "tuotuozu_income_twd"
            }
            temp = temp.rename(columns=rename_map)
            keep_cols = [
                "portfolio","simulation_id","random_seed","year_index","calendar_year","begin_assets_twd",
                "base_portfolio_return_pct","market_return_shift_pct","weighted_ai_excess_shift_pct","scenario_shift_pct","effective_return_pct",
                "portfolio_return_pct","investment_gain_twd","salary_income_twd","tuotuozu_income_twd","inheritance_income_twd","total_income_twd",
                "living_expense_twd","edu_expense_twd","mortgage_twd","withdrawal_twd","end_assets_twd","peak_assets_twd",
                "drawdown_pct","net_cashflow_twd","uncovered_deficit_twd","ruin_flag"
            ]
            logs.extend(temp[keep_cols].to_dict(orient="records"))
    return logs


def build_full_log(
    global_settings: Dict[str, Any],
    current_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
    custom_df: pd.DataFrame,
    scenario_dict: Dict[str, Any],
    current_path: pd.DataFrame,
    recommended_path: pd.DataFrame,
    custom_path: pd.DataFrame,
    validation_df: pd.DataFrame,
    diagnostic_summary: str,
    mc_path_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    assumptions_cols = [
        "ticker","name","hist_10y_cagr_pct","hist_10y_source","model_return_pct","model_return_method",
        "vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct","model_vol_method","confidence","notes","updated_at"
    ]
    warnings = validation_df.loc[validation_df["結果"].isin(["WARNING","FAIL"])].to_dict(orient="records") if not validation_df.empty else []
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "global_settings": global_settings,
        "portfolio_snapshots": {
            "current_portfolio": current_df.to_dict(orient="records"),
            "recommended_portfolio": recommended_df.to_dict(orient="records"),
            "custom_portfolio": custom_df.to_dict(orient="records"),
        },
        "assumptions": {
            "current": current_df[[c for c in assumptions_cols if c in current_df.columns]].to_dict(orient="records"),
            "recommended": recommended_df[[c for c in assumptions_cols if c in recommended_df.columns]].to_dict(orient="records"),
            "custom": custom_df[[c for c in assumptions_cols if c in custom_df.columns]].to_dict(orient="records"),
        },
        "scenario_settings": scenario_dict,
        "simulation_paths": {
            "current": current_path.to_dict(orient="records"),
            "recommended": recommended_path.to_dict(orient="records"),
            "custom": custom_path.to_dict(orient="records"),
            "monte_carlo_current_recommended": mc_path_log,
        },
        "validation_results": {
            "checks": validation_df.to_dict(orient="records"),
            "warnings": warnings,
        },
        "diagnostic_summary": diagnostic_summary,
    }


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def df_from_saved(key: str, default_df: pd.DataFrame, saved: Dict[str, Any]) -> pd.DataFrame:
    value = saved.get(key)
    if not value:
        return default_df.copy()
    try:
        return pd.DataFrame(value)
    except Exception:
        return default_df.copy()


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(bool)
    return out.to_dict(orient="records")


def prepare_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def sanitize_for_json(obj: Any):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return sanitize_for_json(obj.to_dict())
    if pd.isna(obj):
        return None
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    return obj


def prepare_json_download(payload: Any) -> bytes:
    return json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2).encode("utf-8-sig")


def prepare_text_download(text: str) -> bytes:
    return text.encode("utf-8-sig")


def build_validation_bundle_zip(
    *,
    bundle_kind: str,
    settings_payload: Dict[str, Any],
    global_settings: Dict[str, Any],
    scenario_row_dict: Dict[str, Any],
    diagnostic_summary: str,
    validation_df: pd.DataFrame,
    portfolio_context_df: pd.DataFrame,
    scenario_context_runtime_df: pd.DataFrame,
    portfolio_effect_df: pd.DataFrame,
    current_edited: pd.DataFrame,
    recommended_edited: pd.DataFrame,
    custom_edited: pd.DataFrame,
    current_norm: pd.DataFrame,
    recommended_norm: pd.DataFrame,
    custom_norm: pd.DataFrame,
    current_path_df: pd.DataFrame,
    recommended_path_df: pd.DataFrame,
    custom_path_df: pd.DataFrame,
    scenario_table_df: pd.DataFrame,
    light_log: Dict[str, Any],
    full_log: Dict[str, Any] | None = None,
    sims_df: pd.DataFrame | None = None,
    mc_summary_df: pd.DataFrame | None = None,
    mc_metrics: Dict[str, Any] | None = None,
    matrix_df: pd.DataFrame | None = None,
    mc_path_log_records: List[Dict[str, Any]] | None = None,
) -> bytes:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = f"ben_validation_bundle_{bundle_kind}_{timestamp}"
    bio = BytesIO()

    def _writestr(zf: zipfile.ZipFile, name: str, data: bytes | str):
        zf.writestr(f"{root}/{name}", data if isinstance(data, bytes) else str(data).encode("utf-8-sig"))

    def _write_df(zf: zipfile.ZipFile, name: str, df: pd.DataFrame | None):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return
        _writestr(zf, name, prepare_download(df))

    readme = f"""Ben 財務與投資模擬器｜AI 驗證包

用途：
- 一次打包目前畫面實際吃進模擬的主要資料
- 提供給 AI 或人工做交叉驗證
- 避免只看截圖猜模型

這包內含：
1. settings / global 與 scenario runtime
2. edited 與 normalized 後的三組組合
3. deterministic annual paths
4. validation / diagnostic summary / logs
5. 若有啟用重型分析，另含 Monte Carlo summary/detail

包類型：{bundle_kind}
產生時間：{timestamp}
"""
    manifest = {
        "bundle_kind": bundle_kind,
        "generated_at": timestamp,
        "scenario_name": scenario_row_dict.get("scenario_name"),
        "simulation_years": global_settings.get("simulation_years"),
        "has_full_log": full_log is not None,
        "has_mc_detail": isinstance(sims_df, pd.DataFrame) and not sims_df.empty,
        "current_rows": int(len(current_norm)),
        "recommended_rows": int(len(recommended_norm)),
        "custom_rows": int(len(custom_norm)),
        "current_path_years": int(len(current_path_df)),
        "recommended_path_years": int(len(recommended_path_df)),
        "custom_path_years": int(len(custom_path_df)),
    }

    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        _writestr(zf, "README.txt", readme)
        _writestr(zf, "manifest.json", prepare_json_download(manifest))
        _writestr(zf, "logs/diagnostic_summary.txt", prepare_text_download(diagnostic_summary))
        _writestr(zf, "logs/light_log.json", prepare_json_download(light_log))
        if full_log is not None:
            _writestr(zf, "logs/full_log.json", prepare_json_download(full_log))
        if mc_metrics:
            _writestr(zf, "logs/mc_metrics.json", prepare_json_download(mc_metrics))

        _writestr(zf, "settings/settings_payload.json", prepare_json_download(settings_payload))
        _writestr(zf, "settings/global_settings.json", prepare_json_download(global_settings))
        _writestr(zf, "settings/scenario_runtime.json", prepare_json_download(scenario_row_dict))

        _write_df(zf, "contexts/portfolio_context.csv", portfolio_context_df)
        _write_df(zf, "contexts/scenario_context.csv", scenario_context_runtime_df)
        _write_df(zf, "contexts/portfolio_effect.csv", portfolio_effect_df)
        _write_df(zf, "validation/validation_checks.csv", validation_df)

        _write_df(zf, "portfolios/edited/current_edited.csv", current_edited)
        _write_df(zf, "portfolios/edited/recommended_edited.csv", recommended_edited)
        _write_df(zf, "portfolios/edited/custom_edited.csv", custom_edited)
        _write_df(zf, "portfolios/normalized/current_normalized.csv", current_norm)
        _write_df(zf, "portfolios/normalized/recommended_normalized.csv", recommended_norm)
        _write_df(zf, "portfolios/normalized/custom_normalized.csv", custom_norm)
        _write_df(zf, "scenarios/scenario_table.csv", scenario_table_df)

        _write_df(zf, "paths/current_single_path.csv", current_path_df)
        _write_df(zf, "paths/recommended_single_path.csv", recommended_path_df)
        _write_df(zf, "paths/custom_single_path.csv", custom_path_df)

        _write_df(zf, "monte_carlo/scenario_matrix.csv", matrix_df)
        _write_df(zf, "monte_carlo/summary.csv", mc_summary_df)
        _write_df(zf, "monte_carlo/detail.csv", sims_df)
        if mc_path_log_records:
            _write_df(zf, "monte_carlo/path_replay.csv", pd.DataFrame(mc_path_log_records))

    bio.seek(0)
    return bio.getvalue()


@st.cache_data(show_spinner=False)
def cached_load_positions(source):
    return load_positions(source)


@st.cache_data(show_spinner=False)
def cached_load_biz_projection(source):
    return load_business_income_projection(source)


def openpyxl_available() -> bool:
    return importlib.util.find_spec("openpyxl") is not None


def get_biz_excel_diagnostics(mode: str, source_mode: str | None, uploaded_file, built_in_path: Path, projection_list: list[float], projection_preview: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if mode != "Excel 預測":
        return notes
    if source_mode == "使用內建範例 Excel":
        notes.append(f"內建 Excel 路徑：{built_in_path}")
        notes.append(f"內建 Excel 是否存在：{'是' if built_in_path.exists() else '否'}")
    elif source_mode == "上傳 Excel":
        notes.append(f"是否已上傳 Excel：{'是' if uploaded_file is not None else '否'}")
    notes.append(f"openpyxl 是否可用：{'是' if openpyxl_available() else '否'}")
    notes.append(f"目前成功讀到的年度筆數：{len(projection_list)}")
    if projection_preview.empty:
        notes.append("預覽表目前為空。")
    return notes


saved = load_json(SETTINGS_PATH)

with st.sidebar:
    st.header("1) 資料來源")
    source_mode = st.radio(
        "持股來源",
        ["使用內建範例 CSV", "上傳新的 CSV"],
        index=0 if saved.get("source_mode", "使用內建範例 CSV") == "使用內建範例 CSV" else 1,
        help="Current Portfolio 的基礎來源。若你已經有最新券商 CSV，建議直接上傳。",
        key="source_mode",
    )
    uploaded_csv = st.file_uploader("上傳持股 CSV", type=["csv"], key="uploaded_csv")
    fx_rate = sidebar_number_input("USD/TWD 匯率", min_value=20.0, max_value=50.0, value=float(saved.get("fx_rate", 32.0)), step=0.1, digits=1, help="把美元持股換算成台幣時使用。")

    csv_source = SAMPLE_CSV if SAMPLE_CSV.exists() else uploaded_csv
    if source_mode == "上傳新的 CSV" and uploaded_csv is not None:
        csv_source = uploaded_csv

    current_positions_raw, missing_fields = cached_load_positions(str(csv_source) if isinstance(csv_source, Path) else csv_source)
    uploaded_total_usd = float(current_positions_raw["market_value_usd"].sum()) if not current_positions_raw.empty else None
    defaults = default_control_values(uploaded_total_usd, fx_rate)

    st.header("2) 模型控制")
    start_assets_twd = sidebar_number_input("起始流動資產 (TWD)", min_value=0.0, value=float(saved.get("start_assets_twd", defaults["start_assets_twd"])), step=100000.0, digits=1, help="三組方案比較時共用的起跑點。越一致，結果越能比較。")
    simulation_years = st.slider("模擬年數", 5, 30, int(saved.get("simulation_years", defaults["simulation_years"])), help="通常先看 20 年，才能看到退休與教育費高峰。")
    rebalance_frequency_years = st.selectbox("再平衡頻率", options=[0, 1, 2, 3, 5], index=[0,1,2,3,5].index(int(saved.get("rebalance_frequency_years", defaults["rebalance_frequency_years"]))), format_func=lambda x: "不再平衡" if x == 0 else f"每 {x} 年", help="多久把持股比例拉回目標權重。新手建議每 1 年。")
    withdrawal_strategy = st.selectbox("現金不足時提領方式", ["比例賣出", "先賣現金 / ETF", "先賣波動低資產"], index=["比例賣出", "先賣現金 / ETF", "先賣波動低資產"].index(saved.get("withdrawal_strategy", "比例賣出")), help="生活費不夠時，系統要先賣哪一種資產。")
    cash_reserve_target_pct = st.slider("現金保留比重 %", 0.0, 40.0, float(saved.get("cash_reserve_target_pct", defaults["cash_reserve_target_pct"])), step=0.5, help="越高越穩，但成長性通常會下降。")
    monte_carlo_sims = st.slider("Monte Carlo 次數", 100, 1500, int(saved.get("monte_carlo_sims", defaults.get("monte_carlo_sims", 500))), step=100, help="次數越高，勝率更穩定，但會比較慢。")
    enable_heavy_compute = st.toggle("預載重型分析（情境矩陣 / Monte Carlo）", value=bool(saved.get("enable_heavy_compute", False)), help="關掉時，5/6 頁與一般操作會快很多。需要看情境矩陣或 Monte Carlo 時再打開。")

    st.header("3) 本業收入（加薪→退休）")
    salary_annual = sidebar_number_input("本業年薪", min_value=0.0, value=float(saved.get("salary_annual", defaults["salary_annual"])), step=50000.0, digits=1, help="你可支配的本業年度收入，不是公司營收。")
    salary_growth_pct = sidebar_number_input("薪資成長率 %", min_value=-5.0, max_value=20.0, value=float(saved.get("salary_growth_pct", defaults["salary_growth_pct"])), step=0.1, digits=1, help="退休前本業收入每年成長率。新手可先用 2%。")
    retirement_age = st.number_input("退休年齡", min_value=45, max_value=90, value=int(saved.get("retirement_age", defaults["retirement_age"])), step=1, help="到了這個年齡，本業收入會歸零。")

    st.header("4) 妥妥租收入（遞減 / Excel）")
    tuotuozu_mode = st.radio("妥妥租模式", ["手動遞減", "Excel 預測"], index=0 if saved.get("tuotuozu_mode", "手動遞減") == "手動遞減" else 1, help="手動遞減適合快速試算；Excel 預測適合沿用你原本的預測表。")
    biz_projection_list, biz_projection_preview = [], pd.DataFrame(columns=["year", "net_profit_twd"])
    biz_excel_source_mode = saved.get("biz_excel_source_mode", "使用內建範例 Excel")
    uploaded_biz_excel = None
    biz_excel_error_message = ""
    if tuotuozu_mode == "Excel 預測":
        biz_excel_source_mode = st.radio("妥妥租 Excel", ["使用內建範例 Excel", "上傳 Excel"], index=0 if saved.get("biz_excel_source_mode", "使用內建範例 Excel") == "使用內建範例 Excel" else 1, help="如果你 repo 的 data 裡有妥妥租_預測.xlsx，可直接用內建版本。")
        uploaded_biz_excel = st.file_uploader("上傳 妥妥租_預測.xlsx", type=["xlsx"], key="uploaded_biz_excel")
        if not openpyxl_available():
            biz_excel_error_message = "目前環境缺少 openpyxl，pandas 無法讀取 .xlsx。請把 openpyxl 加進 requirements.txt 後重新部署。"
        elif biz_excel_source_mode == "上傳 Excel":
            if uploaded_biz_excel is not None:
                biz_projection_list, biz_projection_preview = cached_load_biz_projection(uploaded_biz_excel)
                if len(biz_projection_list) == 0:
                    biz_excel_error_message = "已上傳 Excel，但沒有成功解析出年度淨利資料。請檢查工作表名稱與欄位。"
            else:
                biz_excel_error_message = "目前選擇『上傳 Excel』，但尚未上傳檔案。"
        else:
            if SAMPLE_BIZ_XLSX.exists():
                biz_projection_list, biz_projection_preview = cached_load_biz_projection(str(SAMPLE_BIZ_XLSX))
                if len(biz_projection_list) == 0:
                    biz_excel_error_message = f"找到內建 Excel，但沒有成功解析內容：{SAMPLE_BIZ_XLSX}"
            else:
                biz_excel_error_message = f"找不到內建 Excel 檔案：{SAMPLE_BIZ_XLSX}"
    biz_excel_diagnostics = get_biz_excel_diagnostics(tuotuozu_mode, biz_excel_source_mode, uploaded_biz_excel, SAMPLE_BIZ_XLSX, biz_projection_list, biz_projection_preview)
    tuotuozu_base_annual = sidebar_number_input("妥妥租目前年度淨利", min_value=0.0, value=float(saved.get("tuotuozu_base_annual", defaults["tuotuozu_base_annual"])), step=100000.0, digits=1, help="當你不用 Excel 時，這個數字就是妥妥租收入的起點。")
    tuotuozu_decay_pct = sidebar_number_input("妥妥租年衰退率 %", min_value=-50.0, max_value=30.0, value=float(saved.get("tuotuozu_decay_pct", defaults["tuotuozu_decay_pct"])), step=0.5, digits=1, help="例如 10 代表每年衰退 10%。")
    tuotuozu_fallback_mode = st.selectbox("Excel 年數用完後", ["continue_decay_from_last_value", "zero_after_list_end"], index=0 if saved.get("tuotuozu_fallback_mode", "continue_decay_from_last_value") == "continue_decay_from_last_value" else 1, format_func=lambda x: "從最後一年繼續遞減" if x == "continue_decay_from_last_value" else "直接歸零", help="若 Excel 只有 10 年、你模擬 20 年，後面要怎麼處理。")

    st.header("5) 支出 / 繼承")
    living_expense_annual = sidebar_number_input("基礎生活費 / 年", min_value=0.0, value=float(saved.get("living_expense_annual", defaults["living_expense_annual"])), step=50000.0, digits=1, help="不含教育費的家庭年度生活支出。")
    inflation_pct = sidebar_number_input("通膨率 %", min_value=0.0, max_value=20.0, value=float(saved.get("inflation_pct", defaults["inflation_pct"])), step=0.1, digits=1, help="生活費逐年上升的速度。")
    edu_phase1_annual = sidebar_number_input("教育費 2026-2033 / 年", min_value=0.0, value=float(saved.get("edu_phase1_annual", defaults["edu_phase1_annual"])), step=50000.0, digits=1, help="教育費中度區間。")
    edu_phase2_annual = sidebar_number_input("教育費 2034-2038 / 年", min_value=0.0, value=float(saved.get("edu_phase2_annual", defaults["edu_phase2_annual"])), step=50000.0, digits=1, help="教育費高峰區間。")
    mortgage_annual = sidebar_number_input("房貸 / 居住成本 / 年", min_value=0.0, value=float(saved.get("mortgage_annual", defaults["mortgage_annual"])), step=50000.0, digits=1, help="如果已經含在生活費，這裡就不要再重複算。")
    inheritance_age = st.number_input("遺產事件年齡", min_value=45, max_value=90, value=int(saved.get("inheritance_age", defaults["inheritance_age"])), step=1, help="從這個年齡開始，繼承租金收入會進來。")
    inherited_rent_monthly = sidebar_number_input("遺產後租金 / 月", min_value=0.0, value=float(saved.get("inherited_rent_monthly", defaults["inherited_rent_monthly"])), step=5000.0, digits=1, help="遺產事件發生後，每月新增多少租金收入。")

    st.header("6) 設定儲存")
    st.caption("此版本會把設定存成 JSON。Cloud 重啟或重新部署後可能需要重新載入，但一般重新整理不會全失。")


st.subheader("資料讀取狀態")
mcols = st.columns(6)
mcols[0].metric("Current 持股筆數", f"{current_positions_raw.shape[0]}")
mcols[1].metric("Current 市值 (USD)", fmt_human(uploaded_total_usd, 1))
mcols[2].metric("Current 市值折 TWD", fmt_human((uploaded_total_usd or 0) * fx_rate, 1))
mcols[3].metric("建模起始資產", fmt_human(start_assets_twd, 1))
mcols[4].metric("現金保留比重", f"{cash_reserve_target_pct:.1f}%")
mcols[5].metric("妥妥租 Excel 年數", f"{len(biz_projection_list)}")

if missing_fields:
    st.warning(f"CSV 缺少欄位：{', '.join(missing_fields)}。已用替代邏輯補足，但建議再檢查。")
else:
    st.success("CSV 欄位映射完整，可直接作為 Current Portfolio 基礎來源。")

if tuotuozu_mode == "手動遞減":
    st.info("目前使用「手動遞減」模式，因此妥妥租 Excel 不會被讀取。這不是壞掉，是你目前的設定。")
elif len(biz_projection_list) > 0:
    st.success(f"妥妥租 Excel 已成功載入，共 {len(biz_projection_list)} 年資料。")
else:
    st.error("已切換到 Excel 預測模式，但目前沒有成功讀到資料。請檢查檔名、Excel 格式或重新上傳。")

current_default = current_positions_raw[[
    "ticker","name","quantity","market_value_usd","cost_basis_usd","unrealized_gain_usd","gain_pct","weight_pct","asset_class","theme","role_bucket","risk_group","ai_direct","expected_return_pct","volatility_pct","sell_priority","include"
]].copy()

saved_current = enrich_assumptions(df_from_saved("current_portfolio", current_default, saved))
saved_recommended = enrich_assumptions(df_from_saved("recommended_portfolio", build_recommended_portfolio(), saved))
saved_custom = enrich_assumptions(df_from_saved("custom_portfolio", build_defensive_portfolio(), saved))
saved_scenarios = df_from_saved("scenario_table", scenario_table(), saved)

editor_cfg = {
    "include": st.column_config.CheckboxColumn("納入模擬"),
    "ticker": st.column_config.TextColumn("Ticker", disabled=True),
    "name": st.column_config.TextColumn("名稱", disabled=True),
    "quantity": st.column_config.NumberColumn("股數", disabled=True),
    "market_value_usd": st.column_config.NumberColumn("市值 USD", format="%.2f", disabled=True),
    "cost_basis_usd": st.column_config.NumberColumn("成本 USD", format="%.2f", disabled=True),
    "unrealized_gain_usd": st.column_config.NumberColumn("未實現損益 USD", format="%.2f", disabled=True),
    "gain_pct": st.column_config.NumberColumn("損益 %", format="%.2f", disabled=True),
    "weight_pct": st.column_config.NumberColumn("權重 %", min_value=0.0, max_value=100.0, step=0.1),
    "asset_class": st.column_config.TextColumn("資產類別", disabled=True),
    "theme": st.column_config.TextColumn("主題", disabled=True),
    "role_bucket": st.column_config.TextColumn("角色", disabled=True),
    "risk_group": st.column_config.TextColumn("風險群組", disabled=True),
    "ai_direct": st.column_config.CheckboxColumn("AI 直接曝險", disabled=True),
    "hist_10y_cagr_pct": st.column_config.NumberColumn("歷史10年年化報酬 %", format="%.2f", disabled=True),
    "hist_10y_source": st.column_config.TextColumn("歷史值來源", disabled=True),
    "model_return_pct": st.column_config.NumberColumn("模型採用報酬 %", min_value=-20.0, max_value=80.0, step=0.5, help="真正進模擬的是這個欄位。"),
    "model_return_method": st.column_config.TextColumn("報酬方法", disabled=True),
    "vol_5y_weekly_pct": st.column_config.NumberColumn("5Y每週年化波動 %", format="%.2f", disabled=True),
    "vol_3y_weekly_pct": st.column_config.NumberColumn("3Y每週年化波動 %", format="%.2f", disabled=True),
    "model_vol_pct": st.column_config.NumberColumn("模型採用波動 %", min_value=0.0, max_value=100.0, step=0.5, help="真正進模擬的是這個欄位。"),
    "model_vol_method": st.column_config.TextColumn("波動方法", disabled=True),
    "confidence": st.column_config.TextColumn("可信度", disabled=True),
    "notes": st.column_config.TextColumn("補充說明"),
    "updated_at": st.column_config.TextColumn("更新時間", disabled=True),
    "sell_priority": st.column_config.NumberColumn("賣出優先序", min_value=0, max_value=10, step=1),
}
DISPLAY_COLS_CURRENT = [
    "include","ticker","name","quantity","market_value_usd","cost_basis_usd","unrealized_gain_usd","gain_pct","weight_pct",
    "asset_class","role_bucket","ai_direct","hist_10y_cagr_pct","hist_10y_source","model_return_pct","model_return_method",
    "vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct","model_vol_method","confidence","notes"
]
DISPLAY_COLS_EDITABLE = [
    "include","ticker","name","weight_pct","asset_class","role_bucket","ai_direct","hist_10y_cagr_pct","hist_10y_source",
    "model_return_pct","model_return_method","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct","model_vol_method","confidence","notes","sell_priority"
]

# 先給所有下游 tab 一組安全的預設值，避免因為前面 tab 尚未互動而導致空白或 NameError。
current_edited = saved_current.copy()
recommended_edited = saved_recommended.copy()
custom_edited = saved_custom.copy()
scenario_df = saved_scenarios.copy()
default_scenario_name = scenario_df["scenario_name"].iloc[0] if not scenario_df.empty else "自訂情境"
scenario_name = default_scenario_name
mode_override = "跟隨情境"
mode_override_value = None

main_tabs = st.tabs(["1. 輸入與持股編輯", "2. 儀表板", "3. 單一情境模擬", "4. 情境矩陣 / Monte Carlo", "5. 假設透明化 / LOG", "6. 驗證 / 除錯", "7. 原始資料與下載"])

with main_tabs[0]:
    st.markdown("## 新手先看")
    with st.expander("欄位說明：哪些高比較好？哪些低比較好？", expanded=False):
        st.markdown("""
- **權重 %**：這檔在整體資產中的比例。越高代表越重要，也代表越集中。  
- **年化報酬 %**：你對這檔的長期報酬假設。不要亂填太高，容易把未來算得像天堂。  
- **波動率 %**：價格起伏大小。越高越刺激，也越容易讓你在壞年份心情不好。  
- **AI 直接曝險**：是否直接受 AI 題材影響。高不是錯，但太高代表重複押同一件事。  
- **賣出優先序**：現金不足時，系統先賣誰。數字越小，越容易先被賣。  
- **Current / Recommended / Custom**：  
  - Current = 你現在真實持股  
  - Recommended = 建議配置  
  - Custom = 你自己改造後的版本
        """)

    sub_tabs = st.tabs(["Current", "Recommended", "Custom", "Current vs Recommended", "妥妥租 Excel 預覽"])
    with sub_tabs[0]:
        current_readable = format_table_df(
            saved_current,
            pct_cols=["gain_pct", "weight_pct", "hist_10y_cagr_pct", "model_return_pct", "vol_5y_weekly_pct", "vol_3y_weekly_pct", "model_vol_pct"],
            human_cols=["market_value_usd", "cost_basis_usd", "unrealized_gain_usd"],
            keep_cols=DISPLAY_COLS_CURRENT,
        )
        st.dataframe(current_readable, width="stretch", hide_index=True)
        with st.expander("進階編輯 Current 欄位", expanded=False):
            current_edited = st.data_editor(saved_current[DISPLAY_COLS_CURRENT + ["theme","risk_group","sell_priority"]], hide_index=True, width="stretch", num_rows="fixed", column_config=editor_cfg, key="current_editor")
        current_sum = current_edited.loc[current_edited["include"], "weight_pct"].sum()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Current 權重總和", fmt_pct(current_sum, 2))
        c2.metric("Current 市值合計 USD", fmt_human(current_edited["market_value_usd"].sum(), 1))
        c3.metric("Current 成本合計 USD", fmt_human(current_edited["cost_basis_usd"].sum(), 1))
        c4.metric("Current 未實現損益 USD", fmt_human(current_edited["unrealized_gain_usd"].sum(), 1))
        c5.metric("Current 市值折 TWD", fmt_human(current_edited["market_value_usd"].sum() * fx_rate, 1))
        if abs(current_sum - 100) > 0.5:
            st.warning("Current 權重總和目前不是 100%。模擬前會自動正規化，但建議你先確認。")
    with sub_tabs[1]:
        recommended_readable = format_table_df(
            saved_recommended,
            pct_cols=["weight_pct", "hist_10y_cagr_pct", "model_return_pct", "vol_5y_weekly_pct", "vol_3y_weekly_pct", "model_vol_pct"],
            keep_cols=DISPLAY_COLS_EDITABLE,
        )
        st.dataframe(recommended_readable, width="stretch", hide_index=True)
        with st.expander("進階編輯 Recommended 欄位", expanded=False):
            recommended_edited = st.data_editor(saved_recommended[DISPLAY_COLS_EDITABLE + ["theme","risk_group"]], hide_index=True, width="stretch", num_rows="dynamic", column_config=editor_cfg, key="recommended_editor")
        rsum = recommended_edited.loc[recommended_edited["include"], "weight_pct"].sum()
        st.metric("Recommended 權重總和", fmt_pct(rsum, 2))
    with sub_tabs[2]:
        custom_readable = format_table_df(
            saved_custom,
            pct_cols=["weight_pct", "hist_10y_cagr_pct", "model_return_pct", "vol_5y_weekly_pct", "vol_3y_weekly_pct", "model_vol_pct"],
            keep_cols=DISPLAY_COLS_EDITABLE,
        )
        st.dataframe(custom_readable, width="stretch", hide_index=True)
        with st.expander("進階編輯 Custom 欄位", expanded=False):
            custom_edited = st.data_editor(saved_custom[DISPLAY_COLS_EDITABLE + ["theme","risk_group"]], hide_index=True, width="stretch", num_rows="dynamic", column_config=editor_cfg, key="custom_editor")
        csum = custom_edited.loc[custom_edited["include"], "weight_pct"].sum()
        st.metric("Custom 權重總和", fmt_pct(csum, 2))
    with sub_tabs[3]:
        comparison_df = build_comparison(current_edited, recommended_edited)
        st.dataframe(format_table_df(comparison_df), width="stretch", hide_index=True)
    with sub_tabs[4]:
        if tuotuozu_mode != "Excel 預測":
            st.warning("目前為「手動遞減」模式，尚未啟用 Excel 預測，因此此處不顯示 Excel 預覽。")
        elif biz_projection_preview.empty:
            st.error(biz_excel_error_message or "已切換到 Excel 預測模式，但目前沒有成功讀到 Excel 資料。")
            if biz_excel_diagnostics:
                st.caption(" / ".join(biz_excel_diagnostics))
        else:
            st.success(f"已成功讀取 {len(biz_projection_preview)} 年妥妥租預測資料。")
            if biz_excel_diagnostics:
                st.caption(" / ".join(biz_excel_diagnostics))
            st.dataframe(format_table_df(biz_projection_preview, human_cols=["net_profit_twd"]), width="stretch", hide_index=True)

    save_col1, save_col2, save_col3 = st.columns([1,1,2])
    if save_col1.button("儲存目前設定", type="primary"):
        payload = {
            "source_mode": source_mode,
            "fx_rate": fx_rate,
            "start_assets_twd": start_assets_twd,
            "simulation_years": simulation_years,
            "rebalance_frequency_years": rebalance_frequency_years,
            "withdrawal_strategy": withdrawal_strategy,
            "cash_reserve_target_pct": cash_reserve_target_pct,
            "monte_carlo_sims": monte_carlo_sims,
            "enable_heavy_compute": enable_heavy_compute,
            "salary_annual": salary_annual,
            "salary_growth_pct": salary_growth_pct,
            "retirement_age": retirement_age,
            "tuotuozu_mode": tuotuozu_mode,
            "biz_excel_source_mode": locals().get("biz_excel_source_mode", "使用內建範例 Excel"),
            "tuotuozu_base_annual": tuotuozu_base_annual,
            "tuotuozu_decay_pct": tuotuozu_decay_pct,
            "tuotuozu_fallback_mode": tuotuozu_fallback_mode,
            "living_expense_annual": living_expense_annual,
            "inflation_pct": inflation_pct,
            "edu_phase1_annual": edu_phase1_annual,
            "edu_phase2_annual": edu_phase2_annual,
            "mortgage_annual": mortgage_annual,
            "inheritance_age": inheritance_age,
            "inherited_rent_monthly": inherited_rent_monthly,
            "current_portfolio": df_to_records(current_edited),
            "recommended_portfolio": df_to_records(recommended_edited),
            "custom_portfolio": df_to_records(custom_edited),
            "scenario_table": df_to_records(saved_scenarios),
        }
        save_json(SETTINGS_PATH, payload)
        st.success("已把目前設定存成 JSON。重新整理後若環境沒有重啟，通常仍可沿用。")
    settings_payload = {
        "source_mode": source_mode,
        "fx_rate": fx_rate,
        "start_assets_twd": start_assets_twd,
        "simulation_years": simulation_years,
        "rebalance_frequency_years": rebalance_frequency_years,
        "withdrawal_strategy": withdrawal_strategy,
        "cash_reserve_target_pct": cash_reserve_target_pct,
        "monte_carlo_sims": monte_carlo_sims,
        "salary_annual": salary_annual,
        "salary_growth_pct": salary_growth_pct,
        "retirement_age": retirement_age,
        "tuotuozu_mode": tuotuozu_mode,
        "tuotuozu_base_annual": tuotuozu_base_annual,
        "tuotuozu_decay_pct": tuotuozu_decay_pct,
        "tuotuozu_fallback_mode": tuotuozu_fallback_mode,
        "living_expense_annual": living_expense_annual,
        "inflation_pct": inflation_pct,
        "edu_phase1_annual": edu_phase1_annual,
        "edu_phase2_annual": edu_phase2_annual,
        "mortgage_annual": mortgage_annual,
        "inheritance_age": inheritance_age,
        "inherited_rent_monthly": inherited_rent_monthly,
        "current_portfolio": df_to_records(current_edited),
        "recommended_portfolio": df_to_records(recommended_edited),
        "custom_portfolio": df_to_records(custom_edited),
    }
    save_col2.download_button("下載設定 JSON", data=json.dumps(settings_payload, ensure_ascii=False, indent=2), file_name="ben_financial_settings.json", mime="application/json")
    uploaded_json = save_col3.file_uploader("載入先前儲存的設定 JSON", type=["json"])
    if uploaded_json is not None:
        try:
            incoming = json.load(uploaded_json)
            save_json(SETTINGS_PATH, incoming)
            st.success("設定 JSON 已載入。請重新整理頁面一次，讓所有欄位回填。")
        except Exception:
            st.error("這份 JSON 無法載入，請確認格式。")

current_norm = prepare_simulation_df(apply_cash_reserve_target(normalize_weights(current_edited.copy()), cash_reserve_target_pct))
recommended_norm = prepare_simulation_df(apply_cash_reserve_target(normalize_weights(recommended_edited.copy()), cash_reserve_target_pct))
custom_norm = prepare_simulation_df(apply_cash_reserve_target(normalize_weights(custom_edited.copy()), cash_reserve_target_pct))

# 先做一輪基礎計算，讓 2~7 tab 都有內容可顯示。
current_metrics = compute_risk_duplicate_metrics(current_norm)
recommended_metrics = compute_risk_duplicate_metrics(recommended_norm)
custom_metrics = compute_risk_duplicate_metrics(custom_norm)
scenario_row = get_scenario_row(pd.DataFrame(scenario_df), scenario_name)
cur_res = simulate_portfolio(current_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override_value, seed=42)
rec_res = simulate_portfolio(recommended_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override_value, seed=43)
cus_res = simulate_portfolio(custom_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override_value, seed=44)
cur_sum = summarize_simulation(cur_res, start_assets_twd)
rec_sum = summarize_simulation(rec_res, start_assets_twd)
cus_sum = summarize_simulation(cus_res, start_assets_twd)
validation_df = build_validation_checks(
    current_edited, recommended_edited, custom_edited,
    rec_res, cur_res, cus_res, tuotuozu_mode, biz_projection_list, retirement_age, edu_phase2_annual, scenario_name
)
if validation_df.empty:
    validation_df = pd.DataFrame([{"檢查項目": "最小檢查集", "結果": "INFO", "說明": "目前尚未產生完整結果，但頁面已正常啟動。"}])
global_settings = {
    "fx_rate": fx_rate,
    "start_assets_twd": start_assets_twd,
    "simulation_years": simulation_years,
    "rebalance_frequency_years": rebalance_frequency_years,
    "withdrawal_strategy": withdrawal_strategy,
    "cash_reserve_target_pct": cash_reserve_target_pct,
    "monte_carlo_sims": monte_carlo_sims,
    "salary_annual": salary_annual,
    "salary_growth_pct": salary_growth_pct,
    "retirement_age": retirement_age,
    "tuotuozu_mode": tuotuozu_mode,
    "tuotuozu_base_annual": tuotuozu_base_annual,
    "tuotuozu_decay_pct": tuotuozu_decay_pct,
    "living_expense_annual": living_expense_annual,
    "inflation_pct": inflation_pct,
    "edu_phase1_annual": edu_phase1_annual,
    "edu_phase2_annual": edu_phase2_annual,
    "mortgage_annual": mortgage_annual,
    "inheritance_age": inheritance_age,
    "inherited_rent_monthly": inherited_rent_monthly,
}
diagnostic_summary = build_diagnostic_summary(
    scenario_name, global_settings, current_norm, recommended_norm, custom_norm, current_metrics, cur_sum, rec_sum, cus_sum, validation_df
)
portfolio_context_df = portfolio_context_rows(current_norm, recommended_norm, custom_norm)
scenario_context_runtime_df = scenario_context_df(scenario_row, scenario_name, mode_override_value)
portfolio_effect_df = portfolio_scenario_effect_rows(current_norm, recommended_norm, custom_norm, scenario_row, mode_override_value)
light_log = build_full_log(
    global_settings, current_norm, recommended_norm, custom_norm, scenario_row.to_dict(),
    cur_res, rec_res, cus_res, validation_df, diagnostic_summary, []
)
light_log = sanitize_for_json(light_log)

sims_df = pd.DataFrame()
mc_metrics: Dict[str, Any] = {}
mc_summary_df = pd.DataFrame()
matrix_df = pd.DataFrame()
full_log_payload = st.session_state.get("full_log_payload")
full_bundle_zip_bytes = st.session_state.get("full_bundle_zip_bytes")

with main_tabs[1]:
    st.markdown("## 儀表板")
    current_metrics = compute_risk_duplicate_metrics(current_norm)
    recommended_metrics = compute_risk_duplicate_metrics(recommended_norm)
    custom_metrics = compute_risk_duplicate_metrics(custom_norm)
    risk_df = pd.DataFrame([
        {"portfolio": "Current", **current_metrics},
        {"portfolio": "Recommended", **recommended_metrics},
        {"portfolio": "Custom", **custom_metrics},
    ])
    c1,c2 = st.columns(2)
    with c1:
        alloc_long = pd.concat([
            current_norm[["ticker","weight_pct"]].assign(portfolio="Current"),
            recommended_norm[["ticker","weight_pct"]].assign(portfolio="Recommended"),
            custom_norm[["ticker","weight_pct"]].assign(portfolio="Custom"),
        ], ignore_index=True)
        st.plotly_chart(px.bar(alloc_long, x="ticker", y="weight_pct", color="portfolio", barmode="group", title="資產配置長條圖"), width="stretch")
    with c2:
        ai_long = risk_df.melt(id_vars="portfolio", value_vars=["直接 AI 曝險 %","ETF 重疊曝險 %","同跌風險程度（0-100）"], var_name="指標", value_name="數值")
        st.plotly_chart(px.bar(ai_long, x="portfolio", y="數值", color="指標", barmode="group", title="AI / 科技集中度圖"), width="stretch")
    st.dataframe(format_table_df(risk_df, pct_cols=["直接 AI 曝險 %","ETF 重疊曝險 %","單一風險主題集中度 %","類別分散度指數 %","單一持股過重程度 %"], human_cols=["同跌風險程度（0-100）"]), width="stretch", hide_index=True)

with main_tabs[2]:
    st.markdown("## 單一情境模擬")
    scenario_df = st.data_editor(saved_scenarios, width="stretch", hide_index=True, num_rows="fixed", key="scenario_editor")
    scenario_name = st.selectbox("選擇要展開的情境", scenario_df["scenario_name"].tolist(), help="這裡選的是情境名稱，不是持股方案。")
    mode_override = st.selectbox("模擬模式（可覆蓋情境內建模式）", ["跟隨情境", "fixed", "monte_carlo", "path"])
    mode_override_value = None if mode_override == "跟隨情境" else mode_override
    scenario_row = get_scenario_row(pd.DataFrame(scenario_df), scenario_name)

    cur_res = simulate_portfolio(current_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override_value, seed=42)
    rec_res = simulate_portfolio(recommended_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override_value, seed=43)
    cus_res = simulate_portfolio(custom_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, mode_override_value, seed=44)

    cur_sum = summarize_simulation(cur_res, start_assets_twd)
    rec_sum = summarize_simulation(rec_res, start_assets_twd)
    cus_sum = summarize_simulation(cus_res, start_assets_twd)

    st.info("這一頁是單一路徑 deterministic 模擬：適合看方向與年度曲線，但不代表機率分布真相。Monte Carlo 請看下一頁。")
    st.markdown("### 組合基礎加權報酬 / 波動 與情境額外加成")
    st.dataframe(format_table_df(portfolio_effect_df, pct_cols=["基礎加權報酬率%","基礎加權波動率%","AI 直接曝險權重%","market_return_shift_pct","weighted_ai_excess_shift_pct","scenario_shift_pct","effective_portfolio_return_pct","effective_portfolio_vol_pct"]), width="stretch", hide_index=True)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Current 最終資產", fmt_human(cur_sum.get("最終資產終值"), 1))
    s2.metric("Recommended 最終資產", fmt_human(rec_sum.get("最終資產終值"), 1))
    s3.metric("Custom 最終資產", fmt_human(cus_sum.get("最終資產終值"), 1))
    s4.metric("Recommended 最大回撤", fmt_pct(rec_sum.get("最大回撤 %"), 1))

    scenario_alerts = []
    if not rec_res.empty and rec_res["portfolio_return_pct"].abs().max() > 0 and abs(float(rec_sum.get("最大回撤 %", 0) or 0)) < 1e-9:
        scenario_alerts.append("Recommended 最大回撤顯示為 0.0%，但年度報酬有波動，這通常代表 drawdown 邏輯或顯示需要再驗。")
    if not cus_res.empty:
        rec_weights = recommended_norm.loc[recommended_norm["include"], ["ticker", "weight_pct"]].sort_values("ticker").reset_index(drop=True)
        cus_weights = custom_norm.loc[custom_norm["include"], ["ticker", "weight_pct"]].sort_values("ticker").reset_index(drop=True)
        if rec_res["end_assets_twd"].round(6).equals(cus_res["end_assets_twd"].round(6)) and not rec_weights.equals(cus_weights):
            scenario_alerts.append("Custom 與 Recommended 的結果完全相同，但兩者權重不同，這很可疑，建議優先看驗證 / 除錯 tab。")
    if scenario_alerts:
        st.warning("\n\n".join(scenario_alerts))

    single_path_summary_df = pd.DataFrame([
        {"方案": "Current", "終值": cur_sum.get("最終資產終值"), "最大回撤%": cur_sum.get("最大回撤 %"), "effective_portfolio_return_pct": float(portfolio_effect_df.loc[portfolio_effect_df["方案"].eq("Current"), "effective_portfolio_return_pct"].iloc[0]), "effective_portfolio_vol_pct": float(portfolio_effect_df.loc[portfolio_effect_df["方案"].eq("Current"), "effective_portfolio_vol_pct"].iloc[0])},
        {"方案": "Recommended", "終值": rec_sum.get("最終資產終值"), "最大回撤%": rec_sum.get("最大回撤 %"), "effective_portfolio_return_pct": float(portfolio_effect_df.loc[portfolio_effect_df["方案"].eq("Recommended"), "effective_portfolio_return_pct"].iloc[0]), "effective_portfolio_vol_pct": float(portfolio_effect_df.loc[portfolio_effect_df["方案"].eq("Recommended"), "effective_portfolio_vol_pct"].iloc[0])},
        {"方案": "Custom", "終值": cus_sum.get("最終資產終值"), "最大回撤%": cus_sum.get("最大回撤 %"), "effective_portfolio_return_pct": float(portfolio_effect_df.loc[portfolio_effect_df["方案"].eq("Custom"), "effective_portfolio_return_pct"].iloc[0]), "effective_portfolio_vol_pct": float(portfolio_effect_df.loc[portfolio_effect_df["方案"].eq("Custom"), "effective_portfolio_vol_pct"].iloc[0])},
    ])
    st.markdown("### 單一路徑結果摘要")
    st.dataframe(format_table_df(single_path_summary_df, pct_cols=["最大回撤%","effective_portfolio_return_pct","effective_portfolio_vol_pct"], human_cols=["終值"]), width="stretch", hide_index=True)

    combo = pd.concat([
        cur_res.assign(portfolio="Current"),
        rec_res.assign(portfolio="Recommended"),
        cus_res.assign(portfolio="Custom"),
    ], ignore_index=True)

    viz_tabs = st.tabs(["資產成長曲線", "現金流缺口圖", "收入拆解", "最大回撤比較", "AI 分析 Prompt"])
    with viz_tabs[0]:
        st.plotly_chart(px.line(combo, x="calendar_year", y="end_assets_twd", color="portfolio", markers=True, title="資產成長曲線"), width="stretch")
    with viz_tabs[1]:
        fig = px.bar(combo, x="calendar_year", y="net_cashflow_twd", color="portfolio", barmode="group", title="現金流缺口圖")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.add_vrect(x0=2034 - 0.5, x1=2038 + 0.5, fillcolor="orange", opacity=0.12, line_width=0, annotation_text="教育費高峰")
        st.plotly_chart(fig, width="stretch")
    with viz_tabs[2]:
        income_combo = pd.concat([
            rec_res[["calendar_year","salary_income_twd","tuotuozu_income_twd","inheritance_income_twd"]].assign(portfolio="Recommended")
        ])
        income_long = income_combo.melt(id_vars=["calendar_year","portfolio"], var_name="收入來源", value_name="金額")
        st.plotly_chart(px.bar(income_long, x="calendar_year", y="金額", color="收入來源", title="Recommended 收入拆解"), width="stretch")
    with viz_tabs[3]:
        st.plotly_chart(px.line(combo, x="calendar_year", y="drawdown_pct", color="portfolio", markers=True, title="最大回撤比較"), width="stretch")
    with viz_tabs[4]:
        ai_prompt = f"""請根據以下模擬結果，做決策型分析，而不是只重述數字。\n\n【本次情境】\n{scenario_name}\n\n【我的目標】\n1. 長期勝率\n2. 不容易中途出局\n3. 教育費高峰時可承受\n4. 大跌時不容易被迫賣股\n5. 整體人生舒服度\n\n【輸入摘要】\n- 起始流動資產: {fmt_human(start_assets_twd,1)} TWD\n- 本業年薪: {fmt_human(salary_annual,1)}\n- 薪資成長率: {fmt_pct(salary_growth_pct,1)}\n- 退休年齡: {retirement_age}\n- 妥妥租模式: {tuotuozu_mode}\n- 妥妥租目前年度淨利: {fmt_human(tuotuozu_base_annual,1)}\n- 妥妥租衰退率: {fmt_pct(tuotuozu_decay_pct,1)}\n- 妥妥租 Excel 年數: {len(biz_projection_list)}\n- 基礎生活費/年: {fmt_human(living_expense_annual,1)}\n- 教育費 2026-2033/年: {fmt_human(edu_phase1_annual,1)}\n- 教育費 2034-2038/年: {fmt_human(edu_phase2_annual,1)}\n- 房貸/年: {fmt_human(mortgage_annual,1)}\n- 現金保留比重: {fmt_pct(cash_reserve_target_pct,1)}\n\n【結果摘要】\nCurrent: 最終資產 {fmt_human(cur_sum.get('最終資產終值'),1)} / 最大回撤 {fmt_pct(cur_sum.get('最大回撤 %'),1)} / 10年後 {fmt_human(cur_sum.get('10 年後淨資產'),1)} / 20年後 {fmt_human(cur_sum.get('20 年後淨資產'),1)} / 人生適配分數 {cur_sum.get('人生適配分數')}\nRecommended: 最終資產 {fmt_human(rec_sum.get('最終資產終值'),1)} / 最大回撤 {fmt_pct(rec_sum.get('最大回撤 %'),1)} / 10年後 {fmt_human(rec_sum.get('10 年後淨資產'),1)} / 20年後 {fmt_human(rec_sum.get('20 年後淨資產'),1)} / 人生適配分數 {rec_sum.get('人生適配分數')}\nCustom: 最終資產 {fmt_human(cus_sum.get('最終資產終值'),1)} / 最大回撤 {fmt_pct(cus_sum.get('最大回撤 %'),1)} / 10年後 {fmt_human(cus_sum.get('10 年後淨資產'),1)} / 20年後 {fmt_human(cus_sum.get('20 年後淨資產'),1)} / 人生適配分數 {cus_sum.get('人生適配分數')}\n\n【請回答】\n1. 先結論：哪個方案最適合我，為什麼？\n2. 分成：事實 / 推論 / 假設\n3. 一定要分析：最大回撤、現金流缺口、資產耗盡風險、人生適配分數\n4. 指出哪個方案比較像「賺得多但難撐」\n5. 指出哪個方案比較像「賺得稍慢但比較舒服」\n6. 若看到任何數據不合理，請提醒我先回頭驗模型\n"""
        st.text_area("可直接複製給 AI 分析的 Prompt", value=ai_prompt, height=420)
        st.download_button("下載這次模擬的 AI Prompt", data=ai_prompt, file_name="ai_analysis_prompt.txt", mime="text/plain")

with main_tabs[3]:
    st.markdown("## 情境矩陣 / Monte Carlo")
    if not enable_heavy_compute:
        st.info("目前已關閉『預載重型分析』，所以這一頁不會自動跑情境矩陣與 Monte Carlo。這樣可以大幅加快 5/6 頁與整體操作速度。")
        st.caption("需要重型分析時，請到左側打開『預載重型分析（情境矩陣 / Monte Carlo）』後再重新整理。")
    else:
        matrix_rows = []
        for _, srow in pd.DataFrame(scenario_df).iterrows():
            cur = simulate_portfolio(current_norm, srow, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, None, seed=42)
            rec = simulate_portfolio(recommended_norm, srow, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years, None, seed=43)
            cur_s = summarize_simulation(cur, start_assets_twd)
            rec_s = summarize_simulation(rec, start_assets_twd)
            matrix_rows.append({
                "情境": srow["scenario_name"],
                "Current 最終資產": cur_s.get("最終資產終值"),
                "Recommended 最終資產": rec_s.get("最終資產終值"),
                "Current 最大回撤%": cur_s.get("最大回撤 %"),
                "Recommended 最大回撤%": rec_s.get("最大回撤 %"),
                "Current 人生適配分數": cur_s.get("人生適配分數"),
                "Recommended 人生適配分數": rec_s.get("人生適配分數"),
            })
        matrix_df = pd.DataFrame(matrix_rows)
        st.dataframe(format_table_df(matrix_df, pct_cols=["Current 最大回撤%","Recommended 最大回撤%"], human_cols=["Current 最終資產","Recommended 最終資產"]), width="stretch", hide_index=True)

        st.info("這一頁是 Monte Carlo 機率分布：適合看 P50 / P5 worst case / 資產耗盡機率，不要和單一路徑混在一起解讀。")
        sims_df, mc_metrics = run_monte_carlo_compare(
            current_norm, recommended_norm, custom_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR,
            salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode,
            living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly,
            withdrawal_strategy, rebalance_frequency_years, mode_override_value, simulations=monte_carlo_sims, seed=42
        )
        mc_summary_df = pd.DataFrame([
            {"方案": "Current", "P50 終值": mc_metrics.get("Current P50 終值"), "P25 終值": mc_metrics.get("Current P25 終值"), "P75 終值": mc_metrics.get("Current P75 終值"), "P5 Worst Case": mc_metrics.get("Current P5 Worst Case 終值"), "資產耗盡機率%": mc_metrics.get("Current 資產耗盡機率"), "最大回撤中位數%": mc_metrics.get("Current 最大回撤中位數")},
            {"方案": "Recommended", "P50 終值": mc_metrics.get("Recommended P50 終值"), "P25 終值": mc_metrics.get("Recommended P25 終值"), "P75 終值": mc_metrics.get("Recommended P75 終值"), "P5 Worst Case": mc_metrics.get("Recommended P5 Worst Case 終值"), "資產耗盡機率%": mc_metrics.get("Recommended 資產耗盡機率"), "最大回撤中位數%": mc_metrics.get("Recommended 最大回撤中位數")},
            {"方案": "Custom", "P50 終值": mc_metrics.get("Custom P50 終值"), "P25 終值": mc_metrics.get("Custom P25 終值"), "P75 終值": mc_metrics.get("Custom P75 終值"), "P5 Worst Case": mc_metrics.get("Custom P5 Worst Case 終值"), "資產耗盡機率%": mc_metrics.get("Custom 資產耗盡機率"), "最大回撤中位數%": mc_metrics.get("Custom 最大回撤中位數")},
        ])
        st.markdown("### Monte Carlo 結果摘要")
        st.dataframe(format_table_df(mc_summary_df, pct_cols=["資產耗盡機率%","最大回撤中位數%"], human_cols=["P50 終值","P25 終值","P75 終值","P5 Worst Case"]), width="stretch", hide_index=True)
        m1,m2,m3 = st.columns(3)
        m1.metric("Recommended 勝過 Current 機率", fmt_pct(mc_metrics.get("Recommended 勝過 Current 機率"),1))
        m2.metric("Recommended 勝過 Custom 機率", fmt_pct(mc_metrics.get("Recommended 勝過 Custom 機率"),1))
        m3.metric("Current 勝過 Custom 機率", fmt_pct(mc_metrics.get("Current 勝過 Custom 機率"),1))
        long_mc = sims_df.melt(value_vars=["current_final_assets_twd","recommended_final_assets_twd","custom_final_assets_twd"], var_name="portfolio", value_name="final_assets_twd")
        st.plotly_chart(px.histogram(long_mc, x="final_assets_twd", color="portfolio", barmode="overlay", nbins=40, title="Monte Carlo 終值分布"), width="stretch")
        dd_long = sims_df.melt(value_vars=["current_max_drawdown_pct","recommended_max_drawdown_pct","custom_max_drawdown_pct"], var_name="portfolio", value_name="max_drawdown_pct")
        st.plotly_chart(px.histogram(dd_long, x="max_drawdown_pct", color="portfolio", barmode="overlay", nbins=40, title="Monte Carlo 最大回撤分布"), width="stretch")

with main_tabs[4]:
    try:
        st.markdown("## 假設透明化 / LOG")
        st.info("這一頁的目的不是秀漂亮圖，而是讓你看得出：每檔標的的模型假設從哪裡來、哪些是假設保守化、哪些其實資料不足。")
        st.markdown("### 目前實際吃進模擬的配置 / 情境")
        st.dataframe(format_table_df(portfolio_context_df, pct_cols=["現金%","加權報酬%","加權波動%"]), width="stretch", hide_index=True)
        st.dataframe(format_table_df(scenario_context_runtime_df, pct_cols=["market_shift%","ai_excess%","max_drawdown_hint%"]), width="stretch", hide_index=True)
        st.dataframe(format_table_df(portfolio_effect_df, pct_cols=["基礎加權報酬率%","基礎加權波動率%","AI 直接曝險權重%","market_return_shift_pct","weighted_ai_excess_shift_pct","scenario_shift_pct","effective_portfolio_return_pct","effective_portfolio_vol_pct"]), width="stretch", hide_index=True)
        audit_choice = st.selectbox("選擇要查看的假設組", ["Current", "Recommended", "Custom", "診斷摘要 / LOG"], key="audit_choice")
        audit_keep = ["ticker","name","classification_bucket","hist_10y_cagr_pct","hist_10y_source","model_return_pct","model_return_method","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct","model_vol_method","confidence","notes","updated_at"]

        if audit_choice == "Current":
            st.dataframe(format_table_df(current_norm, pct_cols=["hist_10y_cagr_pct","model_return_pct","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct"], keep_cols=audit_keep), width="stretch", hide_index=True)
        elif audit_choice == "Recommended":
            st.dataframe(format_table_df(recommended_norm, pct_cols=["hist_10y_cagr_pct","model_return_pct","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct"], keep_cols=audit_keep), width="stretch", hide_index=True)
        elif audit_choice == "Custom":
            st.dataframe(format_table_df(custom_norm, pct_cols=["hist_10y_cagr_pct","model_return_pct","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct"], keep_cols=audit_keep), width="stretch", hide_index=True)
        else:
            light_bundle_zip_bytes = build_validation_bundle_zip(
                bundle_kind="light",
                settings_payload=settings_payload,
                global_settings=global_settings,
                scenario_row_dict=scenario_row.to_dict(),
                diagnostic_summary=diagnostic_summary,
                validation_df=validation_df,
                portfolio_context_df=portfolio_context_df,
                scenario_context_runtime_df=scenario_context_runtime_df,
                portfolio_effect_df=portfolio_effect_df,
                current_edited=current_edited,
                recommended_edited=recommended_edited,
                custom_edited=custom_edited,
                current_norm=current_norm,
                recommended_norm=recommended_norm,
                custom_norm=custom_norm,
                current_path_df=cur_res,
                recommended_path_df=rec_res,
                custom_path_df=cus_res,
                scenario_table_df=pd.DataFrame(scenario_df),
                light_log=light_log,
                sims_df=sims_df,
                mc_summary_df=mc_summary_df,
                mc_metrics=mc_metrics,
                matrix_df=matrix_df,
            )
            d1, d2, d3 = st.columns(3)
            d1.download_button("下載輕量 LOG（JSON）", prepare_json_download(light_log), "simulation_light_log.json", "application/json")
            d2.download_button("下載診斷摘要（txt）", prepare_text_download(diagnostic_summary), "diagnostic_summary.txt", "text/plain")
            d3.download_button("一鍵下載 AI 驗證包 ZIP（輕量）", light_bundle_zip_bytes, f"ben_validation_bundle_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip")
            st.text_area("可直接複製的診斷摘要", diagnostic_summary, height=380)
            st.caption("輕量包會一次打包：設定、情境、三組持股、年度路徑、validation、diagnostic summary、light log，以及已存在的 Monte Carlo 摘要。")
            if st.button("產生完整 LOG / AI 驗證包（較慢，含 Monte Carlo path replay）", key="gen_full_log_btn"):
                with st.spinner("正在產生完整 LOG 與驗證包，這一步會比較慢..."):
                    mc_path_log = build_mc_path_log(
                        scenario_row.to_dict(), df_to_records(current_norm), df_to_records(recommended_norm),
                        simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR, salary_annual, salary_growth_pct,
                        retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list,
                        tuotuozu_fallback_mode, living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual,
                        mortgage_annual, inheritance_age, inherited_rent_monthly, withdrawal_strategy, rebalance_frequency_years,
                        mode_override_value, monte_carlo_sims, 42
                    )
                    full_log = build_full_log(
                        global_settings, current_norm, recommended_norm, custom_norm, scenario_row.to_dict(),
                        cur_res, rec_res, cus_res, validation_df, diagnostic_summary, mc_path_log
                    )
                    full_log = sanitize_for_json(full_log)
                    full_bundle_zip_bytes = build_validation_bundle_zip(
                        bundle_kind="full",
                        settings_payload=settings_payload,
                        global_settings=global_settings,
                        scenario_row_dict=scenario_row.to_dict(),
                        diagnostic_summary=diagnostic_summary,
                        validation_df=validation_df,
                        portfolio_context_df=portfolio_context_df,
                        scenario_context_runtime_df=scenario_context_runtime_df,
                        portfolio_effect_df=portfolio_effect_df,
                        current_edited=current_edited,
                        recommended_edited=recommended_edited,
                        custom_edited=custom_edited,
                        current_norm=current_norm,
                        recommended_norm=recommended_norm,
                        custom_norm=custom_norm,
                        current_path_df=cur_res,
                        recommended_path_df=rec_res,
                        custom_path_df=cus_res,
                        scenario_table_df=pd.DataFrame(scenario_df),
                        light_log=light_log,
                        full_log=full_log,
                        sims_df=sims_df,
                        mc_summary_df=mc_summary_df,
                        mc_metrics=mc_metrics,
                        matrix_df=matrix_df,
                        mc_path_log_records=mc_path_log,
                    )
                    st.session_state["full_log_payload"] = full_log
                    st.session_state["full_bundle_zip_bytes"] = full_bundle_zip_bytes
                    full_log_payload = full_log
                    st.success(f"完整 LOG 已產生，含 {len(mc_path_log)} 筆 path records。")
            full_log_payload = st.session_state.get("full_log_payload")
            full_bundle_zip_bytes = st.session_state.get("full_bundle_zip_bytes")
            if full_log_payload is not None:
                f1, f2 = st.columns(2)
                f1.download_button("下載完整 LOG（JSON）", prepare_json_download(full_log_payload), "simulation_full_log.json", "application/json", key="download_full_log_btn")
                if full_bundle_zip_bytes is not None:
                    f2.download_button("一鍵下載 AI 驗證包 ZIP（完整）", full_bundle_zip_bytes, f"ben_validation_bundle_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip", key="download_full_bundle_zip_btn")
            st.caption("若沒有真實 10Y / 5Y / 3Y 價格資料，系統會清楚標示 unavailable / bucket default，不假裝精準。")
    except Exception as e:
        st.error(f"假設透明化 / LOG 頁產生失敗：{e}")
with main_tabs[5]:
    try:
        st.markdown("## 驗證 / 除錯")
        st.markdown("### 目前實際吃進模擬的配置 / 情境")
        st.dataframe(format_table_df(portfolio_context_df, pct_cols=["現金%","加權報酬%","加權波動%"]), width="stretch", hide_index=True)
        st.dataframe(format_table_df(scenario_context_runtime_df, pct_cols=["market_shift%","ai_excess%","max_drawdown_hint%"]), width="stretch", hide_index=True)
        st.dataframe(format_table_df(portfolio_effect_df, pct_cols=["基礎加權報酬率%","基礎加權波動率%","AI 直接曝險權重%","market_return_shift_pct","weighted_ai_excess_shift_pct","scenario_shift_pct","effective_portfolio_return_pct","effective_portfolio_vol_pct"]), width="stretch", hide_index=True)
        if validation_df.empty:
            validation_df = pd.DataFrame([{"檢查項目": "最小檢查集", "結果": "INFO", "說明": "尚未產生完整模擬結果，但權重與缺值檢查已完成。"}])

        st.info("這一頁是模型的體檢表。若這裡有 FAIL 或 WARNING，就不要急著相信最終資產數字。")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("PASS 數", int((validation_df["結果"] == "PASS").sum()))
        v2.metric("WARNING 數", int((validation_df["結果"] == "WARNING").sum()))
        v3.metric("FAIL 數", int((validation_df["結果"] == "FAIL").sum()))
        v4.metric("INFO 數", int((validation_df["結果"] == "INFO").sum()))

        fail_df = validation_df.loc[validation_df["結果"] == "FAIL"]
        warn_df = validation_df.loc[validation_df["結果"] == "WARNING"]
        if not fail_df.empty:
            st.error("目前有 FAIL 項目，建議先修模型再解讀結果。")
        elif not warn_df.empty:
            st.warning("目前有 WARNING 項目，代表這版模型還有需要人工判讀的地方。")
        else:
            st.success("目前最小檢查集皆通過。")

        st.dataframe(validation_df, width="stretch", hide_index=True)

        sens_rows = []
        for param, low, high, label in [
            ("cash_reserve_target_pct", max(0,cash_reserve_target_pct-4), min(40,cash_reserve_target_pct+4), "現金保留比重"),
            ("salary_growth_pct", max(-5,salary_growth_pct-1), min(20,salary_growth_pct+1), "薪資成長率"),
            ("tuotuozu_decay_pct", max(-50,tuotuozu_decay_pct-3), min(30,tuotuozu_decay_pct+3), "妥妥租衰退率"),
            ("inflation_pct", max(0,inflation_pct-1), min(20,inflation_pct+1), "通膨率"),
            ("edu_phase2_annual", max(0,edu_phase2_annual-200000), edu_phase2_annual+200000, "教育費高峰"),
        ]:
            sens_rows.append({"參數": label, "低值": low, "高值": high, "提醒": "這項對結果常有明顯影響，調整前後請重跑並比較圖表。"})
        st.markdown("### 最敏感 5 個參數（人工提醒版）")
        st.dataframe(format_table_df(pd.DataFrame(sens_rows), human_cols=["低值","高值"]), width="stretch", hide_index=True)
    except Exception as e:
        st.error(f"驗證 / 除錯頁產生失敗：{e}")
with main_tabs[6]:

    st.markdown("## 原始資料與下載")
    st.info("若你要把整包資料丟給 AI 驗證，最省事的是到『5. 假設透明化 / LOG』直接按一鍵下載 AI 驗證包 ZIP。這裡則保留逐項下載。")
    raw_tabs = st.tabs(["Current 正規化後", "Recommended 正規化後", "Custom 正規化後", "情境表", "Monte Carlo 明細"])
    with raw_tabs[0]:
        st.dataframe(format_table_df(current_norm, pct_cols=["weight_pct","hist_10y_cagr_pct","model_return_pct","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct"], human_cols=["market_value_usd","cost_basis_usd","unrealized_gain_usd"]), width="stretch", hide_index=True)
        st.download_button("下載 Current CSV", prepare_download(current_norm), "current_portfolio.csv", "text/csv")
    with raw_tabs[1]:
        st.dataframe(format_table_df(recommended_norm, pct_cols=["weight_pct","hist_10y_cagr_pct","model_return_pct","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct"]), width="stretch", hide_index=True)
        st.download_button("下載 Recommended CSV", prepare_download(recommended_norm), "recommended_portfolio.csv", "text/csv")
    with raw_tabs[2]:
        st.dataframe(format_table_df(custom_norm, pct_cols=["weight_pct","hist_10y_cagr_pct","model_return_pct","vol_5y_weekly_pct","vol_3y_weekly_pct","model_vol_pct"]), width="stretch", hide_index=True)
        st.download_button("下載 Custom CSV", prepare_download(custom_norm), "custom_portfolio.csv", "text/csv")
    with raw_tabs[3]:
        st.dataframe(format_table_df(pd.DataFrame(scenario_df), pct_cols=["market_return_shift_pct","ai_excess_return_pct","early_negative_return_pct","early_ai_penalty_pct","recovery_boost_pct","max_drawdown_hint_pct"], human_cols=[]), width="stretch", hide_index=True)
        st.download_button("下載情境表 CSV", prepare_download(pd.DataFrame(scenario_df)), "scenario_table.csv", "text/csv")
    with raw_tabs[4]:
        try:
            st.dataframe(format_table_df(sims_df, human_cols=["current_final_assets_twd","recommended_final_assets_twd"]), width="stretch", hide_index=True)
            st.download_button("下載 Monte Carlo 明細 CSV", prepare_download(sims_df), "mc_detail.csv", "text/csv")
        except Exception:
            st.info("請先到『情境矩陣 / Monte Carlo』頁跑一次。")
