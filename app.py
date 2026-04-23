
from __future__ import annotations

import json
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
st.title("Ben 財務與投資模擬器 · 收入引擎整合版 v5")
st.caption("加入：Tab 樣式、設定儲存、Excel UX 改善、Current 驗算欄位、AI 分析 Prompt 區塊。")

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


@st.cache_data(show_spinner=False)
def cached_load_positions(source):
    return load_positions(source)


@st.cache_data(show_spinner=False)
def cached_load_biz_projection(source):
    return load_business_income_projection(source)


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
    fx_rate = st.number_input("USD/TWD 匯率", min_value=20.0, max_value=50.0, value=float(saved.get("fx_rate", 32.0)), step=0.1, help="把美元持股換算成台幣時使用。")

    csv_source = SAMPLE_CSV if SAMPLE_CSV.exists() else uploaded_csv
    if source_mode == "上傳新的 CSV" and uploaded_csv is not None:
        csv_source = uploaded_csv

    current_positions_raw, missing_fields = cached_load_positions(str(csv_source) if isinstance(csv_source, Path) else csv_source)
    uploaded_total_usd = float(current_positions_raw["market_value_usd"].sum()) if not current_positions_raw.empty else None
    defaults = default_control_values(uploaded_total_usd, fx_rate)

    st.header("2) 模型控制")
    start_assets_twd = st.number_input("起始流動資產 (TWD)", min_value=0.0, value=float(saved.get("start_assets_twd", defaults["start_assets_twd"])), step=100000.0, help="三組方案比較時共用的起跑點。越一致，結果越能比較。")
    simulation_years = st.slider("模擬年數", 5, 30, int(saved.get("simulation_years", defaults["simulation_years"])), help="通常先看 20 年，才能看到退休與教育費高峰。")
    rebalance_frequency_years = st.selectbox("再平衡頻率", options=[0, 1, 2, 3, 5], index=[0,1,2,3,5].index(int(saved.get("rebalance_frequency_years", defaults["rebalance_frequency_years"]))), format_func=lambda x: "不再平衡" if x == 0 else f"每 {x} 年", help="多久把持股比例拉回目標權重。新手建議每 1 年。")
    withdrawal_strategy = st.selectbox("現金不足時提領方式", ["比例賣出", "先賣現金 / ETF", "先賣波動低資產"], index=["比例賣出", "先賣現金 / ETF", "先賣波動低資產"].index(saved.get("withdrawal_strategy", "比例賣出")), help="生活費不夠時，系統要先賣哪一種資產。")
    cash_reserve_target_pct = st.slider("現金保留比重 %", 0.0, 40.0, float(saved.get("cash_reserve_target_pct", defaults["cash_reserve_target_pct"])), step=0.5, help="越高越穩，但成長性通常會下降。")
    monte_carlo_sims = st.slider(
    "Monte Carlo 次數",
    100,
    1500,
    int(saved.get("monte_carlo_sims", defaults.get("monte_carlo_sims", 500))),
    step=100,
    help="次數越高，勝率更穩定，但會比較慢。"
)

    st.header("3) 本業收入（加薪→退休）")
    salary_annual = st.number_input("本業年薪", min_value=0.0, value=float(saved.get("salary_annual", defaults["salary_annual"])), step=50000.0, help="你可支配的本業年度收入，不是公司營收。")
    salary_growth_pct = st.number_input("薪資成長率 %", min_value=-5.0, max_value=20.0, value=float(saved.get("salary_growth_pct", defaults["salary_growth_pct"])), step=0.1, help="退休前本業收入每年成長率。新手可先用 2%。")
    retirement_age = st.number_input("退休年齡", min_value=45, max_value=90, value=int(saved.get("retirement_age", defaults["retirement_age"])), step=1, help="到了這個年齡，本業收入會歸零。")

    st.header("4) 妥妥租收入（遞減 / Excel）")
    tuotuozu_mode = st.radio("妥妥租模式", ["手動遞減", "Excel 預測"], index=0 if saved.get("tuotuozu_mode", "手動遞減") == "手動遞減" else 1, help="手動遞減適合快速試算；Excel 預測適合沿用你原本的預測表。")
    biz_projection_list, biz_projection_preview = [], pd.DataFrame(columns=["year", "net_profit_twd"])
    if tuotuozu_mode == "Excel 預測":
        biz_excel_source_mode = st.radio("妥妥租 Excel", ["使用內建範例 Excel", "上傳 Excel"], index=0 if saved.get("biz_excel_source_mode", "使用內建範例 Excel") == "使用內建範例 Excel" else 1, help="如果你 repo 的 data 裡有妥妥租_預測.xlsx，可直接用內建版本。")
        uploaded_biz_excel = st.file_uploader("上傳 妥妥租_預測.xlsx", type=["xlsx"], key="uploaded_biz_excel")
        if biz_excel_source_mode == "上傳 Excel" and uploaded_biz_excel is not None:
            biz_projection_list, biz_projection_preview = cached_load_biz_projection(uploaded_biz_excel)
        elif SAMPLE_BIZ_XLSX.exists():
            biz_projection_list, biz_projection_preview = cached_load_biz_projection(str(SAMPLE_BIZ_XLSX))
    tuotuozu_base_annual = st.number_input("妥妥租目前年度淨利", min_value=0.0, value=float(saved.get("tuotuozu_base_annual", defaults["tuotuozu_base_annual"])), step=100000.0, help="當你不用 Excel 時，這個數字就是妥妥租收入的起點。")
    tuotuozu_decay_pct = st.number_input("妥妥租年衰退率 %", min_value=-50.0, max_value=30.0, value=float(saved.get("tuotuozu_decay_pct", defaults["tuotuozu_decay_pct"])), step=0.5, help="例如 10 代表每年衰退 10%。")
    tuotuozu_fallback_mode = st.selectbox("Excel 年數用完後", ["continue_decay_from_last_value", "zero_after_list_end"], index=0 if saved.get("tuotuozu_fallback_mode", "continue_decay_from_last_value") == "continue_decay_from_last_value" else 1, format_func=lambda x: "從最後一年繼續遞減" if x == "continue_decay_from_last_value" else "直接歸零", help="若 Excel 只有 10 年、你模擬 20 年，後面要怎麼處理。")

    st.header("5) 支出 / 繼承")
    living_expense_annual = st.number_input("基礎生活費 / 年", min_value=0.0, value=float(saved.get("living_expense_annual", defaults["living_expense_annual"])), step=50000.0, help="不含教育費的家庭年度生活支出。")
    inflation_pct = st.number_input("通膨率 %", min_value=0.0, max_value=20.0, value=float(saved.get("inflation_pct", defaults["inflation_pct"])), step=0.1, help="生活費逐年上升的速度。")
    edu_phase1_annual = st.number_input("教育費 2026-2033 / 年", min_value=0.0, value=float(saved.get("edu_phase1_annual", defaults["edu_phase1_annual"])), step=50000.0, help="教育費中度區間。")
    edu_phase2_annual = st.number_input("教育費 2034-2038 / 年", min_value=0.0, value=float(saved.get("edu_phase2_annual", defaults["edu_phase2_annual"])), step=50000.0, help="教育費高峰區間。")
    mortgage_annual = st.number_input("房貸 / 居住成本 / 年", min_value=0.0, value=float(saved.get("mortgage_annual", defaults["mortgage_annual"])), step=50000.0, help="如果已經含在生活費，這裡就不要再重複算。")
    inheritance_age = st.number_input("遺產事件年齡", min_value=45, max_value=90, value=int(saved.get("inheritance_age", defaults["inheritance_age"])), step=1, help="從這個年齡開始，繼承租金收入會進來。")
    inherited_rent_monthly = st.number_input("遺產後租金 / 月", min_value=0.0, value=float(saved.get("inherited_rent_monthly", defaults["inherited_rent_monthly"])), step=5000.0, help="遺產事件發生後，每月新增多少租金收入。")

    st.header("6) 設定儲存")
    st.caption("此版本會把設定存成 JSON。Cloud 重啟或重新部署後可能需要重新載入，但一般重新整理不會全失。")


st.subheader("資料讀取狀態")
mcols = st.columns(6)
mcols[0].metric("Current 持股筆數", f"{current_positions_raw.shape[0]}")
mcols[1].metric("Current 市值 (USD)", fmt_num(uploaded_total_usd, 0))
mcols[2].metric("Current 市值折 TWD", fmt_num((uploaded_total_usd or 0) * fx_rate, 0))
mcols[3].metric("建模起始資產", fmt_num(start_assets_twd, 0))
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

saved_current = df_from_saved("current_portfolio", current_default, saved)
saved_recommended = df_from_saved("recommended_portfolio", build_recommended_portfolio(), saved)
saved_custom = df_from_saved("custom_portfolio", build_recommended_portfolio(), saved)
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
    "expected_return_pct": st.column_config.NumberColumn("年化報酬 %", min_value=-20.0, max_value=80.0, step=0.5),
    "volatility_pct": st.column_config.NumberColumn("波動率 %", min_value=0.0, max_value=100.0, step=0.5),
    "asset_class": st.column_config.TextColumn("資產類別", disabled=True),
    "theme": st.column_config.TextColumn("主題", disabled=True),
    "role_bucket": st.column_config.TextColumn("角色", disabled=True),
    "risk_group": st.column_config.TextColumn("風險群組", disabled=True),
    "ai_direct": st.column_config.CheckboxColumn("AI 直接曝險", disabled=True),
    "sell_priority": st.column_config.NumberColumn("賣出優先序", min_value=0, max_value=10, step=1),
}

main_tabs = st.tabs(["1. 輸入與持股編輯", "2. 儀表板", "3. 單一情境模擬", "4. 情境矩陣 / Monte Carlo", "5. 驗證 / 除錯", "6. 原始資料與下載"])

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
        current_edited = st.data_editor(saved_current, hide_index=True, use_container_width=True, num_rows="fixed", column_config=editor_cfg, key="current_editor")
        current_sum = current_edited.loc[current_edited["include"], "weight_pct"].sum()
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Current 權重總和", fmt_pct(current_sum, 2))
        c2.metric("Current 市值合計 USD", fmt_num(current_edited["market_value_usd"].sum(), 0))
        c3.metric("Current 成本合計 USD", fmt_num(current_edited["cost_basis_usd"].sum(), 0))
        c4.metric("Current 未實現損益 USD", fmt_num(current_edited["unrealized_gain_usd"].sum(), 0))
        c5.metric("Current 市值折 TWD", fmt_num(current_edited["market_value_usd"].sum() * fx_rate, 0))
        if abs(current_sum - 100) > 0.5:
            st.warning("Current 權重總和目前不是 100%。模擬前會自動正規化，但建議你先確認。")
    with sub_tabs[1]:
        recommended_edited = st.data_editor(saved_recommended, hide_index=True, use_container_width=True, num_rows="dynamic", column_config={k:v for k,v in editor_cfg.items() if k not in {"quantity","market_value_usd","cost_basis_usd","unrealized_gain_usd","gain_pct"}}, key="recommended_editor")
        rsum = recommended_edited.loc[recommended_edited["include"], "weight_pct"].sum()
        st.metric("Recommended 權重總和", fmt_pct(rsum, 2))
    with sub_tabs[2]:
        custom_edited = st.data_editor(saved_custom, hide_index=True, use_container_width=True, num_rows="dynamic", column_config={k:v for k,v in editor_cfg.items() if k not in {"quantity","market_value_usd","cost_basis_usd","unrealized_gain_usd","gain_pct"}}, key="custom_editor")
        csum = custom_edited.loc[custom_edited["include"], "weight_pct"].sum()
        st.metric("Custom 權重總和", fmt_pct(csum, 2))
    with sub_tabs[3]:
        comparison_df = build_comparison(current_edited, recommended_edited)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    with sub_tabs[4]:
        if tuotuozu_mode != "Excel 預測":
            st.warning("目前為「手動遞減」模式，尚未啟用 Excel 預測，因此此處不顯示 Excel 預覽。")
        elif biz_projection_preview.empty:
            st.error("已切換到 Excel 預測模式，但目前沒有成功讀到 Excel 資料。請檢查檔名、上傳檔案與欄位格式。")
        else:
            st.success(f"已成功讀取 {len(biz_projection_preview)} 年妥妥租預測資料。")
            st.dataframe(biz_projection_preview, use_container_width=True, hide_index=True)

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

current_norm = apply_cash_reserve_target(normalize_weights(current_edited.copy()), cash_reserve_target_pct)
recommended_norm = apply_cash_reserve_target(normalize_weights(recommended_edited.copy()), cash_reserve_target_pct)
custom_norm = apply_cash_reserve_target(normalize_weights(custom_edited.copy()), cash_reserve_target_pct)

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
        st.plotly_chart(px.bar(alloc_long, x="ticker", y="weight_pct", color="portfolio", barmode="group", title="資產配置長條圖"), use_container_width=True)
    with c2:
        ai_long = risk_df.melt(id_vars="portfolio", value_vars=["直接 AI 曝險 %","ETF 重疊曝險 %","同跌風險程度（0-100）"], var_name="指標", value_name="數值")
        st.plotly_chart(px.bar(ai_long, x="portfolio", y="數值", color="指標", barmode="group", title="AI / 科技集中度圖"), use_container_width=True)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

with main_tabs[2]:
    st.markdown("## 單一情境模擬")
    scenario_df = st.data_editor(saved_scenarios, use_container_width=True, hide_index=True, num_rows="fixed", key="scenario_editor")
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

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Current 最終資產", fmt_num(cur_sum.get("最終資產終值"), 0))
    s2.metric("Recommended 最終資產", fmt_num(rec_sum.get("最終資產終值"), 0))
    s3.metric("Custom 最終資產", fmt_num(cus_sum.get("最終資產終值"), 0))
    s4.metric("Recommended 最大回撤", fmt_pct(rec_sum.get("最大回撤 %"), 1))

    combo = pd.concat([
        cur_res.assign(portfolio="Current"),
        rec_res.assign(portfolio="Recommended"),
        cus_res.assign(portfolio="Custom"),
    ], ignore_index=True)

    viz_tabs = st.tabs(["資產成長曲線", "現金流缺口圖", "收入拆解", "最大回撤比較", "AI 分析 Prompt"])
    with viz_tabs[0]:
        st.plotly_chart(px.line(combo, x="calendar_year", y="end_assets_twd", color="portfolio", markers=True, title="資產成長曲線"), use_container_width=True)
    with viz_tabs[1]:
        fig = px.bar(combo, x="calendar_year", y="net_cashflow_twd", color="portfolio", barmode="group", title="現金流缺口圖")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.add_vrect(x0=2034 - 0.5, x1=2038 + 0.5, fillcolor="orange", opacity=0.12, line_width=0, annotation_text="教育費高峰")
        st.plotly_chart(fig, use_container_width=True)
    with viz_tabs[2]:
        income_combo = pd.concat([
            rec_res[["calendar_year","salary_income_twd","tuotuozu_income_twd","inheritance_income_twd"]].assign(portfolio="Recommended")
        ])
        income_long = income_combo.melt(id_vars=["calendar_year","portfolio"], var_name="收入來源", value_name="金額")
        st.plotly_chart(px.bar(income_long, x="calendar_year", y="金額", color="收入來源", title="Recommended 收入拆解"), use_container_width=True)
    with viz_tabs[3]:
        st.plotly_chart(px.line(combo, x="calendar_year", y="drawdown_pct", color="portfolio", markers=True, title="最大回撤比較"), use_container_width=True)
    with viz_tabs[4]:
        ai_prompt = f"""請根據以下模擬結果，做決策型分析，而不是只重述數字。\n\n【本次情境】\n{scenario_name}\n\n【我的目標】\n1. 長期勝率\n2. 不容易中途出局\n3. 教育費高峰時可承受\n4. 大跌時不容易被迫賣股\n5. 整體人生舒服度\n\n【輸入摘要】\n- 起始流動資產: {fmt_num(start_assets_twd,0)} TWD\n- 本業年薪: {fmt_num(salary_annual,0)}\n- 薪資成長率: {fmt_pct(salary_growth_pct,1)}\n- 退休年齡: {retirement_age}\n- 妥妥租模式: {tuotuozu_mode}\n- 妥妥租目前年度淨利: {fmt_num(tuotuozu_base_annual,0)}\n- 妥妥租衰退率: {fmt_pct(tuotuozu_decay_pct,1)}\n- 妥妥租 Excel 年數: {len(biz_projection_list)}\n- 基礎生活費/年: {fmt_num(living_expense_annual,0)}\n- 教育費 2026-2033/年: {fmt_num(edu_phase1_annual,0)}\n- 教育費 2034-2038/年: {fmt_num(edu_phase2_annual,0)}\n- 房貸/年: {fmt_num(mortgage_annual,0)}\n- 現金保留比重: {fmt_pct(cash_reserve_target_pct,1)}\n\n【結果摘要】\nCurrent: 最終資產 {fmt_num(cur_sum.get('最終資產終值'),0)} / 最大回撤 {fmt_pct(cur_sum.get('最大回撤 %'),1)} / 10年後 {fmt_num(cur_sum.get('10 年後淨資產'),0)} / 20年後 {fmt_num(cur_sum.get('20 年後淨資產'),0)} / 人生適配分數 {cur_sum.get('人生適配分數')}\nRecommended: 最終資產 {fmt_num(rec_sum.get('最終資產終值'),0)} / 最大回撤 {fmt_pct(rec_sum.get('最大回撤 %'),1)} / 10年後 {fmt_num(rec_sum.get('10 年後淨資產'),0)} / 20年後 {fmt_num(rec_sum.get('20 年後淨資產'),0)} / 人生適配分數 {rec_sum.get('人生適配分數')}\nCustom: 最終資產 {fmt_num(cus_sum.get('最終資產終值'),0)} / 最大回撤 {fmt_pct(cus_sum.get('最大回撤 %'),1)} / 10年後 {fmt_num(cus_sum.get('10 年後淨資產'),0)} / 20年後 {fmt_num(cus_sum.get('20 年後淨資產'),0)} / 人生適配分數 {cus_sum.get('人生適配分數')}\n\n【請回答】\n1. 先結論：哪個方案最適合我，為什麼？\n2. 分成：事實 / 推論 / 假設\n3. 一定要分析：最大回撤、現金流缺口、資產耗盡風險、人生適配分數\n4. 指出哪個方案比較像「賺得多但難撐」\n5. 指出哪個方案比較像「賺得稍慢但比較舒服」\n6. 若看到任何數據不合理，請提醒我先回頭驗模型\n"""
        st.text_area("可直接複製給 AI 分析的 Prompt", value=ai_prompt, height=420)
        st.download_button("下載這次模擬的 AI Prompt", data=ai_prompt, file_name="ai_analysis_prompt.txt", mime="text/plain")

with main_tabs[3]:
    st.markdown("## 情境矩陣 / Monte Carlo")
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
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    sims_df, mc_metrics = run_monte_carlo_compare(
        current_norm, recommended_norm, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR,
        salary_annual, salary_growth_pct, retirement_age, tuotuozu_mode, tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode,
        living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual, mortgage_annual, inheritance_age, inherited_rent_monthly,
        withdrawal_strategy, rebalance_frequency_years, mode_override_value, simulations=monte_carlo_sims, seed=42
    )
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Recommended 勝過 Current 機率", fmt_pct(mc_metrics.get("B 方案勝過 A 方案的機率"),1))
    m2.metric("Current 資產耗盡機率", fmt_pct(mc_metrics.get("Current 資產耗盡機率"),1))
    m3.metric("Recommended 資產耗盡機率", fmt_pct(mc_metrics.get("Recommended 資產耗盡機率"),1))
    m4.metric("Recommended 中位數終值", fmt_num(mc_metrics.get("Recommended 終值中位數"),0))
    long_mc = sims_df.melt(value_vars=["current_final_assets_twd","recommended_final_assets_twd"], var_name="portfolio", value_name="final_assets_twd")
    st.plotly_chart(px.histogram(long_mc, x="final_assets_twd", color="portfolio", barmode="overlay", nbins=40, title="Monte Carlo 終值分布"), use_container_width=True)

with main_tabs[4]:
    st.markdown("## 驗證 / 除錯")
    checks = []
    for name, df in [("Current", current_edited), ("Recommended", recommended_edited), ("Custom", custom_edited)]:
        s = df.loc[df["include"], "weight_pct"].sum()
        checks.append({"檢查項目": f"{name} 權重總和", "結果": "PASS" if abs(s-100)<=0.5 else "WARNING", "說明": f"目前合計 {s:.2f}%（模擬前會自動正規化）"})
    if not rec_res.empty:
        year_seq_ok = rec_res["calendar_year"].is_monotonic_increasing and rec_res["calendar_year"].nunique() == rec_res.shape[0]
        checks.append({"檢查項目": "年度序列", "結果": "PASS" if year_seq_ok else "FAIL", "說明": "年份應連續且不重複"})
        salary_ok = float(rec_res.iloc[-1]["salary_income_twd"]) == 0.0 if int(rec_res.iloc[-1]["age"]) >= retirement_age else True
        checks.append({"檢查項目": "本業收入退休後歸零", "結果": "PASS" if salary_ok else "FAIL", "說明": "退休後薪資應歸零"})
        if tuotuozu_mode == "Excel 預測":
            checks.append({"檢查項目": "妥妥租 Excel 載入", "結果": "PASS" if len(biz_projection_list)>0 else "FAIL", "說明": f"目前讀到 {len(biz_projection_list)} 年"})
        edu_peak_ok = rec_res.loc[rec_res["calendar_year"].between(2034,2038), "education_expense_twd"].max() >= edu_phase2_annual * 0.9
        checks.append({"檢查項目": "教育費高峰", "結果": "PASS" if edu_peak_ok else "WARNING", "說明": "2034-2038 應明顯高於前段"})
    check_df = pd.DataFrame(checks)
    st.dataframe(check_df, use_container_width=True, hide_index=True)

    sens_rows = []
    for param, low, high, label in [
        ("cash_reserve_target_pct", max(0,cash_reserve_target_pct-4), min(40,cash_reserve_target_pct+4), "現金保留比重"),
        ("salary_growth_pct", max(-5,salary_growth_pct-1), min(20,salary_growth_pct+1), "薪資成長率"),
        ("tuotuozu_decay_pct", max(-50,tuotuozu_decay_pct-3), min(30,tuotuozu_decay_pct+3), "妥妥租衰退率"),
        ("inflation_pct", max(0,inflation_pct-1), min(20,inflation_pct+1), "通膨率"),
        ("edu_phase2_annual", max(0,edu_phase2_annual-200000), edu_phase2_annual+200000, "教育費高峰"),
    ]:
        sens_rows.append({"參數": label, "低值": low, "高值": high, "提醒": "這項對結果常有明顯影響，調整前後請重跑並比較圖表。"})
    st.markdown("### 最敏感 5 個參數（先用人工提醒版）")
    st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

with main_tabs[5]:
    st.markdown("## 原始資料與下載")
    raw_tabs = st.tabs(["Current 正規化後", "Recommended 正規化後", "Custom 正規化後", "情境表", "Monte Carlo 明細"])
    with raw_tabs[0]:
        st.dataframe(current_norm, use_container_width=True, hide_index=True)
        st.download_button("下載 Current CSV", prepare_download(current_norm), "current_portfolio.csv", "text/csv")
    with raw_tabs[1]:
        st.dataframe(recommended_norm, use_container_width=True, hide_index=True)
        st.download_button("下載 Recommended CSV", prepare_download(recommended_norm), "recommended_portfolio.csv", "text/csv")
    with raw_tabs[2]:
        st.dataframe(custom_norm, use_container_width=True, hide_index=True)
        st.download_button("下載 Custom CSV", prepare_download(custom_norm), "custom_portfolio.csv", "text/csv")
    with raw_tabs[3]:
        st.dataframe(pd.DataFrame(scenario_df), use_container_width=True, hide_index=True)
        st.download_button("下載情境表 CSV", prepare_download(pd.DataFrame(scenario_df)), "scenario_table.csv", "text/csv")
    with raw_tabs[4]:
        try:
            st.dataframe(sims_df, use_container_width=True, hide_index=True)
            st.download_button("下載 Monte Carlo 明細 CSV", prepare_download(sims_df), "mc_detail.csv", "text/csv")
        except Exception:
            st.info("請先到『情境矩陣 / Monte Carlo』頁跑一次。")
