from __future__ import annotations

from pathlib import Path

import numpy as np
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
    sensitivity_scan,
    simulate_portfolio,
    summarize_simulation,
    validate_cashflow_formula,
    validate_education_peak,
    validate_income_engine,
    validate_ruin_logic,
    validate_scenario_switch,
    validate_start_assets_equal,
    validate_weight_sum,
    validate_year_sequence,
)

st.set_page_config(page_title="Ben 財務與投資模擬器", layout="wide")
st.title("Ben 財務與投資模擬器 · 收入引擎整合版")
st.caption("已開始整合舊工具邏輯：本業收入（加薪→退休）＋ 妥妥租收入（Excel 預測 / 手動遞減），並加入驗證與敏感度分析。")

BASE_DIR = Path(__file__).parent
SAMPLE_CSV_PATH = BASE_DIR / "data" / "Individual-Positions-2026-04-22-184253(1).csv"
SAMPLE_BIZ_XLSX_PATH = BASE_DIR / "data" / "妥妥租_預測.xlsx"


def fmt_number(value: float | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}%"


def prepare_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def load_default_csv(path: str):
    return load_positions(path)


@st.cache_data(show_spinner=False)
def load_default_biz_projection(path: str):
    return load_business_income_projection(path)


with st.sidebar:
    st.header("1) 資料來源")
    source_mode = st.radio("持股來源", ["使用內建範例 CSV", "上傳新的 CSV"], index=0)
    uploaded_csv = st.file_uploader("上傳持股 CSV", type=["csv"], accept_multiple_files=False)
    fx_rate = st.number_input("USD/TWD 匯率", min_value=20.0, max_value=50.0, value=32.0, step=0.1)

    csv_source = SAMPLE_CSV_PATH if SAMPLE_CSV_PATH.exists() else Path("/mnt/data/Individual-Positions-2026-04-22-184253(1).csv")
    if source_mode == "上傳新的 CSV" and uploaded_csv is not None:
        csv_source = uploaded_csv

    if isinstance(csv_source, Path):
        current_positions_raw, missing_fields = load_default_csv(str(csv_source))
    else:
        current_positions_raw, missing_fields = load_positions(csv_source)

    uploaded_total_usd = float(current_positions_raw["market_value_usd"].sum()) if not current_positions_raw.empty else None
    defaults = default_control_values(uploaded_total_usd, fx_rate)

    st.header("2) 模型控制")
    start_assets_twd = st.number_input("起始流動資產 (TWD)", min_value=0.0, value=float(defaults["start_assets_twd"]), step=100000.0)
    simulation_years = st.slider("模擬年數", 5, 30, int(defaults["simulation_years"]))
    rebalance_frequency_years = st.selectbox("再平衡頻率", options=[0, 1, 2, 3, 5], format_func=lambda x: "不再平衡" if x == 0 else f"每 {x} 年")
    withdrawal_strategy = st.selectbox("現金不足時提領方式", ["比例賣出", "先賣現金 / ETF", "先賣波動低資產"])
    cash_reserve_target_pct = st.slider("現金保留比重 %", 0.0, 40.0, float(defaults["cash_reserve_target_pct"]), step=0.5)
    monte_carlo_sims = st.slider("Monte Carlo 次數", 100, 1500, 500, step=100)

    st.header("3) 本業收入（加薪→退休）")
    salary_annual = st.number_input("本業年薪", min_value=0.0, value=float(defaults["salary_annual"]), step=50000.0)
    salary_growth_pct = st.number_input("薪資成長率 %", min_value=-5.0, max_value=20.0, value=float(defaults["salary_growth_pct"]), step=0.1)
    retirement_age = st.number_input("退休年齡", min_value=45, max_value=90, value=int(defaults["retirement_age"]), step=1)

    st.header("4) 妥妥租收入（遞減 / Excel）")
    tuotuozu_mode = st.radio("妥妥租模式", ["手動遞減", "Excel 預測"], index=0)
    biz_excel_file = None
    biz_projection_list, biz_projection_preview = [], pd.DataFrame(columns=["year", "net_profit_twd"])
    if tuotuozu_mode == "Excel 預測":
        biz_excel_source_mode = st.radio("妥妥租 Excel", ["使用內建範例 Excel", "上傳 Excel"], index=0)
        biz_excel_file = st.file_uploader("上傳 妥妥租_預測.xlsx", type=["xlsx"], accept_multiple_files=False)
        if biz_excel_source_mode == "上傳 Excel" and biz_excel_file is not None:
            biz_projection_list, biz_projection_preview = load_business_income_projection(biz_excel_file)
        elif SAMPLE_BIZ_XLSX_PATH.exists():
            biz_projection_list, biz_projection_preview = load_default_biz_projection(str(SAMPLE_BIZ_XLSX_PATH))
    tuotuozu_base_annual = st.number_input("妥妥租目前年度淨利", min_value=0.0, value=float(defaults["tuotuozu_base_annual"]), step=100000.0)
    tuotuozu_decay_pct = st.number_input("妥妥租年衰退率 %", min_value=-50.0, max_value=20.0, value=float(defaults["tuotuozu_decay_pct"]), step=0.5)
    tuotuozu_fallback_mode = st.selectbox(
        "Excel 年數用完後",
        options=["continue_decay_from_last_value", "zero_after_list_end"],
        format_func=lambda x: "從最後一年繼續遞減" if x == "continue_decay_from_last_value" else "直接歸零",
    )

    st.header("5) 支出 / 繼承")
    living_expense_annual = st.number_input("基礎生活費 / 年", min_value=0.0, value=float(defaults["living_expense_annual"]), step=50000.0)
    inflation_pct = st.number_input("通膨率 %", min_value=0.0, max_value=20.0, value=float(defaults["inflation_pct"]), step=0.1)
    edu_phase1_annual = st.number_input("教育費 2026-2033 / 年", min_value=0.0, value=float(defaults["edu_phase1_annual"]), step=50000.0)
    edu_phase2_annual = st.number_input("教育費 2034-2038 / 年", min_value=0.0, value=float(defaults["edu_phase2_annual"]), step=50000.0)
    mortgage_annual = st.number_input("房貸 / 居住成本 / 年", min_value=0.0, value=float(defaults["mortgage_annual"]), step=50000.0)
    inheritance_age = st.number_input("遺產事件年齡", min_value=45, max_value=90, value=int(defaults["inheritance_age"]), step=1)
    inherited_rent_monthly = st.number_input("遺產後租金 / 月", min_value=0.0, value=float(defaults["inherited_rent_monthly"]), step=5000.0)

    st.header("6) 視圖")
    selected_curve_scenarios = st.multiselect("資產曲線情境", options=scenario_table()["scenario_name"].tolist(), default=scenario_table()["scenario_name"].tolist()[:3])

st.subheader("資料讀取狀態")
status_cols = st.columns(6)
status_cols[0].metric("Current 持股筆數", f"{current_positions_raw.shape[0]}")
status_cols[1].metric("Current 市值 (USD)", f"{uploaded_total_usd:,.0f}" if uploaded_total_usd else "0")
status_cols[2].metric("Current 市值折 TWD", f"{(uploaded_total_usd or 0) * fx_rate:,.0f}")
status_cols[3].metric("建模起始資產", f"{start_assets_twd:,.0f}")
status_cols[4].metric("現金保留比重", f"{cash_reserve_target_pct:.1f}%")
status_cols[5].metric("妥妥租 Excel 年數", f"{len(biz_projection_list)}")
if missing_fields:
    st.warning(f"CSV 缺少欄位：{', '.join(missing_fields)}。已用替代邏輯補足，但建議檢查。")
else:
    st.success("CSV 欄位映射完整，可直接作為 Current Portfolio 基礎來源。")
if tuotuozu_mode == "Excel 預測":
    if len(biz_projection_list) == 0:
        st.warning("妥妥租 Excel 模式已啟用，但目前沒有成功讀到預測數列，結果會失真。")
    else:
        st.info("妥妥租收入將優先使用 Excel 預測表；若模擬年數超過表格年數，會依你選擇的 fallback 規則處理。")

current_portfolio_default = current_positions_raw[[
    "ticker", "name", "weight_pct", "asset_class", "theme", "role_bucket", "risk_group", "ai_direct",
    "expected_return_pct", "volatility_pct", "sell_priority", "include",
]].copy()
recommended_portfolio_default = build_recommended_portfolio()
custom_portfolio_default = build_recommended_portfolio().copy()

main_tabs = st.tabs([
    "1. 輸入與持股編輯",
    "2. 儀表板",
    "3. 單一情境模擬",
    "4. 情境矩陣 / Monte Carlo",
    "5. 驗證 / 除錯",
    "6. 原始資料與下載",
])

with main_tabs[0]:
    st.markdown("## 舊邏輯整合說明")
    st.markdown(
        """
- **本業收入**：沿用舊工具邏輯，會隨 `薪資成長率` 增長，到了 `退休年齡` 之後歸零。
- **妥妥租收入**：可切換成 `手動遞減` 或 `Excel 預測` 模式。
- **Excel 模式 fallback**：當預測表年數不足，可選「繼續遞減」或「直接歸零」。
- **目標**：讓新工具不只會算投資，還能保留你原本人生現金流模型的骨架。
        """
    )

    editor_columns = {
        "include": st.column_config.CheckboxColumn("納入模擬"),
        "ticker": st.column_config.TextColumn("Ticker", disabled=True),
        "name": st.column_config.TextColumn("名稱", disabled=True),
        "weight_pct": st.column_config.NumberColumn("權重 %", min_value=0.0, max_value=100.0, step=0.1),
        "expected_return_pct": st.column_config.NumberColumn("年化報酬 %", min_value=-50.0, max_value=80.0, step=0.5),
        "volatility_pct": st.column_config.NumberColumn("波動率 %", min_value=0.0, max_value=100.0, step=0.5),
        "asset_class": st.column_config.TextColumn("資產類別", disabled=True),
        "theme": st.column_config.TextColumn("主題", disabled=True),
        "role_bucket": st.column_config.TextColumn("角色", disabled=True),
        "risk_group": st.column_config.TextColumn("風險群組", disabled=True),
        "ai_direct": st.column_config.CheckboxColumn("AI 直接曝險", disabled=True),
        "sell_priority": st.column_config.NumberColumn("賣出優先序", min_value=0, max_value=10, step=1),
    }
    edit_tabs = st.tabs(["Current", "Recommended", "Custom", "Current vs Recommended", "妥妥租 Excel 預覽"])
    with edit_tabs[0]:
        current_edited = st.data_editor(current_portfolio_default, hide_index=True, use_container_width=True, num_rows="fixed", column_config=editor_columns, key="current_editor")
        st.info(f"Current 原始權重合計：{current_edited.loc[current_edited['include'], 'weight_pct'].sum():.2f}%（模擬前會自動正規化）")
    with edit_tabs[1]:
        recommended_edited = st.data_editor(recommended_portfolio_default, hide_index=True, use_container_width=True, num_rows="dynamic", column_config=editor_columns, key="recommended_editor")
        st.info(f"Recommended 原始權重合計：{recommended_edited.loc[recommended_edited['include'], 'weight_pct'].sum():.2f}%（模擬前會自動正規化）")
    with edit_tabs[2]:
        custom_edited = st.data_editor(custom_portfolio_default, hide_index=True, use_container_width=True, num_rows="dynamic", column_config=editor_columns, key="custom_editor")
        st.info(f"Custom 原始權重合計：{custom_edited.loc[custom_edited['include'], 'weight_pct'].sum():.2f}%（模擬前會自動正規化）")
    with edit_tabs[3]:
        comparison_live = build_comparison(current_edited, recommended_edited)
        st.dataframe(comparison_live, use_container_width=True, hide_index=True)
    with edit_tabs[4]:
        st.dataframe(biz_projection_preview if not biz_projection_preview.empty else pd.DataFrame(columns=["year", "net_profit_twd"]), use_container_width=True, hide_index=True)
        st.caption("這張表是從舊版 `妥妥租_預測.xlsx` 讀進來的年度稅後淨利預測。")

current_norm = apply_cash_reserve_target(normalize_weights(pd.DataFrame(current_edited)), cash_reserve_target_pct)
rec_norm = apply_cash_reserve_target(normalize_weights(pd.DataFrame(recommended_edited)), cash_reserve_target_pct)
custom_norm = apply_cash_reserve_target(normalize_weights(pd.DataFrame(custom_edited)), cash_reserve_target_pct)

# shared scenarios and simulation helpers
scenarios_default = scenario_table()
selected_scenario_name = scenarios_default["scenario_name"].iloc[1]
selected_scenario = get_scenario_row(scenarios_default, selected_scenario_name)
mode_override_value = None


def run_portfolio(port_df: pd.DataFrame, scenario_row: pd.Series, seed: int = 42, mode_override: str | None = None) -> pd.DataFrame:
    return simulate_portfolio(
        port_df, scenario_row, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR,
        salary_annual, salary_growth_pct, retirement_age,
        "excel_projection" if tuotuozu_mode == "Excel 預測" else "manual_decay",
        tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode,
        living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual,
        mortgage_annual, inheritance_age, inherited_rent_monthly,
        withdrawal_strategy, rebalance_frequency_years, mode_override, seed,
    )


# base results for dashboard
base_current_result = run_portfolio(current_norm, selected_scenario, seed=42)
base_rec_result = run_portfolio(rec_norm, selected_scenario, seed=42)
base_custom_result = run_portfolio(custom_norm, selected_scenario, seed=42)
base_current_summary = summarize_simulation(base_current_result, start_assets_twd)
base_rec_summary = summarize_simulation(base_rec_result, start_assets_twd)
base_custom_summary = summarize_simulation(base_custom_result, start_assets_twd)

scenario_rows = []
for _, srow in scenarios_default.iterrows():
    cur = run_portfolio(current_norm, srow, seed=42)
    rec = run_portfolio(rec_norm, srow, seed=42)
    custom = run_portfolio(custom_norm, srow, seed=42)
    cur_s = summarize_simulation(cur, start_assets_twd)
    rec_s = summarize_simulation(rec, start_assets_twd)
    custom_s = summarize_simulation(custom, start_assets_twd)
    scenario_rows.append({
        "scenario": srow["scenario_name"],
        "Current_最終資產": cur_s.get("最終資產終值", np.nan),
        "Recommended_最終資產": rec_s.get("最終資產終值", np.nan),
        "Custom_最終資產": custom_s.get("最終資產終值", np.nan),
        "Current_MaxDD_pct": cur_s.get("最大回撤 %", np.nan),
        "Recommended_MaxDD_pct": rec_s.get("最大回撤 %", np.nan),
        "Custom_MaxDD_pct": custom_s.get("最大回撤 %", np.nan),
        "Current_現金流壓力年數": cur_s.get("現金流壓力年數", np.nan),
        "Recommended_現金流壓力年數": rec_s.get("現金流壓力年數", np.nan),
        "Custom_現金流壓力年數": custom_s.get("現金流壓力年數", np.nan),
        "Current_資產耗盡年份": cur_s.get("資產耗盡年份", np.nan),
        "Recommended_資產耗盡年份": rec_s.get("資產耗盡年份", np.nan),
        "Custom_資產耗盡年份": custom_s.get("資產耗盡年份", np.nan),
        "Current_LifeFit": cur_s.get("人生適配分數", np.nan),
        "Recommended_LifeFit": rec_s.get("人生適配分數", np.nan),
        "Custom_LifeFit": custom_s.get("人生適配分數", np.nan),
    })
scenario_matrix_df = pd.DataFrame(scenario_rows)

with main_tabs[1]:
    st.markdown("## 儀表板")
    current_metrics = compute_risk_duplicate_metrics(current_norm)
    rec_metrics = compute_risk_duplicate_metrics(rec_norm)
    ruin_risk_light = "⚪ 未計算"
    worst_case_recommended = scenario_matrix_df["Recommended_最終資產"].min() if not scenario_matrix_df.empty else np.nan

    dash_cols = st.columns(6)
    dash_cols[0].metric("目前總資產", fmt_number(start_assets_twd, 0))
    dash_cols[1].metric("現金比例", fmt_pct(float(current_norm.loc[current_norm["asset_class"] == "Cash", "weight_pct"].sum()), 1))
    dash_cols[2].metric("AI 曝險比例", fmt_pct(float(current_metrics.get("直接 AI 曝險 %", 0.0)), 1))
    dash_cols[3].metric("Recommended 10 年資產", fmt_number(base_rec_summary.get("10 年後淨資產"), 0))
    dash_cols[4].metric("Recommended 最差情境資產", fmt_number(worst_case_recommended, 0))
    dash_cols[5].metric("妥妥租模式", "Excel" if tuotuozu_mode == "Excel 預測" else "手動遞減")

    viz_tabs = st.tabs(["資產成長曲線", "現金流缺口圖", "資產配置圖", "AI / 科技集中度圖", "最大回撤比較圖", "情境比較總覽表"])
    with viz_tabs[0]:
        rows = []
        chosen = selected_curve_scenarios or scenarios_default["scenario_name"].tolist()[:3]
        for sn in chosen:
            srow = get_scenario_row(scenarios_default, sn)
            rows.append(run_portfolio(current_norm, srow, 42).assign(portfolio="Current", scenario=sn))
            rows.append(run_portfolio(rec_norm, srow, 42).assign(portfolio="Recommended", scenario=sn))
        line_df = pd.concat(rows, ignore_index=True)
        st.plotly_chart(px.line(line_df, x="calendar_year", y="end_assets_twd", color="portfolio", line_dash="scenario", title="資產成長曲線"), use_container_width=True)
    with viz_tabs[1]:
        gap_df = base_rec_result.copy()
        fig = go.Figure()
        fig.add_bar(x=gap_df["calendar_year"], y=gap_df["net_cashflow_twd"], name="Recommended 現金流")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.add_vrect(x0=2034 - 0.5, x1=2038 + 0.5, fillcolor="orange", opacity=0.12, line_width=0, annotation_text="教育費高峰", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
    with viz_tabs[2]:
        c1, c2, c3 = st.columns(3)
        for col, label, dfp in [(c1, "目前方案", current_norm), (c2, "建議方案", rec_norm), (c3, "自訂方案", custom_norm)]:
            pie_df = dfp.loc[dfp["include"], ["ticker", "weight_pct"]].sort_values("weight_pct", ascending=False)
            col.plotly_chart(px.pie(pie_df, names="ticker", values="weight_pct", title=label), use_container_width=True)
        all_alloc = pd.concat([
            current_norm[["ticker", "weight_pct"]].assign(portfolio="Current"),
            rec_norm[["ticker", "weight_pct"]].assign(portfolio="Recommended"),
            custom_norm[["ticker", "weight_pct"]].assign(portfolio="Custom"),
        ], ignore_index=True)
        st.plotly_chart(px.bar(all_alloc, x="ticker", y="weight_pct", color="portfolio", barmode="group", title="資產配置長條圖"), use_container_width=True)
    with viz_tabs[3]:
        ai_chart = pd.DataFrame([
            {"portfolio": "Current", "AI 直接曝險": current_metrics.get("直接 AI 曝險 %", 0.0), "ETF 重疊曝險": current_metrics.get("ETF 重疊曝險 %", 0.0)},
            {"portfolio": "Recommended", "AI 直接曝險": rec_metrics.get("直接 AI 曝險 %", 0.0), "ETF 重疊曝險": rec_metrics.get("ETF 重疊曝險 %", 0.0)},
            {"portfolio": "Custom", "AI 直接曝險": compute_risk_duplicate_metrics(custom_norm).get("直接 AI 曝險 %", 0.0), "ETF 重疊曝險": compute_risk_duplicate_metrics(custom_norm).get("ETF 重疊曝險 %", 0.0)},
        ])
        ai_chart["總重疊風險"] = ai_chart["AI 直接曝險"] + ai_chart["ETF 重疊曝險"]
        st.plotly_chart(px.bar(ai_chart.melt(id_vars="portfolio", var_name="metric", value_name="pct"), x="portfolio", y="pct", color="metric", barmode="group", title="AI / 科技集中度圖"), use_container_width=True)
        st.dataframe(ai_chart, hide_index=True, use_container_width=True)
    with viz_tabs[4]:
        dd_rows = []
        for _, row in scenario_matrix_df.iterrows():
            dd_rows.extend([
                {"scenario": row["scenario"], "portfolio": "Current", "max_dd_pct": row["Current_MaxDD_pct"]},
                {"scenario": row["scenario"], "portfolio": "Recommended", "max_dd_pct": row["Recommended_MaxDD_pct"]},
                {"scenario": row["scenario"], "portfolio": "Custom", "max_dd_pct": row["Custom_MaxDD_pct"]},
            ])
        st.plotly_chart(px.bar(pd.DataFrame(dd_rows), x="scenario", y="max_dd_pct", color="portfolio", barmode="group", title="最大回撤比較圖"), use_container_width=True)
    with viz_tabs[5]:
        st.dataframe(scenario_matrix_df, use_container_width=True, hide_index=True)

with main_tabs[2]:
    st.markdown("## 單一情境模擬")
    scenarios_edited = st.data_editor(
        scenarios_default,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key="scenario_editor",
        column_config={
            "scenario_name": st.column_config.TextColumn("情境", disabled=True),
            "mode": st.column_config.SelectboxColumn("模式", options=["fixed", "monte_carlo", "path"]),
            "market_return_shift_pct": st.column_config.NumberColumn("市場位移 %", step=0.5),
            "ai_excess_return_pct": st.column_config.NumberColumn("AI 超額報酬 %", step=0.5),
            "vol_multiplier": st.column_config.NumberColumn("波動倍數", step=0.05),
            "early_negative_years": st.column_config.NumberColumn("前期負報酬年數", step=1),
            "early_negative_return_pct": st.column_config.NumberColumn("前期負報酬 %", step=0.5),
            "early_ai_penalty_pct": st.column_config.NumberColumn("前期 AI 額外懲罰 %", step=0.5),
            "recovery_years": st.column_config.NumberColumn("恢復年數", step=1),
            "recovery_boost_pct": st.column_config.NumberColumn("恢復加速 %", step=0.5),
            "business_decay_pct": st.column_config.NumberColumn("妥妥租衰退覆蓋 %", step=0.5),
            "inflation_pct": st.column_config.NumberColumn("通膨 %", step=0.1),
            "education_multiplier": st.column_config.NumberColumn("教育費倍率", step=0.05),
        },
    )
    scenario_name = st.selectbox("選擇要展開的情境", scenarios_edited["scenario_name"].tolist(), index=1)
    selected_scenario = get_scenario_row(pd.DataFrame(scenarios_edited), scenario_name)
    mode_override = st.selectbox("模擬模式覆蓋", ["跟隨情境", "fixed", "monte_carlo", "path"], index=0)
    mode_override_value = None if mode_override == "跟隨情境" else mode_override

    current_result = run_portfolio(current_norm, selected_scenario, seed=42, mode_override=mode_override_value)
    rec_result = run_portfolio(rec_norm, selected_scenario, seed=42, mode_override=mode_override_value)
    custom_result = run_portfolio(custom_norm, selected_scenario, seed=42, mode_override=mode_override_value)
    current_summary = summarize_simulation(current_result, start_assets_twd)
    rec_summary = summarize_simulation(rec_result, start_assets_twd)
    custom_summary = summarize_simulation(custom_result, start_assets_twd)

    cols = st.columns(6)
    cols[0].metric("Current 終值", fmt_number(current_summary.get("最終資產終值"), 0))
    cols[1].metric("Recommended 終值", fmt_number(rec_summary.get("最終資產終值"), 0))
    cols[2].metric("Custom 終值", fmt_number(custom_summary.get("最終資產終值"), 0))
    cols[3].metric("Current 最大回撤", fmt_pct(current_summary.get("最大回撤 %"), 1))
    cols[4].metric("Recommended 最大回撤", fmt_pct(rec_summary.get("最大回撤 %"), 1))
    cols[5].metric("推薦方案人生適配", fmt_number(rec_summary.get("人生適配分數"), 1))

    chart_df = pd.concat([
        current_result.assign(portfolio="Current"),
        rec_result.assign(portfolio="Recommended"),
        custom_result.assign(portfolio="Custom"),
    ], ignore_index=True)

    deep_tabs = st.tabs(["資產曲線", "現金流", "收入拆解", "回撤路徑", "年度明細"])
    with deep_tabs[0]:
        st.plotly_chart(px.line(chart_df, x="calendar_year", y="end_assets_twd", color="portfolio", markers=True, title="年度資產曲線"), use_container_width=True)
    with deep_tabs[1]:
        st.plotly_chart(px.bar(chart_df, x="calendar_year", y="net_cashflow_twd", color="portfolio", barmode="group", title="年度現金流"), use_container_width=True)
    with deep_tabs[2]:
        inc_long = chart_df.melt(id_vars=["calendar_year", "portfolio"], value_vars=["salary_income_twd", "tuotuozu_income_twd", "inheritance_income_twd"], var_name="income_type", value_name="amount_twd")
        st.plotly_chart(px.bar(inc_long, x="calendar_year", y="amount_twd", color="income_type", facet_row="portfolio", title="收入拆解：本業 vs 妥妥租 vs 繼承"), use_container_width=True)
    with deep_tabs[3]:
        st.plotly_chart(px.line(chart_df, x="calendar_year", y="drawdown_pct", color="portfolio", markers=True, title="回撤路徑"), use_container_width=True)
    with deep_tabs[4]:
        st.dataframe(chart_df, use_container_width=True, hide_index=True)

with main_tabs[3]:
    st.markdown("## 情境矩陣 / Monte Carlo")
    st.dataframe(scenario_matrix_df, use_container_width=True, hide_index=True)
    sims_df, mc_metrics = run_monte_carlo_compare(
        current_norm, rec_norm, selected_scenario, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR,
        salary_annual, salary_growth_pct, retirement_age,
        "excel_projection" if tuotuozu_mode == "Excel 預測" else "manual_decay",
        tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode,
        living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual,
        mortgage_annual, inheritance_age, inherited_rent_monthly,
        withdrawal_strategy, rebalance_frequency_years, mode_override_value, monte_carlo_sims, 42,
    )
    mc_cols = st.columns(6)
    mc_cols[0].metric("Recommended 勝過 Current 機率", fmt_pct(mc_metrics.get("B 方案勝過 A 方案的機率"), 1))
    mc_cols[1].metric("Current 勝過 Recommended 機率", fmt_pct(mc_metrics.get("A 方案勝過 B 方案的機率"), 1))
    mc_cols[2].metric("Current 耗盡機率", fmt_pct(mc_metrics.get("Current 資產耗盡機率"), 1))
    mc_cols[3].metric("Recommended 耗盡機率", fmt_pct(mc_metrics.get("Recommended 資產耗盡機率"), 1))
    mc_cols[4].metric("Current 中位數終值", fmt_number(mc_metrics.get("Current 終值中位數"), 0))
    mc_cols[5].metric("Recommended 中位數終值", fmt_number(mc_metrics.get("Recommended 終值中位數"), 0))
    sims_long = sims_df.melt(value_vars=["current_final_assets_twd", "recommended_final_assets_twd"], var_name="portfolio", value_name="final_assets_twd")
    st.plotly_chart(px.histogram(sims_long, x="final_assets_twd", color="portfolio", barmode="overlay", nbins=40, title="Monte Carlo 終值分布"), use_container_width=True)
    st.dataframe(sims_df, use_container_width=True, hide_index=True)

with main_tabs[4]:
    st.markdown("## 驗證 / 除錯")
    # use selected scenario from single-scenario logic if available, else base
    validation_checks = []
    validation_checks.append(validate_weight_sum(pd.DataFrame(current_edited), "Current"))
    validation_checks.append(validate_weight_sum(pd.DataFrame(recommended_edited), "Recommended"))
    validation_checks.append(validate_weight_sum(pd.DataFrame(custom_edited), "Custom"))
    validation_checks.append(validate_year_sequence(rec_result, "Recommended"))
    validation_checks.append(validate_cashflow_formula(rec_result, "Recommended"))
    validation_checks.extend(validate_income_engine(
        rec_result, salary_annual, salary_growth_pct, retirement_age,
        "excel_projection" if tuotuozu_mode == "Excel 預測" else "manual_decay",
        tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode,
        selected_scenario,
    ))
    validation_checks.append(validate_education_peak(rec_result))
    validation_checks.append(validate_ruin_logic(rec_result, "Recommended"))
    validation_checks.append(validate_start_assets_equal(current_result, rec_result))
    alt_scenario = get_scenario_row(scenarios_default, scenarios_default["scenario_name"].iloc[0])
    alt_summary = summarize_simulation(run_portfolio(rec_norm, alt_scenario, 42), start_assets_twd)
    validation_checks.append(validate_scenario_switch(rec_summary, alt_summary, alt_scenario["scenario_name"]))
    validation_checks.append({"check": "手動修改後是否同步更新", "status": "PASS", "detail": "本工具每次互動都重新 rerun，圖表與結果直接綁定目前 editor 狀態。"})

    validation_df = pd.DataFrame(validation_checks)
    st.dataframe(validation_df, use_container_width=True, hide_index=True)
    st.markdown("### 最敏感 5 參數")
    sensitivity_df = sensitivity_scan(
        rec_norm, selected_scenario, simulation_years, start_assets_twd, CURRENT_AGE, CURRENT_YEAR,
        salary_annual, salary_growth_pct, retirement_age,
        "excel_projection" if tuotuozu_mode == "Excel 預測" else "manual_decay",
        tuotuozu_base_annual, tuotuozu_decay_pct, biz_projection_list, tuotuozu_fallback_mode,
        living_expense_annual, inflation_pct, edu_phase1_annual, edu_phase2_annual,
        mortgage_annual, inheritance_age, inherited_rent_monthly,
        withdrawal_strategy, rebalance_frequency_years, mode_override_value, 42,
    ).head(5)
    st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

    uncertainties = []
    if tuotuozu_mode == "Excel 預測" and len(biz_projection_list) == 0:
        uncertainties.append({"不確定點": "妥妥租 Excel 沒有成功讀到資料", "原因": "上傳檔案或工作表欄位不符", "影響": "高，妥妥租收入路徑會失真", "需補資料": "正確的 妥妥租_預測.xlsx 或確認工作表欄位"})
    if tuotuozu_mode == "Excel 預測" and len(biz_projection_list) < simulation_years:
        uncertainties.append({"不確定點": "Excel 預測年數少於模擬年數", "原因": "表格只有前幾年預測", "影響": "中，後段收入取決於 fallback 規則", "需補資料": "更長的預測表或確認 fallback 假設"})
    uncertainties.append({"不確定點": "教育費目前仍採年度分段，不是依孩子實際年齡細分", "原因": "本版先優先整合收入引擎", "影響": "中", "需補資料": "若要完全沿用舊工具，可再整合孩子年齡版教育模組"})
    st.markdown("### 不確定點與資料缺口")
    st.dataframe(pd.DataFrame(uncertainties), use_container_width=True, hide_index=True)

with main_tabs[5]:
    st.markdown("## 原始資料與下載")
    raw_tabs = st.tabs(["Current 原始持股", "Current 正規化後", "Recommended 正規化後", "Custom 正規化後", "妥妥租 Excel 預覽"])
    with raw_tabs[0]:
        st.dataframe(current_positions_raw, use_container_width=True, hide_index=True)
    with raw_tabs[1]:
        st.dataframe(current_norm, use_container_width=True, hide_index=True)
        st.download_button("下載 Current Portfolio CSV", prepare_download_bytes(current_norm), "current_portfolio.csv", "text/csv")
    with raw_tabs[2]:
        st.dataframe(rec_norm, use_container_width=True, hide_index=True)
        st.download_button("下載 Recommended Portfolio CSV", prepare_download_bytes(rec_norm), "recommended_portfolio.csv", "text/csv")
    with raw_tabs[3]:
        st.dataframe(custom_norm, use_container_width=True, hide_index=True)
        st.download_button("下載 Custom Portfolio CSV", prepare_download_bytes(custom_norm), "custom_portfolio.csv", "text/csv")
    with raw_tabs[4]:
        st.dataframe(biz_projection_preview if not biz_projection_preview.empty else pd.DataFrame(columns=["year", "net_profit_twd"]), use_container_width=True, hide_index=True)
        if not biz_projection_preview.empty:
            st.download_button("下載 妥妥租 Excel 預覽 CSV", prepare_download_bytes(biz_projection_preview), "tuotuozu_preview.csv", "text/csv")
