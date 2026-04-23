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
    build_comparison,
    build_recommended_portfolio,
    bucket_exposure,
    compute_etf_overlap,
    compute_risk_duplicate_metrics,
    default_control_values,
    load_positions,
    metric_table,
    normalize_weights,
    run_monte_carlo_compare,
    scenario_table,
    get_scenario_row,
    simulate_portfolio,
    summarize_simulation,
)

st.set_page_config(page_title="Ben 財務與投資模擬器", layout="wide")
st.title("Ben 財務與投資模擬器 · Streamlit 版本")
st.caption("可計算、可驗算、可修改：Current Portfolio vs Recommended Portfolio，多情境 + 現金流 + 資產存續模擬")

sample_path = Path(__file__).parent / "data" / "Individual-Positions-2026-04-22-184253(1).csv"

with st.sidebar:
    st.header("資料與核心參數")
    source_mode = st.radio("持股來源", ["使用內建範例 CSV", "上傳新的 CSV"], index=0)
    uploaded_file = st.file_uploader("上傳持股 CSV", type=["csv"], accept_multiple_files=False)
    fx_rate = st.number_input("USD/TWD 匯率", min_value=20.0, max_value=50.0, value=32.0, step=0.1)

    source = sample_path
    if source_mode == "上傳新的 CSV" and uploaded_file is not None:
        source = uploaded_file

    current_positions_raw, missing_fields = load_positions(source)
    uploaded_total_usd = float(current_positions_raw["market_value_usd"].sum()) if not current_positions_raw.empty else None
    defaults = default_control_values(uploaded_total_usd, fx_rate)

    start_assets_twd = st.number_input("起始流動資產 (TWD)", min_value=0.0, value=float(defaults["start_assets_twd"]), step=100000.0)
    simulation_years = st.slider("模擬年數", 5, 30, int(defaults["simulation_years"]))
    rebalance_frequency_years = st.selectbox("再平衡頻率", options=[0, 1, 2, 3, 5], format_func=lambda x: "不再平衡" if x == 0 else f"每 {x} 年")
    withdrawal_strategy = st.selectbox("現金不足時提領方式", ["比例賣出", "先賣現金 / ETF", "先賣波動低資產"])
    monte_carlo_sims = st.slider("Monte Carlo 次數", 50, 1000, 300, step=50)

    st.header("人生 / 現金流參數")
    business_profit_annual = st.number_input("本業年度淨利", min_value=0.0, value=float(defaults["business_profit_annual"]), step=100000.0)
    business_decay_pct = st.number_input("本業衰退率 %", min_value=0.0, max_value=50.0, value=float(defaults["business_decay_pct"]), step=0.5)
    living_expense_annual = st.number_input("基礎生活費 / 年", min_value=0.0, value=float(defaults["living_expense_annual"]), step=50000.0)
    inflation_pct = st.number_input("通膨率 %", min_value=0.0, max_value=20.0, value=float(defaults["inflation_pct"]), step=0.1)
    edu_phase1_annual = st.number_input("教育費 2026-2033 / 年", min_value=0.0, value=float(defaults["edu_phase1_annual"]), step=50000.0)
    edu_phase2_annual = st.number_input("教育費 2034-2038 / 年", min_value=0.0, value=float(defaults["edu_phase2_annual"]), step=50000.0)
    mortgage_annual = st.number_input("房貸 / 居住成本 / 年", min_value=0.0, value=float(defaults["mortgage_annual"]), step=50000.0)
    inheritance_age = st.number_input("遺產事件年齡", min_value=45, max_value=90, value=int(defaults["inheritance_age"]), step=1)
    inherited_rent_monthly = st.number_input("遺產後租金 / 月", min_value=0.0, value=float(defaults["inherited_rent_monthly"]), step=5000.0)

st.subheader("資料讀取狀態")
status_cols = st.columns(4)
status_cols[0].metric("Current 持股筆數", f"{current_positions_raw.shape[0]}")
status_cols[1].metric("Current 市值 (USD)", f"{uploaded_total_usd:,.0f}" if uploaded_total_usd else "0")
status_cols[2].metric("Current 市值折 TWD", f"{(uploaded_total_usd or 0) * fx_rate:,.0f}")
status_cols[3].metric("建模起始資產", f"{start_assets_twd:,.0f}")
if missing_fields:
    st.warning(f"CSV 缺少欄位：{', '.join(missing_fields)}。已用替代邏輯補足，但建議檢查。")
else:
    st.success("CSV 欄位映射完整，可直接作為 Current Portfolio 基礎來源。")

current_portfolio_default = current_positions_raw[
    [
        "ticker",
        "name",
        "weight_pct",
        "asset_class",
        "theme",
        "role_bucket",
        "risk_group",
        "ai_direct",
        "expected_return_pct",
        "volatility_pct",
        "sell_priority",
        "include",
    ]
].copy()
recommended_portfolio_default = build_recommended_portfolio()

comparison_seed = build_comparison(current_portfolio_default, recommended_portfolio_default)

base_tabs = st.tabs([
    "1. 作品導覽 / 驗算說明",
    "2. 持股與參數編輯",
    "3. 重複風險分析",
    "4. 情境模擬",
    "5. Monte Carlo / 勝率",
    "6. 原始資料",
])

with base_tabs[0]:
    st.markdown(
        """
### 這個工具怎麼算
1. **持股端**：每檔持股都有 `權重 / 預期年化報酬 / 波動率 / 風險群組 / 是否 AI 直接曝險`。
2. **情境端**：每個情境都可以改 `市場位移 / AI 超額報酬 / 波動倍數 / 前幾年負報酬 / 復原期 / 本業衰退 / 通膨 / 教育費倍率`。
3. **現金流端**：每年計算 `本業收入 + 遺產租金 - 生活費 - 教育費 - 房貸`。
4. **資產存續端**：若現金流為負，依照你指定的賣出規則提領。
5. **再平衡端**：按你設定的頻率，把資產拉回目標權重。
6. **勝率端**：Monte Carlo 會用相同隨機種子比較 A/B，避免兩邊不是在同一個世界裡打球。

### 人生適配分數（透明版）
- 35%：沒有提早耗盡資產
- 20%：最大回撤是否過深
- 20%：最終資產相對起始資產的成長
- 15%：最低現金緩衝月數
- 10%：現金流赤字年份比例

> 這不是神諭，是一個透明可改的風險偏好分數。你可以把它當儀表板，不要把它當上帝。
        """
    )

    st.markdown("### Recommended Portfolio 設計理念")
    st.markdown(
        """
- **核心底盤**：Cash / VOO / BRK.B，讓組合不是全靠科技情緒活著。
- **AI 核心**：NVDA / MSFT / AMZN / GOOG，保留主要 AI 商業化與算力路徑。
- **半導體上游 / 網通 / 資料中心**：ASML / AVGO / ANET / EQIX，補足 AI 基礎設施鏈。
- **企業 AI workflow**：NOW / ORCL，補上企業落地與資料層。
- **非 AI 複利資產**：MA / ETN，避免整包都跟同一個敘事一起摔。
        """
    )

with base_tabs[1]:
    st.markdown("## A. 持股與手動調參區")
    editor_tabs = st.tabs(["Current Portfolio", "Recommended Portfolio", "Current vs Recommended 比較"])
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
    with editor_tabs[0]:
        st.caption("這裡已匯入你的真實持股。你可以直接修改權重、年化報酬率、波動率，工具會在模擬前自動正規化權重。")
        current_edited = st.data_editor(
            current_portfolio_default,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config=editor_columns,
            key="current_editor",
        )
        st.info(f"Current 權重合計：{current_edited.loc[current_edited['include'], 'weight_pct'].sum():.2f}%（模擬時會自動正規化）")
    with editor_tabs[1]:
        st.caption("這裡是預設 Recommended Portfolio。你可以直接改成自己的版本。")
        recommended_edited = st.data_editor(
            recommended_portfolio_default,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config=editor_columns,
            key="recommended_editor",
        )
        st.info(f"Recommended 權重合計：{recommended_edited.loc[recommended_edited['include'], 'weight_pct'].sum():.2f}%（模擬時會自動正規化）")
    with editor_tabs[2]:
        comparison_live = build_comparison(current_edited, recommended_edited)
        st.dataframe(comparison_live, use_container_width=True, hide_index=True)
        st.caption("KEEP / ADD / TRIM / EXIT 是依兩邊權重差異自動判定。")

with base_tabs[2]:
    st.markdown("## B. 風險重複度分析")
    current_norm = normalize_weights(pd.DataFrame(current_edited))
    rec_norm = normalize_weights(pd.DataFrame(recommended_edited))

    risk_cols = st.columns(2)
    current_metrics = compute_risk_duplicate_metrics(current_norm)
    rec_metrics = compute_risk_duplicate_metrics(rec_norm)
    risk_metric_df = pd.DataFrame(
        {
            "metric": list(current_metrics.keys()),
            "Current": list(current_metrics.values()),
            "Recommended": [rec_metrics.get(k, np.nan) for k in current_metrics.keys()],
        }
    )
    risk_cols[0].dataframe(risk_metric_df, hide_index=True, use_container_width=True)

    group_tabs = st.tabs(["類別分布", "角色分布", "ETF 重疊", "單一持股過重"])
    with group_tabs[0]:
        c_bucket = bucket_exposure(current_norm, "asset_class").rename(columns={"weight_pct": "Current"})
        r_bucket = bucket_exposure(rec_norm, "asset_class").rename(columns={"weight_pct": "Recommended"})
        merged = pd.merge(c_bucket, r_bucket, on="asset_class", how="outer").fillna(0)
        fig = go.Figure()
        fig.add_bar(name="Current", x=merged["asset_class"], y=merged["Current"])
        fig.add_bar(name="Recommended", x=merged["asset_class"], y=merged["Recommended"])
        fig.update_layout(barmode="group", xaxis_title="資產類別", yaxis_title="權重 %")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(merged, hide_index=True, use_container_width=True)
    with group_tabs[1]:
        c_role = bucket_exposure(current_norm, "role_bucket").rename(columns={"weight_pct": "Current"})
        r_role = bucket_exposure(rec_norm, "role_bucket").rename(columns={"weight_pct": "Recommended"})
        merged_role = pd.merge(c_role, r_role, on="role_bucket", how="outer").fillna(0)
        fig = go.Figure()
        fig.add_bar(name="Current", x=merged_role["role_bucket"], y=merged_role["Current"])
        fig.add_bar(name="Recommended", x=merged_role["role_bucket"], y=merged_role["Recommended"])
        fig.update_layout(barmode="group", xaxis_title="角色", yaxis_title="權重 %")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(merged_role, hide_index=True, use_container_width=True)
    with group_tabs[2]:
        current_overlap = compute_etf_overlap(current_norm)
        rec_overlap = compute_etf_overlap(rec_norm)
        st.markdown("### Current ETF look-through")
        if current_overlap.empty:
            st.info("Current Portfolio 沒有需要展開的 QQQ / VOO 重疊持股。")
        else:
            st.dataframe(current_overlap, hide_index=True, use_container_width=True)
        st.markdown("### Recommended ETF look-through")
        if rec_overlap.empty:
            st.info("Recommended Portfolio 沒有需要展開的 QQQ / VOO 重疊持股。")
        else:
            st.dataframe(rec_overlap, hide_index=True, use_container_width=True)
        st.caption("這裡用的是 QQQ / VOO 的重點持股近似 look-through，方向正確，但不是完整全成分精算。")
    with group_tabs[3]:
        overweight_df = pd.merge(
            current_norm[["ticker", "weight_pct"]].rename(columns={"weight_pct": "Current"}),
            rec_norm[["ticker", "weight_pct"]].rename(columns={"weight_pct": "Recommended"}),
            on="ticker",
            how="outer",
        ).fillna(0)
        overweight_df["MaxWeight"] = overweight_df[["Current", "Recommended"]].max(axis=1)
        overweight_df = overweight_df.sort_values("MaxWeight", ascending=False)
        st.dataframe(overweight_df, hide_index=True, use_container_width=True)
        st.caption("看最大權重與主題集中度時，最容易抓出『其實買了很多，結果都在賭同一件事』的情況。")

with base_tabs[3]:
    st.markdown("## C. 情境模擬")
    scenarios_default = scenario_table()
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
            "business_decay_pct": st.column_config.NumberColumn("本業衰退 %", step=0.5),
            "inflation_pct": st.column_config.NumberColumn("通膨 %", step=0.1),
            "education_multiplier": st.column_config.NumberColumn("教育費倍率", step=0.05),
            "max_drawdown_hint_pct": st.column_config.NumberColumn("情境提示最大回撤 %", step=1.0),
        },
    )
    scenario_name = st.selectbox("選擇要展開的情境", scenarios_edited["scenario_name"].tolist(), index=1)
    selected_scenario = get_scenario_row(pd.DataFrame(scenarios_edited), scenario_name)
    mode_override = st.selectbox("模擬模式（可覆蓋情境內建模式）", ["跟隨情境", "fixed", "monte_carlo", "path"], index=0)
    mode_override_value = None if mode_override == "跟隨情境" else mode_override

    current_result = simulate_portfolio(
        pd.DataFrame(current_norm),
        selected_scenario,
        simulation_years,
        start_assets_twd,
        CURRENT_AGE,
        CURRENT_YEAR,
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
        mode_override_value,
        seed=42,
    )
    rec_result = simulate_portfolio(
        pd.DataFrame(rec_norm),
        selected_scenario,
        simulation_years,
        start_assets_twd,
        CURRENT_AGE,
        CURRENT_YEAR,
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
        mode_override_value,
        seed=42,
    )

    current_summary = summarize_simulation(current_result, start_assets_twd)
    rec_summary = summarize_simulation(rec_result, start_assets_twd)

    summary_cols = st.columns(4)
    summary_cols[0].metric("Current 最終資產", f"{current_summary.get('最終資產終值', 0):,.0f}")
    summary_cols[1].metric("Recommended 最終資產", f"{rec_summary.get('最終資產終值', 0):,.0f}")
    summary_cols[2].metric("Current 最大回撤", f"{current_summary.get('最大回撤 %', 0):.1f}%")
    summary_cols[3].metric("Recommended 最大回撤", f"{rec_summary.get('最大回撤 %', 0):.1f}%")

    summary_cols2 = st.columns(4)
    summary_cols2[0].metric("Current 10 年淨資產", f"{(current_summary.get('10 年後淨資產') or 0):,.0f}")
    summary_cols2[1].metric("Recommended 10 年淨資產", f"{(rec_summary.get('10 年後淨資產') or 0):,.0f}")
    summary_cols2[2].metric("Current 人生適配分數", f"{current_summary.get('人生適配分數', 0):.1f}")
    summary_cols2[3].metric("Recommended 人生適配分數", f"{rec_summary.get('人生適配分數', 0):.1f}")

    chart_df = pd.concat(
        [
            current_result.assign(portfolio="Current"),
            rec_result.assign(portfolio="Recommended"),
        ],
        ignore_index=True,
    )

    line_tabs = st.tabs(["年度資產曲線", "年度現金流", "最大回撤路徑", "年度明細表", "六情境總表"])
    with line_tabs[0]:
        fig = px.line(chart_df, x="calendar_year", y="end_assets_twd", color="portfolio", markers=True, title="年度資產曲線")
        st.plotly_chart(fig, use_container_width=True)
    with line_tabs[1]:
        fig = px.bar(chart_df, x="calendar_year", y="net_cashflow_twd", color="portfolio", barmode="group", title="年度現金流")
        st.plotly_chart(fig, use_container_width=True)
    with line_tabs[2]:
        fig = px.line(chart_df, x="calendar_year", y="drawdown_pct", color="portfolio", markers=True, title="最大回撤路徑")
        st.plotly_chart(fig, use_container_width=True)
    with line_tabs[3]:
        st.dataframe(chart_df, use_container_width=True, hide_index=True)
    with line_tabs[4]:
        scenario_rows = []
        all_scenarios_df = pd.DataFrame(scenarios_edited)
        for _, srow in all_scenarios_df.iterrows():
            cur = simulate_portfolio(
                pd.DataFrame(current_norm),
                srow,
                simulation_years,
                start_assets_twd,
                CURRENT_AGE,
                CURRENT_YEAR,
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
                None,
                seed=42,
            )
            rec = simulate_portfolio(
                pd.DataFrame(rec_norm),
                srow,
                simulation_years,
                start_assets_twd,
                CURRENT_AGE,
                CURRENT_YEAR,
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
                None,
                seed=42,
            )
            cur_s = summarize_simulation(cur, start_assets_twd)
            rec_s = summarize_simulation(rec, start_assets_twd)
            scenario_rows.extend(
                [
                    {
                        "scenario": srow["scenario_name"],
                        "portfolio": "Current",
                        "5y": float(cur.loc[cur["year_index"] == min(5, simulation_years), "end_assets_twd"].iloc[0]),
                        "10y": float(cur.loc[cur["year_index"] == min(10, simulation_years), "end_assets_twd"].iloc[0]),
                        "15y": float(cur.loc[cur["year_index"] == min(15, simulation_years), "end_assets_twd"].iloc[0]),
                        "20y": float(cur.loc[cur["year_index"] == min(20, simulation_years), "end_assets_twd"].iloc[0]),
                        "max_drawdown_pct": cur_s.get("最大回撤 %", np.nan),
                        "life_fit_score": cur_s.get("人生適配分數", np.nan),
                        "ruin_year": cur_s.get("資產耗盡年份", None),
                    },
                    {
                        "scenario": srow["scenario_name"],
                        "portfolio": "Recommended",
                        "5y": float(rec.loc[rec["year_index"] == min(5, simulation_years), "end_assets_twd"].iloc[0]),
                        "10y": float(rec.loc[rec["year_index"] == min(10, simulation_years), "end_assets_twd"].iloc[0]),
                        "15y": float(rec.loc[rec["year_index"] == min(15, simulation_years), "end_assets_twd"].iloc[0]),
                        "20y": float(rec.loc[rec["year_index"] == min(20, simulation_years), "end_assets_twd"].iloc[0]),
                        "max_drawdown_pct": rec_s.get("最大回撤 %", np.nan),
                        "life_fit_score": rec_s.get("人生適配分數", np.nan),
                        "ruin_year": rec_s.get("資產耗盡年份", None),
                    },
                ]
            )
        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)

with base_tabs[4]:
    st.markdown("## D. Monte Carlo / 勝率與人生適配度")
    st.caption("這裡會用相同亂數世界比較 A / B。若情境本身是 fixed 或 path，結果可能會很接近 deterministic。")
    sims_df, mc_metrics = run_monte_carlo_compare(
        pd.DataFrame(current_norm),
        pd.DataFrame(rec_norm),
        selected_scenario,
        simulation_years,
        start_assets_twd,
        CURRENT_AGE,
        CURRENT_YEAR,
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
        mode_override_value,
        simulations=monte_carlo_sims,
        seed=42,
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("A 方案勝過 B 方案機率", f"{mc_metrics['A 方案勝過 B 方案的機率']:.1f}%")
    metric_cols[1].metric("B 方案勝過 A 方案機率", f"{mc_metrics['B 方案勝過 A 方案的機率']:.1f}%")
    metric_cols[2].metric("Recommended 資產耗盡機率", f"{mc_metrics['Recommended 資產耗盡機率']:.1f}%")

    metric_cols2 = st.columns(3)
    metric_cols2[0].metric("Current 資產耗盡機率", f"{mc_metrics['Current 資產耗盡機率']:.1f}%")
    metric_cols2[1].metric("Current 終值中位數", f"{mc_metrics['Current 終值中位數']:,.0f}")
    metric_cols2[2].metric("Recommended 終值中位數", f"{mc_metrics['Recommended 終值中位數']:,.0f}")

    hist = px.histogram(
        sims_df.melt(value_vars=["current_final_assets_twd", "recommended_final_assets_twd"], var_name="portfolio", value_name="final_assets_twd"),
        x="final_assets_twd",
        color="portfolio",
        barmode="overlay",
        nbins=40,
        title="Monte Carlo 終值分布",
    )
    st.plotly_chart(hist, use_container_width=True)
    st.dataframe(sims_df, use_container_width=True, hide_index=True)

with base_tabs[5]:
    st.markdown("## 原始資料與匯出")
    raw_tabs = st.tabs(["Current 原始持股", "Current 編輯後", "Recommended 編輯後", "計算公式速查"])
    with raw_tabs[0]:
        st.dataframe(current_positions_raw, use_container_width=True, hide_index=True)
    with raw_tabs[1]:
        st.dataframe(pd.DataFrame(current_norm), use_container_width=True, hide_index=True)
        st.download_button(
            "下載 Current Portfolio CSV",
            pd.DataFrame(current_norm).to_csv(index=False).encode("utf-8-sig"),
            file_name="current_portfolio_streamlit.csv",
            mime="text/csv",
        )
    with raw_tabs[2]:
        st.dataframe(pd.DataFrame(rec_norm), use_container_width=True, hide_index=True)
        st.download_button(
            "下載 Recommended Portfolio CSV",
            pd.DataFrame(rec_norm).to_csv(index=False).encode("utf-8-sig"),
            file_name="recommended_portfolio_streamlit.csv",
            mime="text/csv",
        )
    with raw_tabs[3]:
        st.markdown(
            """
### 公式速查
- **持股權重** = 個別市值 / 組合總市值
- **年度投資資產** = 年初投資資產 × (1 + 當年報酬)
- **年度現金流** = 本業收入 + 遺產租金 - 生活費 - 教育費 - 房貸
- **提領**：若年度現金流為負，依賣出規則從現金 / ETF / 其他資產提領
- **最大回撤** = 當前資產 / 歷史高點 - 1
- **情境勝率** = Monte Carlo 中 A 終值 > B 終值 的比例
- **人生適配分數** = 35% 資產存續 + 20% 最大回撤 + 20% 終值成長 + 15% 現金緩衝 + 10% 現金流穩定
            """
        )
