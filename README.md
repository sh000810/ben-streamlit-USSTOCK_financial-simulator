# Ben 財務與投資模擬器 v7

這版加入：

- 全系統數字格式統一：KPI 用 K / M / B，表格用千分位
- 假設透明化（Assumption Audit）
- 每檔股票顯示：歷史值 / 模型值 / 方法 / 信心 / 補充說明
- 一鍵下載完整 LOG（JSON）
- 可直接複製的診斷摘要
- 驗證 / 除錯頁補上最小檢查集，不再空白
- 透明化原則：若沒有真實 10Y / 5Y / 3Y 價格資料，系統會明確標示 unavailable / bucket default，不假裝精準

## 啟動
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
