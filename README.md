# Ben 財務與投資模擬器（Streamlit）

## 啟動方式
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 這版有什麼
- Current Portfolio：讀取真實 CSV，做欄位映射、分類、權重化
- Recommended Portfolio：預設建議配置，可手動修改
- 情境模擬：fixed / monte_carlo / path
- 現金流模擬：本業衰退、生活費通膨、教育費 bulge、房貸、遺產租金
- 資產存續：現金不足時依規則提領
- 風險分析：AI 直接曝險、ETF look-through 重疊、集中度、同跌風險
- Monte Carlo：A/B 勝率、資產耗盡機率、終值中位數

## 重要提醒
- ETF look-through 目前為 QQQ / VOO 重點持股近似版，不是完整全成分精算。
- 模型為透明版本，適合調參與決策，不是投資保證。
- 起始資產可用你的持股折算 TWD，或手動覆寫成你的真實流動資產。

## 建議的下一步
1. 把更多帳戶的持股 CSV 合併
2. 把真實台股 / 現金 / 債券一起納入
3. 若要更完整的勝率分析，可再接歷史報酬分布與完整 ETF 成分表
