Ben 財務與投資模擬器 v7.4 修補重點

1. Excel 預測讀不到
- requirements.txt 新增 openpyxl
- app.py 會明確顯示：
  - 內建 Excel 路徑
  - 檔案是否存在
  - openpyxl 是否可用
  - 目前讀到幾年資料
- 若是環境缺 openpyxl，畫面會直接提示，而不是只說「讀不到 Excel」

2. 假設透明化 / LOG 讀很久
- 主因不是 LOG 表格本身，而是 Streamlit tabs 不是 lazy load。
- v7.3 的 main_tabs[3] 仍會在每次 rerun 時先跑完整的「情境矩陣 / Monte Carlo」。
- 這會拖慢整個 app，包括你切到 5 / 6 頁。
- v7.4 新增：
  - 左側 toggle：預載重型分析（情境矩陣 / Monte Carlo）
  - 預設關閉
  - 關掉時，tab 4 不會自動跑重型分析，5 / 6 頁會快很多

3. 這版檔案
- app.py：v7.4
- core.py：沿用 v7.3
- requirements.txt：加入 openpyxl

部署後建議
1) 重新部署
2) 先確認 Excel 預測訊息是否變得明確
3) 預載重型分析先保持關閉
4) 再測 5 / 6 頁速度
