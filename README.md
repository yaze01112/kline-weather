# K線氣象台

用 [Kronos](https://github.com/shiyu-coder/Kronos)（Kronos-small）每天預報 `stocks.json` 裡的台股未來 20 個交易日，網頁像天氣預報一樣顯示上漲機率、80% 區間和過去 6 次預報的命中紀錄。

- `predict.py`：抓日 K（FinMind，備援 Yahoo）→ 跑 32 條模擬路徑 → 滾動回測 → 寫出 `docs/forecast.json`
- `.github/workflows/forecast.yml`：週一到週五台灣時間 16:17 自動執行，也可以在 Actions 頁手動按 Run workflow
- `docs/index.html`：GitHub Pages 網頁，讀 `forecast.json`

要改追蹤的股票，編輯 `stocks.json`。設定 repo secret `FINMIND_TOKEN` 可以提高 FinMind 額度（不設也能跑）。

預報只是模型輸出，不是投資建議。實測 Kronos 在日線上對大漲過的股票偏空，回看期因此用 120 日。
