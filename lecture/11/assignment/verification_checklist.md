# Assignment #6 驗證紀錄

驗證日期：2026-10-29

## 題目 PDF 逐頁對應

`Assignment_6.pdf` 實際頁數 **6 頁**（投影片 p.40–45），每頁都有對應的驗收項目：

| PDF 頁 | 投影片 | 題目內容 | 實際結果 |
| --- | --- | --- | --- |
| 1 | p.40 | 目的：基準一年後糖尿病疾病進展之預測器；臨床醫學定量評估 | 報告第一節「目的」照述 |
| 2 | p.41 | 資料集 `load_diabetes()`，442 筆、10 維、target 25–346、特徵已預先縮放 | 442×10；target `min=25.0, max=346.0`；各特徵平方和 = 1（測試驗證） |
| 3 | p.42 | 三種演算法；切分 80/20；MSE 與 R²；線性 baseline、PolynomialFeatures degree=2、PyTorch MLP | 353/89 筆；三模型皆輸出 MSE 與 R²；degree=2 展開 65 維；MLP 為 `nn.Sequential` |
| 4 | p.43 | 預期結果 1：三種模型測試集量化指標比較表；測試集約 89 筆；最佳 R² ≥ 0.46 | 89 筆；最佳 **R² 0.5033**（PyTorch MLP Regressor） |
| 5 | p.44 | 預期結果 2：長條圖，標題註明測試集，長條上方標四位小數 R² | `assignment6_model_comparison.png`，標題 `Comparison of Regression Models (Test Set)`，標註 0.4526 / 0.4037 / 0.5033 |
| 6 | p.45 | 最佳 R² 通常落在 0.45 ~ 0.55；各模型種類的 R² 參考區間 | 最佳 0.5033 落在區間內；多項式 0.4037 落在 p.45 所列的 0.30~0.42；MLP 落在 0.46~0.53 |

## 實際數值

程式輸出（`11_assignment6.py`）：

```
========== 各模型效能比較 ==========
[Linear Regression] MSE: 2900.17 | R² Score: 0.4526
[Polynomial Regression] MSE: 3159.34 | R² Score: 0.4037
[PyTorch MLP Regressor] MSE: 2631.53 | R² Score: 0.5033
```

| 項目 | 數值 |
| --- | --- |
| 訓練 / 測試筆數 | 353 / **89** |
| 最佳模型 | PyTorch MLP Regressor |
| 最佳 R² | **0.5033**（門檻 0.46） |
| MLP 架構 | 10 → 32 → 16 → 1，ReLU，1,105 個參數 |
| MLP 最佳 epoch | 158（上限 2000，patience 100） |
| 多項式展開維度 | 65（10 原始 + 10 平方 + 45 交互作用） |

**與投影片數值的比對**：投影片 p.43 的 Linear Regression 為 MSE 2900.19 / R² 0.4526，
本次實測 2900.17 / 0.4526 —— R² 完全相同，MSE 差 0.02（四捨五入與 BLAS 誤差），
確認 `test_size=0.2, random_state=42` 的切分設定與投影片一致。

多項式回歸本次為 3159.34 / 0.4037，投影片為 3096.03 / 0.4156，差異來自特徵縮放步驟的
擺放位置；兩者都落在投影片 p.45 所列的 0.30 ~ 0.42 區間內，量級相同。

## 交叉驗證與測試集結論不一致（已在報告第九節據實說明）

| 模型 | CV R²（353 筆訓練集，5-fold） | 測試集 R²（89 筆） |
| --- | --- | --- |
| 多元線性回歸 | 0.4804 ± 0.0409 | 0.4526 |
| PyTorch MLP (32,16) | 0.4361 ± 0.0304 | 0.5033 |

交叉驗證認為線性回歸較好，測試集認為 MLP 較好，結論相反。測試集僅 89 筆，
CV 標準差本身就有 ±0.04，兩者約 5 個百分點的差距落在雜訊量級內。
報告第九節說明了為何不依測試集分數回頭更換模型（那等同於把測試集當驗證集使用）。

## 逐項驗證

- [x] **題目頁數比對**：重新以 `fitz` 取得頁數 = 6，與上表記載一致；6 頁全部對應到驗收項目。
- [x] **測試套件**：`cd lecture/11/assignment && python -m unittest test_11_assignment6`
      → **Ran 25 tests, OK**（12.0 秒）
- [x] **突變檢測**：見下方「突變檢測結果」。
- [x] **繳交 zip 乾淨目錄執行**：解壓到 scratchpad 的空目錄後直接
      `python 11_assignment6.py`，重現完全相同的三行比較表與長條圖，
      不依賴任何未繳交的檔案。
- [x] **報告 PDF 逐頁渲染檢查**：11 頁全部以 Read 渲染確認。
      修正了兩處：第 3 頁 R² 公式（兩層連加的分式）壓到下一段文字，`space` 由 0.082 調到 0.108；
      第 5 頁 mono 區塊的「各模型效能比較」因 DejaVuSansMono 無中文字符而顯示為空白
      （`report_style.mono()` 已修，詳見 A5 的驗證紀錄）。
- [x] **`report_style` 測試**：`cd lecture && python -m unittest test_report_style`
      → **Ran 20 tests, OK**
- [x] **報告字型比對**：
      `['Arial-BoldMT', 'ArialMT', 'ArialUnicodeMS', 'Cmb10', 'Cmex10', 'Cmmi10', 'Cmr10', 'Cmsy10', 'DejaVuSansMono']`
      —— 與 Assignment #3 / #5 相同，無任何 `Heiti`。
      第一版曾出現 `STIXGeneral-Regular`，追查後確認是第 3 頁多項式公式的 `\dots` 省略號
      在 Computer Modern 裡沒有對應 glyph 而 fallback；改用 `...` 後即走 `Cmr10`，字型集合與其他報告一致。
- [x] **繳交 zip UTF-8 旗標**：以 `scripts/make_submission.py` 打包，
      `515661055_陳欣怡_Assignment6_結果報告.pdf` 的 UTF-8 旗標為「是」。

## 本次修掉的實作問題

**`build_mlp` 的預設參數在定義時就綁定了 `HIDDEN_SIZES`**，導致改善空間研究裡四種
隱藏層容量跑出完全相同的數字（0.4361 ± 0.0304）。原本 `11_assignment6_explore.py`
是透過改寫模組層級的 `HIDDEN_SIZES` 來切換容量，但預設值早已在函式定義時求值，
改寫根本沒有生效。

修法：`build_mlp(n_features, hidden_sizes=None)`，在函式內才讀模組變數；
`train_mlp` 增加 `hidden_sizes` 參數並傳遞下去；explore 改為直接傳參數，不再 monkeypatch。
新增兩項測試鎖住（`test_build_mlp_respects_the_requested_hidden_sizes`、
`test_train_mlp_passes_the_requested_hidden_sizes_through`）。

**Ridge 在本環境回傳 nan**：sklearn 1.0.2 的 `_solve_cholesky` 呼叫
`scipy.linalg.solve(..., sym_pos=True)`，而 scipy 1.13 已移除該參數，fit 直接拋
`TypeError`，`cross_val_score` 把它吞成 nan。這是環境版本不相容，不是實作問題。
改善研究改用 `Ridge(solver="svd")` 繞過 cholesky 路徑即正常。
（繳交的 `11_assignment6.py` 不使用 Ridge，不受影響。）

## 突變檢測結果

故意改壞關鍵常數與邏輯，確認每個突變至少讓一個測試失敗（腳本跑完會自動還原原始檔案）：

| 突變 | 捕捉它的測試 | 結果 |
| --- | --- | --- |
| `Y_LIMITS` 改成 (0.0, 1.0) | `BarChartStyleTests` | 測試失敗 ✅ |
| 長條圖標註改成 2 位小數 | `BarChartStyleTests` | 測試失敗 ✅ |
| `PolynomialFeatures` degree 2 → 3 | `PolynomialTests` | 測試失敗 ✅ |
| 拿掉提早停止的 `break` | `MLPTests` | **第一次存活**，修正測試後捕捉 ✅ |
| `train_mlp` 不把 `hidden_sizes` 傳下去 | `MLPTests` | 測試失敗 ✅ |
| 比較表標題改成 `===== ... =====` | `ComparisonTableTests` | 測試失敗 ✅ |

### 一個存活的突變（已修正測試）

`test_mlp_stops_early_instead_of_running_every_epoch` 原本的斷言是：

```python
self.assertLess(history["best_epoch"], module.MAX_EPOCHS)
```

把 `break` 改成 `pass` 之後，迴圈會跑滿 2000 輪，但「最佳輪次」仍然是第 158 輪，
所以 `best_epoch < MAX_EPOCHS` 依然成立，測試照樣通過——這個斷言根本驗不到提早停止。

改成檢查迴圈真的中斷了：

```python
self.assertLess(len(history["valid_losses"]), module.MAX_EPOCHS)
self.assertEqual(len(history["valid_losses"]), history["best_epoch"] + module.PATIENCE)
```

重跑同一個突變 → `AssertionError: 2000 not less than 2000`，確認現在抓得到。
修正後 11 個突變**全數被捕捉**。

## 繳交內容

`515661055_陳欣怡_Assignment6.zip`（0.43 MB）：

| 檔案 | 大小 | UTF-8 旗標 |
| --- | --- | --- |
| `11_assignment6.py` | 10,221 | 否（純 ASCII 檔名） |
| `515661055_陳欣怡_Assignment6_結果報告.pdf` | 469,409 | 是 |
