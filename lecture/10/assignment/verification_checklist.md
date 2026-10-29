# Assignment #5 驗證紀錄

驗證日期：2026-10-29

## 題目 PDF 逐頁對應

`Assignment_5.pdf` 實際頁數 **7 頁**（投影片 p.50–56），每頁都有對應的驗收項目：

| PDF 頁 | 投影片 | 題目內容 | 實際結果 |
| --- | --- | --- | --- |
| 1 | p.50 | 與 Assignment #3 相同，特徵選取換成特徵提取 (PCA) | 管線由 `SelectFromModel` 換成 `PCA`，其餘流程不變 |
| 2 | p.51 | 資料集 `load_digits()`，1797 張 8×8、64 維 [0,16]、10 類 | 1797 筆、64 維，測試集 360 筆 |
| 3 | p.52 | LogReg 分類器；切分 80/20；降維後維度 ≤ 40；測試集正確率 ≥ 95%；`joblib.dump()` 儲存管線 | 1437/360 筆；**40 維**；**0.9611**；`digits_pipeline_lr_pca.pkl` |
| 4 | p.53 | 只繳模型評估程式與模型檔，不得含訓練或模型選取程式碼 | 拆成 `10_assignment5_train.py`（不繳）與 `10_assignment5_eval.py`（繳交） |
| 5 | p.54 | 預期結果 1：輸出 PCA 維度與總解釋變異量（維度須 ≤ 40） | **40 維、95.18%**，與投影片範例輸出數值完全相同 |
| 6 | p.55 | 預期結果 2：測試集分類報告 + 視覺化混淆矩陣（accuracy ≥ 0.95） | accuracy 0.96；`assignment5_confusion_matrix.png` |
| 7 | p.56 | 預期結果 3：誤判樣本總數 + 繪出所有誤判樣本，須與混淆矩陣對應 | 「測試集總共誤判了 14 張圖片。」14 張全繪出，與混淆矩陣非對角線總和一致 |

## 實際數值

| 項目 | 數值 |
| --- | --- |
| 最佳超參數 | `pca__n_components=40`、`classifier__C=1.0` |
| 交叉驗證正確率 | 0.9638 ± 0.0068（5-fold StratifiedKFold） |
| 測試集正確率 | **0.9611**（346 / 360） |
| PCA 維度 | **40**（原 64 維） |
| 總解釋變異量 | **95.18%** |
| 誤判樣本數 | **14** 張 |
| 混淆矩陣非對角線總和 | 14（與誤判數一致） |
| 各類 f1-score 最低 | 數字 8，0.90 |
| 模型檔大小 | 26,540 bytes |

各維度的解釋變異量（實測）：10 維 59.23%、20 維 79.59%、30 維 89.55%、40 維 95.18%。

## 逐項驗證

- [x] **題目頁數比對**：重新以 `fitz` 取得頁數 = 7，與上表記載一致；7 頁全部對應到驗收項目。
- [x] **測試套件**：`cd lecture/10/assignment && python -m unittest test_10_assignment5`
      → **Ran 26 tests, OK**（158.8 秒）
- [x] **跨作業回歸**：A3（lecture/07）、A5（lecture/10）、A6（lecture/11）測試各自執行，皆通過。
- [x] **突變檢測**：見下方「突變檢測結果」。
- [x] **繳交 zip 乾淨目錄執行**：解壓到 scratchpad 的空目錄後直接
      `python 10_assignment5_eval.py`，輸出 accuracy 0.9611、PCA 40 維 / 95.18%、
      誤判 14 張，不依賴任何未繳交的檔案。
- [x] **報告 PDF 逐頁渲染檢查**：14 頁全部以 Read 渲染確認。
      修正了兩處：第 3 頁 EVR 公式（含分式與連加）壓到下一段文字，`space` 由 0.072 調到 0.100；
      第 6 頁 mono 區塊的中文因 DejaVuSansMono 無中文字符而顯示為空白（見下）。
- [x] **`report_style` 測試**：`cd lecture && python -m unittest test_report_style`
      → **Ran 20 tests, OK**（含本次新增的 2 項 mono 測試）
- [x] **報告字型比對**：
      `['Arial-BoldMT', 'ArialMT', 'ArialUnicodeMS', 'Cmb10', 'Cmex10', 'Cmmi10', 'Cmr10', 'Cmsy10', 'DejaVuSansMono']`
      —— 與 Assignment #3 的字型集合**完全相同**，無任何 `Heiti`。
- [x] **繳交 zip UTF-8 旗標**：以 `scripts/make_submission.py` 打包，
      `515661055_陳欣怡_Assignment5_結果報告.pdf` 的 UTF-8 旗標為「是」。

## 本次修掉的 report_style 問題

`report_style.mono()` 原本固定使用 `family="monospace"`（DejaVuSansMono），
該字型**沒有中文字符**。拿它排中文時，PDF 的文字層仍抽得到字，畫出來的卻是空白 glyph，
所以只檢查文字抽取結果或字型名稱都不會發現——必須檢查「含中文的 span 用了哪個字型」。

A3 的 mono 只放純 ASCII 的分類報告，因此一直沒踩到。本次 A5 第 5 頁要輸出
「PCA 提取後的特徵維度 ...」才暴露出來。

修法：`mono()` 逐行判斷，含中文的行改用 Arial Unicode。純 ASCII 行的行為完全不變
（已由 `test_ascii_lines_stay_in_the_monospace_font` 鎖住），A1/A3 的報告不受影響。

## 與 Assignment #3 的對照

| | A3（特徵選取） | A5（特徵提取） |
| --- | --- | --- |
| CV 正確率 | 0.9638 | 0.9638 |
| 測試集正確率 | 0.9750 | 0.9611 |
| 誤判數 | 9 | 14 |

交叉驗證完全相同，測試集差 5 筆（約 1.4 個百分點），與 CV 標準差（±0.007）同量級，
不足以判定何者較優。報告第十二節據實說明了這點。

## 突變檢測結果

故意改壞關鍵常數與邏輯，確認每個突變至少讓一個測試失敗（腳本跑完會自動還原原始檔案）：

| 突變 | 捕捉它的測試 | 結果 |
| --- | --- | --- |
| `MAX_COMPONENTS` 40 → 50 | `ComponentBudgetTests` | 測試失敗 ✅ |
| 搜尋空間放入超過上限的 64 維 | `ComponentBudgetTests` | 測試失敗 ✅ |
| `RANDOM_STATE` 42 → 7 | `SplitContractTests` | 測試失敗 ✅ |
| 解釋變異量改成 1 位小數 | `ExplainedVarianceTests` | 測試失敗 ✅ |
| 混淆矩陣軸標改成 `Predicted` | `ExpectedFigureStyleTests` | 測試失敗 ✅ |

A5 的 5 個突變全數被捕捉。A6 的 6 個突變見該作業的驗證紀錄。

## 繳交內容

`515661055_陳欣怡_Assignment5.zip`（0.54 MB）：

| 檔案 | 大小 | UTF-8 旗標 |
| --- | --- | --- |
| `10_assignment5_eval.py` | 6,689 | 否（純 ASCII 檔名） |
| `digits_pipeline_lr_pca.pkl` | 26,540 | 否（純 ASCII 檔名） |
| `515661055_陳欣怡_Assignment5_結果報告.pdf` | 570,031 | 是 |
