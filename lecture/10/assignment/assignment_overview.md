# Assignment #5 作業說明

## 目前進度

本作業已完成。程式包含資料切分、特徵縮放、PCA 特徵提取、GridSearchCV 模型選取、
管線儲存與模型評估；另已產生混淆矩陣圖、誤判樣本圖、解釋變異量補充圖、結果報告與提交 ZIP。

題目 PDF 共 7 頁（投影片 p.50–56），逐頁驗收紀錄見 `verification_checklist.md`。

## 這份作業在學什麼

本次作業的實作任務與 Assignment #3 (Lecture 6) 相同，唯一的差別是
**將特徵選取換成特徵提取（使用 PCA）**。

1. 以 `sklearn.datasets.load_digits()` 載入 1797 張 8×8 灰階影像。
2. 以 Logistic Regression 作為分類器演算法。
3. 切分 80% 訓練（用於交叉驗證）與 20% 測試，種子固定為 42。
4. 降維後的特徵空間維度 ≤ 40。
5. 在測試集（約 360 筆）上的整體正確率須達 95% 以上。
6. 以 `joblib.dump()` 將訓練好的最佳模型（指管線）儲存成檔案。
7. 輸出 PCA 降維後的維度與總解釋變異量。
8. 輸出測試集的分類報告，並視覺化呈現混淆矩陣。
9. 輸出誤判的樣本總數，繪出所有誤判樣本（標示真實標籤與預測結果），且必須與混淆矩陣對應。

## 特徵選取 vs 特徵提取

| | Assignment #3 | Assignment #5 |
| --- | --- | --- |
| 手段 | 特徵選取（SelectFromModel + L1） | 特徵提取（PCA） |
| 結果 | 從 64 個像素挑出 40 個 | 把 64 維投影成 40 個主成分 |
| 可解釋性 | 高，可指出用了哪幾格 | 低，每維都是 64 像素的線性組合 |
| CV 正確率 | 0.9638 | 0.9638 |
| 測試集正確率 | 0.9750 | 0.9611 |

兩者交叉驗證完全相同，測試集差 5 筆（約 1.4 個百分點），屬標準差量級。

## 繳交規定（與 Assignment #3 相同）

把**模型評估部分的程式碼與模型儲存檔**獨立出來打包成單一 `.zip`：

- **不要**包含任何訓練或模型選取程式碼
- 助教將直接執行模型評估程式批閱作業（沒有時間進行交叉驗證與微調）

因此程式拆成兩支：`10_assignment5_train.py`（訓練，不繳交）與
`10_assignment5_eval.py`（評估，繳交）。

## 資料集

`sklearn.datasets.load_digits()`，無外部 CSV 檔。

| 項目 | 說明 |
| --- | --- |
| 樣本數 | 1797 張 8×8 灰階影像 |
| 特徵 | 64 維像素值，整數範圍 [0, 16] |
| 標籤 | 0–9 共 10 類 |
| 訓練／測試 | 1437 筆 / 360 筆（`test_size=0.2, random_state=42`） |

## 模型

```
StandardScaler → PCA → LogisticRegression
```

以 5-fold StratifiedKFold 交叉驗證搜尋 `pca__n_components`（20 / 30 / 40）
與 `classifier__C`（0.1 / 1.0 / 10.0）共 9 組組合。

最佳組合為 `n_components=40`、`C=1.0`，交叉驗證正確率 0.9638，測試集正確率 0.9611，
總解釋變異量 95.18%，測試集共誤判 14 張。

> `n_components=40` 與 95.18% 與投影片 p.54 的預期輸出數值完全一致。

## 預計輸出

- `digits_pipeline_lr_pca.pkl`：以 `joblib.dump()` 儲存的完整管線
- `assignment5_confusion_matrix.png`：測試集混淆矩陣
- `assignment5_misclassified.png`：所有誤判樣本（每列 5 張，標題為真實標籤，左下角紅字為預測）
- `assignment5_explained_variance.png`：報告補充圖（投影片未要求），各主成分與累積解釋變異量
- `assignment5_results.txt`：維度、解釋變異量、分類報告、混淆矩陣與誤判總數
- `assignment5_improvement_study.json`：改善空間研究的量測結果
- `515661055_陳欣怡_Assignment5_結果報告.pdf`：十三頁結果報告（含改善空間討論）
- `515661055_陳欣怡_Assignment5.zip`：提交檔案

## 檔案

- `Assignment_5.pdf`：原始作業題目
- `10_assignment5_train.py`：訓練、PCA、模型選取與管線儲存（**不繳交**）
- `10_assignment5_eval.py`：模型評估程式（**繳交**）
- `test_10_assignment5.py`：切分契約、維度上限、模型儲存、圖表樣式與繳交限制測試
- `10_assignment5_explore.py`：改善空間研究，輸出量測 JSON（**不繳交**）
- `build_report.py`：結果報告 PDF 產生器（版面共用 `lecture/report_style.py`）
- `verification_checklist.md`：完成後的逐項驗證紀錄
