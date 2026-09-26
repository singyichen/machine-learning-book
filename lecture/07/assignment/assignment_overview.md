# Assignment #3 作業說明

## 目前進度

本作業已完成。程式包含資料切分、特徵縮放、L1 特徵選取、GridSearchCV 模型選取、
管線儲存與模型評估；另已產生特徵選取遮罩圖、混淆矩陣圖、結果報告與提交 ZIP。

## 這份作業在學什麼

主題是以 scikit-learn 建立低解析度手寫數字 (0–9) 的多類別分類器，應用情境為
光學字元辨識（OCR）。假設字元定位與切割已完成，本作業僅負責字元辨識任務。

1. 以 `sklearn.datasets.load_digits()` 載入 1797 張 8×8 灰階影像。
2. 以 Logistic Regression 作為分類器演算法。
3. 切分 80% 訓練（用於交叉驗證）與 20% 測試，種子固定為 42。
4. 至少淘汰 20 個以上的特徵，即僅能使用 44 個以下的像素值。
5. 在測試集（約 360 筆）上的整體正確率須達 95% 以上。
6. 以 `joblib.dump()` 將訓練好的最佳模型（指整條管線）儲存成檔案。
7. 視覺化特徵選取遮罩，並輸出測試集的分類報告與混淆矩陣。

## 繳交規定（與前兩次不同）

老師要求把**模型評估部分的程式碼與模型儲存檔**獨立出來打包成單一 `.zip`：

- **不要**包含任何訓練或模型選取程式碼
- 助教將直接執行模型評估程式批閱作業（沒有時間進行交叉驗證與微調）
- 評估程式僅載入 `train_test_split()` 與 metrics 相關函式，以相同的
  `random_state=42` 重建測試集，再 `joblib.load()` 載入模型進行評估

因此程式拆成兩支：`07_assignment3_train.py`（訓練，不繳交）與
`07_assignment3_eval.py`（評估，繳交）。

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
StandardScaler → SelectFromModel(L1 LogisticRegression) → LogisticRegression
```

以 5-fold StratifiedKFold 交叉驗證搜尋 `selector__max_features`（32 / 40 / 44）
與 `classifier__C`（0.1 / 1.0 / 10.0）共 9 組組合。

最佳組合為 `max_features=40`、`C=1.0`，交叉驗證正確率 0.9638，測試集正確率 0.9750。

## 預計輸出

- `digits_pipeline_lr.pkl`：以 `joblib.dump()` 儲存的完整管線
- `assignment3_feature_mask.png`：8×8 特徵選取遮罩圖，標題含選取比例 40/64
- `assignment3_confusion_matrix.png`：測試集混淆矩陣
- `assignment3_results.txt`：分類報告與混淆矩陣文字輸出
- `assignment3_improvement_study.json`：改善空間研究的量測結果
- `515661055_陳欣怡_Assignment3_結果報告.pdf`：十二頁結果報告（含改善空間討論）
- `515661055_陳欣怡_Assignment3.zip`：提交檔案

## 檔案

- `Assignment_3.pdf`：原始作業題目
- `07_assignment3_train.py`：訓練、特徵選取、模型選取與管線儲存（**不繳交**）
- `07_assignment3_eval.py`：模型評估程式（**繳交**）
- `test_07_assignment3.py`：切分契約、特徵上限、模型儲存與繳交限制測試
- `07_assignment3_explore.py`：改善空間研究，輸出量測 JSON（**不繳交**）
- `build_report.py`：結果報告 PDF 產生器（版面共用 `lecture/report_style.py`）
- `verification_checklist.md`：完成後的逐項驗證紀錄
