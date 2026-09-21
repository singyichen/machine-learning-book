# Assignment #2 作業說明

## 目前進度

本作業已完成。程式包含手刻 Logistic Regression、特徵工程、標準化、
binary cross-entropy、梯度下降、學習率比較、圖表與待驗收晶圓預測；另已產生
結果報告與提交 ZIP。所有核心流程均未使用 scikit-learn。

## 這份作業在學什麼

主題是使用手刻的 Logistic Regression 建立晶圓驗收分類器。作業明確要求不使用 scikit-learn。

1. 使用 `wat_train.csv` 的 15 筆歷史晶圓資料訓練模型。
2. 以氧化層厚度偏差 `x1` 與微量金屬污染度 `x2` 判斷晶圓是否合格。
3. 留意厚度偏差無論過大或過小都可能不合格，因此需比較 `abs(x1)` 或 `x1 ** 2` 等特徵轉換。
4. 僅以訓練集計算標準化所需的平均值與標準差，再把相同統計值套用到待驗收資料。
5. 以梯度下降訓練 Logistic Regression，並嘗試不同學習率。
6. 繪製 Loss vs. Epochs 與訓練資料的決策邊界。
7. 對 `wat_test.csv` 的 5 筆晶圓輸出通過機率與 Pass／Fail 結果。

## 資料欄位

### `wat_train.csv`

| 欄位 | 說明 |
| --- | --- |
| `x1` | 氧化層厚度偏差（Å） |
| `x2` | 微量金屬污染度（ppm） |
| `y` | 晶圓狀態，`1` 為 Pass、`0` 為 Fail |

### `wat_test.csv`

| 欄位 | 說明 |
| --- | --- |
| `Wafer ID` | 待驗收晶圓識別碼 |
| `x1` | 氧化層厚度偏差（Å） |
| `x2` | 微量金屬污染度（ppm） |

## 預計輸出

- `assignment2_loss_curves.png`：九組特徵轉換／學習率的訓練損失曲線
- `assignment2_decision_boundary.png`：決策邊界與訓練樣本分布圖
- `assignment2_predictions.csv`：DataFrame 格式的晶圓通過機率與驗收結果
- `assignment2_results.txt`：特徵／學習率比較與驗收結果文字輸出
- `Assignment2_陳欣怡_結果報告.pdf`：八頁結果報告
- `Assignment2_陳欣怡.zip`：提交檔案

## 檔案

- `Assignment_2.pdf`：原始作業題目
- `wat_train.csv`：15 筆訓練資料
- `wat_test.csv`：5 筆待驗收資料
- `05_assignment2.py`：完整模型訓練、比較、預測與視覺化程式
- `test_05_assignment2.py`：資料契約、數學函數、前處理與整體流程測試
- `verification_checklist.md`：完成後的逐項驗證紀錄
