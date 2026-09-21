# Assignment #2 驗證 Checklist

驗證日期：2026-09-21

驗證環境：`.venv`（Python 3.9 / pandas 1.4.4 / numpy 1.21.2 / matplotlib 3.5.3）

驗證對象：程式、兩份 CSV、兩張圖、預測 CSV／文字結果、八頁 PDF 報告與提交 ZIP

## 一、原始素材

- [x] `Assignment_2.pdf` 已放入作業目錄
- [x] `wat_train.csv` 已放入作業目錄
- [x] `wat_test.csv` 已放入作業目錄
- [x] 訓練集為 15 筆，欄位為 `x1`、`x2`、`y`
- [x] 待驗收資料為 5 筆，欄位為 `Wafer ID`、`x1`、`x2`
- [x] 兩份資料皆無缺失值
- [x] 訓練標籤只包含 `0` 與 `1`

## 二、程式限制

- [x] 已建立不依賴 scikit-learn 的資料載入與驗證骨架
- [x] 手刻 Logistic Regression 與數值穩定 sigmoid；極端輸入 `[-1000, 0, 1000]` 的單元測試通過
- [x] 手刻 binary cross-entropy loss 與 full-batch gradient descent；BCE 手算值與可線性分離資料學習測試通過
- [x] 比較學習率 `0.01`、`0.10`、`0.30`；`assignment2_loss_curves.png` 與結果表皆含三組比較

## 三、資料前處理

- [x] 比較原始 `x1`、`abs(x1)` 與 `x1 ** 2`：最佳學習率下訓練正確率依序為 73.33%、100%、100%；最終選用物理意義較直觀且符合題目參考輸出的 `abs(x1)`
- [x] `Standardizer.fit()` 僅對訓練集呼叫；單元測試以刻意不同的測試資料確認 `mean_`／`scale_` 只來自訓練集
- [x] 待驗收資料使用 `abs(x1)` 並只呼叫既有 scaler 的 `transform()`，未重新計算統計值

## 四、預期輸出

- [x] `assignment2_loss_curves.png` 已產生（1800 × 1200 px），含九條曲線、標題、座標軸、網格與圖例
- [x] `assignment2_decision_boundary.png` 已產生（1600 × 1200 px），含 Pass／Fail 區域、15 筆訓練樣本、標題、座標軸與圖例
- [x] `assignment2_predictions.csv` 與終端 DataFrame 已輸出五筆通過機率及 Pass／Fail；數值與題目範例一致：99.65%、4.91%、0.06%、9.55%、100.00%
- [x] 權重固定由零初始化且無隨機步驟；單元測試連續執行兩次完整流程並確認 DataFrame 完全一致

## 五、提交前檢查

- [x] 已在 `lecture/05/assignment` 直接執行 `05_assignment2.py`，exit code 0 且四項輸出檔成功產生
- [x] 兩張 PNG 已逐張目視檢查，標題、座標軸、圖例完整，無裁切或重疊
- [x] 結果欄位為 `Wafer ID`、`x1`、`x2`、`abs_x1`、`Pass Probability`、`Result`，與題目範例一致
- [x] `Assignment2_陳欣怡_結果報告.pdf`（A4、8 頁）與 `Assignment2_陳欣怡.zip`（7 個提交檔）已產生；PDF 已逐頁渲染檢查

---

**總結**：9 項自動化測試全數通過；最終模型使用 `abs(x1)`、`x2`、
learning rate `0.3` 與 `1,000` epochs，訓練正確率為 100%，五筆待驗收晶圓的
機率與分類均重現題目投影片範例。
