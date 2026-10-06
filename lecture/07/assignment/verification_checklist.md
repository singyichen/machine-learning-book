# Assignment #3 驗證 Checklist

驗證日期：2026-09-26（2026-10-06 補驗預期結果 3）

驗證環境：`.venv`（Python 3.9 / scikit-learn 1.0.2 / numpy 1.26.4 / scipy 1.13.1 /
joblib 1.5.3 / matplotlib 3.5.3）

> numpy 原為 1.21.2，其內建 OpenBLAS 在 ARM64 的 dgemm 核心有緩衝區溢位，
> 於 2026-09-26 升級。升級後全套測試通過，繳交的 `.pkl` 結果不變（0.9750）。

驗證對象：訓練程式、評估程式、模型檔、三張圖、文字結果、十三頁 PDF 報告與提交 ZIP

## 一、原始素材

- [x] `Assignment_3.pdf` 已放入作業目錄（**7 頁**，投影片 p.55–61）
      > 2026-09-26 的版本只對照到 p.60，漏掉 p.61「預期結果 3」；2026-10-06 補上。
- [x] 資料集直接由 `sklearn.datasets.load_digits()` 載入，無外部 CSV
- [x] 確認資料為 1797 筆 × 64 維，像素值範圍 [0, 16]，標籤為 0–9 共 10 類

## 二、資料切分

- [x] `train_test_split(test_size=0.2, random_state=42)`：1437 筆訓練 / 360 筆測試
- [x] 測試集各類別筆數為 33、28、33、34、46、47、35、34、30、40，
      與作業投影片參考輸出的 support 欄完全一致
- [x] 訓練程式與評估程式各自切分，單元測試確認兩者得到同一組測試集
      （`y_test[:10] = [6, 9, 3, 7, 2, 1, 5, 2, 5, 2]`，以 sklearn 獨立計算為參考值）

## 三、特徵縮放與特徵選取

- [x] `StandardScaler` 僅在管線內對訓練折計算統計值，測試集只做 transform
- [x] `SelectFromModel` 以 L1 Logistic Regression (liblinear, C=0.1) 估計重要性，
      `threshold=-inf` 搭配 `max_features` 確保選取數量可控且可重現
- [x] 最終選用 **40/64** 個像素，淘汰 **24** 個，符合「至少淘汰 20 個、僅用 44 個以下」
- [x] `validate_feature_budget()` 在超過 44 個時直接拋出 `ValueError`；單元測試涵蓋 44（通過）與 45（拋錯）

## 四、模型選取

- [x] 5-fold `StratifiedKFold` 交叉驗證，搜尋 3 × 3 = 9 組參數組合
- [x] 最佳組合 `max_features=40`、`C=1.0`，CV 平均正確率 0.9638
- [x] 測試集全程未參與調參，僅在最後評估時使用一次

## 五、預期輸出

- [x] `assignment3_feature_mask.png` 已產生，8×8 遮罩、0/1 標註齊全，
      標題為 `Feature Selection Mask (Selected: 40/64)`，含選取比例且在 44/64 以下
- [x] 遮罩圖比照投影片 p.59 不畫座標軸，64 格皆標 0/1（樣式測試檢查 `axison` 與文字內容）
- [x] `assignment3_confusion_matrix.png` 比照投影片 p.60：100 格（含 0）都標數字、無標題，
      軸標為 `Predicted label` / `True label`（樣式測試檢查）
- [x] 分類報告 10 個類別 f1-score 皆 ≥ 0.95，macro avg 與 weighted avg 均為 0.98
- [x] 測試集整體正確率 **0.9750**，達成 95% 門檻
- [x] 預期結果 3：評估程式輸出「測試集總共誤判了 9 張圖片。」
- [x] `assignment3_misclassified.png` 比照投影片 p.61：每列 5 張、共 2 列，標題 `True: x`，
      左下角紅字為預測結果；9 組（真實→預測）為 9→3、7→9、3→5、9→8、4→1、5→9、8→5、6→5、5→6，
      與混淆矩陣 9 個非對角線格逐一對應（非對角線總和 = 360 − 351 = 9）
- [x] 三張 PNG 已逐張目視檢查並與投影片截圖比對，無裁切或重疊

## 六、模型儲存

- [x] 以 `joblib.dump()` 儲存**整條 Pipeline**（含 scaler 與 selector），非僅分類器
- [x] 單元測試重新載入 `.pkl` 並確認預測結果與原管線完全一致
- [x] 突變測試：改為只存分類器會讓測試失敗
      （`X has 64 features, but LogisticRegression is expecting 40 features`）

## 七、繳交限制（本次新增）

- [x] `07_assignment3_eval.py` 未匯入 `GridSearchCV`、`StratifiedKFold`、`SelectFromModel`、
      `StandardScaler`、`LogisticRegression` 等訓練／模型選取符號（AST 掃描驗證）
- [x] 評估程式中沒有任何 `.fit()` 或 `.fit_transform()` 呼叫（AST 掃描驗證）
- [x] 將評估程式與 `.pkl` 單獨複製到空資料夾後以子行程執行，exit code 0 且輸出 accuracy
- [x] 解壓提交 ZIP 至乾淨資料夾直接執行，exit code 0，輸出 Accuracy 0.9750 與誤判 9 張，並產生三張圖

## 八、測試與突變檢測

- [x] 本作業 23 項自動化測試全數通過（新增 5 項預期結果圖樣式與誤判樣本測試）
- [x] 全專案測試合計 64 項全數通過，無回歸（A1 8 項、A2 15 項、A3 23 項、`test_report_style` 18 項）
- [x] 突變檢測 5 項全部被測試攔截：
  - `RANDOM_STATE` 42 → 7（切分契約測試失敗）
  - `MAX_FEATURES` 44 → 64（特徵上限測試失敗）
  - 遮罩標題移除選取比例（標題測試失敗）
  - 只儲存分類器而非管線（模型往返測試失敗）
  - 評估端 `TEST_SIZE` 0.2 → 0.25（測試集形狀測試失敗）
- [x] 2026-10-06 新增突變 7 項全部被攔截：誤判判斷 `!=` → `==`、混淆矩陣不標 0、
      每列 5 張 → 4 張、預測字改黑色、遮罩圖保留座標軸、標題誤用預測值、文字結果漏印誤判數

## 八之二、改善空間研究

- [x] `07_assignment3_explore.py` 量測 18 組設定，結果寫入 `assignment3_improvement_study.json`
- [x] 所有比較均在訓練集上以 5-fold 交叉驗證進行，測試集僅在最後做一次確認性評估
- [x] 單元測試比對研究的基準組與實際存下的 `.pkl`，避免兩者漂移
- [x] 主要結論：放寬到 64 個特徵只多 0.4pp；改用 RBF 核 SVM 可達 0.9896，
      顯示瓶頸為線性決策邊界而非特徵數量
- [x] 報告產生器已加入頁面溢出偵測，內容超出可用高度會直接拋錯而非靜默裁切

## 九、提交前檢查

- [x] 在 `lecture/07/assignment` 直接執行 `07_assignment3_train.py`，exit code 0，模型檔成功產生
- [x] 在同一目錄執行 `07_assignment3_eval.py`，exit code 0，四項輸出檔成功產生
- [x] `515661055_陳欣怡_Assignment3_結果報告.pdf`（A4、13 頁，新增「八、預期結果 3：誤判樣本」）已逐頁渲染檢查，
      版面比照 Assignment #1（粗體標題、分隔線、淡藍表頭表格、mathtext 公式、頁碼置右下）；
      內嵌字型為 ArialUnicodeMS、Arial-BoldMT、ArialMT、Cm*、DejaVuSansMono，無 Heiti
- [x] `515661055_陳欣怡_Assignment3.zip`（3 個提交檔）已產生，中文檔名已設 UTF-8 旗標

---

**總結**：最終模型以 40 個像素（淘汰 24 個）、Logistic Regression `C=1.0`
達成測試集 **97.50%** 的整體正確率，360 筆中誤判 9 張。特徵淘汰數、使用像素數、測試集正確率、
模型儲存方式、繳交內容限制與誤判樣本對應共六項驗收條件全數通過，題目三項預期結果皆已產出。
