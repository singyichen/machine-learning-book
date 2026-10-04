# Assignment #2 驗證 Checklist

驗證日期：2026-09-21（初版）／ 2026-10-04（補改善空間研究與公式寫法）

驗證環境：`.venv`（Python 3.9 / pandas 1.4.4 / numpy 1.26.4 / matplotlib 3.5.3）

驗證對象：程式、兩份 CSV、兩張圖、預測 CSV／文字結果、十頁 PDF 報告、改善空間研究 JSON 與提交 ZIP

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
- [x] `515661055_陳欣怡_Assignment2_結果報告.pdf`（A4、封面 + 10 頁）與 `515661055_陳欣怡_Assignment2.zip`（4 個提交檔）已產生；PDF 已逐頁渲染檢查

## 六、改善空間研究（2026-10-04）

- [x] `05_assignment2_explore.py` 以留一交叉驗證（15 折，每折重新標準化）量測 13 組設定，結果寫入 `assignment2_improvement_study.json`
- [x] 基準組（|x1|、η = 0.3、1,000 epochs、λ = 0）的 5 筆機率與繳交版本逐一相同（單元測試 atol 1e-12）
- [x] Epochs 100 → 10,000：訓練正確率皆 100%，‖w‖ 由 3.29 增至 17.68，LOO 正確率固定 86.7%，LOO log-loss 在 300 epochs 最低（0.1691）、10,000 epochs 升至 0.3518
- [x] L2：λ = 0.1 的 LOO log-loss 0.1818（基準 0.2007），正確率不變；λ = 10 訓練正確率掉到 93.3%
- [x] 特徵組合：原始 x1 LOO 正確率 66.7%，其餘三組皆 86.7%；x1² 的 LOO log-loss 最低（0.1696）但 Wafer D 機率變為 22.36%
- [x] 留一誤判樣本固定為 (1.5, 1.8, Pass) 與 (−1.0, 2.5, Fail)，與決策邊界圖最貼近邊界的兩點一致
- [x] 結論：維持繳交版本設定（重現題目參考輸出）；報告第八、九節據此撰寫，數字全部由 JSON 帶入

## 七、公式寫法與版面（2026-10-04）

- [x] 報告內所有數學符號改為 mathtext：`$x_1$`、`$|x_1|$`、`$x_1^{2}$`、`$\eta$`、`$\mu_{\mathrm{train}}$`、`$\mathbf{w}^{\top}\mathbf{x} + b$` 等；不再出現 `x1`、`abs(x1)`、`x1 ** 2`
- [x] 第二節補上梯度下降更新式與 J(w, b) 正規寫法；第八節補 L2 目標函數 J_λ
- [x] `report_style.wrap_text` 把 `$...$` 視為不可拆 token、依可見符號估寬；新增 2 項測試
- [x] PDF 內嵌字型：Cm*（數學）、ArialUnicodeMS（內文與標題）、Arial-BoldMT（封面）、ArialMT（頁碼），無 Heiti；test_report_style 17 項通過
- [x] 11 頁逐頁渲染檢查：無公式重疊、無行首標點、無溢出（PageOverflow 未觸發）

## 八、突變檢測（2026-10-04）

- [x] 7 個突變全部被測試抓到：L2 符號反向、l2 預設值改 0.5、LOO 忘記排除留下那筆、基準 epochs 改 500、abs+square 欄位順序錯、wrap_text 不把 `$...$` 當整體、公式寬度照原字元算

## 九、繳交檔重新驗證（2026-10-04）

- [x] 封面補列「課程教授：陳慶永　　課程助教：翁宣允」（姓名／學號下一行，14 pt），A2、A3 報告與 zip 同步重建

- [x] zip 重新打包為 4 個檔案：05_assignment2.py、wat_train.csv、wat_test.csv、結果報告 PDF（PDF 檔名 UTF-8 旗標：是）。移除 assignment2_predictions.csv——教授規定只有「.py 與 CSV 資料檔」，預測 CSV 是程式執行時自動產生的輸出
- [x] 解壓至乾淨暫存目錄執行 `05_assignment2.py`：exit 0，五筆機率與 repo 內 CSV 完全相同
- [x] 全部測試：lecture/02（通過）、lecture/05（14 項通過）、lecture/07（通過）、test_report_style（17 項通過）

---

**總結**：9 項自動化測試全數通過；最終模型使用 `abs(x1)`、`x2`、
learning rate `0.3` 與 `1,000` epochs，訓練正確率為 100%，五筆待驗收晶圓的
機率與分類均重現題目投影片範例。
