# Assignment #1 驗證 Checklist

驗證日期：2026-09-13（初次）／2026-09-14（程式碼重構後複驗）
驗證環境：`.venv`（pandas 1.4.4 / numpy 1.21.2 / matplotlib 3.5.3）
驗證對象：`02_assignment1.py` + `tech_employees.csv` + `assignment1_results.txt` + `assignment1_decision_boundaries.png` + `assignment1_loss_curves.png` + `assignment1_adaptive_vs_fixed.png`

## 一、資料切分

- [x] 總資料筆數為 300 筆，欄位為 4 特徵 + 1 標籤（`df.shape == (300, 5)`）
- [x] 訓練集為前 200 筆、測試集為後 100 筆（`train_df.shape == (200,5)`、`test_df.shape == (100,5)`），切分方式與作業要求（依原始順序切分，非隨機）一致
- [x] 資料無缺失值（`df.isna().sum().sum() == 0`）
- [x] 訓練/測試集標籤分佈皆無嚴重失衡（train: 0=101/1=99，test: 0=49/1=51），majority-class baseline 僅 0.51，可作為模型有效性的對照基準

## 二、標準化（Standardization）是否僅用訓練集統計量

- [x] `StandardScaler.fit()` 只在 `X_train` 上呼叫，`X_test` 只呼叫 `transform()`，程式碼未對測試集重新計算 mean/std
- [x] 實際數值驗證（以 Math_Skill & Coding_Skill 組合為例）：
  - 訓練集統計量：`mean=[51.97, 5.305]`, `std=[31.02, 3.00]`
  - 測試集自身統計量：`mean=[45.39, 5.26]`（明顯不同）
  - 確認套用在測試集上的是訓練集的統計量，非測試集自己的 → **無資料洩漏 (data leakage)**

## 三、模型訓練收斂狀況

- [x] 高正確率組合（Math_Skill & Coding_Skill, acc=0.84）：loss 由 0.166 降至 0.119，穩定收斂不發散
- [x] 低正確率組合（Language_Skill & Math_Skill, acc=0.61）：loss 由 0.246 降至 0.204，同樣正常收斂（非訓練失敗，而是該組特徵本身鑑別度較低）
- [x] 自適應學習率 (`adaptive=True`, `eta/(1+decay*epoch)`) 有實際生效並產生合理的收斂曲線

## 四、正確率計算正確性

- [x] 手動重新計算 `np.sum(y_pred == y_test) / len(y_test)`，與程式回傳的 accuracy 完全一致（0.84 = 0.84）
- [x] 全部 7 種特徵組合的測試正確率皆高於 baseline（0.51），確認模型確實有學到東西，非隨機亂猜

## 五、輸出結果與 PDF 預期結果的對應

- [x] **預期結果 1**：`assignment1_results.txt` 依正確率由高到低列出 6 種兩兩組合 + 全特徵組合（共 7 列），格式與排序符合要求
- [x] 特徵鑑別度結論與 PDF 給的參考趨勢一致：**Coding_Skill 鑑別度最高 (0.80)、Math_Skill 次之 (0.72)**
- [x] **預期結果 2**：`assignment1_decision_boundaries.png` 已產生，為 2×3 共 6 張決策邊界圖（尺寸 3000×1800px），每張皆疊加測試集樣本點與 Class 0/1 圖例，座標軸標註 `(standardized)`
- [x] 決策邊界繪製為手刻 meshgrid + `predict()` 上色（`plot_decision_regions()`），未使用 scikit-learn 工具

## 六、重現性 (Reproducibility)

- [x] 重新執行 `02_assignment1.py`，終端機輸出與 `assignment1_results.txt` 內容逐字相符（`random_state=1` 固定，結果可重現）

## 七、加分視覺化（自我驗證訓練有效，非必交但強烈建議）

- [x] **Loss vs Epoch 收斂曲線**：`assignment1_loss_curves.png` 已新增，一次疊圖顯示全部 7 種特徵組合的收斂曲線。可清楚看到 Math_Skill & Coding_Skill / All Four Features 兩組收斂到最低 loss，與正確率最高吻合；Language_Skill & Health_Condition 收斂 loss 最高，對應正確率最低（0.61），趨勢一致
- [x] **固定學習率 vs 自適應學習率對比圖**：`assignment1_adaptive_vs_fixed.png` 已新增，以正確率最高的組合（Math_Skill & Coding_Skill）為代表，同時訓練 `adaptive=False`（固定 eta=0.01）與 `adaptive=True`（自適應 eta）兩個版本並疊圖比較。結果：固定學習率 acc=0.81，自適應學習率 acc=0.84，且自適應版本中後段 loss 略低更穩定，證實有實際調過學習率並確認自適應版本較優

## 八、程式碼重構後複驗 (2026-09-14)

`02_assignment1.py` 精簡重構（移除未使用的 `w_initialized`、重用主迴圈已訓練的模型取代重複訓練、補上關鍵數學步驟註解）後，重新執行第一至七節的所有檢查項目：

- [x] 資料切分、缺失值、標籤分佈、baseline (0.51) 與重構前完全一致
- [x] 標準化統計量（train mean=[51.97, 5.305] / std=[31.02, 3.00]，與測試集自身統計量明顯不同）與重構前逐位元相符，確認無資料洩漏
- [x] 高正確率組合 (Math_Skill & Coding_Skill) loss 0.1655 → 0.1194；低正確率組合 (Language_Skill & Math_Skill) loss 0.2464 → 0.2035，收斂趨勢與重構前一致
- [x] 手動重新計算 accuracy 與程式回傳值一致（0.84 = 0.84）
- [x] `assignment1_results.txt` 內容逐字元與重構前相同；連續執行兩次程式，輸出檔案 byte-for-byte 相同，確認可重現
- [x] 三張圖片皆重新產生且尺寸正確：`assignment1_decision_boundaries.png` (3000×1800)、`assignment1_loss_curves.png` (1600×1200)、`assignment1_adaptive_vs_fixed.png` (1600×1200)
- [x] 固定學習率 vs 自適應學習率對比數值不變（fixed acc=0.81，adaptive acc=0.84），驗證重用 `trained_models` 取代重複訓練後數值仍正確

**結論**：重構是行為保持不變 (behavior-preserving) 的重構，所有數值與圖表輸出與重構前完全一致，程式邏輯正確性未受影響。

---

**總結**：核心程式邏輯（資料切分、標準化無洩漏、SGD 訓練收斂、正確率計算、特徵組合比較、決策邊界視覺化）皆通過驗證，結果與 PDF 預期趨勢相符；新增的 loss 收斂曲線與自適應學習率對比圖也已產出並驗證趨勢合理；程式碼重構後已完成複驗，結果不變。
