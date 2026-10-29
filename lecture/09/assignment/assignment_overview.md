# Assignment #4 作業說明

## 目前進度

兩種方法皆已訓練完成並產出繳交用的參數檔與測試腳本。

題目 PDF 共 4 頁（投影片 p.67–70），逐頁驗收紀錄見 `verification_checklist.md`。

## 這份作業在學什麼

改善 CelebA 微笑分類任務的性能，目標正確率 > 90%。課堂範例
（Sample code 08-3.py，即本 repo 的 `lecture/09/09_3.py`）的四層 CNN
大約落在 88% 上下，本作業要以兩種不同的手段把它推過門檻。

**兩種方法皆需實作，且資料集條件須一致：**

- 僅使用 CelebA **訓練集的前 16,000 筆** samples 進行訓練（可以使用資料增強技術）
- 使用 CelebA **測試集的前 10,000 筆** samples 進行測試

### 方法一：調整網路架構

投影片 p.67 列出的手段：加入批次正規化 (`nn.BatchNorm2d`)、調整網路寬度或深度、
調整 Dropout 位置或機率。

本實作相對課堂範例的三處調整：

1. 每層卷積後加入 `nn.BatchNorm2d`（共 4 層）
2. 卷積層的 Dropout 由 0.5 降為 0.1 / 0.2
3. 分類頭（Linear）之前補一層 `Dropout(0.3)`

第 2 點是關鍵：課堂範例在卷積層用 0.5，等於每次前向傳播丟掉一半的特徵圖通道；
對只有 32 / 64 個通道的淺層而言強度過大。批次正規化本身已帶有正則化效果，
Dropout 可以放鬆，改把正則化集中在參數量最大的全連接層之前。

### 方法二：ResNet-18 預訓練模型微調

`torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)`，加上三項必要調整：

1. 縮放輸入影像尺寸到 224 × 224
2. 影像正規化：採用 ImageNet 的 mean / std
3. 修改輸出層：`Linear(512, 1) + Sigmoid`，對應二分類問題

學習率刻意調小到 1e-4，避免一開始的大梯度把預訓練權重沖掉。

## 繳交規定

總共**四個檔案**打包成單一 `.zip` 檔上傳：

| 檔案 | 說明 |
| --- | --- |
| `model_1.pth` | 方法一的參數檔 |
| `model_1_test.py` | 方法一的測試腳本 |
| `model_2.pth` | 方法二的參數檔 |
| `model_2_test.py` | 方法二的測試腳本 |

測試腳本載入模型參數的寫法參考 Sample code 08-4.py（本 repo 的 `lecture/09/09_4.py`）。
兩支測試腳本都是**自給自足**的單檔腳本，不 import 任何共用模組
（zip 裡只有這四個檔案，沒有別的）。

> **注意**：`model_2.pth` 是 ResNet-18（1,118 萬參數），約 45 MB，
> 繳交 zip 因此偏大。這是作業指定使用 ResNet-18 的必然結果，
> 上傳前請確認作業系統的檔案大小限制。

## 資料集：CelebA

| 用途 | 來源 | 筆數 |
| --- | --- | --- |
| 訓練 | CelebA train split 前 16,000 筆 | 16,000 |
| 驗證 | CelebA valid split 前 1,000 筆 | 1,000 |
| 測試 | CelebA test split 前 10,000 筆 | 10,000 |

標籤取自 40 維屬性中的第 31 項（Smiling）。測試集的 10,000 筆中
Not Smiling 4,985 筆、Smiling 5,015 筆 —— 與投影片 p.69 分類報告的 support 欄完全吻合。

### 本機取得 CelebA 的問題（重要）

官方 CelebA 託管在 Google Drive，長期處於
「Too many users have viewed or downloaded this file recently」的配額封鎖狀態，
`torchvision` 的 `download=True` 在本機一律失敗（已實測確認）。

解法：`prepare_celeba_local.py`（不繳交）改由 HuggingFace 的 `eurecom-ds/celeba`
鏡像取得影像與 40 維屬性，重建成 `torchvision.datasets.CelebA` 期望的目錄結構。
該鏡像的 train/validation/test 筆數為 162,770 / 19,867 / 19,962，與官方
`list_eval_partition.txt` 完全一致。只解出本作業用到的 27,000 張影像，
不必下載整包 8.9 GB。

`celeba_local_patch.py`（不繳交）負責把 torchvision 硬編碼的 md5 換成本機檔案的實際值
—— 重建的標註檔內容正確但行尾格式不同，md5 對不上會被判定為 corrupted。

**繳交的 `model_1_test.py` / `model_2_test.py` 不依賴這兩支腳本**：
它們呼叫標準的 `datasets.CelebA(..., download=True)`，與課堂範例一字不差，
助教環境中既有的 CelebA 即可直接使用。

## 訓練結果

| 項目 | 方法一 | 方法二 |
| --- | --- | --- |
| 輸入尺寸 | 64 × 64 | 224 × 224 |
| Epochs | 30 | 5 |
| 學習率 | 1e-3 | 1e-4 |
| Weight decay | 0 | 1e-4 |
| 最佳 epoch | 28 | 3 |
| 驗證集 loss | 0.2076 | 0.1788 |
| 驗證集正確率 | 92.8% | 92.8% |

兩者都依**驗證集 loss** 保留最佳的一輪權重再存檔，避免交出過擬合那一輪的參數。
測試集全程未參與訓練與任何選擇。

## 關於 93% 這個天花板

題目 p.67 註明：「微笑分類的平均正確率通常落在 93.0% ~ 93.5% 左右，
正確率無法再隨模型升級而改善的主因在於資料集本身的標籤錯誤率。」

CelebA 的屬性標註由標註者主觀判定，「微笑」與「沒微笑」之間本來就存在大量灰色地帶
（抿嘴、半笑、側臉）。這類樣本無論模型多強都無法穩定分對，因為連標準答案本身都不一致。
從誤判樣本圖可以直接看到這點。

## 檔案

| 檔案 | 繳交 | 說明 |
| --- | --- | --- |
| `Assignment_4.pdf` | | 原始作業題目 |
| `model_1.pth` / `model_1_test.py` | ✅ | 方法一的參數檔與測試腳本 |
| `model_2.pth` / `model_2_test.py` | ✅ | 方法二的參數檔與測試腳本 |
| `09_assignment4_common.py` | | 兩種方法共用的資料與模型定義 |
| `09_assignment4_train.py` | | 訓練腳本（`--method 1` / `--method 2`） |
| `09_assignment4_explore.py` | | 改善空間研究，輸出量測 JSON |
| `test_09_assignment4.py` | | 資料契約、架構、權重對應與繳交形式測試 |
| `prepare_celeba_local.py` | | 本機 CelebA 重建（見上） |
| `celeba_local_patch.py` | | 讓 torchvision 接受重建的資料集 |
| `build_report.py` | | 結果報告 PDF 產生器 |
| `verification_checklist.md` | | 完成後的逐項驗證紀錄 |
