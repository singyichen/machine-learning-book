---
name: ml-course-assignment
description: 這門機器學習課程的作業製作與繳交流程（陳欣怡 / 學號 515661055）。當使用者提到某週的作業、Assignment #N、給你一份 Assignment_N.pdf、要你實作課堂作業、產生結果報告、打包繳交 zip，或問「第 N 週作業要放哪」時，務必使用這支 skill——即使他們沒有明講「作業流程」。這裡記錄了資料夾編號與週次的對應、zip 中文檔名編碼、報告版面規格等一看就會踩的細節。
---

# 機器學習課程作業流程

這門課每週一份作業，結構高度重複。這支 skill 記的是**重複踩過的坑**，不是通則——
TDD 的做法請用 `superpowers:test-driven-development`，程式架構看 repo 本身就知道。

## 先確認資料夾編號

作業放在 `lecture/<週次>/assignment/`。**週次不等於投影片編號**：

```
投影片 04_Logistic Regression.pdf  →  lecture/05/   （第 5 週）
投影片 06_Model Selection.pdf      →  lecture/07/   （第 7 週）
```

投影片編號比週次小 1。使用者通常會直接說「第 7 週」，照他說的建 `lecture/07/`。
若不確定，拿 Downloads 裡的 `NN_*.py` 和既有 `lecture/MM/MM_*.py` 做 diff 比對就能確認對應關係。

歷史對照：Assignment #1 → `lecture/02/`，#2 → `lecture/05/`，#3 → `lecture/07/`。

## 流程

1. **讀題目 PDF**。用 Read 工具搭配 `pages` 參數逐頁看圖，別只抽文字——題目的
   預期輸出常以截圖呈現（圖表長相、報告格式、門檻數字）。把驗收條件列成表。

2. **先確認繳交規定再動工**。每次都可能不同。Assignment #3 就要求「只繳模型評估
   程式與模型檔，不得包含任何訓練或模型選取程式碼」，這會直接決定程式要拆成幾支。

3. **TDD 實作**。測試要能真的抓到錯——寫完跑一次突變檢測（故意改壞常數、改壞
   儲存邏輯），確認每個突變至少讓一個測試失敗。測試檔命名 `test_NN_assignmentM.py`。

4. **產生圖表與文字結果**，逐張用 Read 目視檢查，確認標題、座標軸、圖例完整。

5. **產生結果報告 PDF**：寫 `build_report.py`，版面共用 `lecture/report_style.py`。

6. **打包繳交 zip**：用 `scripts/make_submission.py`（見下方）。

7. **驗證後提交**：跑完整測試套件、解壓 zip 實際執行、PDF 逐頁渲染檢查，再 commit。

## 固定產出清單

放在 `lecture/<週次>/assignment/`：

| 檔案 | 說明 |
|---|---|
| `Assignment_N.pdf` | 原始題目 |
| `NN_assignmentM.py` | 實作（依繳交規定可能要拆成 train / eval 兩支） |
| `test_NN_assignmentM.py` | 測試 |
| `build_report.py` | 報告產生器（不繳交） |
| `assignmentM_*.png` / `.txt` / `.csv` | 圖表與結果 |
| `515661055_陳欣怡_AssignmentM_結果報告.pdf` | 結果報告 |
| `515661055_陳欣怡_AssignmentM.zip` | 繳交檔 |
| `assignment_overview.md` | 作業說明與進度 |
| `verification_checklist.md` | 逐項驗證紀錄（完成後才寫，要寫實際數值） |

## 繳交 zip

檔名規則：`學號_姓名_AssignmentN.zip` → `515661055_陳欣怡_Assignment3.zip`

**一定要用 `scripts/make_submission.py`，不要用 `zip` 指令。** macOS 的 `zip` 不會設
UTF-8 檔名旗標，老師在 Windows 解壓會看到亂碼；`zip -UN=UTF8` 在系統內建版本也不支援。
Python `zipfile` 對非 ASCII 檔名會自動設旗標。

```bash
python .claude/skills/ml-course-assignment/scripts/make_submission.py \
  lecture/07/assignment 515661055_陳欣怡_Assignment3.zip \
  07_assignment3_eval.py digits_pipeline_lr.pkl 515661055_陳欣怡_Assignment3_結果報告.pdf
```

腳本會印出每個檔案的 UTF-8 旗標供確認。結果報告 PDF 前兩次都有放進 zip，沿用即可
（題目沒禁止，且方便老師看推導過程）。

## 結果報告版面

以 Assignment #1 的報告為基準，共用模組在 `lecture/report_style.py`，用 matplotlib
PdfPages 繪製。`build_report.py` 照這個骨架寫：

```python
sys.path.insert(0, str(ASSIGNMENT_DIR.parent.parent))   # lecture/
import report_style

report = report_style.Report(metadata={...})
report.cover(標題, 副標, 課程週次, "姓名：陳欣怡　　學號：515661055", [資料集, 模型])
report.page("一、作業說明")
report.subheading(...); report.text(...); report.bullets([...])
report.math(r"$...$", size=14, space=0.06)      # space 要依公式高度調
report.table(表頭, 列, 欄寬比例)
report.figure_image(圖路徑, height=0.5, caption="檔名 — 說明文字")
report.mono(分類報告之類的等寬文字)
report.save(輸出路徑)
```

頁面結構慣例（約 8–9 頁）：封面 / 一、作業說明 / 二、數學觀念 / 三、訓練與評估流程 /
四～七、各項預期結果 / 八、總結（含驗收項目對照表與主要發現）。

**數學式用 mathtext**（`r"$\dfrac{a}{b}$"`）。`report_style.py` 已設定
`matplotlib.rcParams["mathtext.fontset"] = "cm"`，公式才會以 Computer Modern
呈現、和 A1 的襯線外觀一致——**不要拿掉這行**。matplotlib 的預設值是 `dejavusans`
（無襯線），語法一樣能編譯、圖也畫得出來，但外觀和 A1 完全不同，光看版面結構不會發現。
分式與連加符號的實際高度遠大於字級，`space` 給不夠會壓到下一段文字——產生後一定要
渲染檢查。

報告數值直接從程式的回傳值帶入，不要手抄，才不會改完程式忘了改報告。

## 這個環境的地雷

- **`n_jobs=-1` 會讓 GridSearchCV 崩掉**（loky worker SIGBUS）。省略該參數用單行程，
  這種規模的資料幾秒就跑完。
- **模組檔名以數字開頭不能 import**，測試要用 `importlib.util.spec_from_file_location`
  載入。既有測試檔都有 `load_module()` helper 可抄。
- **中文字型**：粗體 `/System/Library/Fonts/STHeiti Medium.ttc`，
  內文 `/System/Library/Fonts/Supplemental/Arial Unicode.ttf`。
- **matplotlib 會噴一堆 PyparsingDeprecationWarning**，是環境問題不是程式問題，
  跑測試時過濾掉即可，不用去修。
- 虛擬環境在 `.venv/`，用 `.venv/bin/python`。沒有 pytest，用 `python -m unittest`。

## 提交前驗證

這幾項每次都要實際跑過，不要用推論代替：

- [ ] 跑全部三份以上的作業測試，確認沒有回歸（各自 `cd` 到目錄後 `python -m unittest`）
- [ ] 突變檢測：故意改壞關鍵常數與邏輯，確認測試會失敗
- [ ] 解壓繳交 zip 到**乾淨的暫存目錄**直接執行，確認不依賴任何未繳交的檔案
- [ ] 報告 PDF 逐頁用 Read 渲染檢查，特別注意公式重疊與行首標點
- [ ] 比對三份報告內嵌的字型，確認風格真的一致（版面對了不代表字型對了）：

      ```bash
      .venv/bin/python -c "
      import fitz
      for p in ['lecture/02/assignment/Assignment1_陳欣怡_結果報告.pdf', ...]:
          d = fitz.open(p)
          print(p, sorted({f[3].split('+')[-1] for pg in d for f in pg.get_fonts()}))
      "
      ```

      三份都應該出現 `Cmr10` / `Cmmi10` / `Cmsy10`（數學式）與 `STHeitiTC-Medium`（粗體中文）。
      若看到 `DejaVuSans-Oblique`，表示 mathtext 字型設定沒生效。
- [ ] `verification_checklist.md` 寫入實際數值，而非「已完成」

## Commit

在 `main` 上直接 commit 並 push（這是個人課程 repo，歷史都是如此）。
commit message 用中文描述成果與驗證結果，附上實際數字。

已確認過的慣例，不用再問：

- 繳交 zip 與模型 `.pkl` **保留在版控中**，當作繳交紀錄，不要建議加進 `.gitignore`。
- Assignment #1 的報告檔名維持舊格式 `Assignment1_陳欣怡_結果報告.pdf`（已繳交，不補學號前綴）。
  學號前綴規則只適用 #2 之後。
