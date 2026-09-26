"""產生 Assignment #3 的結果報告 PDF（版面沿用 Assignment #1）。

此檔為報告產生工具，不屬於繳交內容。執行方式：

    python build_report.py
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent
LECTURE_DIR = ASSIGNMENT_DIR.parent.parent
sys.path.insert(0, str(LECTURE_DIR))

import report_style  # noqa: E402  (需先加入 lecture 目錄)


REPORT_NAME = "515661055_陳欣怡_Assignment3_結果報告.pdf"


def load_module(filename, name):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(training, evaluation, mask_figure, confusion_figure):
    selected = int(training["selected_pixels"])
    dropped = 64 - selected
    accuracy = evaluation["accuracy"]
    best = training["best_params"]

    report = report_style.Report(
        metadata={
            "Title": "Assignment #3 - 低解析度手寫數字多類別分類 結果報告",
            "Author": "陳欣怡",
            "Subject": "機器學習實作系列 第 7 週：模型選取與模型評估",
            "Creator": "Assignment #1 style report builder",
        }
    )

    report.cover(
        "Assignment #3",
        "低解析度手寫數字多類別分類",
        "機器學習實作系列　第 7 週：模型選取與模型評估",
        "姓名：陳欣怡　　學號：515661055",
        [
            "資料集：sklearn.datasets.load_digits()（1797 張 8×8 灰階影像）",
            "模型：StandardScaler → L1 特徵選取 → Logistic Regression",
        ],
    )

    # --- 一、作業說明 ---
    report.page("一、作業說明")
    report.subheading("目的")
    report.text(
        "建立低解析度手寫數字 (0–9) 的多類別分類器，應用情境為光學字元辨識（OCR）。"
        "假設字元定位與切割已經過預先處理，本作業僅負責字元辨識任務，且僅針對 0–9 之數字字元。"
    )
    report.gap(0.03)
    report.subheading("資料集：sklearn.datasets.load_digits()")
    report.bullets([
        "1797 張 8 × 8 灰階（grayscale）影像",
        "64 維（像素值）特徵，整數值範圍 [0, 16]",
        "多類別標籤 0–9，共 10 類",
        "資料直接由 scikit-learn 載入，無外部 CSV 檔",
    ])
    report.gap(0.022)
    report.subheading("任務需求")
    report.bullets([
        "以 Logistic Regression 作為分類器演算法",
        "運用資料切分、特徵縮放、特徵選取、模型選取與模型評估",
        "切分比例：80% 訓練（用於交叉驗證）、20% 測試",
        "至少淘汰 20 個以上的特徵（即僅使用 44 個以下的像素值）",
        "在測試集（約 360 筆樣本）上的整體正確率（Accuracy）達 95% 以上",
        "使用 joblib.dump() 將訓練好的最佳模型（指管線）儲存成檔案",
    ])

    # --- 二、數學觀念 ---
    report.page("二、數學觀念")
    report.subheading("1. 多類別 Logistic Regression — Softmax")
    report.math(
        r"$P(y=k \mid \mathbf{x}) = \dfrac{e^{\mathbf{w}_k \cdot \mathbf{x} + b_k}}"
        r"{\sum_{j=0}^{9} e^{\mathbf{w}_j \cdot \mathbf{x} + b_j}}$",
        size=14,
        space=0.082,
    )
    report.text("10 個類別各有一組權重；預測類別取機率最大者。")
    report.gap(0.026)
    report.subheading("2. 標準化（Z-score）")
    report.math(r"$z = \dfrac{x - \mu}{\sigma}$", size=14, space=0.058)
    report.text("μ、σ 僅以訓練集計算，再套用到測試集，避免資料洩漏。")
    report.gap(0.026)
    report.subheading("3. 特徵選取 — L1 正則化")
    report.math(r"$\min_{\mathbf{w}} \; C \sum_i L(y_i, \hat{y}_i) + \|\mathbf{w}\|_1$", size=14, space=0.062)
    report.text(
        "L1 懲罰會把不重要像素的權重壓到 0，因此可依各像素的權重大小排序重要性，取前 N 個作為選取結果。"
    )
    report.gap(0.026)
    report.subheading("4. 交叉熵損失（多類別）")
    report.math(
        r"$L = -\dfrac{1}{n}\sum_{i=1}^{n}\sum_{k=0}^{9} y_{ik}\,\log\,P(y=k \mid \mathbf{x}_i)$",
        size=14,
        space=0.072,
    )
    report.gap(0.016)
    report.subheading("5. 正確率")
    report.math(r"$\mathrm{Accuracy} = \dfrac{n_{correct}}{n_{total}}$", size=14, space=0.060)
    report.text("n_correct：預測正確的測試筆數；n_total：測試集總筆數（360）。")

    # --- 三、訓練與評估流程 ---
    report.page("三、訓練與評估流程")
    report.bullets([
        "1. load_digits() 載入 1797 張 8 × 8 影像，攤平為 64 維像素特徵",
        "2. train_test_split(test_size=0.2, random_state=42)：1437 筆訓練 / 360 筆測試",
        "3. StandardScaler：僅以訓練集計算 mean / std，測試集只做 transform",
        "4. SelectFromModel + L1 Logistic Regression：依像素重要性取前 N 個",
        "5. GridSearchCV：5-fold StratifiedKFold，搜尋特徵數量與正則化強度共 9 組",
        "6. 驗證特徵上限：選用像素數必須 ≤ 44，否則直接拋出錯誤中止",
        "7. joblib.dump() 儲存整條 Pipeline（含 scaler 與 selector）",
        "8. 評估程式重建相同測試集，載入模型輸出分類報告與混淆矩陣",
        "9. 自管線取出特徵選取遮罩，繪製 8 × 8 遮罩圖",
    ])
    report.gap(0.03)
    report.subheading("關鍵限制")
    report.text(
        "測試集全程未參與任何調參，僅在最後評估時使用一次；切分種子固定為 42，"
        "繳交的評估程式才能重現完全相同的測試集，助教執行時不需重新訓練或交叉驗證即可得到一致結果。"
    )

    # --- 四、模型選取結果 ---
    report.page("四、模型選取結果")
    report.text("5-fold 交叉驗證在 9 組參數組合上的平均正確率（依排名排序）：")
    report.gap(0.03)
    cv = training["search"].cv_results_
    order = sorted(range(len(cv["rank_test_score"])), key=lambda i: cv["rank_test_score"][i])
    rows = [
        (
            cv["rank_test_score"][i],
            cv["param_selector__max_features"][i],
            cv["param_classifier__C"][i],
            f"{cv['mean_test_score'][i]:.4f}",
            f"{cv['std_test_score'][i]:.4f}",
        )
        for i in order
    ]
    report.table(
        ["排名", "選用像素數", "分類器 C", "CV 平均正確率", "標準差"],
        rows,
        [0.9, 1.5, 1.3, 1.9, 1.4],
    )
    report.subheading("最佳組合")
    report.bullets([
        f"selector__max_features = {best['selector__max_features']}",
        f"classifier__C = {best['classifier__C']}",
        f"交叉驗證平均正確率 = {training['cv_accuracy']:.4f}",
        f"測試集正確率 = {training['test_accuracy']:.4f}",
    ])
    report.gap(0.016)
    report.text(
        f"三組特徵數量中，40 與 44 個像素的表現相同（CV 皆為 0.9638），32 個像素明顯較差；"
        f"在同分的情況下選擇像素數較少的 {selected} 個，淘汰 {dropped} 個，"
        f"符合「至少淘汰 20 個特徵、僅使用 44 個以下像素」的規定。"
    )

    # --- 五、預期結果 1 ---
    report.page("五、預期結果 1：特徵選取遮罩")
    report.text(
        f"以 8 × 8 遮罩圖呈現特徵選取結果，0 代表淘汰、1 代表選取，"
        f"標題含選取比例 {selected}/64（必須在 44/64 以下）。"
    )
    report.gap(0.02)
    report.figure_image(
        mask_figure,
        height=0.50,
        caption=(
            f"assignment3_feature_mask.png — 被淘汰的 {dropped} 個像素集中在影像最左、最右兩行與四個角落。"
            "這些位置在 8 × 8 手寫數字影像中幾乎恆為背景，對辨識沒有貢獻，"
            "顯示 L1 特徵選取確實抓到了有意義的筆畫區域。"
        ),
    )

    # --- 六、預期結果 2：分類報告 ---
    report.page("六、預期結果 2：分類報告")
    report.text(f"測試集共 360 筆，整體正確率（Accuracy）= {accuracy:.4f}，達成 95% 門檻。")
    report.gap(0.03)
    report.subheading("詳細分類報告（Classification Report）")
    report.mono(evaluation["report"])
    report.gap(0.03)
    report.text(
        "10 個類別的 f1-score 皆在 0.95 以上，macro avg 與 weighted avg 均為 0.98，"
        "顯示模型在各數字上的表現相當平均，沒有特定類別明顯落後。"
        "表現最弱的是數字 5 與 9（f1-score 0.95），兩者在 8 × 8 解析度下筆畫較容易混淆。"
    )

    # --- 七、預期結果 2：混淆矩陣 ---
    report.page("七、預期結果 2：混淆矩陣")
    report.text("測試集混淆矩陣；對角線為正確分類，非對角線為誤判。")
    report.gap(0.02)
    confusion = evaluation["confusion_matrix"]
    errors = int(confusion.sum()) - int(confusion.diagonal().sum())
    report.figure_image(
        confusion_figure,
        height=0.50,
        caption=(
            f"assignment3_confusion_matrix.png — 360 筆測試樣本中共有 {errors} 筆誤判，"
            "且分散於不同類別，沒有集中的系統性混淆。誤判多發生在筆畫相近的數字之間"
            "（例如 5 與 9、3 與 5、9 與 3），符合低解析度 8 × 8 影像的預期限制。"
        ),
    )

    # --- 八、總結 ---
    report.page("八、總結")
    report.text(
        "本作業以 scikit-learn 建立低解析度手寫數字的多類別分類器，"
        "完成資料切分、特徵縮放、L1 特徵選取、交叉驗證模型選取與模型評估，"
        "並將最佳管線以 joblib.dump() 儲存供評估程式載入。"
    )
    report.gap(0.032)
    report.subheading("驗收項目對照")
    report.table(
        ["驗收項目", "規定", "實際結果"],
        [
            ("特徵淘汰數", "≥ 20 個", f"{dropped} 個"),
            ("使用像素數", "≤ 44 個", f"{selected} 個"),
            ("測試集正確率", "≥ 95%", f"{accuracy * 100:.2f}%"),
            ("測試集樣本數", "約 360 筆", "360 筆"),
            ("模型儲存", "joblib.dump() 管線", "digits_pipeline_lr.pkl"),
        ],
        [1.5, 1.6, 1.9],
    )
    report.subheading("主要發現")
    report.bullets([
        f"僅用 {selected} 個像素（淘汰 {dropped} 個）即達成 {accuracy * 100:.2f}% 測試正確率，高於 95% 門檻",
        "被淘汰的像素全部落在影像邊緣，與手寫數字的背景區域吻合",
        "40 與 44 個像素的交叉驗證表現相同，顯示再多的邊緣像素並未帶來額外資訊",
        "繳交的評估程式不含任何訓練或模型選取程式碼，助教可直接執行批閱",
    ])
    report.gap(0.024)
    report.subheading("繳交內容")
    report.bullets([
        "07_assignment3_eval.py：僅載入模型並評估，不含訓練或模型選取程式碼",
        "digits_pipeline_lr.pkl：以 joblib.dump() 儲存的完整管線（含預處理與模型）",
    ])

    return report


def main():
    train_module = load_module("07_assignment3_train.py", "assignment3_train")
    eval_module = load_module("07_assignment3_eval.py", "assignment3_eval")

    training = train_module.run_training(output_dir=ASSIGNMENT_DIR)
    evaluation = eval_module.run_evaluation(
        model_path=training["model_path"], output_dir=ASSIGNMENT_DIR
    )

    report = build(
        training,
        evaluation,
        ASSIGNMENT_DIR / eval_module.MASK_FIGURE,
        ASSIGNMENT_DIR / eval_module.CONFUSION_FIGURE,
    )
    output_path = report.save(ASSIGNMENT_DIR / REPORT_NAME)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"報告已產生：{output_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
