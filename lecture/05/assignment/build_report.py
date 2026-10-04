"""產生 Assignment #2 的結果報告 PDF（版面沿用 Assignment #1）。

此檔為報告產生工具，不屬於繳交內容。執行方式：

    python build_report.py

內文、項目與表格中的數學符號一律以 mathtext（$...$）書寫，report_style 會把
整段 $...$ 視為不可拆的 token，並以 Computer Modern 字型渲染。
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent
LECTURE_DIR = ASSIGNMENT_DIR.parent.parent
sys.path.insert(0, str(LECTURE_DIR))

import report_style  # noqa: E402  (需先加入 lecture 目錄)


REPORT_NAME = "515661055_陳欣怡_Assignment2_結果報告.pdf"
STUDY_NAME = "assignment2_improvement_study.json"

TRANSFORMATION_LABELS = {"raw": "$x_1$", "abs": "$|x_1|$", "square": "$x_1^{2}$"}


def load_module(filename, name):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pct(value):
    return f"{value * 100:.2f}%"


def build(result, study):
    comparison = result["comparison"]
    predictions = result["predictions"]
    experiments = study["experiments"]

    def rows_of(group):
        return [row for row in experiments if row["group"] == group]

    def row_of(group, **match):
        for row in rows_of(group):
            if all(row["config"][key] == value for key, value in match.items()):
                return row
        raise KeyError(f"{group}: {match}")

    def mark(row):
        return " ←" if row["baseline"] else ""

    baseline = row_of("l2", l2=0.0)

    report = report_style.Report(
        metadata={
            "Title": "Assignment #2 - Logistic Regression 晶圓驗收分類 結果報告",
            "Author": "陳欣怡",
            "Subject": "機器學習實作系列 第 5 週：Logistic Regression",
            "Creator": "Assignment #1 style report builder",
        }
    )

    report.cover(
        "Assignment #2",
        "Logistic Regression 晶圓驗收分類",
        "機器學習實作系列　第 5 週：Logistic Regression",
        ["課程教授：陳慶永　　課程助教：翁宣允", "姓名：陳欣怡　　學號：515661055"],
        [
            "資料集：wat_train.csv（15 筆）／ wat_test.csv（5 筆）",
            "模型：自行實作 Logistic Regression（未使用 scikit-learn）",
        ],
    )

    # --- 一、作業說明 ---
    report.page("一、作業說明")
    report.subheading("目的")
    report.text(
        "使用自行實作的 Logistic Regression 分類模型，依氧化層厚度偏差與微量金屬污染度，"
        "預測晶圓在驗收測試中為「合格（Pass）」或「不合格（Fail）」，並輸出每筆晶圓的通過機率。"
    )
    report.gap(0.03)
    report.subheading("資料集：wat_train.csv / wat_test.csv")
    report.bullets([
        "wat_train.csv：15 筆歷史晶圓資料",
        "$x_1$：氧化層厚度偏差（Å），目標值為 0",
        "$x_2$：微量金屬污染度（ppm），數值越高越容易失效",
        "$y$：晶圓狀態，1 = Pass、0 = Fail",
        "wat_test.csv：5 筆待驗收晶圓資料，欄位為 Wafer ID、$x_1$、$x_2$",
    ])
    report.gap(0.022)
    report.subheading("任務需求")
    report.bullets([
        "不使用 scikit-learn，自行實作 sigmoid、binary cross-entropy 與梯度下降",
        "比較 $x_1$、$|x_1|$、$x_1^{2}$ 三種特徵表示",
        "僅以訓練集計算標準化統計值，再套用至待驗收資料",
        "比較不同學習率 $\\eta$，繪製 Loss 曲線與決策邊界",
        "以 DataFrame 格式輸出通過機率及 Pass／Fail 結果",
    ])

    # --- 二、數學觀念 ---
    report.page("二、數學觀念")
    report.subheading("1. 特徵工程", level=2)
    report.text(
        "$x_1$ 的正負號代表偏差方向，但製程風險取決於偏差幅度，"
        "因此比較三種特徵表示 $\\phi(x_1)$，並與 $x_2$ 組成特徵向量 $\\mathbf{x}$："
    )
    report.math(r"$\phi(x_1) \in \{\, x_1,\ |x_1|,\ x_1^{2} \,\}, \qquad "
                r"\mathbf{x} = \left(\phi(x_1),\, x_2\right)$", size=14, space=0.058)
    report.subheading("2. 標準化（Z-score）", level=2)
    report.math(r"$z = \dfrac{x - \mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}$",
                size=14, space=0.058)
    report.text(
        "$\\mu_{\\mathrm{train}}$ 與 $\\sigma_{\\mathrm{train}}$ 僅由訓練資料計算；"
        "待驗收資料只套用相同統計值，避免資料洩漏。"
    )
    report.gap(0.014)
    report.subheading("3. Logistic Regression", level=2)
    report.math(r"$p = \sigma(z) = \dfrac{1}{1 + e^{-z}}, \qquad "
                r"z = \mathbf{w}^{\top}\mathbf{x} + b$", size=14, space=0.058)
    report.text("當 $p \\geq 0.5$ 時預測為 Pass，否則預測為 Fail；$p$ 同時作為晶圓的通過機率。")
    report.gap(0.014)
    report.subheading("4. Binary Cross-Entropy 與梯度下降", level=2)
    report.math(r"$J(\mathbf{w}, b) = -\dfrac{1}{n}\sum_{i=1}^{n}"
                r"\left[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \,\right]$", size=14, space=0.068)
    report.math(r"$\mathbf{w} \leftarrow \mathbf{w} - \eta\,\dfrac{1}{n}\sum_{i=1}^{n}"
                r"(p_i - y_i)\,\mathbf{x}_i, \qquad "
                r"b \leftarrow b - \eta\,\dfrac{1}{n}\sum_{i=1}^{n}(p_i - y_i)$",
                size=14, space=0.068)
    report.text(
        "每個 epoch 使用全部 $n = 15$ 筆訓練資料計算梯度（full-batch），"
        "以學習率 $\\eta$ 更新 $\\mathbf{w}$ 與 $b$，使 $J$ 下降。"
    )
    report.gap(0.014)
    report.subheading("5. 正確率", level=2)
    report.math(r"$\mathrm{Accuracy} = \dfrac{n_{\mathrm{correct}}}{n_{\mathrm{total}}}$",
                size=14, space=0.056)
    report.text("$n_{\\mathrm{total}}$：訓練集總筆數（15）。")

    # --- 三、訓練與評估流程 ---
    report.page("三、訓練與評估流程")
    report.bullets([
        "1. 讀取 wat_train.csv 與 wat_test.csv，驗證欄位、缺失值與標籤",
        "2. 分別建立 $x_1$、$|x_1|$、$x_1^{2}$ 三種特徵資料",
        "3. 僅用 Train 計算 $\\mu$／$\\sigma$，將相同統計值套用到 Train 與 Test",
        "4. 將 $\\mathbf{w}$ 與 $b$ 固定初始化為 0，確保結果可重現",
        "5. 分別以 $\\eta = 0.01$、$0.10$、$0.30$ 訓練 1,000 epochs",
        "6. 比較訓練正確率、初始 loss、最終 loss 與收斂曲線",
        "7. 選用 $|x_1|$、$x_2$ 與 $\\eta = 0.3$ 的模型",
        "8. 產生 Loss vs. Epochs 與決策邊界圖",
        "9. 預測 5 筆晶圓的通過機率，輸出 DataFrame、CSV 與文字結果",
    ])
    report.gap(0.03)
    report.subheading("關鍵限制", level=2)
    report.text(
        "標準化的平均值與標準差只能用訓練集計算，避免資料洩漏；"
        "權重固定由零初始化且流程中無任何隨機步驟，因此每次執行的結果完全一致。"
    )

    # --- 四、特徵轉換與學習率比較 ---
    report.page("四、特徵轉換與學習率比較")
    report.text("所有模型皆訓練 1,000 epochs；正確率與 loss 均以 15 筆訓練資料計算。")
    report.gap(0.03)
    rows = [
        (
            TRANSFORMATION_LABELS[row["Feature Transformation"]],
            f"{row['Learning Rate']:.2f}",
            _pct(row["Training Accuracy"]),
            f"{row['Initial Loss']:.6f}",
            f"{row['Final Loss']:.6f}",
        )
        for _, row in comparison.iterrows()
    ]
    report.table(
        ["特徵轉換", "學習率 $\\eta$", "訓練正確率", "初始 Loss", "最終 Loss"],
        rows,
        [1.3, 1.0, 1.4, 1.3, 1.3],
    )
    best_raw = comparison[comparison["Feature Transformation"] == "raw"]["Training Accuracy"].max()
    report.subheading("結論", level=2)
    report.text(
        f"原始 $x_1$ 因正、負偏差互相抵銷，最高訓練正確率僅 {_pct(best_raw)}。"
        "$|x_1|$ 與 $x_1^{2}$ 都能表達偏差幅度並達到 100%；"
        "最終選擇物理意義較直觀、且能重現題目參考輸出的 $|x_1|$。"
        "$\\eta = 0.3$ 收斂最快且未發散。"
    )

    # --- 五、決策邊界視覺化 ---
    report.page("五、決策邊界視覺化")
    report.text(
        "以最終模型（$|x_1|$、$x_2$、$\\eta = 0.3$）繪製決策邊界。"
        "上圖比照題目預期結果，畫在標準化座標上；下圖換回原始單位，便於對照製程數值。"
    )
    report.gap(0.012)
    report.figure_image(
        ASSIGNMENT_DIR / "assignment2_decision_boundary_std.png",
        height=0.31,
        caption=(
            "assignment2_decision_boundary_std.png — 標準化座標上的決策區域；"
            "Class 0 為 Fail、Class 1 為 Pass，15 筆訓練樣本皆位於正確區域。"
        ),
    )
    report.gap(0.006)
    report.figure_image(
        ASSIGNMENT_DIR / "assignment2_decision_boundary.png",
        height=0.31,
        caption=(
            "assignment2_decision_boundary.png — 同一模型在 $|x_1|$（Å）與 $x_2$（ppm）原始單位上的決策邊界，"
            "訓練正確率 100%。"
        ),
    )

    # --- 六、Loss 收斂曲線 ---
    selected_final = comparison[
        (comparison["Feature Transformation"] == "abs") & (comparison["Learning Rate"] == 0.3)
    ]["Final Loss"].iloc[0]
    raw_final = comparison[comparison["Feature Transformation"] == "raw"]["Final Loss"].min()
    report.page("六、Loss 收斂曲線")
    report.text(
        "上圖為最終模型的 binary cross-entropy 隨 epoch 變化（比照題目預期結果）；"
        "下圖疊上九組特徵轉換／學習率組合，作為第四節比較的輔助。"
    )
    report.gap(0.012)
    report.figure_image(
        ASSIGNMENT_DIR / "assignment2_loss_curve_selected.png",
        height=0.31,
        caption=(
            f"assignment2_loss_curve_selected.png — 最終模型由 0.693 收斂至 {selected_final:.4f}，"
            "前 100 個 epoch 下降最快，之後平緩。"
        ),
    )
    report.gap(0.006)
    report.figure_image(
        ASSIGNMENT_DIR / "assignment2_loss_curves.png",
        height=0.31,
        caption=(
            f"assignment2_loss_curves.png — 原始 $x_1$ 最終停留在約 {raw_final:.3f}；"
            "$|x_1|$ 與 $x_1^{2}$ 持續下降，$\\eta = 0.3$ 收斂最快。"
            "Loss 只能確認收斂，無法判斷泛化能力，第八、九節以留一交叉驗證補做檢驗。"
        ),
    )

    # --- 七、待驗收結果 ---
    report.page("七、待驗收結果")
    report.text("以最終模型預測 5 筆全新晶圓；分類閾值為 $p \\geq 0.5$。")
    report.gap(0.03)
    prediction_rows = [
        (
            row["Wafer ID"],
            f"{row['x1']:.1f}",
            f"{row['x2']:.1f}",
            f"{row['abs_x1']:.1f}",
            _pct(row["Pass Probability"]),
            row["Result"],
        )
        for _, row in predictions.iterrows()
    ]
    report.table(
        ["Wafer ID", "$x_1$", "$x_2$", "$|x_1|$", "通過機率 $p$", "結果"],
        prediction_rows,
        [1.4, 0.9, 0.9, 1.1, 1.4, 0.9],
    )
    report.subheading("主要發現")
    report.bullets([
        f"偏差幅度 $|x_1|$ 比帶號偏差更具鑑別度，訓練正確率由 {_pct(best_raw)} 提升至 100%",
        "Wafer A 與 Wafer E 的偏差幅度小且污染度低，判定為 Pass",
        "Wafer B 偏差幅度達 4.5、Wafer C 污染度達 4.8，皆判定為 Fail",
        "五筆預測機率與題目投影片的參考輸出完全一致",
    ])

    # --- 八、改善空間討論（一）---
    epochs_rows = rows_of("epochs")
    l2_rows = rows_of("l2")
    first_epoch, last_epoch = epochs_rows[0], epochs_rows[-1]
    best_loo_epoch = min(epochs_rows, key=lambda row: row["loocv_log_loss"])
    l2_small = row_of("l2", l2=0.1)
    l2_large = l2_rows[-1]

    report.page("八、改善空間討論（一）：訓練輪次與正則化")
    report.text(
        f"訓練集僅 {study['n_train']} 筆、待驗收資料沒有標籤，無法另外切出驗證集。"
        "以下以留一交叉驗證（LOO）估計泛化能力：每折以 14 筆重新計算"
        "標準化統計值並訓練，預測被留下的 1 筆，15 折合計。待驗收晶圓的機率只作觀察，"
        "不用於挑選設定。標示 ← 者為繳交版本。"
    )
    report.gap(0.022)
    report.subheading("訓練輪次：loss 收斂不等於模型變好", level=2)
    report.table(
        ["Epochs", "訓練正確率", "最終 Loss", "$\\|\\mathbf{w}\\|_2$",
         "LOO 正確率", "LOO log-loss", "Wafer D $p$"],
        [
            (row["label"] + mark(row), _pct(row["training_accuracy"]), f"{row['final_loss']:.4f}",
             f"{row['weight_norm']:.2f}", _pct(row["loocv_accuracy"]),
             f"{row['loocv_log_loss']:.4f}", _pct(row["probabilities"]["Wafer D"]))
            for row in epochs_rows
        ],
        [1.0, 1.15, 1.05, 0.9, 1.15, 1.2, 1.15],
        size=9.5,
    )
    report.text(
        "訓練資料在 $|x_1|$–$x_2$ 平面上線性可分，binary cross-entropy 沒有有限的最小值："
        f"epochs 越多，loss 持續下降、$\\|\\mathbf{{w}}\\|_2$ 由 {first_epoch['weight_norm']:.2f} "
        f"增至 {last_epoch['weight_norm']:.2f}，但 LOO 正確率始終是 {_pct(baseline['loocv_accuracy'])}；"
        f"LOO log-loss 在 {best_loo_epoch['label']} epochs 最低（{best_loo_epoch['loocv_log_loss']:.3f}），"
        f"之後反而升到 {last_epoch['loocv_log_loss']:.3f}，Wafer D 的通過機率也從 "
        f"{_pct(first_epoch['probabilities']['Wafer D'])} 一路壓到 "
        f"{_pct(last_epoch['probabilities']['Wafer D'])}。"
        "這正是題目投影片提醒的情況：loss 收斂只代表權重還在變大、模型對自己越來越有信心，"
        "不代表分類能力提升。"
    )
    report.gap(0.022)
    report.subheading("L2 正則化：限制權重大小", level=2)
    report.math(r"$J_{\lambda}(\mathbf{w}, b) = J(\mathbf{w}, b) + "
                r"\dfrac{\lambda}{2n}\,\|\mathbf{w}\|_2^{2}$", size=13, space=0.056)
    report.table(
        ["$\\lambda$", "訓練正確率", "最終 Loss", "$\\|\\mathbf{w}\\|_2$",
         "LOO 正確率", "LOO log-loss", "Wafer D $p$"],
        [
            (f"{row['config']['l2']:g}" + mark(row), _pct(row["training_accuracy"]),
             f"{row['final_loss']:.4f}", f"{row['weight_norm']:.2f}",
             _pct(row["loocv_accuracy"]), f"{row['loocv_log_loss']:.4f}",
             _pct(row["probabilities"]["Wafer D"]))
            for row in l2_rows
        ],
        [1.0, 1.15, 1.05, 0.9, 1.15, 1.2, 1.15],
        size=9.5,
    )
    report.text(
        f"$\\lambda = 0.1$ 把 $\\|\\mathbf{{w}}\\|_2$ 壓回 {l2_small['weight_norm']:.2f}，"
        f"LOO log-loss 由 {baseline['loocv_log_loss']:.3f} 降至 {l2_small['loocv_log_loss']:.3f}，"
        f"LOO 正確率不變；$\\lambda \\geq 1$ 開始欠擬合，$\\lambda = {l2_large['config']['l2']:g}$ "
        f"連訓練正確率都掉到 {_pct(l2_large['training_accuracy'])}。"
    )

    # --- 九、改善空間討論（二）---
    feature_rows = rows_of("feature_set")
    square = row_of("feature_set", transformation="square")
    raw = row_of("feature_set", transformation="raw")
    misclassified = baseline["loocv_misclassified"]

    report.page("九、改善空間討論（二）：特徵組合與最終取捨")
    report.subheading("特徵組合（皆為 $\\eta = 0.3$、1,000 epochs、$\\lambda = 0$）", level=2)
    report.table(
        ["特徵組合", "訓練正確率", "LOO 正確率", "LOO log-loss", "Wafer B $p$", "Wafer D $p$"],
        [
            (row["label"] + mark(row), _pct(row["training_accuracy"]), _pct(row["loocv_accuracy"]),
             f"{row['loocv_log_loss']:.4f}", _pct(row["probabilities"]["Wafer B"]),
             _pct(row["probabilities"]["Wafer D"]))
            for row in feature_rows
        ],
        [1.7, 1.15, 1.15, 1.2, 1.15, 1.15],
        size=9.5,
    )
    sample_text = "、".join(
        f"$({m['x1']:g},\\ {m['x2']:g})$（實際 {'Pass' if m['y'] else 'Fail'}，LOO 機率 {_pct(m['probability'])}）"
        for m in misclassified
    )
    report.text(
        f"除原始 $x_1$（LOO 正確率 {_pct(raw['loocv_accuracy'])}）外，三種組合的 LOO 正確率都是 "
        f"{_pct(baseline['loocv_accuracy'])}，且誤判的始終是同 {len(misclassified)} 筆："
        f"{sample_text}，即第五節決策邊界圖中最貼近邊界的兩點；任何設定都無法同時把這兩筆分對。"
        f"$x_1^{{2}}$ 的 LOO log-loss 最低（{square['loocv_log_loss']:.3f}），但它把 Wafer D 的機率從 "
        f"{_pct(baseline['probabilities']['Wafer D'])} 推到 {_pct(square['probabilities']['Wafer D'])}、"
        f"Wafer B 從 {_pct(baseline['probabilities']['Wafer B'])} 壓到 {_pct(square['probabilities']['Wafer B'])}："
        "五筆分類結果不變，但置信度差異明顯；並列 $|x_1|$ 與 $x_1^{2}$ 則無額外好處。"
    )
    report.gap(0.022)
    report.subheading("為何維持繳交版本的設定", level=2)
    report.bullets([
        "所有調整都沒有改變 LOO 正確率；改變的只有置信度（log-loss）",
        f"log-loss 的差距（{best_loo_epoch['loocv_log_loss']:.2f}–{baseline['loocv_log_loss']:.2f}）"
        "在 15 筆資料下約等於一筆樣本的機率變動，不足以據此換設定",
        "以 LOO 為準最值得採用的是 $\\lambda = 0.1$ 或 $x_1^{2}$，但兩者都會使五筆機率偏離題目的參考輸出",
        "本作業的目的是正確實作流程並重現參考結果，故維持 $|x_1|$、$\\eta = 0.3$、1,000 epochs、$\\lambda = 0$",
        f"若用於實務，建議加入 $\\lambda \\approx 0.1$ 或提早停止（約 {best_loo_epoch['label']} epochs），"
        "避免輸出過度自信的機率",
    ])

    # --- 十、總結 ---
    report.page("十、總結")
    report.text(
        "本作業從零實作 Logistic Regression（未使用 scikit-learn），"
        "完成資料驗證、特徵工程、標準化、梯度下降訓練、學習率比較、"
        "決策邊界與 loss 曲線視覺化，並輸出 5 筆待驗收晶圓的通過機率與 Pass／Fail 結果。"
    )
    report.gap(0.032)
    report.subheading("驗收項目對照")
    report.table(
        ["驗收項目", "規定", "實際結果"],
        [
            ("函式庫限制", "不使用 scikit-learn", "sigmoid、BCE、GD 皆手刻"),
            ("特徵工程", "考慮 $|x_1|$ 或 $x_1^{2}$", "三種表示皆比較，採用 $|x_1|$"),
            ("特徵縮放", "統計值只能來自訓練集", "Standardizer 僅 fit 訓練集"),
            ("學習率", "嘗試不同學習率", "$\\eta = 0.01$、$0.1$、$0.3$ 共九組"),
            ("預期結果 1", "Loss 曲線與決策邊界圖", "兩張圖皆已產生"),
            ("預期結果 2", "DataFrame 輸出機率與結果", "五筆與投影片範例完全一致"),
        ],
        [1.0, 1.6, 1.9],
    )
    report.subheading("主要發現")
    report.bullets([
        f"偏差幅度 $|x_1|$ 是關鍵特徵：訓練正確率 {_pct(best_raw)} → 100%，LOO 正確率 "
        f"{_pct(raw['loocv_accuracy'])} → {_pct(baseline['loocv_accuracy'])}",
        "資料線性可分時 loss 會無限下降，epochs 越多只是權重越大、機率越極端（第八節）",
        "L2 正則化或提早停止能改善置信度，但 LOO 正確率不變，故繳交版本維持原設定（第九節）",
        "五筆待驗收晶圓的通過機率與分類結果與題目參考輸出完全一致",
    ])
    report.gap(0.024)
    report.subheading("繳交內容")
    report.bullets([
        "05_assignment2.py：資料驗證、特徵工程、標準化、模型訓練、比較、預測與繪圖",
        "wat_train.csv、wat_test.csv：題目提供的 CSV 資料檔",
        "執行 05_assignment2.py 會自動產生兩張圖、assignment2_predictions.csv 與 assignment2_results.txt",
    ])

    return report


def main():
    assignment = load_module("05_assignment2.py", "assignment2")
    result = assignment.run_assignment(data_dir=ASSIGNMENT_DIR, output_dir=ASSIGNMENT_DIR)

    study_path = ASSIGNMENT_DIR / STUDY_NAME
    if study_path.exists():
        study = json.loads(study_path.read_text(encoding="utf-8"))
    else:
        explore = load_module("05_assignment2_explore.py", "assignment2_explore")
        study = explore.run_study(data_dir=ASSIGNMENT_DIR, output_dir=ASSIGNMENT_DIR)

    report = build(result, study)
    output_path = report.save(ASSIGNMENT_DIR / REPORT_NAME)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"報告已產生：{output_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
