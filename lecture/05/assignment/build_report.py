"""產生 Assignment #2 的結果報告 PDF（版面沿用 Assignment #1）。

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


REPORT_NAME = "515661055_陳欣怡_Assignment2_結果報告.pdf"


def load_module(filename, name):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(result):
    comparison = result["comparison"]
    predictions = result["predictions"]

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
        "姓名：陳欣怡　　學號：515661055",
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
        "x1：氧化層厚度偏差（Å），目標值為 0",
        "x2：微量金屬污染度（ppm），數值越高越容易失效",
        "y：晶圓狀態，1 = Pass、0 = Fail",
        "wat_test.csv：5 筆待驗收晶圓資料，欄位為 Wafer ID、x1、x2",
    ])
    report.gap(0.022)
    report.subheading("任務需求")
    report.bullets([
        "不使用 scikit-learn，自行實作 sigmoid、BCE 與梯度下降",
        "比較 x1、abs(x1)、x1 ** 2 三種特徵表示",
        "僅以訓練集計算標準化統計值，再套用至待驗收資料",
        "比較不同學習率，繪製 Loss 曲線與決策邊界",
        "以 DataFrame 格式輸出通過機率及 Pass／Fail 結果",
    ])

    # --- 二、數學觀念 ---
    report.page("二、數學觀念")
    report.subheading("1. 特徵工程")
    report.text(
        "x1 的正負號代表偏差方向，但製程風險取決於偏差幅度。"
        "因此比較原始 x1、絕對值 abs(x1) 與平方 x1 ** 2 三種表示。"
    )
    report.gap(0.022)
    report.subheading("2. 標準化（Z-score）")
    report.math(r"$z = \dfrac{x - \mu_{train}}{\sigma_{train}}$", size=14, space=0.058)
    report.text(
        "μ 與 σ 僅由訓練資料計算；待驗收資料只使用相同統計值進行 transform，避免資料洩漏。"
    )
    report.gap(0.022)
    report.subheading("3. Logistic Regression")
    report.math(r"$p = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \dfrac{1}{1 + e^{-z}}$",
                size=14, space=0.058)
    report.text("當 p ≥ 0.5 時預測為 Pass，否則預測為 Fail；p 同時作為晶圓的通過機率。")
    report.gap(0.022)
    report.subheading("4. Binary Cross-Entropy 與梯度下降")
    report.math(r"$\mathrm{BCE} = -\dfrac{1}{n}\sum_{i=1}^{n}"
                r"\left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]$", size=14, space=0.070)
    report.text("每個 epoch 使用全部 15 筆訓練資料計算梯度，更新 weights 與 bias，使 BCE 逐步下降。")
    report.gap(0.016)
    report.subheading("5. 正確率")
    report.math(r"$\mathrm{Accuracy} = \dfrac{n_{correct}}{n_{total}}$", size=14, space=0.058)
    report.text("n_total：訓練集總筆數（15）。")

    # --- 三、訓練與評估流程 ---
    report.page("三、訓練與評估流程")
    report.bullets([
        "1. 讀取 wat_train.csv 與 wat_test.csv，驗證欄位、缺失值與標籤",
        "2. 分別建立 x1、abs(x1)、x1 ** 2 三種特徵資料",
        "3. 僅用 Train 計算 mean／std，將相同統計值套用到 Train 與 Test",
        "4. 將 weights 與 bias 固定初始化為 0，確保結果可重現",
        "5. 分別以 learning rate = 0.01、0.10、0.30 訓練 1,000 epochs",
        "6. 比較訓練正確率、初始 loss、最終 loss 與收斂曲線",
        "7. 選用 abs(x1)、x2 與 learning rate = 0.3 的模型",
        "8. 產生 Loss vs. Epochs 與決策邊界圖",
        "9. 預測 5 筆晶圓的通過機率，輸出 DataFrame、CSV 與文字結果",
    ])
    report.gap(0.03)
    report.subheading("關鍵限制")
    report.text(
        "標準化的平均值與標準差只能用訓練集計算，避免資料洩漏；"
        "權重固定由零初始化且流程中無任何隨機步驟，因此每次執行的結果完全一致。"
    )

    # --- 四、特徵轉換與學習率比較 ---
    report.page("四、特徵轉換與學習率比較")
    report.text("所有模型皆訓練 1,000 epochs；正確率與 loss 均以 15 筆訓練資料計算。")
    report.gap(0.03)
    transformation_labels = {"raw": "x1", "abs": "abs(x1)", "square": "x1 ** 2"}
    rows = [
        (
            transformation_labels[row["Feature Transformation"]],
            f"{row['Learning Rate']:.2f}",
            f"{row['Training Accuracy'] * 100:.2f}%",
            f"{row['Initial Loss']:.6f}",
            f"{row['Final Loss']:.6f}",
        )
        for _, row in comparison.iterrows()
    ]
    report.table(
        ["特徵轉換", "學習率", "訓練正確率", "初始 Loss", "最終 Loss"],
        rows,
        [1.3, 1.0, 1.4, 1.3, 1.3],
    )
    report.subheading("結論")
    report.text(
        "原始 x1 因正、負偏差互相抵銷，最高訓練正確率僅 73.33%。"
        "abs(x1) 與 x1 ** 2 都能表達偏差幅度並達到 100%；"
        "最終選擇物理意義較直觀、且能重現題目參考輸出的 abs(x1)。"
        "learning rate = 0.3 收斂最快且未發散。"
    )

    # --- 五、決策邊界視覺化 ---
    report.page("五、決策邊界視覺化")
    report.text("以最終模型（abs(x1)、x2、learning rate = 0.3）繪製決策邊界。")
    report.gap(0.02)
    report.figure_image(
        ASSIGNMENT_DIR / "assignment2_decision_boundary.png",
        height=0.48,
        caption=(
            "assignment2_decision_boundary.png — 使用 abs(x1) 與 x2 的決策邊界。"
            "藍色區域預測為 Pass、紅色區域預測為 Fail；"
            "15 筆訓練樣本皆位於正確區域，訓練正確率為 100%。"
        ),
    )

    # --- 六、Loss 收斂曲線 ---
    report.page("六、Loss 收斂曲線")
    report.text("九組特徵轉換／學習率組合的 BCE loss 隨 epoch 變化。")
    report.gap(0.02)
    report.figure_image(
        ASSIGNMENT_DIR / "assignment2_loss_curves.png",
        height=0.44,
        caption=(
            "assignment2_loss_curves.png — 原始 x1 最終停留在約 0.498；"
            "abs(x1) 與 x1 ** 2 的 loss 持續下降。學習率 0.3 在本資料上收斂最快。"
        ),
    )
    report.gap(0.016)
    report.subheading("觀察")
    report.text(
        "Loss 曲線可確認訓練是否收斂，但只有訓練資料，無法據此判斷過擬合或泛化能力；"
        "本作業重點為正確實作 Logistic Regression 與資料前處理流程。"
    )

    # --- 七、待驗收結果與總結 ---
    report.page("七、待驗收結果與總結")
    report.text("以最終模型預測 5 筆全新晶圓；分類閾值為 0.5。")
    report.gap(0.03)
    prediction_rows = [
        (
            row["Wafer ID"],
            f"{row['x1']:.1f}",
            f"{row['x2']:.1f}",
            f"{row['abs_x1']:.1f}",
            f"{row['Pass Probability'] * 100:.2f}%",
            row["Result"],
        )
        for _, row in predictions.iterrows()
    ]
    report.table(
        ["Wafer ID", "x1", "x2", "abs_x1", "通過機率", "結果"],
        prediction_rows,
        [1.4, 0.9, 0.9, 1.1, 1.4, 0.9],
    )
    report.subheading("主要發現")
    report.bullets([
        "偏差幅度（abs(x1)）比帶號偏差更具鑑別度，正確率由 73.33% 提升至 100%",
        "Wafer A 與 Wafer E 的偏差幅度小且污染度低，判定為 Pass",
        "Wafer B 偏差幅度達 4.5、Wafer C 污染度達 4.8，皆判定為 Fail",
        "五筆預測機率與題目投影片的參考輸出完全一致",
    ])
    report.gap(0.022)
    report.subheading("總結")
    report.text(
        "本作業從零實作 Logistic Regression（未使用 scikit-learn），"
        "完成資料驗證、特徵工程、標準化、梯度下降訓練、學習率比較、"
        "決策邊界與 loss 曲線視覺化，並輸出 5 筆待驗收晶圓的通過機率與 Pass／Fail 結果。"
    )

    return report


def main():
    assignment = load_module("05_assignment2.py", "assignment2")
    result = assignment.run_assignment(data_dir=ASSIGNMENT_DIR, output_dir=ASSIGNMENT_DIR)

    report = build(result)
    output_path = report.save(ASSIGNMENT_DIR / REPORT_NAME)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"報告已產生：{output_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
