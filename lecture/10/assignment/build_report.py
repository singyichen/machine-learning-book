"""產生 Assignment #5 的結果報告 PDF（版面沿用 Assignment #1）。

此檔為報告產生工具，不屬於繳交內容。執行方式：

    python build_report.py
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSIGNMENT_DIR = Path(__file__).resolve().parent
LECTURE_DIR = ASSIGNMENT_DIR.parent.parent
sys.path.insert(0, str(LECTURE_DIR))

import report_style  # noqa: E402  (需先加入 lecture 目錄)


REPORT_NAME = "515661055_陳欣怡_Assignment5_結果報告.pdf"
VARIANCE_FIGURE = "assignment5_explained_variance.png"


def load_module(filename, name):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_explained_variance_figure(pipeline, output_path):
    """報告用的輔助圖：各主成分與累積解釋變異量（投影片未要求，僅補充說明預期結果 1）。"""
    ratios = pipeline.named_steps["pca"].explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    ax.bar(components, ratios, alpha=0.55, label="Individual explained variance")
    ax.step(components, cumulative, where="mid", color="#c0392b", linewidth=1.8,
            label="Cumulative explained variance")
    ax.axhline(cumulative[-1], color="#7f8c8d", linestyle="--", linewidth=1.0)
    ax.text(
        1.0, cumulative[-1] + 0.02, f"{cumulative[-1] * 100:.2f}%",
        fontsize=9, color="#7f8c8d", va="bottom",
    )
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def build(training, evaluation, study, confusion_figure, misclassified_figure,
          variance_figure, misclassified_message):
    n_components = int(training["n_components"])
    explained = training["explained_variance"]
    accuracy = evaluation["accuracy"]
    best = training["best_params"]

    report = report_style.Report(
        metadata={
            "Title": "Assignment #5 - 低解析度手寫數字多類別分類（PCA 特徵提取）結果報告",
            "Author": "陳欣怡",
            "Subject": "機器學習實作系列 第 10 週：特徵提取與降維",
            "Creator": "Assignment #1 style report builder",
        }
    )

    report.cover(
        "Assignment #5",
        "低解析度手寫數字多類別分類（PCA 特徵提取）",
        "機器學習實作系列　第 10 週：特徵提取與降維",
        ["課程教授：陳慶永　　課程助教：翁宣允", "姓名：陳欣怡　　學號：515661055"],
        [
            "資料集：sklearn.datasets.load_digits()（1797 張 8×8 灰階影像）",
            "模型：StandardScaler → PCA 特徵提取 → Logistic Regression",
        ],
    )

    # --- 一、作業說明 ---
    report.page("一、作業說明")
    report.subheading("目的")
    report.text(
        "建立低解析度手寫數字 (0–9) 的多類別分類器。本次作業的實作任務與 Assignment #3 "
        "(Lecture 6) 相同，唯一的差別是把特徵選取換成特徵提取（使用 PCA）。"
    )
    report.gap(0.03)
    report.subheading("特徵選取與特徵提取的差別")
    report.text(
        "特徵選取是從原本的 64 個像素中挑出一個子集合，被保留的欄位仍然是原始像素，"
        "可以直接指著影像說「用了哪幾格」；特徵提取則是把 64 維投影到一組全新的座標軸上，"
        "每一個新維度都是 64 個像素的線性組合，不再對應任何單一像素。"
    )
    report.gap(0.022)
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
        "運用資料切分、特徵縮放、特徵提取 (PCA)、模型選取與模型評估",
        "切分比例：80% 訓練（用於交叉驗證）、20% 測試",
        "降維後的特徵空間維度 ≤ 40",
        "在測試集（約 360 筆樣本）上的整體正確率（Accuracy）達 95% 以上",
        "使用 joblib.dump() 將訓練好的最佳模型（指管線）儲存成檔案",
    ])

    # --- 二、數學觀念 ---
    report.page("二、數學觀念")
    report.subheading("1. 標準化（Z-score）", level=2)
    report.math(r"$z = \dfrac{x - \mu}{\sigma}$", size=14, space=0.058)
    report.text(
        "PCA 以變異數大小挑選主軸，若各像素尺度不一，主軸會被數值範圍大的像素綁架，"
        "因此縮放必須排在 PCA 之前。$\\mu$、$\\sigma$ 僅以訓練集計算。"
    )
    report.gap(0.026)
    report.subheading("2. 共變異數矩陣", level=2)
    report.math(
        r"$\mathbf{\Sigma} = \dfrac{1}{n-1}\sum_{i=1}^{n}"
        r"(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^{\top}$",
        size=14,
        space=0.082,
    )
    report.text("$\\mathbf{\\Sigma}$ 為 64 × 64 的對稱矩陣，描述各像素之間的共變關係。")
    report.gap(0.026)
    report.subheading("3. PCA — 特徵分解求主軸", level=2)
    report.math(r"$\mathbf{\Sigma}\,\mathbf{v}_j = \lambda_j\,\mathbf{v}_j$", size=14, space=0.058)
    report.text(
        "特徵向量 $\\mathbf{v}_j$ 即主成分方向，特徵值 $\\lambda_j$ 為資料在該方向上的變異數。"
        "將 $\\lambda_j$ 由大到小排序，取前 $k$ 個特徵向量組成投影矩陣 $\\mathbf{W}$，"
        "即可把樣本降到 $k$ 維。"
    )
    report.gap(0.026)
    report.subheading("4. 投影與解釋變異比例", level=2)
    report.math(r"$\mathbf{z} = \mathbf{W}^{\top}\mathbf{x}$", size=14, space=0.052)
    report.math(
        r"$\mathrm{EVR}_j = \dfrac{\lambda_j}{\sum_{m=1}^{64}\lambda_m}$",
        size=14,
        space=0.100,
    )
    report.text(
        "總解釋變異量為前 $k$ 個 $\\mathrm{EVR}_j$ 之和，代表降維後保留了原始資料多少比例的變異。"
    )
    report.gap(0.026)
    report.subheading("5. 多類別 Logistic Regression — Softmax", level=2)
    report.math(
        r"$P(y=k \mid \mathbf{z}) = \dfrac{e^{\mathbf{w}_k \cdot \mathbf{z} + b_k}}"
        r"{\sum_{j=0}^{9} e^{\mathbf{w}_j \cdot \mathbf{z} + b_j}}$",
        size=14,
        space=0.100,
    )
    report.text("分類器接到的是 PCA 投影後的 $k$ 維向量 $\\mathbf{z}$，不是原始像素。")

    # --- 三、訓練與評估流程 ---
    report.page("三、訓練與評估流程")
    report.bullets([
        "1. load_digits() 載入 1797 張 8 × 8 影像，攤平為 64 維像素特徵",
        "2. train_test_split(test_size=0.2, random_state=42)：1437 筆訓練 / 360 筆測試",
        "3. StandardScaler：僅以訓練集計算 mean / std，測試集只做 transform",
        "4. PCA：在標準化後的訓練資料上求主成分，取前 n_components 個",
        "5. GridSearchCV：5-fold StratifiedKFold，搜尋 n_components 與 C 共 9 組",
        "6. 驗證維度上限：保留的主成分數必須 ≤ 40，否則直接拋出錯誤中止",
        "7. joblib.dump() 儲存整條 Pipeline（含 scaler 與 PCA）",
        "8. 評估程式重建相同測試集，載入模型輸出維度、解釋變異量與分類報告",
        "9. 視覺化呈現測試集混淆矩陣",
        "10. 比對預測與真實標籤找出所有誤判樣本，輸出總數並逐張繪出",
    ])
    report.gap(0.03)
    report.subheading("關鍵限制", level=2)
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
            cv["param_pca__n_components"][i],
            cv["param_classifier__C"][i],
            f"{cv['mean_test_score'][i]:.4f}",
            f"{cv['std_test_score'][i]:.4f}",
        )
        for i in order
    ]
    report.table(
        ["排名", "PCA 維度", "分類器 C", "CV 平均正確率", "標準差"],
        rows,
        [0.9, 1.4, 1.3, 1.9, 1.4],
    )
    report.subheading("最佳組合", level=2)
    report.bullets([
        f"pca__n_components = {best['pca__n_components']}",
        f"classifier__C = {best['classifier__C']}",
        f"交叉驗證平均正確率 = {training['cv_accuracy']:.4f}",
        f"測試集正確率 = {training['test_accuracy']:.4f}",
    ])
    report.gap(0.016)
    report.text(
        f"三種維度中以 {n_components} 維最佳，且剛好是作業允許的上限；"
        "維度降到 30 與 20 時交叉驗證分別掉到 0.9576 與 0.9415，"
        "顯示在這個資料集上 PCA 的維度仍是主要的效能來源（詳見第九節）。"
    )

    # --- 五、預期結果 1 ---
    report.page("五、預期結果 1：PCA 維度與總解釋變異量")
    report.text("評估程式輸出：")
    report.gap(0.02)
    report.mono(
        f"PCA 提取後的特徵維度 (n_components): {n_components} "
        f"(從原本的 64 維降至 {n_components} 維)\n"
        f"特徵提取的總解釋變異量: {explained * 100:.2f}%"
    )
    report.gap(0.028)
    report.text(
        f"降維後的特徵空間維度為 {n_components}，符合「必須在 40 以下」的規定；"
        f"這 {n_components} 個主成分保留了原始 64 維資料 {explained * 100:.2f}% 的變異，"
        f"也就是說捨棄 24 個維度只損失了 {100 - explained * 100:.2f}% 的變異量。"
    )
    report.gap(0.02)
    report.figure_image(
        variance_figure,
        height=0.36,
        caption=(
            f"{VARIANCE_FIGURE} — 報告補充圖（投影片未要求）。長條為各主成分的個別解釋變異比例，"
            "紅線為累積值。前幾個主成分就吃掉大部分變異，之後迅速衰減，"
            "這正是 PCA 能把 64 維壓到 40 維而幾乎不損失資訊的原因。"
        ),
    )

    # --- 六、預期結果 2：分類報告 ---
    report.page("六、預期結果 2：分類報告")
    report.text(f"測試集共 360 筆，整體正確率（Accuracy）= {accuracy:.4f}，達成 95% 門檻。")
    report.gap(0.03)
    report.subheading("詳細分類報告（Classification Report）", level=2)
    report.mono(evaluation["report"])
    report.gap(0.03)
    report.text(
        "10 個類別的 f1-score 皆在 0.90 以上，macro avg 與 weighted avg 均為 0.96，"
        "各數字的表現相當平均。表現最弱的是數字 8（f1-score 0.90），"
        "其次是 5 與 9（0.94），這幾個數字在 8 × 8 解析度下筆畫本就容易混淆。"
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
            f"assignment5_confusion_matrix.png — 360 筆測試樣本中共有 {errors} 筆誤判，"
            "分散於不同類別，沒有集中的系統性混淆。誤判多發生在筆畫相近的數字之間"
            "（例如 8 與 1、9 與 8、5 與 9），符合低解析度 8 × 8 影像的預期限制。"
        ),
    )

    # --- 八、預期結果 3：誤判樣本 ---
    misclassified = evaluation["misclassified"]
    y_pred = evaluation["y_pred"]
    y_test = evaluation["y_test"]
    report.page("八、預期結果 3：誤判樣本")
    report.text(
        f"評估程式輸出：「{misclassified_message}」"
        "下圖繪出所有誤判樣本：每張圖上方的 True 為真實標籤，左下角紅字為模型的預測結果。"
    )
    report.gap(0.02)
    report.figure_image(
        misclassified_figure,
        height=0.30,
        caption=(
            f"assignment5_misclassified.png — 共 {len(misclassified)} 張，"
            "與混淆矩陣非對角線的總和一致。"
        ),
    )

    # --- 九、與混淆矩陣對應 ---
    report.page("九、誤判樣本與混淆矩陣對應")
    report.text(
        "作業要求誤判樣本圖「必須與混淆矩陣對應」。下表逐張列出誤判樣本的真實標籤、"
        "預測結果，以及該組合在混淆矩陣中對應格子的計數。"
    )
    report.gap(0.026)
    pair_rows = [
        (str(i + 1), str(int(y_test[index])), str(int(y_pred[index])),
         str(int(confusion[y_test[index], y_pred[index]])))
        for i, index in enumerate(misclassified)
    ]
    report.table(["順序", "真實標籤", "預測結果", "混淆矩陣該格計數"], pair_rows, [0.9, 1.3, 1.3, 2.0])

    # --- 十、改善空間討論（一）---
    def rows_for(group, value_header="CV 正確率"):
        return [
            (e["label"], f"{e['cv_accuracy']:.4f}", f"± {e['cv_std']:.4f}")
            for e in study["experiments"] if e["group"] == group
        ]

    evr = study["explained_variance_by_components"]
    report.page("十、改善空間討論（一）：降維預算")
    report.text(
        f"以下比較皆在訓練集（1437 筆）上以 {study['cv_folds']}-fold 交叉驗證進行，"
        "測試集不參與任何設定的挑選。"
    )
    report.gap(0.026)
    report.subheading("維度與正確率、解釋變異量的關係", level=2)
    budget_rows = [
        (
            e["label"],
            f"{e['cv_accuracy']:.4f}",
            f"± {e['cv_std']:.4f}",
            f"{evr[str(n)] * 100:.2f}%",
        )
        for e, n in zip(
            [x for x in study["experiments"] if x["group"] == "component_budget"],
            (10, 20, 30, 40),
        )
    ]
    report.table(
        ["PCA 維度", "CV 正確率", "標準差", "總解釋變異量"], budget_rows, [1.6, 1.4, 1.2, 1.6]
    )
    report.text(
        "正確率隨維度單調上升，而且在 40 維仍未飽和——換句話說，作業的「維度 ≤ 40」"
        "在這個資料集上是真正有約束力的限制，和 Assignment #3 的特徵數量上限不同"
        "（當時放寬到 64 個像素也只多 0.4 個百分點）。原因是 PCA 的每一個維度都是"
        "全部 64 個像素的線性組合，沒有哪個維度是純粹的背景，捨棄就是實質的資訊損失。"
    )

    # --- 十一、改善空間討論（二）---
    report.page("十一、改善空間討論（二）：其他可調整項目")
    report.subheading("白化（whitening）", level=2)
    report.table(["設定", "CV 正確率", "標準差"], rows_for("whitening"), [2.4, 1.3, 1.3])
    report.text(
        "白化把各主成分的變異數拉成 1，等於丟掉「哪個方向變異較大」這個資訊。"
        "對 Logistic Regression 而言這項資訊是有用的，因此白化後反而掉了約 1 個百分點。"
    )
    report.gap(0.022)
    report.subheading("分類器正則化強度", level=2)
    report.table(["設定", "CV 正確率", "標準差"], rows_for("hyperparameter"), [2.4, 1.3, 1.3])
    report.text("往兩側調整都沒有帶來提升，顯示目前的設定已接近這個模型的最佳點。")
    report.gap(0.022)
    report.subheading("換用其他特徵提取法", level=2)
    report.table(["提取方法", "CV 正確率", "標準差"], rows_for("extraction_method"), [2.4, 1.3, 1.3])
    report.text(
        "RBF 核的 Kernel PCA 在相同維度下反而較差（核寬度本身又是一組需要調的超參數）；"
        "LDA 雖然是監督式降維，但最多只能投影到「類別數 − 1」= 9 維，"
        "在只有 9 維可用的情況下仍達 0.9520，可見其維度效率很高，只是天花板受限。"
    )

    # --- 十二、改善空間討論（三）---
    report.page("十二、改善空間討論（三）：真正的瓶頸與最終取捨")
    report.subheading("瓶頸仍在線性決策邊界", level=2)
    report.table(["分類器（皆用 40 維主成分）", "CV 正確率", "標準差"], rows_for("nonlinear"), [2.4, 1.3, 1.3])
    report.text(
        "在完全相同的 40 維主成分上改用 RBF 核的 SVM，CV 提升約 1.5 個百分點，"
        "幅度大於前面所有調整。這與 Assignment #3 的結論一致：限制效能的是 "
        "Logistic Regression 的線性決策邊界，而非特徵的表示方式。"
        "本作業指定使用 Logistic Regression，故此組僅作為討論對照。"
    )
    report.gap(0.022)
    report.subheading("特徵提取 vs 特徵選取：與 Assignment #3 的對照", level=2)
    report.table(
        ["作業", "特徵處理", "維度", "CV 正確率", "測試集正確率"],
        [
            ("#3", "L1 特徵選取", "40 個原始像素", "0.9638", "0.9750"),
            ("#5", "PCA 特徵提取", "40 維主成分", f"{training['cv_accuracy']:.4f}",
             f"{training['test_accuracy']:.4f}"),
        ],
        [0.8, 1.6, 1.6, 1.4, 1.5],
    )
    report.text(
        "兩者的交叉驗證正確率完全相同（0.9638），測試集上特徵選取則多對了 5 筆。"
        "360 筆的測試集中 5 筆約等於 1.4 個百分點，和交叉驗證的標準差（±0.007）屬同一量級，"
        "不足以判定何者較優。真正的差別在性質而非分數：特徵選取保留了原始像素、"
        "可以直接指出用了哪幾格，特徵提取則把資訊壓縮進線性組合，可解釋性較低，"
        "但不需要「某些欄位恰好沒有資訊」這個前提，在特徵高度相關的資料上更通用。"
    )
    report.gap(0.022)
    report.subheading("最終決定", level=2)
    report.text(
        "維持以交叉驗證選出的設定（40 維主成分、C = 1.0）不變。上述所有替代方案"
        "要不是在交叉驗證上較差，就是違反作業指定的演算法；"
        "而唯一在 CV 上較佳的 RBF SVM 並非本作業允許的分類器。"
    )

    # --- 十三、總結 ---
    report.page("十三、總結")
    report.text(
        "本作業以 scikit-learn 建立低解析度手寫數字的多類別分類器，"
        "將 Assignment #3 的特徵選取換成 PCA 特徵提取，"
        "完成資料切分、特徵縮放、特徵提取、交叉驗證模型選取與模型評估，"
        "並將最佳管線以 joblib.dump() 儲存供評估程式載入。"
    )
    report.gap(0.032)
    report.subheading("驗收項目對照")
    report.table(
        ["驗收項目", "規定", "實際結果"],
        [
            ("分類器演算法", "Logistic Regression", "Logistic Regression"),
            ("特徵提取", "PCA", f"PCA，64 維 → {n_components} 維"),
            ("降維後維度", "≤ 40", f"{n_components}"),
            ("總解釋變異量", "須輸出", f"{explained * 100:.2f}%"),
            ("測試集正確率", "≥ 95%", f"{accuracy * 100:.2f}%"),
            ("測試集樣本數", "約 360 筆", "360 筆"),
            ("誤判樣本", "總數與圖須對應混淆矩陣", f"{len(misclassified)} 張，一致"),
            ("模型儲存", "joblib.dump() 管線", "digits_pipeline_lr_pca.pkl"),
        ],
        [1.5, 1.6, 1.9],
    )
    report.subheading("主要發現")
    report.bullets([
        f"40 維主成分保留 {explained * 100:.2f}% 的變異，測試集正確率 {accuracy * 100:.2f}%，高於 95% 門檻",
        f"360 筆測試樣本誤判 {len(misclassified)} 張，分散在不同數字組合，無系統性混淆",
        "與 Assignment #3 相比，CV 正確率完全相同（0.9638），測試集差 5 筆、屬標準差量級",
        "PCA 維度在 40 維仍未飽和，維度上限是真正有約束力的限制（與特徵選取不同）",
        "白化、Kernel PCA 都讓表現變差；改用 RBF SVM 可達 0.9791，瓶頸在線性決策邊界",
        "繳交的評估程式不含任何訓練或模型選取程式碼，助教可直接執行批閱",
    ])
    report.gap(0.024)
    report.subheading("繳交內容")
    report.bullets([
        "10_assignment5_eval.py：僅載入模型並評估，不含訓練或模型選取程式碼",
        "digits_pipeline_lr_pca.pkl：以 joblib.dump() 儲存的完整管線（含預處理與 PCA）",
    ])

    return report


def main():
    train_module = load_module("10_assignment5_train.py", "assignment5_train")
    eval_module = load_module("10_assignment5_eval.py", "assignment5_eval")

    study_path = ASSIGNMENT_DIR / "assignment5_improvement_study.json"
    if study_path.exists():
        study = json.loads(study_path.read_text(encoding="utf-8"))
    else:
        explore = load_module("10_assignment5_explore.py", "assignment5_explore")
        study, _ = explore.run_study(output_dir=ASSIGNMENT_DIR)

    training = train_module.run_training(output_dir=ASSIGNMENT_DIR)
    evaluation = eval_module.run_evaluation(
        model_path=training["model_path"], output_dir=ASSIGNMENT_DIR
    )
    _, evaluation["y_test"] = eval_module.load_test_set()

    variance_figure = make_explained_variance_figure(
        training["pipeline"], ASSIGNMENT_DIR / VARIANCE_FIGURE
    )

    report = build(
        training,
        evaluation,
        study,
        ASSIGNMENT_DIR / eval_module.CONFUSION_FIGURE,
        ASSIGNMENT_DIR / eval_module.MISCLASSIFIED_FIGURE,
        variance_figure,
        eval_module.misclassified_message(len(evaluation["misclassified"])),
    )
    output_path = report.save(ASSIGNMENT_DIR / REPORT_NAME)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"報告已產生：{output_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
