"""產生 Assignment #6 的結果報告 PDF（版面沿用 Assignment #1）。

此檔為報告產生工具，不屬於繳交內容。執行方式：

    python build_report.py
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent
LECTURE_DIR = ASSIGNMENT_DIR.parent.parent
sys.path.insert(0, str(LECTURE_DIR))

import report_style  # noqa: E402  (需先加入 lecture 目錄)


REPORT_NAME = "515661055_陳欣怡_Assignment6_結果報告.pdf"


def load_module(filename, name):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(results, study, comparison_figure, mlp_best_epoch):
    linear = results["Linear Regression"]
    polynomial = results["Polynomial Regression"]
    mlp = results["PyTorch MLP Regressor"]
    best_name = max(results, key=lambda name: results[name]["r2"])
    best_r2 = results[best_name]["r2"]

    report = report_style.Report(
        metadata={
            "Title": "Assignment #6 - 糖尿病疾病進展回歸預測 結果報告",
            "Author": "陳欣怡",
            "Subject": "機器學習實作系列 第 11 週：回歸分析",
            "Creator": "Assignment #1 style report builder",
        }
    )

    report.cover(
        "Assignment #6",
        "基準一年後糖尿病疾病進展之預測器",
        "機器學習實作系列　第 11 週：回歸分析",
        ["課程教授：陳慶永　　課程助教：翁宣允", "姓名：陳欣怡　　學號：515661055"],
        [
            "資料集：sklearn.datasets.load_diabetes()（442 筆、10 維特徵）",
            "模型：多元線性回歸 / 多項式回歸 (degree=2) / PyTorch MLP Regressor",
        ],
    )

    # --- 一、作業說明 ---
    report.page("一、作業說明")
    report.subheading("目的")
    report.text(
        "建立基準一年後糖尿病疾病進展之預測器。應用情境為臨床醫學定量評估"
        "（Clinical Quantitative Assessment）：假設病患的基礎生理特徵與化驗數據已由醫院"
        "預先採集完成，本作業僅負責建立回歸模型來預測一年後的疾病進展指數。"
    )
    report.gap(0.03)
    report.subheading("資料集：sklearn.datasets.load_diabetes()")
    report.bullets([
        "442 筆病患臨床紀錄",
        "10 維生理與化驗特徵：年齡、性別、BMI、血壓及 6 種血清化驗數據",
        "各項特徵已預先經過縮放（每項特徵的平方和皆為 1）",
        "連續型目標值：基準一年後疾病進展的定量測量值，數值範圍介於 25 到 346",
    ])
    report.gap(0.022)
    report.subheading("任務需求")
    report.bullets([
        "以三種不同的演算法建立回歸預測器並進行效能比較",
        "運用資料切分、特徵縮放與模型評估",
        "切分比例：80% 訓練集、20% 測試集",
        "模型評估：$\\mathrm{MSE}$ 與 $R^{2}$",
        "多元線性回歸：使用基礎線性模型作為效能 Baseline",
        "多項式回歸：以 PolynomialFeatures 設定 degree=2 進行特徵轉換",
        "MLP 非線性回歸：以 PyTorch 實作 MLP regressor",
        "在測試集（約 89 筆樣本）上三者之中最佳的 $R^{2}$ 分數必須達 0.46 以上",
    ])

    # --- 二、數學觀念 ---
    report.page("二、數學觀念")
    report.subheading("1. 多元線性回歸", level=2)
    report.math(
        r"$\hat{y} = w_0 + \sum_{j=1}^{10} w_j x_j = \mathbf{w}^{\top}\mathbf{x} + w_0$",
        size=14,
        space=0.078,
    )
    report.text("以最小平方法求解，使殘差平方和最小的 $\\mathbf{w}$ 有封閉解。")
    report.gap(0.026)
    report.subheading("2. 多項式特徵轉換（degree = 2）", level=2)
    report.math(
        r"$\phi(\mathbf{x}) = (x_1,...,x_{10},\; x_1^{2},...,x_{10}^{2},"
        r"\; x_1 x_2,...,x_9 x_{10})$",
        size=13,
        space=0.072,
    )
    report.text(
        "10 個原始特徵展開為 10 個原始項 + 10 個平方項 + 45 個交互作用項 = 65 維，"
        "再對 $\\phi(\\mathbf{x})$ 做線性回歸。模型對 $\\mathbf{w}$ 仍是線性的，"
        "但對 $\\mathbf{x}$ 已是非線性。"
    )
    report.gap(0.026)
    report.subheading("3. 均方誤差（MSE）", level=2)
    report.math(
        r"$\mathrm{MSE} = \dfrac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^{2}$",
        size=14,
        space=0.072,
    )
    report.text("單位是目標值的平方，數值越小越好，但不具跨資料集的可比性。")
    report.gap(0.026)
    report.subheading("4. 決定係數（$R^{2}$ Score）", level=2)
    report.math(
        r"$R^{2} = 1 - \dfrac{\sum_i (y_i - \hat{y}_i)^{2}}{\sum_i (y_i - \bar{y})^{2}}$",
        size=14,
        space=0.108,
    )
    report.text(
        "分母是「永遠猜平均值」這個基準模型的誤差。$R^{2} = 1$ 代表完美預測，"
        "$R^{2} = 0$ 代表和猜平均值一樣差，負值代表比猜平均值還差。"
    )
    report.gap(0.026)
    report.subheading("5. MLP 非線性回歸", level=2)
    report.math(
        r"$\hat{y} = \mathbf{W}_3\,\sigma(\mathbf{W}_2\,\sigma(\mathbf{W}_1\mathbf{x}"
        r" + \mathbf{b}_1) + \mathbf{b}_2) + b_3$",
        size=13,
        space=0.072,
    )
    report.text(
        "$\\sigma$ 為 ReLU 激活函數。隱藏層讓模型能學到特徵之間的非線性交互作用，"
        "但參數量也隨之暴增，在小資料集上極易過擬合。"
    )

    # --- 三、訓練與評估流程 ---
    report.page("三、訓練與評估流程")
    report.bullets([
        "1. load_diabetes() 載入 442 筆、10 維特徵與連續型目標值",
        "2. train_test_split(test_size=0.2, random_state=42)：353 筆訓練 / 89 筆測試",
        "3. StandardScaler：僅以訓練集計算 mean / std，測試集只做 transform",
        "4. 模型一 — LinearRegression：直接在 10 維特徵上做最小平方法",
        "5. 模型二 — PolynomialFeatures(degree=2) → StandardScaler → LinearRegression",
        "6. 模型三 — PyTorch MLP：10 → 32 → 16 → 1，ReLU，Adam 最佳化",
        "7. MLP 自訓練集再切 20% 作為驗證集，依驗證集 loss 提早停止",
        "8. 三個模型在完全相同的 89 筆測試集上計算 MSE 與 $R^{2}$",
        "9. 輸出量化指標比較表，並以長條圖視覺化 $R^{2}$ 對比",
    ])
    report.gap(0.03)
    report.subheading("MLP 的防過擬合設計", level=2)
    report.text(
        f"353 筆訓練樣本撐不起大網路，因此刻意採用極小架構（10 → 32 → 16 → 1）、"
        f"加上 weight decay（$\\lambda = 10^{{-3}}$）與提早停止（patience = 100）。"
        f"本次訓練在第 {mlp_best_epoch} 個 epoch 取得最佳驗證集 loss，該輪權重即為最終模型。"
    )
    report.gap(0.022)
    report.subheading("關鍵限制", level=2)
    report.text(
        "測試集全程未參與任何訓練、調參或提早停止的判斷，僅在最後評估時使用一次。"
        "MLP 提早停止所用的驗證集是從訓練集內部再切出來的，與測試集完全隔離。"
    )

    # --- 四、預期結果 1 ---
    report.page("四、預期結果 1：三種模型的量化指標比較")
    report.text("程式輸出的測試集量化指標比較表：")
    report.gap(0.02)
    module = load_module("11_assignment6.py", "assignment6_for_report")
    report.mono(module.format_comparison_table(results))
    report.gap(0.028)
    report.subheading("比較表", level=2)
    report.table(
        ["模型", "MSE", "$R^{2}$ Score", "相對 Baseline"],
        [
            ("多元線性回歸 (Baseline)", f"{linear['mse']:.2f}", f"{linear['r2']:.4f}", "—"),
            ("多項式回歸 (degree=2)", f"{polynomial['mse']:.2f}", f"{polynomial['r2']:.4f}",
             f"{polynomial['r2'] - linear['r2']:+.4f}"),
            ("PyTorch MLP Regressor", f"{mlp['mse']:.2f}", f"{mlp['r2']:.4f}",
             f"{mlp['r2'] - linear['r2']:+.4f}"),
        ],
        [2.2, 1.3, 1.4, 1.4],
    )
    report.text(
        f"三者之中最佳為 {best_name}，$R^{{2}} = {best_r2:.4f}$，"
        f"達成「必須達 0.46 以上」的規定，也落在投影片 p.45 所列的 "
        f"「此糖尿病資料集測試集的最佳 $R^{{2}}$ 通常落在 0.45 ~ 0.55 之間」區間內。"
    )

    # --- 五、預期結果 2 ---
    report.page("五、預期結果 2：效能對比長條圖")
    report.text(
        "以長條圖直觀比較三種模型在測試集上的 $R^{2}$ Score，"
        "標題註明測試集字樣，長條上方精準標示四位小數的 $R^{2}$ 數值。"
    )
    report.gap(0.02)
    report.figure_image(
        comparison_figure,
        height=0.42,
        caption=(
            "assignment6_model_comparison.png — 三種模型的 $R^{2}$ 對比。"
            "多項式回歸雖然模型最複雜，分數卻是三者中最低；"
            "MLP 以極小的架構拿下最高分。"
        ),
    )
    report.gap(0.02)
    report.text(
        "這張圖直接印證投影片 p.44 的註記：「在機器學習中，模型不是越酷炫、越複雜就越好。」"
        "下一節說明多項式回歸為何反而退步。"
    )

    # --- 六、模型分析 ---
    report.page("六、模型分析：多項式回歸為何反而退步")
    report.subheading("維度災難", level=2)
    report.text(
        "degree = 2 的特徵轉換把 10 維展開成 65 維，但訓練樣本只有 353 筆，"
        "樣本數與參數量之比從 35:1 掉到約 5:1。在沒有正規化的情況下，"
        "模型有足夠的自由度去擬合訓練集中的雜訊，因此訓練誤差下降、測試誤差反而上升。"
    )
    report.gap(0.026)
    report.subheading("交叉驗證上的證據", level=2)
    poly_rows = [
        (e["label"], f"{e['cv_r2']:.4f}", f"± {e['cv_std']:.4f}")
        for e in study["experiments"] if e["group"] == "polynomial_fix"
    ]
    report.table(["設定", "CV $R^{2}$", "標準差"], poly_rows, [2.6, 1.3, 1.3])
    report.text(
        "不加正規化的 degree=2 在交叉驗證上只有 0.3314，而且標準差高達 ±0.0782，"
        "是所有實驗中最不穩定的一組——這正是過擬合的典型徵兆。"
        "加上 Ridge (alpha=100) 後回升到 0.4498，證實問題出在缺少正規化而非特徵轉換本身。"
    )
    report.gap(0.022)
    report.subheading("與投影片的對照", level=2)
    report.text(
        f"投影片 p.45 預期多項式回歸 (degree=2) 的 $R^{{2}}$ 落在 0.30 ~ 0.42（或負值），"
        f"本次測試集結果為 {polynomial['r2']:.4f}，落在該區間內；"
        f"投影片的範例輸出為 0.4156，差異來自特徵縮放步驟的擺放位置，量級相同。"
    )

    # --- 七、改善空間討論（一）---
    def rows_for(group):
        return [
            (e["label"], f"{e['cv_r2']:.4f}", f"± {e['cv_std']:.4f}")
            for e in study["experiments"] if e["group"] == group
        ]

    report.page("七、改善空間討論（一）：正規化")
    report.text(
        f"以下比較皆在訓練集（353 筆）上以 {study['cv_folds']}-fold 交叉驗證進行，"
        "測試集不參與任何設定的挑選。"
    )
    report.gap(0.026)
    report.table(["設定", "CV $R^{2}$", "標準差"], rows_for("regularization"), [2.6, 1.3, 1.3])
    report.text(
        "投影片 p.45 指出 Lasso / Ridge 回歸「表現最穩定」，區間 0.47 ~ 0.50。"
        "本次交叉驗證確實落在該區間（0.4797 ~ 0.4812），但與未正規化的線性回歸"
        "（0.4804）差距不到 0.2 個百分點，遠小於標準差（±0.04）。"
        "原因是 load_diabetes 的 10 個特徵本來就經過縮放、數量也少，"
        "共線性造成的係數膨脹有限，正規化沒有太多發揮空間。"
    )

    # --- 八、改善空間討論（二）---
    report.page("八、改善空間討論（二）：MLP 容量與其他演算法")
    report.subheading("MLP 隱藏層容量", level=2)
    report.table(["架構", "CV $R^{2}$", "標準差"], rows_for("mlp_capacity"), [2.6, 1.3, 1.3])
    report.text(
        "投影片 p.45 說 MLP「架構必須非常小，且必須提早停止，否則必過擬合」。"
        "實測四種容量的交叉驗證差距都在 2 個百分點以內、且彼此標準差重疊，"
        "在有提早停止與 weight decay 的前提下，容量並不是主要的效能來源。"
        "最大的 (256, 128, 64) 反而最差，與投影片的提醒一致。"
    )
    report.gap(0.022)
    report.subheading("作業指定以外的演算法", level=2)
    report.table(["演算法", "CV $R^{2}$", "標準差"], rows_for("other_algorithms"), [2.6, 1.3, 1.3])
    report.text(
        "投影片 p.45 預期 KNN (調參後) 可達 0.48 ~ 0.53、隨機森林 0.45 ~ 0.52。"
        "本次交叉驗證的結果（KNN 0.3985 ~ 0.4151、隨機森林 0.4256）略低於該區間，"
        "但都沒有超越線性模型。這兩組僅作為討論對照，不在作業指定的三種演算法之內。"
    )

    # --- 九、改善空間討論（三）---
    report.page("九、改善空間討論（三）：測試集分數的可信度")
    report.subheading("交叉驗證與測試集的結論並不一致", level=2)
    cv_linear = next(e["cv_r2"] for e in study["experiments"] if e["label"] == "多元線性回歸（基準）")
    cv_mlp = next(e["cv_r2"] for e in study["experiments"] if e["label"] == "隱藏層 (32, 16)（基準）")
    report.table(
        ["模型", "CV $R^{2}$（353 筆訓練集）", "測試集 $R^{2}$（89 筆）"],
        [
            ("多元線性回歸", f"{cv_linear:.4f}", f"{linear['r2']:.4f}"),
            ("PyTorch MLP Regressor", f"{cv_mlp:.4f}", f"{mlp['r2']:.4f}"),
        ],
        [2.0, 2.2, 2.0],
    )
    report.text(
        f"交叉驗證認為線性回歸較好（{cv_linear:.4f} 對 {cv_mlp:.4f}），"
        f"測試集卻認為 MLP 較好（{mlp['r2']:.4f} 對 {linear['r2']:.4f}）——兩者結論相反。"
    )
    report.gap(0.022)
    report.subheading("為什麼會這樣", level=2)
    report.text(
        f"測試集只有 {study['n_test']} 筆。交叉驗證的標準差本身就有 ±0.04，"
        "而單次 80/20 切分的波動只會更大。兩個模型在測試集上約 5 個百分點的差距，"
        "完全落在這個雜訊量級之內，不足以判定 MLP 真的比線性回歸好。"
    )
    report.gap(0.022)
    report.subheading("最終決定", level=2)
    report.text(
        "三個模型全部依作業指定的演算法與參數建立，不依測試集分數回頭挑選設定。"
        "若為了提高測試集分數而反覆比較、再據以更換模型或超參數，"
        "等同於把測試集當成驗證集使用，交出去的分數就會是被高估的。"
        "因此維持原設定不變，並在此據實說明：本次最佳的 $R^{2}$ 雖然達標，"
        "但以 89 筆測試集而言，這個排名本身帶有相當的不確定性。"
    )

    # --- 十、總結 ---
    report.page("十、總結")
    report.text(
        "本作業以三種不同的演算法建立糖尿病疾病進展的回歸預測器並比較效能，"
        "完成資料切分、特徵縮放與模型評估，輸出 MSE 與 $R^{2}$ 的量化指標比較表，"
        "並以長條圖視覺化三者的 $R^{2}$ 對比。"
    )
    report.gap(0.032)
    report.subheading("驗收項目對照")
    report.table(
        ["驗收項目", "規定", "實際結果"],
        [
            ("演算法數量", "三種不同演算法", "線性 / 多項式 / MLP"),
            ("切分比例", "80% 訓練、20% 測試", "353 筆 / 89 筆"),
            ("評估指標", "MSE 與 $R^{2}$", "兩者皆輸出"),
            ("多項式回歸", "PolynomialFeatures degree=2", "degree=2，65 維"),
            ("MLP 實作", "以 PyTorch 實作", "10→32→16→1，ReLU"),
            ("最佳 $R^{2}$", "≥ 0.46", f"{best_r2:.4f}"),
            ("測試集樣本數", "約 89 筆", "89 筆"),
            ("長條圖", "標題註明測試集、標四位小數", "已符合"),
        ],
        [1.6, 1.8, 1.6],
    )
    report.subheading("主要發現")
    report.bullets([
        f"最佳模型為 {best_name}，測試集 $R^{{2}} = {best_r2:.4f}$，高於 0.46 門檻",
        f"多元線性回歸重現投影片的 Baseline（MSE {linear['mse']:.2f}、$R^{{2}}$ {linear['r2']:.4f}）",
        f"多項式回歸 $R^{{2}}$ 反而掉到 {polynomial['r2']:.4f}：65 維對上 353 筆樣本，是維度災難",
        "加上 Ridge (alpha=100) 可把多項式回歸的 CV 從 0.3314 拉回 0.4498，證實問題在缺少正規化",
        "MLP 四種容量的 CV 差距都在 2 個百分點內，容量不是主要效能來源（詳見第八節）",
        "交叉驗證與測試集對「線性 vs MLP」的結論相反，89 筆測試集的排名帶有相當不確定性",
    ])
    report.gap(0.024)
    report.subheading("繳交內容")
    report.bullets([
        "11_assignment6.py：三種回歸模型的建立、評估、比較表與長條圖",
        "515661055_陳欣怡_Assignment6_結果報告.pdf：本報告",
    ])

    return report


def main():
    module = load_module("11_assignment6.py", "assignment6")

    study_path = ASSIGNMENT_DIR / "assignment6_improvement_study.json"
    if study_path.exists():
        study = json.loads(study_path.read_text(encoding="utf-8"))
    else:
        explore = load_module("11_assignment6_explore.py", "assignment6_explore")
        study, _ = explore.run_study(output_dir=ASSIGNMENT_DIR)

    results = module.main(output_dir=ASSIGNMENT_DIR)

    report = build(
        results,
        study,
        ASSIGNMENT_DIR / module.COMPARISON_FIGURE,
        results["PyTorch MLP Regressor"]["best_epoch"],
    )
    output_path = report.save(ASSIGNMENT_DIR / REPORT_NAME)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"報告已產生：{output_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
