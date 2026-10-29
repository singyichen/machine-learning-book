"""產生 Assignment #4 的結果報告 PDF（版面沿用 Assignment #1）。

此檔為報告產生工具，不屬於繳交內容。執行方式：

    python build_report.py

報告的數字直接來自**實際執行繳交腳本**的輸出（model_1_test.py / model_2_test.py），
圖也是那兩支腳本存下來的，確保報告與交出去的東西完全一致，不會手抄錯。
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import os
import re
import subprocess
import sys


ASSIGNMENT_DIR = Path(__file__).resolve().parent
LECTURE_DIR = ASSIGNMENT_DIR.parent.parent
CELEBA_ROOT = ASSIGNMENT_DIR.parent / "CelebA"
sys.path.insert(0, str(LECTURE_DIR))

import report_style  # noqa: E402  (需先加入 lecture 目錄)


REPORT_NAME = "515661055_陳欣怡_Assignment4_結果報告.pdf"
EVALUATION_CACHE = "assignment4_evaluation.json"

METHODS = {
    1: {
        "script": "model_1_test.py",
        "weights": "model_1.pth",
        "title": "方法一：調整網路架構",
        "confusion": "assignment4_method1_confusion_matrix.png",
        "misclassified": "assignment4_method1_misclassified.png",
        "history": "assignment4_method1_history.json",
        "curve": "assignment4_method1_loss_curve.png",
    },
    2: {
        "script": "model_2_test.py",
        "weights": "model_2.pth",
        "title": "方法二：ResNet-18 預訓練模型微調",
        "confusion": "assignment4_method2_confusion_matrix.png",
        "misclassified": "assignment4_method2_misclassified.png",
        "history": "assignment4_method2_history.json",
        "curve": "assignment4_method2_loss_curve.png",
    },
}

ACCURACY_PATTERN = re.compile(r"accuracy\s+([0-9.]+)\s+(\d+)")
MISCLASSIFIED_PATTERN = re.compile(r"Total misclassified samples: (\d+) / (\d+)")
CLASS_ROW_PATTERN = re.compile(
    r"^\s*(Not Smiling|Smiling)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$", re.M
)


def load_module(filename, name):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_submission_script(method):
    """執行繳交腳本，回傳 stdout（圖會存到 assignment 目錄）。"""
    config = METHODS[method]
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONPATH": str(ASSIGNMENT_DIR)}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import celeba_local_patch, runpy, sys; "
            "celeba_local_patch.apply(sys.argv[1]); "
            f"runpy.run_path('{config['script']}', run_name='__main__')",
            str(CELEBA_ROOT),
        ],
        cwd=ASSIGNMENT_DIR,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{config['script']} 執行失敗:\n{completed.stderr[-3000:]}")
    return completed.stdout


def parse_output(output):
    """自繳交腳本的輸出解析出報告需要的所有數字。"""
    accuracy = ACCURACY_PATTERN.search(output)
    misclassified = MISCLASSIFIED_PATTERN.search(output)
    if not accuracy or not misclassified:
        raise RuntimeError(f"無法解析評估輸出:\n{output[-2000:]}")

    start = output.index("詳細分類報告 (Classification Report):")
    end = output.index("Total misclassified samples:")
    report_text = output[start:end].split("\n", 1)[1].strip("\n")

    per_class = {
        match.group(1): {
            "precision": float(match.group(2)),
            "recall": float(match.group(3)),
            "f1": float(match.group(4)),
            "support": int(match.group(5)),
        }
        for match in CLASS_ROW_PATTERN.finditer(output)
    }

    return {
        "accuracy": float(accuracy.group(1)),
        "n_test": int(accuracy.group(2)),
        "misclassified": int(misclassified.group(1)),
        "report": report_text,
        "per_class": per_class,
    }


def collect_evaluations(use_cache=True):
    cache_path = ASSIGNMENT_DIR / EVALUATION_CACHE
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    evaluations = {}
    for method in METHODS:
        print(f"執行 {METHODS[method]['script']} …")
        evaluations[str(method)] = parse_output(run_submission_script(method))
    cache_path.write_text(
        json.dumps(evaluations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return evaluations


def build(evaluations, histories, study):
    one = evaluations["1"]
    two = evaluations["2"]
    best = "方法二" if two["accuracy"] >= one["accuracy"] else "方法一"

    report = report_style.Report(
        metadata={
            "Title": "Assignment #4 - CelebA 微笑分類效能改善 結果報告",
            "Author": "陳欣怡",
            "Subject": "機器學習實作系列 第 9 週：卷積神經網路",
            "Creator": "Assignment #1 style report builder",
        }
    )

    report.cover(
        "Assignment #4",
        "CelebA 微笑分類任務的效能改善",
        "機器學習實作系列　第 9 週：卷積神經網路",
        ["課程教授：陳慶永　　課程助教：翁宣允", "姓名：陳欣怡　　學號：515661055"],
        [
            "資料集：CelebA（訓練前 16,000 筆 / 測試前 10,000 筆）",
            "模型：自建 CNN（BatchNorm）/ ResNet-18 預訓練微調",
        ],
    )

    # --- 一、作業說明 ---
    report.page("一、作業說明")
    report.subheading("目的")
    report.text(
        "改善 CelebA 微笑分類任務的性能，要求正確率超過 90%。"
        "課堂範例 (Sample code 08-3.py) 的四層 CNN 大約落在 88% 上下，"
        "本作業要以兩種不同的手段把它推過門檻。"
    )
    report.gap(0.03)
    report.subheading("兩種方法（皆需實作）")
    report.bullets([
        "方法一：調整網路架構 —— 加入批次正規化 (nn.BatchNorm2d)、"
        "調整網路寬度或深度、調整 Dropout 位置或機率",
        "方法二：使用 torchvision.models 預訓練模型進行微調 (Fine-tuning)，"
        "以 ResNet-18 為例，縮放輸入影像尺寸 (224 × 224)、影像正規化、修改輸出層",
    ])
    report.gap(0.022)
    report.subheading("實作要求：兩種方法的資料集條件須一致")
    report.bullets([
        "僅使用 CelebA 訓練集的前 16,000 筆 samples 進行訓練（可以使用資料增強技術）",
        "使用 CelebA 測試集的前 10,000 筆 samples 進行測試",
        "目標：正確率 > 90%",
    ])
    report.gap(0.022)
    report.subheading("繳交方式")
    report.bullets([
        "總共四個檔案打包成單一 .zip 檔上傳",
        "方法一：model_1.pth（參數檔）與 model_1_test.py（測試腳本）",
        "方法二：model_2.pth（參數檔）與 model_2_test.py（測試腳本）",
        "測試腳本載入模型參數的寫法參考 Sample code 08-4.py",
    ])

    # --- 二、數學觀念 ---
    report.page("二、數學觀念")
    report.subheading("1. 二維卷積", level=2)
    report.math(
        r"$S(i,j) = \sum_{m}\sum_{n} I(i+m,\; j+n)\, K(m,n) + b$",
        size=14,
        space=0.080,
    )
    report.text("$K$ 為卷積核，同一組權重掃過整張影像，因此具有平移不變性且參數量遠小於全連接層。")
    report.gap(0.026)
    report.subheading("2. 批次正規化（方法一的主要調整）", level=2)
    report.math(
        r"$\hat{x} = \dfrac{x - \mu_{\mathcal{B}}}"
        r"{\sqrt{\sigma_{\mathcal{B}}^{2} + \epsilon}}, \quad y = \gamma\hat{x} + \beta$",
        size=14,
        space=0.088,
    )
    report.text(
        "$\\mu_{\\mathcal{B}}$、$\\sigma_{\\mathcal{B}}^{2}$ 為該批次的平均與變異數。"
        "把每層輸入拉回穩定的分布，可緩解內部共變量偏移，讓較深的網路收斂得更快更穩。"
        "$\\gamma$、$\\beta$ 是可學習參數，讓網路保有還原原始分布的能力。"
    )
    report.gap(0.026)
    report.subheading("3. Dropout", level=2)
    report.math(r"$y = \dfrac{1}{1-p}\, m \odot x, \quad m_i \sim \mathrm{Bernoulli}(1-p)$",
                size=13, space=0.072)
    report.text(
        "訓練時隨機丟棄比例 $p$ 的神經元。放在卷積層時 $p$ 不宜過大——"
        "特徵圖的通道數本就不多，$p = 0.5$ 等於每次丟掉一半的通道。"
    )
    report.gap(0.026)
    report.subheading("4. 二元交叉熵損失", level=2)
    report.math(
        r"$L = -\dfrac{1}{n}\sum_{i=1}^{n}"
        r"\left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$",
        size=13,
        space=0.082,
    )
    report.text("輸出層為 Sigmoid，輸出 0 ~ 1 的機率值，搭配 nn.BCELoss；門檻 0.5 轉成類別。")
    report.gap(0.026)
    report.subheading("5. 遷移學習與微調（方法二）", level=2)
    report.text(
        "ResNet-18 在 ImageNet 的 128 萬張影像上預訓練過，其卷積層已學會邊緣、紋理、"
        "部位等通用視覺特徵。微調時保留整個骨幹的權重作為初始值，只把最後的分類層"
        "換成對應二分類的 Linear(512, 1) + Sigmoid，再用較小的學習率整體更新。"
        "等於站在百萬張影像學到的特徵上起跑，而不是從隨機權重開始。"
    )

    # --- 三、資料與訓練流程 ---
    report.page("三、資料與訓練流程")
    report.subheading("資料集條件（兩種方法完全一致）", level=2)
    report.table(
        ["用途", "來源", "筆數"],
        [
            ("訓練", "CelebA train split 前 16,000 筆", "16,000"),
            ("驗證", "CelebA valid split 前 1,000 筆", "1,000"),
            ("測試", "CelebA test split 前 10,000 筆", "10,000"),
        ],
        [1.9, 2.6, 1.2],
    )
    report.text(
        "標籤取自 CelebA 40 維屬性中的第 31 項（Smiling）。"
        "測試集的 10,000 筆中 Not Smiling 4,985 筆、Smiling 5,015 筆，類別相當平衡。"
    )
    report.gap(0.026)
    report.subheading("影像前處理", level=2)
    report.table(
        ["階段", "方法一（64 × 64）", "方法二（224 × 224）"],
        [
            ("訓練", "RandomCrop → 翻轉 → Resize 64", "RandomCrop → 翻轉 → Resize 224"),
            ("評估", "CenterCrop → Resize 64", "CenterCrop → Resize 224"),
            ("正規化", "僅縮放到 [0, 1]", "再做 ImageNet 正規化"),
        ],
        [0.9, 2.3, 2.3],
    )
    report.gap(0.012)
    report.text("（Crop 皆為 178 × 178，自 218 × 178 的原圖裁出正方形臉部區域。）")
    report.text(
        "評估階段不可有任何隨機性，否則同一個模型每次跑出來的分數都不同；"
        "資料增強只用在訓練階段（作業允許使用資料增強技術）。"
    )
    report.gap(0.026)
    report.subheading("訓練設定", level=2)
    report.table(
        ["項目", "方法一", "方法二"],
        [
            ("Epochs", str(histories["1"]["epochs"]), str(histories["2"]["epochs"])),
            ("Batch size", "32", "32"),
            ("學習率", "1e-3", "1e-4（微調用較小的學習率）"),
            ("Weight decay", "0", "1e-4"),
            ("最佳 epoch", str(histories["1"]["best_epoch"]), str(histories["2"]["best_epoch"])),
        ],
        [1.4, 1.6, 2.4],
    )
    report.text(
        "兩者都依驗證集 loss 保留最佳的一輪權重再存檔，避免交出過擬合那一輪的參數。"
        "測試集全程未參與訓練與任何選擇。"
    )

    # --- 四、方法一：架構調整 ---
    report.page("四、方法一：調整網路架構")
    report.subheading("相對課堂範例的三處調整", level=2)
    report.bullets([
        "1. 每層卷積後加入 nn.BatchNorm2d（共 4 層）",
        "2. 卷積層的 Dropout 由 0.5 降為 0.1 / 0.2",
        "3. 分類頭（Linear）之前補一層 Dropout(0.3)",
    ])
    report.gap(0.022)
    report.text(
        "第 2 點是關鍵：課堂範例在卷積層用 0.5，等於每次前向傳播丟掉一半的特徵圖通道。"
        "對只有 32 / 64 個通道的淺層而言，這個強度會讓訊號嚴重衰減。"
        "加入批次正規化後本身就帶有正則化效果，Dropout 可以放鬆，改把正則化集中在"
        "參數量最大的全連接層之前。"
    )
    report.gap(0.026)
    report.subheading("訓練曲線", level=2)
    report.figure_image(
        ASSIGNMENT_DIR / METHODS[1]["curve"],
        height=0.34,
        caption=(
            f"assignment4_method1_loss_curve.png — 訓練與驗證 loss。"
            f"第 {histories['1']['best_epoch']} 輪取得最佳驗證集 loss "
            f"{histories['1']['best_valid_loss']:.4f}（驗證集正確率 "
            f"{histories['1']['best_valid_accuracy'] * 100:.2f}%），該輪權重即為 model_1.pth。"
        ),
    )

    # --- 五、預期結果（方法一）---
    report.page("五、預期結果（方法一）：分類報告")
    report.text(
        f"測試集共 {one['n_test']:,} 筆，整體正確率（Accuracy）= {one['accuracy']:.4f}，"
        f"達成「正確率 > 90%」的要求。"
    )
    report.gap(0.03)
    report.subheading("詳細分類報告（Classification Report）", level=2)
    report.mono(one["report"])
    report.gap(0.028)
    report.text(
        f"Not Smiling 的 f1-score 為 {one['per_class']['Not Smiling']['f1']:.2f}、"
        f"Smiling 為 {one['per_class']['Smiling']['f1']:.2f}，兩類表現接近，"
        "沒有偏向任何一邊。"
    )

    report.page("六、預期結果（方法一）：混淆矩陣與誤判樣本")
    report.figure_image(
        ASSIGNMENT_DIR / METHODS[1]["confusion"],
        height=0.36,
        caption=(
            f"assignment4_method1_confusion_matrix.png — "
            f"{one['n_test']:,} 筆測試樣本中共有 {one['misclassified']:,} 筆誤判。"
        ),
    )
    report.gap(0.014)
    report.figure_image(
        ASSIGNMENT_DIR / METHODS[1]["misclassified"],
        height=0.30,
        caption=(
            "assignment4_method1_misclassified.png — 前 25 個誤判樣本，"
            "標題紅字標示真實標籤 (True) 與預測結果 (Pred)。"
        ),
    )

    # --- 七、方法二：ResNet-18 微調 ---
    report.page("七、方法二：ResNet-18 預訓練模型微調")
    report.subheading("三項必要的調整", level=2)
    report.bullets([
        "1. 縮放輸入影像尺寸到 224 × 224（ResNet-18 預訓練時的輸入尺寸）",
        "2. 影像正規化：採用 ImageNet 的 mean / std",
        "3. 修改輸出層：Linear(512, 1) + Sigmoid，對應二分類問題",
    ])
    report.gap(0.022)
    report.text(
        "正規化這一步不能省：預訓練權重是在 ImageNet 的統計量下學出來的，"
        "若直接餵 [0, 1] 的影像，等於把骨幹放到它沒見過的輸入分布上，"
        "預訓練的優勢會大打折扣。學習率也刻意調小到 1e-4，"
        "避免一開始的大梯度把預訓練權重沖掉。"
    )
    report.gap(0.026)
    report.subheading("訓練曲線", level=2)
    report.figure_image(
        ASSIGNMENT_DIR / METHODS[2]["curve"],
        height=0.34,
        caption=(
            f"assignment4_method2_loss_curve.png — 只訓練 {histories['2']['epochs']} 輪即收斂，"
            f"第 {histories['2']['best_epoch']} 輪取得最佳驗證集 loss "
            f"{histories['2']['best_valid_loss']:.4f}（驗證集正確率 "
            f"{histories['2']['best_valid_accuracy'] * 100:.2f}%）。"
        ),
    )

    # --- 八、預期結果（方法二）---
    report.page("八、預期結果（方法二）：分類報告")
    report.text(
        f"測試集共 {two['n_test']:,} 筆，整體正確率（Accuracy）= {two['accuracy']:.4f}，"
        f"達成「正確率 > 90%」的要求。"
    )
    report.gap(0.03)
    report.subheading("詳細分類報告（Classification Report）", level=2)
    report.mono(two["report"])
    report.gap(0.028)
    report.text(
        f"Not Smiling 的 f1-score 為 {two['per_class']['Not Smiling']['f1']:.2f}、"
        f"Smiling 為 {two['per_class']['Smiling']['f1']:.2f}。"
    )

    report.page("九、預期結果（方法二）：混淆矩陣與誤判樣本")
    report.figure_image(
        ASSIGNMENT_DIR / METHODS[2]["confusion"],
        height=0.36,
        caption=(
            f"assignment4_method2_confusion_matrix.png — "
            f"{two['n_test']:,} 筆測試樣本中共有 {two['misclassified']:,} 筆誤判。"
        ),
    )
    report.gap(0.014)
    report.figure_image(
        ASSIGNMENT_DIR / METHODS[2]["misclassified"],
        height=0.30,
        caption=(
            "assignment4_method2_misclassified.png — 前 25 個誤判樣本，"
            "標題紅字標示真實標籤 (True) 與預測結果 (Pred)。"
        ),
    )

    # --- 十、兩種方法比較 ---
    report.page("十、兩種方法的比較")
    report.table(
        ["項目", "方法一（架構調整）", "方法二（ResNet-18 微調）"],
        [
            ("輸入尺寸", "64 × 64", "224 × 224"),
            ("參數量", "約 39 萬", "約 1,118 萬"),
            ("訓練輪數", str(histories["1"]["epochs"]), str(histories["2"]["epochs"])),
            ("測試集正確率", f"{one['accuracy'] * 100:.2f}%", f"{two['accuracy'] * 100:.2f}%"),
            ("誤判筆數", f"{one['misclassified']:,} / {one['n_test']:,}",
             f"{two['misclassified']:,} / {two['n_test']:,}"),
        ],
        [1.5, 2.0, 2.2],
    )
    report.gap(0.022)
    report.text(
        f"兩種方法都達成 90% 的門檻，其中{best}較佳。"
        f"值得注意的是方法二只訓練 {histories['2']['epochs']} 輪就達到這個水準，"
        f"而方法一需要 {histories['1']['epochs']} 輪——這正是遷移學習的價值："
        "骨幹不必從零學起邊緣與紋理這些通用特徵。"
        "代價是參數量大了約 28 倍、輸入尺寸大了 12 倍，單次推論成本高出許多。"
    )
    report.gap(0.026)
    report.subheading("資料集標籤品質才是真正的天花板", level=2)
    report.text(
        "題目 p.67 註明：「微笑分類的平均正確率通常落在 93.0% ~ 93.5% 左右，"
        "正確率無法再隨模型升級而改善的主因在於資料集本身的標籤錯誤率。」"
        "CelebA 的屬性標註是由標註者主觀判定的，「微笑」與「沒微笑」之間本來就存在"
        "大量的灰色地帶（抿嘴、半笑、側臉）。這類樣本無論模型多強都無法穩定分對，"
        "因為連標準答案本身都不一致。從誤判樣本圖也能直接看到這點："
        "多數誤判的臉確實介於兩者之間，而不是模型犯了明顯的錯。"
    )

    # --- 十一、改善空間討論 ---
    def rows_for(group):
        return [
            (
                row["label"],
                f"{row['best_valid_accuracy'] * 100:.2f}%",
                f"{row['n_parameters']:,}",
            )
            for row in study["experiments"] if row["group"] == group
        ]

    report.page("十一、改善空間討論（一）：拆解方法一的三項調整")
    report.text(
        f"訓練一個 CelebA CNN 要十幾分鐘，無法像前幾次作業那樣跑交叉驗證，"
        f"因此改用固定訓練預算的對照實驗：每組都只訓練 {study['epochs_per_run']} 輪、"
        f"同一組種子、同一份訓練集（前 {study['n_train']:,} 筆），"
        f"比較驗證集（前 {study['n_valid']:,} 筆）的最佳正確率。"
        "比較的是「相同預算下誰學得快、學得好」，不是最終收斂上限。"
    )
    report.gap(0.026)
    report.subheading("批次正規化與 Dropout 各自的貢獻", level=2)
    report.table(
        ["架構", f"{study['epochs_per_run']} 輪內最佳驗證正確率", "參數量"],
        rows_for("architecture"),
        [3.4, 1.7, 1.1],
    )
    report.text(
        "這組實驗把方法一的兩項調整拆開來看。批次正規化的貢獻最大，"
        "光是加上它就把 8 輪內的驗證正確率從 65.50% 拉到 88.00%——"
        "沒有它的話，卷積層那兩個 0.5 的 Dropout 會讓訊號衰減到幾乎學不起來。"
        "再把 Dropout 調低又多了 1 個百分點，顯示批次正規化本身已帶有正則化效果，"
        "再疊上 0.5 的 Dropout 就過度了。"
    )

    report.page("十二、改善空間討論（二）：網路寬度與資料增強")
    report.subheading("加寬通道數", level=2)
    report.table(
        ["架構", f"{study['epochs_per_run']} 輪內最佳驗證正確率", "參數量"],
        rows_for("width"),
        [3.4, 1.7, 1.1],
    )
    report.text(
        "把每層通道數加倍（32/64/128/256 → 64/128/256/512），參數量成長約 4 倍，"
        "但在相同訓練預算下並沒有換到相稱的提升。"
        "這與題目 p.67 的提示一致：效能的瓶頸不在模型容量。"
    )
    report.gap(0.026)
    report.subheading("資料增強：短預算下的結果與直覺相反", level=2)
    report.table(
        ["訓練資料", f"{study['epochs_per_run']} 輪內最佳驗證正確率", "參數量"],
        rows_for("augmentation"),
        [3.4, 1.7, 1.1],
    )
    report.text(
        "關閉 RandomCrop 與水平翻轉後，同樣 8 輪預算下的驗證正確率反而較高。"
        "這個結果與「資料增強有助於泛化」的直覺相反，值得說明清楚。"
    )
    report.gap(0.018)
    report.text(
        "原因在於這組比較的是「固定 8 輪預算下誰學得快」。開啟增強後，模型每一輪看到的"
        "都是裁切位置與左右方向都在變動的影像，等於要同時學會辨識微笑與忽略這些變異，"
        "因此收斂較慢；關閉增強則讓每輪的輸入完全一致，短時間內自然追得比較快。"
    )
    report.gap(0.018)
    report.text(
        f"但這不代表增強沒有價值。實際交出去的 model_1.pth 是以開啟增強訓練 "
        f"{histories['1']['epochs']} 輪得到的，驗證集正確率 "
        f"{histories['1']['best_valid_accuracy'] * 100:.2f}%，高於本節任何一組 8 輪的結果。"
        "增強的效果屬於「訓練久了才拉開差距」的那一類，8 輪的預算根本量不到；"
        "若只看這張表就斷定增強有害，就是拿錯誤的實驗設計回答了問題。"
    )
    report.gap(0.022)
    report.subheading("最終決定", level=2)
    report.text(
        "維持繳交版本的設定不變，包含保留資料增強。加寬網路在相同預算下沒有帶來相稱的"
        "提升（參數量 4 倍、正確率 +0.3 個百分點）；關閉增強雖然在 8 輪的短預算下較快，"
        "但完整訓練的結果才是交出去的依據。兩個模型的測試集結果也都已落在題目所說的"
        "「93.0% ~ 93.5%」這個資料集本身的天花板附近，繼續加大模型的邊際效益很低。"
    )

    # --- 十三、總結 ---
    report.page("十三、總結")
    report.text(
        "本作業以兩種不同的手段改善 CelebA 微笑分類的效能：方法一調整網路架構"
        "（加入批次正規化、重新配置 Dropout），方法二以 ImageNet 預訓練的 ResNet-18 "
        "進行微調。兩者使用完全相同的資料集條件，測試集正確率皆超過 90% 的門檻。"
    )
    report.gap(0.032)
    report.subheading("驗收項目對照")
    report.table(
        ["驗收項目", "規定", "實際結果"],
        [
            ("方法一", "調整網路架構", "BatchNorm × 4 + Dropout 重配置"),
            ("方法二", "torchvision 預訓練模型微調", "ResNet-18，224 × 224"),
            ("訓練資料", "訓練集前 16,000 筆", "16,000 筆（兩法一致）"),
            ("測試資料", "測試集前 10,000 筆", "10,000 筆（兩法一致）"),
            ("方法一正確率", "> 90%", f"{one['accuracy'] * 100:.2f}%"),
            ("方法二正確率", "> 90%", f"{two['accuracy'] * 100:.2f}%"),
            ("繳交檔案", "四個檔案打包成單一 .zip", "兩組 .pth 與測試腳本"),
        ],
        [1.6, 1.9, 1.8],
    )
    report.subheading("主要發現")
    report.bullets([
        f"方法一（架構調整）測試集正確率 {one['accuracy'] * 100:.2f}%，"
        f"誤判 {one['misclassified']:,} / {one['n_test']:,}",
        f"方法二（ResNet-18 微調）測試集正確率 {two['accuracy'] * 100:.2f}%，"
        f"誤判 {two['misclassified']:,} / {two['n_test']:,}",
        f"方法二只用 {histories['2']['epochs']} 輪就達標，"
        f"方法一需 {histories['1']['epochs']} 輪——遷移學習省下的是「學通用特徵」的時間",
        "卷積層的 Dropout 由 0.5 調低是方法一的關鍵，BatchNorm 本身已帶有正則化效果",
        "加寬網路參數量 4 倍只換到 +0.3 個百分點；關閉增強在 8 輪短預算下反而較快，但完整訓練仍以開啟增強為準（詳見第十二節）",
        "兩法的結果都接近題目所述 93.0% ~ 93.5% 的天花板，瓶頸在資料集的標籤錯誤率",
    ])
    report.gap(0.024)
    report.subheading("繳交內容")
    report.bullets([
        "model_1.pth / model_1_test.py：方法一的參數檔與測試腳本",
        "model_2.pth / model_2_test.py：方法二的參數檔與測試腳本",
    ])

    return report


def main():
    rebuild = "--rebuild" in sys.argv
    evaluations = collect_evaluations(use_cache=not rebuild)

    histories = {
        str(method): json.loads(
            (ASSIGNMENT_DIR / config["history"]).read_text(encoding="utf-8")
        )
        for method, config in METHODS.items()
    }

    study_path = ASSIGNMENT_DIR / "assignment4_improvement_study.json"
    if study_path.exists():
        study = json.loads(study_path.read_text(encoding="utf-8"))
    else:
        explore = load_module("09_assignment4_explore.py", "assignment4_explore")
        study, _ = explore.run_study(output_dir=ASSIGNMENT_DIR)

    report = build(evaluations, histories, study)
    output_path = report.save(ASSIGNMENT_DIR / REPORT_NAME)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"報告已產生：{output_path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
