"""Assignment #5 - 改善空間研究（不繳交）。

回答「在作業限制內還能做得更好嗎」。泛化能力一律以訓練集上的 5-fold 交叉驗證
估計；測試集只在最後確認一次，不參與任何設定的挑選。

輸出：assignment5_improvement_study.json（報告的「改善空間討論」章節由此帶數字）
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ASSIGNMENT_DIR = Path(__file__).resolve().parent
STUDY_FILENAME = "assignment5_improvement_study.json"
RANDOM_STATE = 42
MAX_COMPONENTS = 40


def _load_train_module():
    path = ASSIGNMENT_DIR / "10_assignment5_train.py"
    spec = spec_from_file_location("assignment5_train_for_study", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_config():
    """交出去的那條管線的設定；研究的所有比較都以它為基準。"""
    return {"n_components": 40, "classifier_C": 1.0}


EXPERIMENTS = [
    # 1. 降維預算：維度上限是作業硬規定，但更低的維度會不會反而更好？
    {"group": "component_budget", "label": "PCA 10 維", "n_components": 10},
    {"group": "component_budget", "label": "PCA 20 維", "n_components": 20},
    {"group": "component_budget", "label": "PCA 30 維", "n_components": 30},
    {"group": "component_budget", "label": "PCA 40 維（上限）", "n_components": 40},
    # 2. 白化：PCA 把主軸的變異數拉成 1，對線性分類器的條件數有幫助嗎？
    {"group": "whitening", "label": "不白化（基準）", "whiten": False},
    {"group": "whitening", "label": "白化 whiten=True", "whiten": True},
    # 3. 分類器正則化強度
    {"group": "hyperparameter", "label": "C = 0.1", "classifier_C": 0.1},
    {"group": "hyperparameter", "label": "C = 1.0（基準）", "classifier_C": 1.0},
    {"group": "hyperparameter", "label": "C = 10.0", "classifier_C": 10.0},
    {"group": "hyperparameter", "label": "C = 100.0", "classifier_C": 100.0},
    # 4. 換一種特徵提取法（仍維持 <= 40 維）
    {"group": "extraction_method", "label": "PCA（基準）", "extractor": "pca"},
    {"group": "extraction_method", "label": "Kernel PCA (RBF)", "extractor": "kernel_pca"},
    {"group": "extraction_method", "label": "LDA（9 維上限）", "extractor": "lda"},
    # 5. 換掉線性分類器 —— 作業指定 Logistic Regression，這組只量「還差多少」
    {"group": "nonlinear", "label": "Logistic Regression（基準）", "classifier": "logistic"},
    {"group": "nonlinear", "label": "RBF SVM", "classifier": "svm"},
]


def _build(spec):
    """依單一實驗設定組出管線；未指定的欄位沿用基準組。"""
    base = baseline_config()
    n_components = spec.get("n_components", base["n_components"])
    classifier_C = spec.get("classifier_C", base["classifier_C"])
    whiten = spec.get("whiten", False)
    extractor = spec.get("extractor", "pca")

    if extractor == "kernel_pca":
        step = KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=0.01,
            random_state=RANDOM_STATE,
        )
    elif extractor == "lda":
        # LDA 最多只能投影到 (類別數 - 1) = 9 維，本身就遠低於 40 維上限。
        step = LinearDiscriminantAnalysis(n_components=9)
    else:
        step = PCA(n_components=n_components, whiten=whiten, random_state=RANDOM_STATE)

    if spec.get("classifier") == "svm":
        model = SVC(C=classifier_C, kernel="rbf", gamma="scale", random_state=RANDOM_STATE)
    else:
        model = LogisticRegression(
            C=classifier_C, max_iter=5000, random_state=RANDOM_STATE
        )

    return Pipeline([("scaler", StandardScaler()), ("pca", step), ("classifier", model)])


def run_study(output_dir=None):
    train = _load_train_module()
    X, y = train.load_digit_images()
    X_train, X_test, y_train, y_test = train.split_train_test(X, y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for spec in EXPERIMENTS:
        pipeline = _build(spec)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        results.append(
            {
                "group": spec["group"],
                "label": spec["label"],
                "cv_accuracy": float(scores.mean()),
                "cv_std": float(scores.std()),
            }
        )

    # 測試集只在最後確認一次，確認交出去的設定沒有被高估。
    chosen = _build({})
    chosen.fit(X_train, y_train)
    final_test_accuracy = float((chosen.predict(X_test) == y_test).mean())

    study = {
        "baseline": baseline_config(),
        "cv_folds": 5,
        "experiments": results,
        "final_test_accuracy": final_test_accuracy,
        "explained_variance_by_components": {
            str(n): float(
                PCA(n_components=n, random_state=RANDOM_STATE)
                .fit(StandardScaler().fit_transform(X_train))
                .explained_variance_ratio_.sum()
            )
            for n in (10, 20, 30, 40)
        },
    }

    destination = Path(output_dir) if output_dir is not None else ASSIGNMENT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / STUDY_FILENAME
    path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    return study, path


def main():
    study, path = run_study()
    by_group = {}
    for row in study["experiments"]:
        by_group.setdefault(row["group"], []).append(row)
    for group, rows in by_group.items():
        print(f"\n[{group}]")
        for row in rows:
            print(f"  {row['label']:<28} CV {row['cv_accuracy']:.4f} ± {row['cv_std']:.4f}")
    print(f"\n最終測試集正確率: {study['final_test_accuracy']:.4f}")
    print(f"研究結果已寫入: {path}")


if __name__ == "__main__":
    main()
