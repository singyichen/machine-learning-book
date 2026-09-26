"""Assignment #3 - 改善空間研究。

量測「在作業限制內還能做得更好嗎」，結果寫成 assignment3_improvement_study.json，
供 build_report.py 產生報告的討論章節。

所有比較都在訓練集上以 5-fold 交叉驗證進行；測試集只在最後做一次確認性評估，
不用於挑選任何設定。此檔為分析工具，不屬於繳交內容。

    python 07_assignment3_explore.py
"""

from datetime import date
from pathlib import Path
import json
import warnings

import numpy as np
from scipy.ndimage import shift as ndshift
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ASSIGNMENT_DIR = Path(__file__).resolve().parent
STUDY_FILENAME = "assignment3_improvement_study.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
FOLDS = 5
TOTAL_PIXELS = 64

# 繳交管線實際採用的設定，其餘實驗都以此為基準比較。
BASELINE = {"selector_C": 0.1, "max_features": 40, "classifier_C": 1.0}

# 影像邊角有少數像素在整個訓練集恆為 0，ANOVA F 檢定對這些像素會除以零。
# 這正是特徵選取幾乎不損失資訊的原因，屬預期行為，不需修正。
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def baseline_config():
    return dict(BASELINE)


def _lr(C=1.0):
    return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, C=C)


def _l1_selector(C, k):
    return SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=RANDOM_STATE),
        threshold=-np.inf,
        max_features=k,
    )


def _pipeline(selector, classifier):
    steps = [("scaler", StandardScaler())]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


# group 決定這筆結果會出現在報告的哪一張表。
EXPERIMENTS = [
    # --- 特徵數量上限的代價 ---
    {"group": "feature_budget", "label": "64 個（不受作業限制）", "baseline": False,
     "make": lambda: _pipeline(None, _lr(1.0))},
    {"group": "feature_budget", "label": "44 個（作業上限）", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.1, 44), _lr(1.0))},
    {"group": "feature_budget", "label": "40 個（現況選用）", "baseline": True,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(1.0))},
    {"group": "feature_budget", "label": "32 個", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.1, 32), _lr(1.0))},

    # --- 換特徵選取方法（都固定 44 個）---
    {"group": "selection_method", "label": "L1 權重排序（現況做法）", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.1, 44), _lr(1.0))},
    {"group": "selection_method", "label": "ANOVA F 檢定 (SelectKBest)", "baseline": False,
     "make": lambda: _pipeline(SelectKBest(f_classif, k=44), _lr(1.0))},
    {"group": "selection_method", "label": "互資訊 (mutual information)", "baseline": False,
     "make": lambda: _pipeline(SelectKBest(mutual_info_classif, k=44), _lr(1.0))},
    {"group": "selection_method", "label": "RFE 遞迴特徵消除", "baseline": False,
     "make": lambda: _pipeline(RFE(_lr(1.0), n_features_to_select=44, step=4), _lr(1.0))},

    # --- 超參數 ---
    {"group": "hyperparameter", "label": "分類器 C = 0.3", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(0.3))},
    {"group": "hyperparameter", "label": "分類器 C = 1.0（現況）", "baseline": True,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(1.0))},
    {"group": "hyperparameter", "label": "分類器 C = 3.0", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(3.0))},
    {"group": "hyperparameter", "label": "L1 選取器 C = 0.01", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.01, 40), _lr(1.0))},
    {"group": "hyperparameter", "label": "L1 選取器 C = 0.1（現況）", "baseline": True,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(1.0))},
    {"group": "hyperparameter", "label": "L1 選取器 C = 0.5", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.5, 40), _lr(1.0))},

    # --- 資料擴增（需要特別處理，見 _score_augmented）---
    {"group": "augmentation", "label": "不擴增（現況）", "baseline": True,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(1.0))},
    {"group": "augmentation", "label": "影像平移 1 像素，訓練資料 ×5", "baseline": False,
     "augment": True,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(1.0))},

    # --- 非線性模型對照（作業指定用 Logistic Regression，此組僅供討論）---
    {"group": "nonlinear", "label": "Logistic Regression（線性，現況）", "baseline": True,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), _lr(1.0))},
    {"group": "nonlinear", "label": "SVM (RBF 核)，同樣只用 40 個像素", "baseline": False,
     "make": lambda: _pipeline(_l1_selector(0.1, 40), SVC(kernel="rbf", random_state=RANDOM_STATE))},
]


def augment_by_shifting(X, y):
    """把 8x8 影像上下左右各平移 1 像素，訓練資料變 5 倍。"""
    images = X.reshape(-1, 8, 8)
    features, labels = [X], [y]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        moved = np.array(
            [ndshift(image, (dy, dx), order=0, mode="constant", cval=0.0) for image in images]
        )
        features.append(moved.reshape(-1, TOTAL_PIXELS))
        labels.append(y)
    return np.vstack(features), np.concatenate(labels)


def _score_augmented(pipeline, X_train, y_train, cv):
    """擴增只施加在訓練折，驗證折保持原樣，避免把擴增樣本也拿去驗證。"""
    scores = []
    for train_index, validation_index in cv.split(X_train, y_train):
        X_aug, y_aug = augment_by_shifting(X_train[train_index], y_train[train_index])
        model = clone(pipeline).fit(X_aug, y_aug)
        scores.append((model.predict(X_train[validation_index]) == y_train[validation_index]).mean())
    return np.array(scores)


def run_study(output_dir=None):
    destination = Path(output_dir) if output_dir is not None else ASSIGNMENT_DIR
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for spec in EXPERIMENTS:
        pipeline = spec["make"]()
        if spec.get("augment"):
            scores = _score_augmented(pipeline, X_train, y_train, cv)
        else:
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        results.append({
            "group": spec["group"],
            "label": spec["label"],
            "baseline": spec["baseline"],
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
        })
        print(f"  {spec['label']:<34} CV = {scores.mean():.4f} ± {scores.std():.4f}", flush=True)

    # 測試集只在這裡用一次，確認「CV 較佳」不等於「測試集較佳」。
    print("\n  測試集確認（不用於挑選設定）：", flush=True)
    test_results = []
    for label, pipeline in [
        ("現況：L1 選 40 個 + LR", _pipeline(_l1_selector(0.1, 40), _lr(1.0))),
        ("CV 最佳：ANOVA F 選 44 個 + LR", _pipeline(SelectKBest(f_classif, k=44), _lr(1.0))),
    ]:
        accuracy = float((pipeline.fit(X_train, y_train).predict(X_test) == y_test).mean())
        errors = int(round((1 - accuracy) * len(y_test)))
        test_results.append({"label": label, "accuracy": accuracy, "errors": errors})
        print(f"    {label:<32} 測試集 = {accuracy:.4f}（錯 {errors} 筆）", flush=True)

    study = {
        "generated": date.today().isoformat(),
        "cv_folds": FOLDS,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "baseline": baseline_config(),
        "experiments": results,
        "test_set": test_results,
    }
    output_path = destination / STUDY_FILENAME
    output_path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    return study, output_path


def main():
    print("改善空間研究（5-fold 交叉驗證，僅使用訓練集）\n")
    _, output_path = run_study()
    print(f"\n結果已寫入：{output_path}")


if __name__ == "__main__":
    main()
