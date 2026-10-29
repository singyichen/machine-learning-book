"""Assignment #6 - 改善空間研究（不繳交）。

回答「在作業限制內還能做得更好嗎」。糖尿病資料集只有 442 筆，單次 80/20 切分的
測試集僅 89 筆，換個種子分數就會跳動好幾個百分點，所以泛化能力一律以訓練集上的
5-fold 交叉驗證估計；測試集只在最後確認一次，不參與任何設定的挑選。

輸出：assignment6_improvement_study.json（報告的「改善空間討論」章節由此帶數字）
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ASSIGNMENT_DIR = Path(__file__).resolve().parent
STUDY_FILENAME = "assignment6_improvement_study.json"
RANDOM_STATE = 42


def _load_assignment():
    path = ASSIGNMENT_DIR / "11_assignment6.py"
    spec = spec_from_file_location("assignment6_for_study", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPERIMENTS = [
    # 1. 正規化：投影片 p.45 說 Lasso / Ridge 表現最穩定（0.47 ~ 0.50）
    {"group": "regularization", "label": "多元線性回歸（基準）", "model": "linear"},
    {"group": "regularization", "label": "Ridge (alpha=1.0)", "model": "ridge", "alpha": 1.0},
    {"group": "regularization", "label": "Ridge (alpha=10.0)", "model": "ridge", "alpha": 10.0},
    {"group": "regularization", "label": "Lasso (alpha=0.1)", "model": "lasso", "alpha": 0.1},
    {"group": "regularization", "label": "Lasso (alpha=1.0)", "model": "lasso", "alpha": 1.0},
    {"group": "regularization", "label": "ElasticNet (alpha=0.1)", "model": "elasticnet",
     "alpha": 0.1},
    # 2. 多項式回歸的維度災難：加上正規化能救回多少？
    {"group": "polynomial_fix", "label": "degree=2 不正規化（基準）", "model": "poly_linear"},
    {"group": "polynomial_fix", "label": "degree=2 + Ridge (alpha=10)", "model": "poly_ridge",
     "alpha": 10.0},
    {"group": "polynomial_fix", "label": "degree=2 + Ridge (alpha=100)", "model": "poly_ridge",
     "alpha": 100.0},
    {"group": "polynomial_fix", "label": "degree=2 + Lasso (alpha=1.0)", "model": "poly_lasso",
     "alpha": 1.0},
    # 3. MLP 容量：投影片 p.45 說架構必須非常小，實際驗證一次
    {"group": "mlp_capacity", "label": "隱藏層 (8,)", "model": "mlp", "hidden": (8,)},
    {"group": "mlp_capacity", "label": "隱藏層 (32, 16)（基準）", "model": "mlp",
     "hidden": (32, 16)},
    {"group": "mlp_capacity", "label": "隱藏層 (128, 64)", "model": "mlp", "hidden": (128, 64)},
    {"group": "mlp_capacity", "label": "隱藏層 (256, 128, 64)", "model": "mlp",
     "hidden": (256, 128, 64)},
    # 4. 作業指定以外的演算法，量一下「還差多少」
    {"group": "other_algorithms", "label": "KNN (k=9)", "model": "knn", "k": 9},
    {"group": "other_algorithms", "label": "KNN (k=15)", "model": "knn", "k": 15},
    {"group": "other_algorithms", "label": "隨機森林 (300 棵)", "model": "forest"},
]


def _build(spec):
    alpha = spec.get("alpha", 1.0)
    kind = spec["model"]

    regressors = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=alpha, solver="svd"),
        "lasso": Lasso(alpha=alpha, max_iter=20000, random_state=RANDOM_STATE),
        "elasticnet": ElasticNet(alpha=alpha, max_iter=20000, random_state=RANDOM_STATE),
        "knn": KNeighborsRegressor(n_neighbors=spec.get("k", 9)),
        "forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
    }

    if kind.startswith("poly_"):
        tail = {
            "poly_linear": LinearRegression(),
            "poly_ridge": Ridge(alpha=alpha, solver="svd"),
            "poly_lasso": Lasso(alpha=alpha, max_iter=50000, random_state=RANDOM_STATE),
        }[kind]
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("regressor", tail),
            ]
        )

    return Pipeline([("scaler", StandardScaler()), ("regressor", regressors[kind])])


def _cv_r2_for_mlp(assignment, X_train, y_train, hidden, cv):
    """MLP 不是 sklearn estimator，手動跑交叉驗證。"""
    scores = []
    for fit_index, score_index in cv.split(X_train):
        history = assignment.train_mlp(
            X_train[fit_index], y_train[fit_index], hidden_sizes=tuple(hidden)
        )
        prediction = history["predict"](X_train[score_index])
        scores.append(assignment.evaluate_predictions(y_train[score_index], prediction)["r2"])
    return np.array(scores)


def run_study(output_dir=None):
    assignment = _load_assignment()
    X, y = assignment.load_diabetes_data()
    X_train, X_test, y_train, y_test = assignment.split_train_test(X, y)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for spec in EXPERIMENTS:
        if spec["model"] == "mlp":
            scores = _cv_r2_for_mlp(assignment, X_train, y_train, spec["hidden"], cv)
        else:
            scores = cross_val_score(_build(spec), X_train, y_train, cv=cv, scoring="r2")
        results.append(
            {
                "group": spec["group"],
                "label": spec["label"],
                "cv_r2": float(scores.mean()),
                "cv_std": float(scores.std()),
            }
        )

    # 測試集只在最後確認一次，確認交出去的三個模型沒有被高估。
    final = assignment.run_all_models()
    study = {
        "cv_folds": 5,
        "experiments": results,
        "final_test_metrics": {
            name: {"mse": row["mse"], "r2": row["r2"]} for name, row in final.items()
        },
        "mlp_best_epoch": final["PyTorch MLP Regressor"]["best_epoch"],
        "n_test": int(len(y_test)),
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
            print(f"  {row['label']:<32} CV R² {row['cv_r2']:+.4f} ± {row['cv_std']:.4f}")
    print("\n最終測試集：")
    for name, metrics in study["final_test_metrics"].items():
        print(f"  {name:<24} MSE {metrics['mse']:8.2f} | R² {metrics['r2']:.4f}")
    print(f"\n研究結果已寫入: {path}")


if __name__ == "__main__":
    main()
