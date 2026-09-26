"""Assignment #3 - 手寫數字 (0-9) 多類別分類器：訓練與模型選取。

此腳本負責資料切分、特徵縮放、特徵選取、模型選取與模型儲存，
並以 joblib.dump() 將最佳管線輸出成 digits_pipeline_lr.pkl。

注意：依作業規定，繳交的 ZIP 只包含模型評估程式與模型檔，
      不得包含本檔案的任何訓練或模型選取程式碼。
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 44
TOTAL_PIXELS = 64
MODEL_FILENAME = "digits_pipeline_lr.pkl"


def load_digit_images():
    """載入 1797 張 8x8 灰階手寫數字影像（64 維像素特徵）。"""
    digits = load_digits()
    return digits.data, digits.target


def split_train_test(X, y):
    """80% 訓練 / 20% 測試；種子固定，評估程式才能重現同一組測試集。"""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def validate_feature_budget(mask, max_features=MAX_FEATURES):
    """確認特徵選取遮罩符合「至少淘汰 20 個特徵」的規定。"""
    selected = int(np.count_nonzero(mask))
    if selected > max_features:
        raise ValueError(
            f"feature selection kept {selected} pixels, "
            f"but the assignment allows at most {max_features}"
        )
    return mask


def build_pipeline():
    """特徵縮放 -> L1 特徵選取 -> Logistic Regression 分類器。"""
    selector_estimator = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "selector",
                SelectFromModel(
                    selector_estimator,
                    threshold=-np.inf,
                    max_features=MAX_FEATURES,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            ),
        ]
    )


def select_best_model(X_train, y_train):
    """在訓練集上以 5-fold 交叉驗證挑選最佳超參數組合。"""
    grid = {
        "selector__max_features": [32, 40, MAX_FEATURES],
        "classifier__C": [0.1, 1.0, 10.0],
    }
    search = GridSearchCV(
        build_pipeline(),
        param_grid=grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )
    search.fit(X_train, y_train)
    validate_feature_budget(extract_feature_mask(search.best_estimator_))
    return search


def extract_feature_mask(pipeline):
    """取出 64 維的特徵選取遮罩（True 代表該像素被選用）。"""
    return pipeline.named_steps["selector"].get_support()


def save_pipeline(pipeline, output_dir=None):
    """以 joblib.dump() 儲存整條管線（含前處理與模型）。"""
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    joblib.dump(pipeline, model_path, compress=3)
    return model_path


def run_training(output_dir=None):
    """完整訓練流程：切分 -> 交叉驗證模型選取 -> 驗證特徵上限 -> 儲存管線。"""
    X, y = load_digit_images()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    search = select_best_model(X_train, y_train)
    pipeline = search.best_estimator_
    mask = extract_feature_mask(pipeline)
    model_path = save_pipeline(pipeline, output_dir)

    return {
        "search": search,
        "pipeline": pipeline,
        "best_params": search.best_params_,
        "cv_accuracy": float(search.best_score_),
        "test_accuracy": float((pipeline.predict(X_test) == y_test).mean()),
        "feature_mask": mask,
        "selected_pixels": int(mask.sum()),
        "model_path": model_path,
    }


def main():
    result = run_training()
    print("最佳超參數:", result["best_params"])
    print(f"交叉驗證正確率: {result['cv_accuracy']:.4f}")
    print(f"測試集正確率  : {result['test_accuracy']:.4f}")
    print(f"選用像素數    : {result['selected_pixels']}/{TOTAL_PIXELS}")
    print(f"成功將管線（含預處理與模型）儲存至: '{result['model_path']}'")


if __name__ == "__main__":
    main()
