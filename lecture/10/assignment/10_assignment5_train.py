"""Assignment #5 - 手寫數字 (0-9) 多類別分類器：訓練與模型選取（PCA 特徵提取）。

此腳本負責資料切分、特徵縮放、特徵提取 (PCA)、模型選取與模型儲存，
並以 joblib.dump() 將最佳管線輸出成 digits_pipeline_lr_pca.pkl。

與 Assignment #3 的差別：特徵「選取」(SelectFromModel) 換成特徵「提取」(PCA)。
前者是從 64 個像素裡挑出子集合，後者是把 64 維投影到新的主成分座標系。

注意：依作業規定，繳交的 ZIP 只包含模型評估程式與模型檔，
      不得包含本檔案的任何訓練或模型選取程式碼。
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_COMPONENTS = 40
TOTAL_PIXELS = 64
MODEL_FILENAME = "digits_pipeline_lr_pca.pkl"

PARAM_GRID = {
    "pca__n_components": [20, 30, 40],
    "classifier__C": [0.1, 1.0, 10.0],
}


def load_digit_images():
    """載入 1797 張 8x8 灰階手寫數字影像（64 維像素特徵）。"""
    digits = load_digits()
    return digits.data, digits.target


def split_train_test(X, y):
    """80% 訓練 / 20% 測試；種子固定，評估程式才能重現同一組測試集。"""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def validate_component_budget(n_components, max_components=MAX_COMPONENTS):
    """確認 PCA 降維後的維度符合「特徵空間維度 <= 40」的規定。"""
    if n_components > max_components:
        raise ValueError(
            f"PCA kept {n_components} components, "
            f"but the assignment allows at most {max_components}"
        )
    return n_components


def build_pipeline():
    """特徵縮放 -> PCA 特徵提取 -> Logistic Regression 分類器。

    PCA 以變異數為準挑主軸，像素尺度不一會讓主軸被大尺度像素綁架，
    所以 StandardScaler 必須排在 PCA 之前。
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=MAX_COMPONENTS, random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            ),
        ]
    )


def select_best_model(X_train, y_train):
    """在訓練集上以 5-fold 交叉驗證挑選最佳超參數組合。"""
    search = GridSearchCV(
        build_pipeline(),
        param_grid=PARAM_GRID,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
    )
    search.fit(X_train, y_train)
    validate_component_budget(extract_n_components(search.best_estimator_))
    return search


def extract_n_components(pipeline):
    """取出 PCA 實際保留的主成分數量。"""
    return int(pipeline.named_steps["pca"].n_components_)


def extract_explained_variance(pipeline):
    """取出 PCA 的總解釋變異量（各主成分解釋變異比例之和）。"""
    return float(pipeline.named_steps["pca"].explained_variance_ratio_.sum())


def save_pipeline(pipeline, output_dir=None):
    """以 joblib.dump() 儲存整條管線（含前處理與模型）。"""
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / MODEL_FILENAME
    joblib.dump(pipeline, model_path, compress=3)
    return model_path


def run_training(output_dir=None):
    """完整訓練流程：切分 -> 交叉驗證模型選取 -> 驗證維度上限 -> 儲存管線。"""
    X, y = load_digit_images()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    search = select_best_model(X_train, y_train)
    pipeline = search.best_estimator_
    model_path = save_pipeline(pipeline, output_dir)

    return {
        "search": search,
        "pipeline": pipeline,
        "best_params": search.best_params_,
        "cv_accuracy": float(search.best_score_),
        "test_accuracy": float((pipeline.predict(X_test) == y_test).mean()),
        "n_components": extract_n_components(pipeline),
        "explained_variance": extract_explained_variance(pipeline),
        "model_path": model_path,
    }


def main():
    result = run_training()
    print("最佳超參數:", result["best_params"])
    print(f"交叉驗證正確率: {result['cv_accuracy']:.4f}")
    print(f"測試集正確率  : {result['test_accuracy']:.4f}")
    print(f"PCA 維度      : {result['n_components']}/{TOTAL_PIXELS}")
    print(f"總解釋變異量  : {result['explained_variance'] * 100:.2f}%")
    print(f"成功將管線（含預處理與模型）儲存至: '{result['model_path']}'")


if __name__ == "__main__":
    main()
