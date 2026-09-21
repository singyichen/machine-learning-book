"""Assignment #2 - 手刻 Logistic Regression 晶圓驗收分類器。

使用氧化層厚度偏差與微量金屬污染度預測晶圓是否合格。模型、損失函數、
梯度下降、標準化與決策邊界皆自行實作，不使用 scikit-learn。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


TRAIN_COLUMNS = ["x1", "x2", "y"]
TEST_COLUMNS = ["Wafer ID", "x1", "x2"]
TRANSFORM_LABELS = {
    "raw": "x1 (raw)",
    "abs": "abs(x1)",
    "square": "x1 ** 2",
}


def validate_assignment_data(train_df, test_df):
    """確認兩份 CSV 符合作業指定的欄位與基本資料條件。"""
    if list(train_df.columns) != TRAIN_COLUMNS:
        raise ValueError(
            f"wat_train.csv columns must be {TRAIN_COLUMNS}, "
            f"got {list(train_df.columns)}"
        )
    if list(test_df.columns) != TEST_COLUMNS:
        raise ValueError(
            f"wat_test.csv columns must be {TEST_COLUMNS}, "
            f"got {list(test_df.columns)}"
        )
    if train_df.isna().any().any() or test_df.isna().any().any():
        raise ValueError("Assignment data must not contain missing values")
    if not set(train_df["y"]).issubset({0, 1}):
        raise ValueError("wat_train.csv labels must be 0 or 1")


def load_assignment_data(data_dir=None):
    """從指定目錄讀取並驗證訓練集與待驗收資料。"""
    base_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent
    train_df = pd.read_csv(base_dir / "wat_train.csv")
    test_df = pd.read_csv(base_dir / "wat_test.csv")
    validate_assignment_data(train_df, test_df)
    return train_df, test_df


def sigmoid(values):
    """以避免 overflow 的寫法計算 sigmoid。"""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def binary_cross_entropy(y_true, probabilities):
    """計算 binary cross-entropy，並裁切機率以避免 log(0)。"""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    return float(
        np.mean(
            -y_true * np.log(probabilities)
            - (1.0 - y_true) * np.log(1.0 - probabilities)
        )
    )


class Standardizer:
    """只以 fit 資料計算統計值的 Z-score 標準化器。"""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        if np.any(self.scale_ == 0):
            raise ValueError("Cannot standardize a constant feature")
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LogisticRegressionGD:
    """以 full-batch gradient descent 訓練的二元 Logistic Regression。"""

    def __init__(self, learning_rate=0.1, epochs=1000):
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.weights_ = np.zeros(X.shape[1], dtype=float)
        self.bias_ = 0.0
        self.losses_ = []

        for _ in range(self.epochs):
            probabilities = sigmoid(X @ self.weights_ + self.bias_)
            self.losses_.append(binary_cross_entropy(y, probabilities))
            errors = probabilities - y
            self.weights_ -= self.learning_rate * (X.T @ errors) / len(y)
            self.bias_ -= self.learning_rate * errors.mean()
        return self

    def predict_proba(self, X):
        return sigmoid(np.asarray(X, dtype=float) @ self.weights_ + self.bias_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def build_features(frame, transformation):
    """依指定方式轉換 x1，並與原始 x2 組成兩欄特徵。"""
    x1 = frame["x1"].to_numpy(dtype=float)
    if transformation == "raw":
        transformed_x1 = x1
    elif transformation == "abs":
        transformed_x1 = np.abs(x1)
    elif transformation == "square":
        transformed_x1 = x1 ** 2
    else:
        raise ValueError(f"Unknown feature transformation: {transformation}")
    return np.column_stack((transformed_x1, frame["x2"].to_numpy(dtype=float)))


def compare_training_options(
    train_df,
    transformations=("raw", "abs", "square"),
    learning_rates=(0.01, 0.1, 0.3),
    epochs=1000,
):
    """訓練所有特徵轉換與學習率組合，回傳摘要與可重用的模型。"""
    y = train_df["y"].to_numpy(dtype=int)
    rows = []
    runs = {}
    for transformation in transformations:
        X = build_features(train_df, transformation)
        scaler = Standardizer().fit(X)
        X_std = scaler.transform(X)
        for learning_rate in learning_rates:
            model = LogisticRegressionGD(learning_rate, epochs).fit(X_std, y)
            accuracy = float(np.mean(model.predict(X_std) == y))
            key = (transformation, float(learning_rate))
            runs[key] = {"model": model, "scaler": scaler, "X_std": X_std}
            rows.append(
                {
                    "Feature Transformation": transformation,
                    "Learning Rate": float(learning_rate),
                    "Training Accuracy": accuracy,
                    "Initial Loss": model.losses_[0],
                    "Final Loss": model.losses_[-1],
                }
            )
    return pd.DataFrame(rows), runs


def _plot_loss_curves(comparison, runs, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in comparison.to_dict("records"):
        transformation = row["Feature Transformation"]
        learning_rate = row["Learning Rate"]
        losses = runs[(transformation, learning_rate)]["model"].losses_
        ax.plot(
            range(1, len(losses) + 1),
            losses,
            label=f"{TRANSFORM_LABELS[transformation]}, eta={learning_rate:g}",
            linewidth=1.4,
        )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Logistic Regression: Loss vs. Epochs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_decision_boundary(train_df, model, scaler, output_path):
    """在 abs(x1)-x2 原始單位平面繪製模型的決策區域。"""
    X = build_features(train_df, "abs")
    y = train_df["y"].to_numpy(dtype=int)
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(0, X[:, 0].max() + 0.5, 300),
        np.linspace(0, X[:, 1].max() + 0.5, 300),
    )
    grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
    regions = model.predict(scaler.transform(grid)).reshape(x1_grid.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(("#f5b8b5", "#b9d5f2"))
    ax.contourf(x1_grid, x2_grid, regions, alpha=0.65, cmap=cmap)
    styles = {
        0: ("#b22222", "s", "Fail (0)"),
        1: ("#174a8b", "o", "Pass (1)"),
    }
    for label, (color, marker, name) in styles.items():
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            color=color,
            marker=marker,
            edgecolor="black",
            s=58,
            label=name,
        )
    ax.set_xlabel("Absolute Oxide Thickness Deviation |x1| (Angstrom)")
    ax.set_ylabel("Metal Contamination x2 (ppm)")
    ax.set_title("Wafer Acceptance Decision Boundary")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _format_results(comparison, predictions):
    comparison_text = comparison.to_string(
        index=False,
        formatters={
            "Learning Rate": "{:.2f}".format,
            "Training Accuracy": "{:.2%}".format,
            "Initial Loss": "{:.6f}".format,
            "Final Loss": "{:.6f}".format,
        },
    )
    display_predictions = predictions.copy()
    display_predictions["Pass Probability"] = display_predictions[
        "Pass Probability"
    ].map(lambda value: f"{value:.2%}")
    prediction_text = display_predictions.to_string(index=False)
    return (
        "Feature / learning-rate comparison\n"
        "==================================\n"
        f"{comparison_text}\n\n"
        "Selected model: abs(x1), learning rate=0.3\n"
        "Reason: abs(x1) is physically interpretable, reaches 100% training accuracy, "
        "and reproduces the assignment's reference output.\n\n"
        "Wafer acceptance results\n"
        "========================\n"
        f"{prediction_text}\n"
    )


def run_assignment(data_dir=None, output_dir=None, epochs=1000):
    """執行完整作業流程並寫出圖表與驗收結果。"""
    source_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent
    destination = Path(output_dir) if output_dir is not None else source_dir
    destination.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_assignment_data(source_dir)

    comparison, runs = compare_training_options(train_df, epochs=epochs)
    selected = runs[("abs", 0.3)]
    X_test = build_features(test_df, "abs")
    X_test_std = selected["scaler"].transform(X_test)
    probabilities = selected["model"].predict_proba(X_test_std)
    predictions = test_df.copy()
    predictions.insert(3, "abs_x1", np.abs(predictions["x1"]))
    predictions["Pass Probability"] = probabilities
    predictions["Result"] = np.where(probabilities >= 0.5, "Pass", "Fail")

    _plot_loss_curves(
        comparison,
        runs,
        destination / "assignment2_loss_curves.png",
    )
    _plot_decision_boundary(
        train_df,
        selected["model"],
        selected["scaler"],
        destination / "assignment2_decision_boundary.png",
    )
    predictions.to_csv(destination / "assignment2_predictions.csv", index=False)
    report_text = _format_results(comparison, predictions)
    (destination / "assignment2_results.txt").write_text(report_text, encoding="utf-8")

    return {
        "comparison": comparison,
        "predictions": predictions,
        "model": selected["model"],
        "scaler": selected["scaler"],
    }


def main():
    result = run_assignment()
    print(_format_results(result["comparison"], result["predictions"]))


if __name__ == "__main__":
    main()
