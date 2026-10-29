"""Assignment #6 - 基準一年後糖尿病疾病進展之預測器。

應用情境為臨床醫學定量評估：假設病患的基礎生理特徵與化驗數據已由醫院預先採集
完成，本作業僅負責建立回歸模型來預測一年後的疾病進展指數。

以三種演算法建立回歸預測器並比較效能：
  1. 多元線性回歸    —— 基礎線性模型，作為效能 baseline
  2. 多項式回歸      —— PolynomialFeatures(degree=2) 特徵轉換
  3. MLP 非線性回歸  —— 以 PyTorch 實作的 MLP regressor

執行方式：

    python 11_assignment6.py
"""

from pathlib import Path
import random

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend 必須先設定)


TEST_SIZE = 0.2
RANDOM_STATE = 42
VALID_SIZE = 0.2          # 自訓練集再切出的驗證集，供 MLP 提早停止使用
MAX_EPOCHS = 2000
PATIENCE = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN_SIZES = (32, 16)

COMPARISON_FIGURE = "assignment6_model_comparison.png"
RESULTS_FILENAME = "assignment6_results.txt"
MODEL_ORDER = ("Linear Regression", "Polynomial Regression", "PyTorch MLP Regressor")
BAR_COLORS = ("#4c72b0", "#dd8452", "#55a868")
Y_LIMITS = (-0.2, 1.0)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_diabetes_data():
    """442 筆病患臨床紀錄，10 維生理與化驗特徵，目標值介於 25 到 346。"""
    X, y = load_diabetes(return_X_y=True)
    return X, y


def split_train_test(X, y):
    """80% 訓練集 / 20% 測試集（約 89 筆）；種子固定以利重現。"""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def evaluate_predictions(y_true, y_pred):
    """模型評估指標：均方誤差 (MSE) 與決定係數 (R² Score)。"""
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# 模型 1：多元線性回歸（效能 baseline）
# --------------------------------------------------------------------------
def build_linear_model():
    """load_diabetes 的特徵已預先縮放，縮放步驟仍保留以統一三個模型的流程。"""
    return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])


# --------------------------------------------------------------------------
# 模型 2：多項式回歸（PolynomialFeatures degree=2）
# --------------------------------------------------------------------------
def build_polynomial_model():
    """10 維展開成 65 維（10 原始 + 10 平方 + 45 交互作用）後再做線性回歸。"""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


# --------------------------------------------------------------------------
# 模型 3：PyTorch MLP regressor
# --------------------------------------------------------------------------
def build_mlp(n_features, hidden_sizes=None):
    """刻意的小架構：442 筆樣本撐不起大網路，參數一多就直接背答案。"""
    hidden_sizes = HIDDEN_SIZES if hidden_sizes is None else hidden_sizes
    layers = []
    in_features = n_features
    for hidden in hidden_sizes:
        layers += [nn.Linear(in_features, hidden), nn.ReLU()]
        in_features = hidden
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


def train_mlp(X_train, y_train, hidden_sizes=None):
    """以訓練集內部切出的驗證集做提早停止，回傳含 predict() 的訓練歷程。

    測試集完全不參與訓練與模型挑選；predict() 會自行套用訓練集統計量做標準化，
    並把標準化後的輸出還原回原始尺度。
    """
    set_seed()

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train, y_train, test_size=VALID_SIZE, random_state=RANDOM_STATE
    )

    x_scaler = StandardScaler().fit(X_fit)
    y_mean, y_std = float(y_fit.mean()), float(y_fit.std())

    def to_tensor(X, y=None):
        features = torch.tensor(x_scaler.transform(X), dtype=torch.float32)
        if y is None:
            return features
        target = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
        return features, target

    X_fit_t, y_fit_t = to_tensor(X_fit, y_fit)
    X_valid_t, y_valid_t = to_tensor(X_valid, y_valid)

    model = build_mlp(X_train.shape[1], hidden_sizes)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_losses, valid_losses = [], []
    best_loss, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss = loss_fn(model(X_fit_t), y_fit_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(X_valid_t), y_valid_t))
        valid_losses.append(valid_loss)

        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break   # 提早停止：驗證集連續 PATIENCE 輪沒有進步

    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            standardized = model(to_tensor(X)).squeeze(1).numpy()
        return standardized * y_std + y_mean

    return {
        "model": model,
        "predict": predict,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "n_train": len(y_fit),
        "n_valid": len(y_valid),
    }


# --------------------------------------------------------------------------
# 三模型比較
# --------------------------------------------------------------------------
def run_all_models():
    """訓練三種模型並在同一組測試集上評估，回傳依投影片順序排列的結果。"""
    set_seed()
    X, y = load_diabetes_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    results = {}

    linear = build_linear_model().fit(X_train, y_train)
    y_pred = linear.predict(X_test)
    results["Linear Regression"] = {**evaluate_predictions(y_test, y_pred), "y_pred": y_pred}

    polynomial = build_polynomial_model().fit(X_train, y_train)
    y_pred = polynomial.predict(X_test)
    results["Polynomial Regression"] = {**evaluate_predictions(y_test, y_pred), "y_pred": y_pred}

    history = train_mlp(X_train, y_train)
    y_pred = history["predict"](X_test)
    results["PyTorch MLP Regressor"] = {
        **evaluate_predictions(y_test, y_pred),
        "y_pred": y_pred,
        "best_epoch": history["best_epoch"],
    }

    return {name: results[name] for name in MODEL_ORDER}


def format_comparison_table(results):
    """預期結果 1：三種模型的測試集量化指標比較表（樣式比照投影片 p.43）。"""
    lines = ["========== 各模型效能比較 =========="]
    for name, row in results.items():
        lines.append(f"[{name}] MSE: {row['mse']:.2f} | R² Score: {row['r2']:.4f}")
    return "\n".join(lines)


def make_comparison_figure(results):
    """預期結果 2：三種模型 R² Score 的長條圖（樣式比照投影片 p.44）。"""
    names = list(results)
    scores = [results[name]["r2"] for name in names]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    bars = ax.bar(names, scores, color=BAR_COLORS[: len(names)], width=0.6)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("R² Score")
    ax.set_title("Comparison of Regression Models (Test Set)")
    ax.set_ylim(*Y_LIMITS)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _format_results(results):
    best_name = max(results, key=lambda name: results[name]["r2"])
    lines = [
        "Assignment #6 - 基準一年後糖尿病疾病進展之預測器",
        "=" * 56,
        "",
        "資料集: sklearn.datasets.load_diabetes()（442 筆、10 維特徵）",
        f"切分比例: {int((1 - TEST_SIZE) * 100)}% 訓練集 / {int(TEST_SIZE * 100)}% 測試集"
        f"（測試集 {len(results[best_name]['y_pred'])} 筆）",
        "",
        format_comparison_table(results),
        "",
        f"最佳模型: {best_name}（R² Score {results[best_name]['r2']:.4f}）",
        "",
    ]
    return "\n".join(lines)


def main(output_dir=None):
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)

    results = run_all_models()

    print()
    print(format_comparison_table(results))
    print()

    figure = make_comparison_figure(results)
    figure.savefig(destination / COMPARISON_FIGURE)
    plt.close(figure)

    (destination / RESULTS_FILENAME).write_text(_format_results(results), encoding="utf-8")
    print(f"長條圖已儲存至: '{destination / COMPARISON_FIGURE}'")
    print(f"文字結果已儲存至: '{destination / RESULTS_FILENAME}'")
    return results


if __name__ == "__main__":
    main()
