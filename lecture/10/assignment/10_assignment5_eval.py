"""Assignment #5 - 手寫數字 (0-9) 多類別分類器：模型評估（PCA 特徵提取）。

本檔為繳交用的模型評估程式，僅載入已訓練好的管線 digits_pipeline_lr_pca.pkl
並在保留的測試集上評估，不包含任何訓練或模型選取程式碼。

執行方式（與 digits_pipeline_lr_pca.pkl 放在同一資料夾）：

    python 10_assignment5_eval.py
"""

from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend 必須先設定)


TEST_SIZE = 0.2
RANDOM_STATE = 42
TOTAL_PIXELS = 64
IMAGE_SHAPE = (8, 8)
MODEL_FILENAME = "digits_pipeline_lr_pca.pkl"
CONFUSION_FIGURE = "assignment5_confusion_matrix.png"
MISCLASSIFIED_FIGURE = "assignment5_misclassified.png"
MISCLASSIFIED_COLUMNS = 5
RESULTS_FILENAME = "assignment5_results.txt"


def load_test_set():
    """重建與訓練時相同的 20% 保留測試集（約 360 筆），僅取測試部分。"""
    digits = load_digits()
    _, X_test, _, y_test = train_test_split(
        digits.data, digits.target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_test, y_test


def load_model(model_path=None):
    """載入含預處理與模型的完整管線。"""
    path = (
        Path(model_path)
        if model_path is not None
        else Path(__file__).resolve().parent / MODEL_FILENAME
    )
    return joblib.load(path)


def extract_n_components(model):
    """自管線取出 PCA 實際保留的主成分數量。"""
    return int(model.named_steps["pca"].n_components_)


def total_explained_variance(model):
    """總解釋變異量：各保留主成分的解釋變異比例之和。"""
    return float(model.named_steps["pca"].explained_variance_ratio_.sum())


def dimension_line(n_components, total_pixels=TOTAL_PIXELS):
    return (
        f"PCA 提取後的特徵維度 (n_components): {n_components} "
        f"(從原本的 {total_pixels} 維降至 {n_components} 維)"
    )


def explained_variance_line(ratio):
    return f"特徵提取的總解釋變異量: {ratio * 100:.2f}%"


def make_confusion_matrix_figure(matrix):
    """視覺化呈現測試集混淆矩陣，每格（含 0）都標數字（樣式比照投影片 p.55）。"""
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    ax.imshow(matrix, cmap="Blues", alpha=0.3)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.tight_layout()
    return fig


def find_misclassified(y_test, y_pred):
    """回傳測試集中預測錯誤的樣本索引（依測試集順序）。"""
    return np.flatnonzero(np.asarray(y_pred) != np.asarray(y_test))


def misclassified_message(count):
    return f"測試集總共誤判了 {count} 張圖片。"


def make_misclassified_figure(X_test, y_test, y_pred, indices):
    """繪出所有誤判樣本：標題為真實標籤，左下角紅字為預測結果（樣式比照投影片 p.56）。"""
    rows = max(1, -(-len(indices) // MISCLASSIFIED_COLUMNS))
    fig, axes = plt.subplots(
        rows,
        MISCLASSIFIED_COLUMNS,
        figsize=(2.4 * MISCLASSIFIED_COLUMNS, 2.8 * rows),
        dpi=200,
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, index in zip(axes.ravel(), indices):
        ax.imshow(X_test[index].reshape(IMAGE_SHAPE), cmap="gray")
        ax.set_title(f"True: {y_test[index]}", fontsize=11)
        ax.text(
            0.04,
            0.04,
            str(y_pred[index]),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="red",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "edgecolor": "none"},
        )
    fig.tight_layout(h_pad=1.5)
    return fig


def save_figure(fig, output_path):
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def evaluate(model, X_test, y_test):
    """在測試集上評估：正確率、分類報告、混淆矩陣與誤判樣本。"""
    y_pred = model.predict(X_test)
    return {
        "y_pred": y_pred,
        "misclassified": find_misclassified(y_test, y_pred),
        "accuracy": float((y_pred == y_test).mean()),
        "report": classification_report(y_test, y_pred, digits=2),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def _format_results(result):
    lines = [
        "Assignment #5 - 低解析度手寫數字 (0-9) 多類別分類器（PCA 特徵提取）",
        "=" * 56,
        "",
        f"測試集樣本數      : {len(result['y_pred'])}",
        dimension_line(result["n_components"]),
        explained_variance_line(result["explained_variance"]),
        f"整體正確率 (Accuracy): {result['accuracy']:.4f}",
        "",
        "詳細分類報告 (Classification Report):",
        result["report"],
        "",
        "混淆矩陣 (Confusion Matrix):",
        np.array2string(result["confusion_matrix"]),
        "",
        misclassified_message(len(result["misclassified"])),
        "",
    ]
    return "\n".join(lines)


def run_evaluation(model_path=None, output_dir=None):
    """載入模型、評估測試集，並輸出兩張圖與文字結果。"""
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    X_test, y_test = load_test_set()
    result = evaluate(model, X_test, y_test)
    result["n_components"] = extract_n_components(model)
    result["explained_variance"] = total_explained_variance(model)

    save_figure(
        make_confusion_matrix_figure(result["confusion_matrix"]), destination / CONFUSION_FIGURE
    )
    save_figure(
        make_misclassified_figure(X_test, y_test, result["y_pred"], result["misclassified"]),
        destination / MISCLASSIFIED_FIGURE,
    )
    (destination / RESULTS_FILENAME).write_text(_format_results(result), encoding="utf-8")

    result["output_dir"] = destination
    return result


def main():
    result = run_evaluation()
    print("模型與預處理管線載入成功！")
    print(_format_results(result))


if __name__ == "__main__":
    main()
