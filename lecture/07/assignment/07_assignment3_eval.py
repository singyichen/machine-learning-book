"""Assignment #3 - 手寫數字 (0-9) 多類別分類器：模型評估。

本檔為繳交用的模型評估程式，僅載入已訓練好的管線 digits_pipeline_lr.pkl
並在保留的測試集上評估，不包含任何訓練或模型選取程式碼。

執行方式（與 digits_pipeline_lr.pkl 放在同一資料夾）：

    python 07_assignment3_eval.py
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
MODEL_FILENAME = "digits_pipeline_lr.pkl"
MASK_FIGURE = "assignment3_feature_mask.png"
CONFUSION_FIGURE = "assignment3_confusion_matrix.png"
MISCLASSIFIED_FIGURE = "assignment3_misclassified.png"
MISCLASSIFIED_COLUMNS = 5
RESULTS_FILENAME = "assignment3_results.txt"


def load_test_set():
    """重建與訓練時相同的 20% 保留測試集（約 360 筆），僅取測試部分。"""
    digits = load_digits()
    _, X_test, _, y_test = train_test_split(
        digits.data, digits.target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_test, y_test


def load_model(model_path=None):
    """載入含預處理與模型的完整管線。"""
    path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / MODEL_FILENAME
    return joblib.load(path)


def extract_feature_mask(model):
    """自管線取出 64 維特徵選取遮罩（True 代表該像素被選用）。"""
    return model.named_steps["selector"].get_support()


def feature_mask_title(mask):
    return f"Feature Selection Mask (Selected: {int(np.count_nonzero(mask))}/{TOTAL_PIXELS})"


def make_feature_mask_figure(mask):
    """以 8x8 遮罩圖呈現特徵選取結果：0 為淘汰、1 為選取（樣式比照投影片 p.59）。"""
    grid = np.asarray(mask, dtype=int).reshape(IMAGE_SHAPE)

    fig, ax = plt.subplots(figsize=(6.0, 6.2), dpi=200)
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
    for row in range(IMAGE_SHAPE[0]):
        for column in range(IMAGE_SHAPE[1]):
            value = grid[row, column]
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value else "#1f3b63",
            )
    ax.set_title(feature_mask_title(mask), fontsize=12, pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_confusion_matrix_figure(matrix):
    """視覺化呈現測試集混淆矩陣，每格（含 0）都標數字（樣式比照投影片 p.60）。"""
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
    """繪出所有誤判樣本：標題為真實標籤，左下角紅字為預測結果（樣式比照投影片 p.61）。"""
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


def _format_results(result, mask):
    lines = [
        "Assignment #3 - 低解析度手寫數字 (0-9) 多類別分類器",
        "=" * 56,
        "",
        f"測試集樣本數      : {len(result['y_pred'])}",
        f"選用像素數        : {int(np.count_nonzero(mask))}/{TOTAL_PIXELS}"
        f"（淘汰 {TOTAL_PIXELS - int(np.count_nonzero(mask))} 個）",
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
    """載入模型、評估測試集，並輸出三張圖與文字結果。"""
    destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    X_test, y_test = load_test_set()
    mask = extract_feature_mask(model)
    result = evaluate(model, X_test, y_test)

    save_figure(make_feature_mask_figure(mask), destination / MASK_FIGURE)
    save_figure(
        make_confusion_matrix_figure(result["confusion_matrix"]), destination / CONFUSION_FIGURE
    )
    save_figure(
        make_misclassified_figure(X_test, y_test, result["y_pred"], result["misclassified"]),
        destination / MISCLASSIFIED_FIGURE,
    )
    (destination / RESULTS_FILENAME).write_text(
        _format_results(result, mask), encoding="utf-8"
    )

    result["feature_mask"] = mask
    result["output_dir"] = destination
    return result


def main():
    result = run_evaluation()
    print("模型與預處理管線載入成功！")
    print(_format_results(result, result["feature_mask"]))


if __name__ == "__main__":
    main()
