"""Assignment #2 - 改善空間研究。

量測「在作業限制內還能做得更好嗎」，結果寫成 assignment2_improvement_study.json，
供 build_report.py 產生報告的討論章節，報告數字不用手抄。

訓練集只有 15 筆、待驗收資料沒有標籤，因此泛化能力一律以訓練集上的
留一交叉驗證（leave-one-out cross-validation, LOOCV）估計：每一折都重新以 14 筆
計算標準化統計值並訓練，再預測被留下的那 1 筆。待驗收的 5 筆晶圓只用來觀察
各設定給出的機率如何變動，不用於挑選任何設定。

所有模型皆為 05_assignment2.py 的手刻 Logistic Regression，未使用 scikit-learn。
此檔為分析工具，不屬於繳交內容。

    python 05_assignment2_explore.py
"""

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json

import numpy as np


ASSIGNMENT_DIR = Path(__file__).resolve().parent
STUDY_FILENAME = "assignment2_improvement_study.json"

# 繳交版本實際採用的設定，其餘實驗都以此為基準比較。
BASELINE = {"transformation": "abs", "learning_rate": 0.3, "epochs": 1000, "l2": 0.0}

EPOCH_OPTIONS = (100, 300, 1000, 3000, 10000)
L2_OPTIONS = (0.0, 0.1, 1.0, 10.0)
FEATURE_SET_OPTIONS = ("raw", "abs", "square", "abs+square")
FEATURE_SET_LABELS = {
    "raw": "$x_1$、$x_2$",
    "abs": "$|x_1|$、$x_2$",
    "square": "$x_1^{2}$、$x_2$",
    "abs+square": "$|x_1|$、$x_1^{2}$、$x_2$",
}


def _load_assignment():
    spec = spec_from_file_location("assignment2", ASSIGNMENT_DIR / "05_assignment2.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assignment2 = _load_assignment()


def baseline_config():
    return dict(BASELINE)


def build_feature_set(frame, name):
    """除作業的三種單一轉換外，另支援 abs 與 square 並列的三特徵組合。"""
    if name == "abs+square":
        absolute = assignment2.build_features(frame, "abs")
        squared = assignment2.build_features(frame, "square")
        return np.column_stack((absolute[:, 0], squared[:, 0], absolute[:, 1]))
    return assignment2.build_features(frame, name)


def fit_configuration(train_df, test_df, transformation, learning_rate, epochs, l2):
    """以指定設定訓練一次，回傳訓練指標與 5 筆待驗收晶圓的通過機率。"""
    X = build_feature_set(train_df, transformation)
    y = train_df["y"].to_numpy(dtype=int)
    scaler = assignment2.Standardizer().fit(X)
    X_std = scaler.transform(X)
    model = assignment2.LogisticRegressionGD(learning_rate, epochs, l2=l2).fit(X_std, y)

    X_test_std = scaler.transform(build_feature_set(test_df, transformation))
    probabilities = model.predict_proba(X_test_std)
    return {
        "training_accuracy": float(np.mean(model.predict(X_std) == y)),
        "final_loss": float(model.losses_[-1]),
        "weight_norm": float(np.linalg.norm(model.weights_)),
        "probabilities": {
            wafer: float(probability)
            for wafer, probability in zip(test_df["Wafer ID"], probabilities)
        },
    }


def leave_one_out(train_df, transformation, learning_rate, epochs, l2):
    """留一交叉驗證：每折以其餘樣本重新標準化並訓練，預測被留下的那一筆。"""
    X = build_feature_set(train_df, transformation)
    y = train_df["y"].to_numpy(dtype=int)
    held_out = np.empty(len(y), dtype=float)
    for index in range(len(y)):
        mask = np.arange(len(y)) != index
        scaler = assignment2.Standardizer().fit(X[mask])
        model = assignment2.LogisticRegressionGD(learning_rate, epochs, l2=l2).fit(
            scaler.transform(X[mask]), y[mask]
        )
        held_out[index] = model.predict_proba(scaler.transform(X[index : index + 1]))[0]
    return {
        "accuracy": float(np.mean((held_out >= 0.5).astype(int) == y)),
        "log_loss": assignment2.binary_cross_entropy(y, held_out),
        "held_out_probabilities": held_out.tolist(),
    }


def _experiment(group, label, train_df, test_df, config, baseline):
    fitted = fit_configuration(train_df, test_df, **config)
    cv = leave_one_out(train_df, **config)
    y = train_df["y"].to_numpy(dtype=int)
    misclassified = [
        {"x1": float(row.x1), "x2": float(row.x2), "y": int(row.y), "probability": float(p)}
        for row, p in zip(train_df.itertuples(index=False), cv["held_out_probabilities"])
        if int(p >= 0.5) != int(row.y)
    ]
    return {
        "group": group,
        "label": label,
        "baseline": baseline,
        "config": config,
        "training_accuracy": fitted["training_accuracy"],
        "final_loss": fitted["final_loss"],
        "weight_norm": fitted["weight_norm"],
        "loocv_accuracy": cv["accuracy"],
        "loocv_log_loss": cv["log_loss"],
        "loocv_misclassified": misclassified,
        "probabilities": fitted["probabilities"],
    }


def run_study(data_dir=None, output_dir=None, epochs_scale=1.0):
    """執行三組實驗並寫出 JSON。`epochs_scale` 供測試縮短訓練時間。"""
    source_dir = Path(data_dir) if data_dir is not None else ASSIGNMENT_DIR
    destination = Path(output_dir) if output_dir is not None else source_dir
    destination.mkdir(parents=True, exist_ok=True)
    train_df, test_df = assignment2.load_assignment_data(source_dir)

    def scaled(epochs):
        return max(int(round(epochs * epochs_scale)), 1)

    experiments = []
    for epochs in EPOCH_OPTIONS:
        config = {**baseline_config(), "epochs": scaled(epochs)}
        experiments.append(
            _experiment("epochs", f"{epochs:,}", train_df, test_df, config,
                        baseline=epochs == BASELINE["epochs"])
        )
    for l2 in L2_OPTIONS:
        config = {**baseline_config(), "epochs": scaled(BASELINE["epochs"]), "l2": l2}
        experiments.append(
            _experiment("l2", f"$\\lambda = {l2:g}$", train_df, test_df, config,
                        baseline=l2 == BASELINE["l2"])
        )
    for name in FEATURE_SET_OPTIONS:
        config = {**baseline_config(), "epochs": scaled(BASELINE["epochs"]),
                  "transformation": name}
        experiments.append(
            _experiment("feature_set", FEATURE_SET_LABELS[name], train_df, test_df, config,
                        baseline=name == BASELINE["transformation"])
        )

    study = {
        "generated": date.today().isoformat(),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "wafer_ids": test_df["Wafer ID"].tolist(),
        "baseline": baseline_config(),
        "experiments": experiments,
    }
    (destination / STUDY_FILENAME).write_text(
        json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return study


def _print_summary(study):
    wafers = study["wafer_ids"]
    for group in ("epochs", "l2", "feature_set"):
        print(f"\n[{group}]")
        header = f"{'label':<28}{'train':>7}{'loss':>10}{'|w|':>8}{'LOO acc':>9}{'LOO loss':>10}  " + "  ".join(
            f"{wafer[-1]:>7}" for wafer in wafers
        )
        print(header)
        for row in study["experiments"]:
            if row["group"] != group:
                continue
            marker = " ←" if row["baseline"] else ""
            probabilities = "  ".join(f"{row['probabilities'][w] * 100:6.2f}%" for w in wafers)
            print(
                f"{row['label']:<28}{row['training_accuracy'] * 100:6.1f}%"
                f"{row['final_loss']:10.4f}{row['weight_norm']:8.2f}"
                f"{row['loocv_accuracy'] * 100:8.1f}%{row['loocv_log_loss']:10.4f}  "
                f"{probabilities}{marker}"
            )


def main():
    study = run_study()
    _print_summary(study)
    print(f"\n已寫入 {ASSIGNMENT_DIR / STUDY_FILENAME}")


if __name__ == "__main__":
    main()
