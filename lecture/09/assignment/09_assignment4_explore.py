"""Assignment #4 - 改善空間研究（不繳交）。

回答「在作業限制內還能做得更好嗎」。訓練一個 CelebA CNN 要十幾分鐘，無法像前幾次
作業那樣做交叉驗證，因此改用**固定訓練預算**的對照實驗：每組都只訓練 EPOCHS 輪、
用同一組種子、同一份訓練集（前 16,000 筆），比較驗證集（前 1,000 筆）的最佳表現。

這樣比較的是「相同預算下誰學得快、學得好」，不是最終收斂上限；報告會據實說明。
測試集完全不參與，只在最後由繳交的評估腳本使用一次。

輸出：assignment4_improvement_study.json

執行方式（約 20 分鐘）：

    python 09_assignment4_explore.py
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import gc
import json
import time

import torch
import torch.nn as nn


ASSIGNMENT_DIR = Path(__file__).resolve().parent
STUDY_FILENAME = "assignment4_improvement_study.json"
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


def _load(name, filename):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load("assignment4_common_for_study", "09_assignment4_common.py")
train_script = _load("assignment4_train_for_study", "09_assignment4_train.py")


def build_sample_code_model():
    """課堂範例 08-3.py 的原始架構：無批次正規化、卷積層 Dropout 0.5。"""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.AvgPool2d(kernel_size=8), nn.Flatten(),
        nn.Linear(256, 1), nn.Sigmoid(),
    )


def build_batchnorm_heavy_dropout():
    """只加批次正規化，Dropout 維持課堂範例的 0.5，用來拆開兩項調整的貢獻。"""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.5),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.5),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AvgPool2d(kernel_size=8), nn.Flatten(),
        nn.Linear(256, 1), nn.Sigmoid(),
    )


def build_wider_model():
    """交出去的設定再加寬一倍通道數，測試「調整網路寬度」這個手段的效果。"""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.1),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.AvgPool2d(kernel_size=8), nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(512, 1), nn.Sigmoid(),
    )


EXPERIMENTS = [
    {"group": "architecture", "label": "課堂範例原架構（無 BN、Dropout 0.5）",
     "builder": build_sample_code_model, "augment": True},
    {"group": "architecture", "label": "加入 BatchNorm（Dropout 仍 0.5）",
     "builder": build_batchnorm_heavy_dropout, "augment": True},
    {"group": "architecture", "label": "BatchNorm + 調低 Dropout（繳交版本）",
     "builder": common.build_model_1, "augment": True},
    {"group": "width", "label": "繳交版本再加寬一倍通道",
     "builder": build_wider_model, "augment": True},
    {"group": "augmentation", "label": "繳交版本、關閉資料增強",
     "builder": common.build_model_1, "augment": False},
]


def run_one(spec, device, data):
    """在固定預算下訓練一組設定，回傳驗證集的最佳表現。

    `data` 是五組共用的 (train_subset, valid_subset, 增強轉換, 評估轉換)。
    每組重新載入 CelebA 會把 202,599 行的屬性表重讀五次，16GB 的機器撐不住
    （實際被 OOM killer 殺過兩次），因此資料只在 run_study 載入一次。
    """
    common.set_seed(42)

    train_data, valid_data, transform_train, transform_eval = data
    # 關閉增強的那組改用評估用的確定性轉換（CenterCrop，不翻轉）
    train_data.dataset.transform = transform_train if spec["augment"] else transform_eval

    train_loader, valid_loader, _ = common.make_loaders(
        train_data, valid_data, valid_data, BATCH_SIZE
    )

    model = spec["builder"]().to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss, best_accuracy, best_epoch = float("inf"), 0.0, 0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_script.train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        valid_loss, valid_accuracy = train_script.validate(
            valid_loader, model, loss_fn, device
        )
        if valid_loss < best_loss:
            best_loss, best_accuracy, best_epoch = valid_loss, valid_accuracy, epoch
        print(
            f"    epoch {epoch}/{EPOCHS} valid loss {valid_loss:.4f} acc {valid_accuracy:.4f}",
            flush=True,
        )

    result = {
        "group": spec["group"],
        "label": spec["label"],
        "best_valid_loss": best_loss,
        "best_valid_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - started, 1),
    }

    # 模型與 loader 每組都重建，用完就放掉；dataset 本身是共用的，不在這裡刪。
    del model, optimizer, train_loader, valid_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def run_study(output_dir=None, limit=None):
    device = common.get_device()
    print(f"Using {device} device；每組固定訓練 {EPOCHS} epochs\n", flush=True)

    # 資料只載入一次，五組共用。
    transform_train, transform_eval = common.build_transforms(
        common.IMAGE_SIZE_SCRATCH, normalize=False
    )
    train_data, valid_data, _ = common.load_datasets(
        common.IMAGE_SIZE_SCRATCH, normalize=False, include_test=False
    )
    data = (train_data, valid_data, transform_train, transform_eval)

    destination = Path(output_dir) if output_dir is not None else ASSIGNMENT_DIR
    partial_path = destination / "assignment4_improvement_partial.json"
    results = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
    done = {row["label"] for row in results}

    ran = 0
    for spec in EXPERIMENTS:
        if spec["label"] in done:
            print(f"[{spec['label']}] 已完成，跳過", flush=True)
            continue
        print(f"[{spec['label']}]", flush=True)
        results.append(run_one(spec, device, data))
        # 逐組存檔：這支腳本被 OOM killer 中斷過，存檔後重跑只需補未完成的組別
        partial_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(flush=True)
        ran += 1
        if limit is not None and ran >= limit:
            # --limit 1 讓外層 shell 每組開一個新行程：行程結束才會把 MPS 與
            # dataset 佔的記憶體完全還給系統，這台 16GB 的機器需要這樣跑。
            print(f"已完成 {ran} 組，本次執行結束（尚有組別未跑）", flush=True)
            return None, partial_path

    study = {
        "epochs_per_run": EPOCHS,
        "n_train": common.N_TRAIN,
        "n_valid": common.N_VALID,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "experiments": results,
    }

    path = destination / STUDY_FILENAME
    path.write_text(json.dumps(study, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    return study, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="本次執行最多跑幾組（搭配逐組存檔，讓每組跑在獨立行程裡省記憶體）",
    )
    arguments = parser.parse_args()

    study, path = run_study(limit=arguments.limit)
    if study is None:
        print(f"部分結果已存入: {path}")
        return
    print("=" * 70)
    for row in study["experiments"]:
        print(
            f"{row['label']:<36} valid acc {row['best_valid_accuracy']:.4f}  "
            f"({row['n_parameters']:,} 參數, {row['seconds']:.0f}s)"
        )
    print(f"\n研究結果已寫入: {path}")


if __name__ == "__main__":
    main()
