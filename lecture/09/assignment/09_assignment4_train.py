"""Assignment #4 - CelebA 微笑分類器訓練（不繳交）。

兩種方法共用這支訓練腳本，以 --method 選擇：

    python 09_assignment4_train.py --method 1    # 調整網路架構（加入 BatchNorm）
    python 09_assignment4_train.py --method 2    # ResNet-18 預訓練模型微調

產出 model_1.pth / model_2.pth（繳交）與對應的 loss curve 圖。
依驗證集 loss 保留最佳權重，避免交出過擬合那一輪的參數。
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import copy
import json
import time

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSIGNMENT_DIR = Path(__file__).resolve().parent


def _load(name, filename):
    spec = spec_from_file_location(name, ASSIGNMENT_DIR / filename)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load("assignment4_common", "09_assignment4_common.py")


METHODS = {
    1: {
        "name": "方法一：調整網路架構（BatchNorm + Dropout 調整）",
        "model_file": "model_1.pth",
        "curve_file": "assignment4_method1_loss_curve.png",
        "image_size": common.IMAGE_SIZE_SCRATCH,
        "normalize": False,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    },
    2: {
        "name": "方法二：ResNet-18 預訓練模型微調",
        "model_file": "model_2.pth",
        "curve_file": "assignment4_method2_loss_curve.png",
        "image_size": common.IMAGE_SIZE_RESNET,
        "normalize": True,
        "batch_size": 32,
        "epochs": 5,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
    },
}


def build_model(method):
    return common.build_model_1() if method == 1 else common.build_model_2()


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        prediction = model(X)[:, 0]
        loss = loss_fn(prediction, y.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    return total / len(dataloader)


def validate(dataloader, model, loss_fn, device):
    model.eval()
    total, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            probability = model(X)[:, 0]
            total += loss_fn(probability, y.float()).item()
            correct += int(((probability >= 0.5).float() == y).sum())
            seen += len(y)
    return total / len(dataloader), correct / seen


def run_training(method, output_dir=None, epochs=None):
    config = METHODS[method]
    destination = Path(output_dir) if output_dir is not None else ASSIGNMENT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    n_epochs = epochs if epochs is not None else config["epochs"]

    common.set_seed(42)
    device = common.get_device()
    print(f"Using {device} device")
    print(config["name"])

    train_data, valid_data, _ = common.load_datasets(
        config["image_size"], config["normalize"], include_test=False
    )
    train_loader, valid_loader, _ = common.make_loaders(
        train_data, valid_data, valid_data, config["batch_size"]
    )

    model = build_model(method).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    train_losses, valid_losses, valid_accuracies = [], [], []
    best_loss, best_epoch, best_state = float("inf"), 0, None

    print("Start Training...")
    for epoch in range(n_epochs):
        started = time.time()
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        valid_loss, valid_accuracy = validate(valid_loader, model, loss_fn, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_accuracy:.4f} | "
            f"{time.time() - started:.0f}s"
        )
    print("Done!\n")

    model.load_state_dict(best_state)
    model_path = destination / config["model_file"]
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path} (best epoch {best_epoch})")

    figure = make_loss_curve_figure(train_losses, valid_losses, config["name"])
    figure.savefig(destination / config["curve_file"])
    plt.close(figure)

    history = {
        "method": method,
        "epochs": n_epochs,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best_valid_accuracy": valid_accuracies[best_epoch - 1],
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
    }
    (destination / f"assignment4_method{method}_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return history


def make_loss_curve_figure(train_losses, valid_losses, title):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax.plot(epochs, valid_losses, label="Valid Loss", marker="s")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", type=int, choices=[1, 2], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    arguments = parser.parse_args()

    history = run_training(arguments.method, epochs=arguments.epochs)
    print(
        f"最佳 epoch {history['best_epoch']}，驗證集 loss {history['best_valid_loss']:.4f}，"
        f"驗證集正確率 {history['best_valid_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
