"""Assignment #4 - 兩種方法共用的資料與模型定義（不繳交）。

作業規定兩種方法的資料集條件須一致：
  - 僅使用 CelebA 訓練集的前 16,000 筆 samples 進行訓練（可用資料增強）
  - 使用 CelebA 測試集的前 10,000 筆 samples 進行測試

繳交的 model_1_test.py / model_2_test.py 是自給自足的單檔腳本，不 import 本模組；
本模組只服務訓練腳本與測試程式，讓兩邊的架構定義有單一來源可比對。
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2


CELEBA_ROOT = "CelebA"
N_TRAIN = 16000
N_TEST = 10000
N_VALID = 1000
SMILING_ATTR_INDEX = 31
LABELS_MAP = {0: "Not Smiling", 1: "Smiling"}

# 方法一沿用課堂範例的 64x64 輸入；方法二的 ResNet-18 需要 224x224 與 ImageNet 正規化。
IMAGE_SIZE_SCRATCH = 64
IMAGE_SIZE_RESNET = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


get_smile = lambda attr: attr[SMILING_ATTR_INDEX]  # noqa: E731


def build_transforms(image_size, normalize):
    """回傳 (訓練用含增強的轉換, 評估用轉換)。

    CenterCrop(178) 先把 218x178 的原圖裁成正方形臉部區域，再縮到模型輸入大小；
    訓練版改用 RandomCrop + 水平翻轉做資料增強（作業允許）。
    """
    tail = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if normalize:
        tail.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    transform_train = v2.Compose(
        [
            v2.RandomCrop([178, 178]),
            v2.RandomHorizontalFlip(),
            v2.Resize([image_size, image_size]),
            *tail,
        ]
    )
    transform_eval = v2.Compose(
        [
            v2.CenterCrop([178, 178]),
            v2.Resize([image_size, image_size]),
            *tail,
        ]
    )
    return transform_train, transform_eval


def resolve_root(root=CELEBA_ROOT):
    """找出本機 CelebA 的位置，並讓 torchvision 接受重建版的標註檔。

    訓練腳本在 assignment/ 底下執行，而資料集放在 lecture/09/CelebA（與課堂範例
    09_3.py 的相對路徑一致），所以要往上一層找。繳交的測試腳本不走這裡，
    它們直接用 root="CelebA"。
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for candidate in (Path(root), here.parent / "CelebA", here / "CelebA"):
        if (candidate / "celeba" / "img_align_celeba").is_dir():
            try:
                import celeba_local_patch

                celeba_local_patch.apply(candidate)
            except ImportError:
                pass
            return str(candidate)
    return str(root)


def load_datasets(image_size, normalize, root=CELEBA_ROOT, download=False, include_test=True):
    """取得訓練前 16000 / 驗證前 1000 / 測試前 10000 的子集合。

    每個 CelebA 物件都會把五個標註檔（202,599 行）讀成 tensor，建一個就是數十 MB，
    所以只訓練時用 `include_test=False` 省掉一份（回傳的 test 為 None）。
    """
    root = resolve_root(root)
    transform_train, transform_eval = build_transforms(image_size, normalize)

    def celeba(split, transform):
        return datasets.CelebA(
            root,
            split=split,
            target_type="attr",
            download=download,
            transform=transform,
            target_transform=get_smile,
        )

    train = Subset(celeba("train", transform_train), torch.arange(N_TRAIN))
    valid = Subset(celeba("valid", transform_eval), torch.arange(N_VALID))
    test = Subset(celeba("test", transform_eval), torch.arange(N_TEST)) if include_test else None
    return train, valid, test


def make_loaders(train, valid, test, batch_size):
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(valid, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def build_model_1():
    """方法一：在課堂範例的四層 CNN 上加入批次正規化並調整 Dropout。

    相對課堂範例 (09_3.py) 的三處調整：
      1. 每層卷積後加 nn.BatchNorm2d —— 穩定各層輸入分布，可用較大學習率且收斂更快
      2. 卷積層的 Dropout 由 0.5 降到 0.1/0.2 —— 0.5 對卷積特徵圖過強，等於丟掉一半通道
      3. 分類頭前補一層 Dropout(0.3) —— 正則化改放在參數最多的全連接層之前
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.1),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=8),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


def build_model_2(weights="default"):
    """方法二：ResNet-18 預訓練模型微調，輸出層改成二分類（1 個 logit + Sigmoid）。"""
    if weights == "default":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    return model


def predict_all(model, dataloader, device, keep_images=False):
    """在 dataloader 上推論，回傳 (preds, trues, images)。"""
    model.eval()
    all_preds, all_trues, all_images = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            probability = model(X.to(device))[:, 0]
            all_preds.extend((probability >= 0.5).float().cpu().numpy())
            all_trues.extend(y.numpy())
            if keep_images:
                all_images.extend(X)
    return np.array(all_preds), np.array(all_trues), all_images
