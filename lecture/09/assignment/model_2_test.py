"""Assignment #4 方法二（ResNet-18 預訓練模型微調）- 模型評估腳本。

載入 model_2.pth，在 CelebA 測試集的前 10,000 筆 samples 上評估微笑分類效能。

與方法一的差別：
  1. 輸入影像縮放為 224 x 224（ResNet-18 預訓練時的輸入尺寸）
  2. 以 ImageNet 的平均值與標準差做影像正規化
  3. 輸出層改成 Linear(512, 1) + Sigmoid 以對應二分類問題

執行方式（與 model_2.pth 放在同一資料夾）：

    python model_2_test.py
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2


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


set_seed(42)

# 1. 自動偵測可用硬體加速 (CUDA / MPS / CPU)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

batch_size = 32

labels_map = {0: "Not Smiling", 1: "Smiling"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 2. 取得 Test 資料集（作業規定：測試集的前 10,000 筆 samples）
get_smile = lambda attr: attr[31]  # noqa: E731

transform = v2.Compose(
    [
        v2.CenterCrop([178, 178]),
        v2.Resize([224, 224]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

test_data = datasets.CelebA(
    "CelebA",
    split="test",
    target_type="attr",
    download=True,
    transform=transform,
    target_transform=get_smile,
)

# 3. 創建 Test DataLoader
number_of_test_samples = 10000
test_dataset = Subset(test_data, torch.arange(number_of_test_samples))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. 定義 NN Model（ResNet-18 骨幹，輸出層改為二分類）
#    評估階段不需要下載預訓練權重，model_2.pth 已包含微調後的完整參數。
model = models.resnet18(weights=None)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
model = model.to(device)

# 5. 載入已訓練模型
model.load_state_dict(torch.load("model_2.pth", weights_only=True, map_location=device))


# 6. 定義 Test Loop
def test_loop(dataloader, model):
    model.eval()

    all_preds = []
    all_trues = []
    all_images = []

    with torch.no_grad():
        for X, y in dataloader:
            X_dev = X.to(device)
            pred = model(X_dev)[:, 0]

            preds = (pred >= 0.5).float().cpu().numpy()
            trues = y.numpy()

            all_preds.extend(preds)
            all_trues.extend(trues)
            all_images.extend(X)

    return np.array(all_preds), np.array(all_trues), all_images


def denormalize(image):
    """把 ImageNet 正規化後的張量還原回 [0, 1]，誤判樣本圖才看得出原影像。"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


# 7. 測試模型
print("Start Testing...")
preds, trues, images = test_loop(test_dataloader, model)
print("Done!\n")

print("\n詳細分類報告 (Classification Report):")
print(classification_report(trues, preds, target_names=[labels_map[i] for i in range(2)]))

# ==========================================
# 視覺化 (Visualization)
# ==========================================
# 1. 找出所有分類錯誤的樣本索引
misclassified_indices = np.where(preds != trues)[0]
print(f"Total misclassified samples: {len(misclassified_indices)} / {number_of_test_samples}")

# 2. 繪製前 25 個分類錯誤的樣本
plt.figure(figsize=(12, 12))
for i, idx in enumerate(misclassified_indices[:25]):
    plt.subplot(5, 5, i + 1)
    img = denormalize(images[idx]).permute(1, 2, 0)
    true_label = labels_map[trues[idx]]
    pred_label = labels_map[preds[idx]]

    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color="red")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("First 25 Misclassified Samples", y=1.02, fontsize=14, weight="bold")
plt.savefig("assignment4_method2_misclassified.png", bbox_inches="tight")
plt.show()

# 3. 繪製混淆矩陣 (Confusion Matrix)
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(labels_map.values()),
    yticklabels=list(labels_map.values()),
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix on Test Dataset", fontsize=14, weight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("assignment4_method2_confusion_matrix.png")
plt.show()
