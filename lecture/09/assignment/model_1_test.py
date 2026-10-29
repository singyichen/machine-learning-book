"""Assignment #4 方法一（調整網路架構）- 模型評估腳本。

載入 model_1.pth，在 CelebA 測試集的前 10,000 筆 samples 上評估微笑分類效能。

網路架構相對課堂範例 (Sample code 08-3.py) 的三處調整：
  1. 每層卷積後加入 nn.BatchNorm2d
  2. 卷積層的 Dropout 由 0.5 降為 0.1 / 0.2
  3. 分類頭（Linear）之前補一層 Dropout(0.3)

執行方式（與 model_1.pth 放在同一資料夾）：

    python model_1_test.py
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
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

# 2. 取得 Test 資料集（作業規定：測試集的前 10,000 筆 samples）
get_smile = lambda attr: attr[31]  # noqa: E731

transform = v2.Compose(
    [
        v2.CenterCrop([178, 178]),
        v2.Resize([64, 64]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
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

# 4. 定義 NN Model（批次正規化 + 調整後的 Dropout）
model = nn.Sequential(
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
    # 全域平均池化層 (GAP)
    nn.AvgPool2d(kernel_size=8),
    nn.Flatten(),
    # 輸出層：Dropout + FC + Sigmoid
    nn.Dropout(0.3),
    nn.Linear(256, 1),
    nn.Sigmoid(),
).to(device)

# 5. 載入已訓練模型
model.load_state_dict(torch.load("model_1.pth", weights_only=True, map_location=device))


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
    img = images[idx].permute(1, 2, 0)
    true_label = labels_map[trues[idx]]
    pred_label = labels_map[preds[idx]]

    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color="red")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("First 25 Misclassified Samples", y=1.02, fontsize=14, weight="bold")
plt.savefig("assignment4_method1_misclassified.png", bbox_inches="tight")
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
plt.savefig("assignment4_method1_confusion_matrix.png")
plt.show()
