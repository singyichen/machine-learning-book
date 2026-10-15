import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import random
def set_seed(seed=42):
    # 1. 固定 Python 內建亂數
    random.seed(seed)
    
    # 2. 固定 NumPy 亂數
    np.random.seed(seed)
    
    # 3. 固定 PyTorch CPU 與 GPU 亂數
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # 4. 固定 GPU 底層演算法 (cuDNN)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. 針對 Mac MPS 或新型 PyTorch 版本的確定性設定 (遇到不支援的算子時只警告不報錯)
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

# MNIST 類別標籤文字
labels_map = {
    0: "Not Smiling",
    1: "Smiling"
}

# 2. 取得 Test 資料集
get_smile = lambda attr: attr[31]
 
transform = v2.Compose([
    v2.CenterCrop([178, 178]),
    v2.Resize([64, 64]),
    v2.ToImage(),                          
    v2.ToDtype(torch.float32, scale=True)
])

test_data = datasets.CelebA("CelebA", 
    split='test', 
    target_type='attr', 
    download=True, 
    transform=transform,
    target_transform=get_smile)

# 3. 創建 Test DataLoader
number_of_test_samples = 10000
test_dataset = Subset(test_data, torch.arange(number_of_test_samples)) 
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 4. 定義 NN Model
model = nn.Sequential(
    # 四層 Conv + ReLU + Pool + Dropout 堆疊
    nn.Conv2d(3, 32, kernel_size=3, padding=1),   nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),  nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),nn.ReLU(),
    
    # 全域平均池化層 (GAP)
    nn.AvgPool2d(kernel_size=8), 
    nn.Flatten(),
    
    # 輸出層：FC + Sigmoid
    nn.Linear(256, 1),
    nn.Sigmoid() # 輸出 0~1 之間的機率值，搭配 nn.BCELoss 進行訓練
).to(device)  # 將模型移至指定硬體

# 5. 載入已訓練模型
model.load_state_dict(torch.load("model_CNN_CelebA.pth", weights_only=True))

# 6. 定義 Test Loop
def test_loop(dataloader, model):
    model.eval()

    all_preds = []
    all_trues = []
    all_images = []

    with torch.no_grad():
        for X, y in dataloader:
            X_dev, y_dev = X.to(device), y.to(device)
            pred = model(X_dev)[:,0]

            # 將結果移回 CPU 並轉為 NumPy 方便後續統計與繪圖
            preds = (pred>=0.5).float().cpu().numpy()
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
    # 設定標題，顯示真實標籤與預測標籤
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color="red")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("First 25 Misclassified Samples", y=1.02, fontsize=14, weight="bold")
plt.show()

# 3. 繪製混淆矩陣 (Confusion Matrix)
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(10, 8))
# 使用 seaborn 的 heatmap 呈現
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
plt.show()
