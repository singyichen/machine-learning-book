import torch
import numpy as np
import torch.nn as nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

# Hyperparameters
learning_rate = 1e-3
batch_size = 32
epochs = 30

# 2. 取得訓練資料集 (資料增強轉換)
get_smile = lambda attr: attr[31]
 
transform_train = v2.Compose([
    v2.RandomCrop([178, 178]),      # 1. 隨機裁剪
    v2.RandomHorizontalFlip(),      # 2. 隨機水平翻轉
    v2.Resize([64, 64]),            # 3. 縮放到模型指定大小
    v2.ToImage(),                   # 4. 將圖片轉換為 Tensor 圖像物件
    v2.ToDtype(torch.float32, scale=True)  # 5. 轉換資料型態並縮放到 [0, 1]
])

transform = v2.Compose([
    v2.CenterCrop([178, 178]),
    v2.Resize([64, 64]),
    v2.ToImage(),                          
    v2.ToDtype(torch.float32, scale=True)
])

training_data = datasets.CelebA("CelebA", 
    split='train', 
    target_type='attr', 
    download=True, 
    transform=transform_train,
    target_transform=get_smile)

valid_data = datasets.CelebA("CelebA", 
    split='valid', 
    target_type='attr',
    download=True, 
    transform=transform,
    target_transform=get_smile)

train_dataset = Subset(training_data, torch.arange(16000)) 
valid_dataset = Subset(valid_data, torch.arange(1000)) 

# 3. 創建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

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

# 5. 定義 Train Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 資料移至指定硬體

        pred = model(X)[:,0]
        loss = loss_fn(pred, y.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 6. 定義 Validation Loop
def valid_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    valid_loss = 0

    all_preds = []
    all_trues = []
    all_images = []

    with torch.no_grad():
        for X, y in dataloader:
            X_dev, y_dev = X.to(device), y.to(device)
            pred = model(X_dev)[:,0]

            valid_loss += loss_fn(pred, y_dev.float()).item()

            # 將結果移回 CPU 並轉為 NumPy 方便後續統計與繪圖
            preds = (pred>=0.5).float().cpu().numpy()
            trues = y.numpy()

            all_preds.extend(preds)
            all_trues.extend(trues)
            all_images.extend(X)

    avg_loss = valid_loss / num_batches
    return avg_loss, np.array(all_preds), np.array(all_trues), all_images

# 7. 選擇 Loss function 與 Optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# 8. 訓練與測試模型
train_loss_history = []
valid_loss_history = []

print("Start Training...")
for t in range(epochs):
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    valid_loss, preds, trues, images = valid_loop(valid_dataloader, model, loss_fn)

    train_loss_history.append(train_loss)
    valid_loss_history.append(valid_loss)

    print(f"Epoch {t+1:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
print("Done!\n")

# 儲存模型
torch.save(model.state_dict(), "model_CNN_CelebA.pth")
print("Saved PyTorch Model State to model_CelebA.pth")

# ==========================================
# 視覺化 (Visualization)
# ==========================================
# 繪製 Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_loss_history, label="Train Loss", marker="o")
plt.plot(range(1, epochs + 1), valid_loss_history, label="Valid Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


