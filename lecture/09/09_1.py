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
batch_size = 64
epochs = 20

# MNIST 類別標籤文字
labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}

# 2. 取得訓練資料集
training_data = datasets.MNIST(
    root="MNIST",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

valid_dataset = Subset(training_data, torch.arange(10000)) 
train_dataset = Subset(training_data, torch.arange(10000, len(training_data)))

# 3. 創建 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 4. 定義 NN Model
model = nn.Sequential(
    # 第一層卷積與池化 (輸入通道=1, 輸出通道=32)
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    # 第二層卷積與池化 (輸入通道=32, 輸出通道=64)
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    # 分類器
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, 10) # 輸出 Logits，交由 nn.CrossEntropyLoss 處理
).to(device)  # 將模型移至指定硬體


# model = nn.Sequential()
# model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
# model.add_module('relu1', nn.ReLU())        
# model.add_module('pool1', nn.MaxPool2d(kernel_size=2))   
# model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
# model.add_module('relu2', nn.ReLU())        
# model.add_module('pool2', nn.MaxPool2d(kernel_size=2))      
# model.add_module('flatten', nn.Flatten()) 
# model.add_module('fc1', nn.Linear(3136, 1024)) 
# model.add_module('relu3', nn.ReLU()) 
# model.add_module('dropout', nn.Dropout(p=0.5)) 
# model.add_module('fc2', nn.Linear(1024, 10)) 
# model = model.to(device) 

# 5. 定義 Train Loop
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # 資料移至指定硬體

        pred = model(X)
        loss = loss_fn(pred, y)
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
            pred = model(X_dev)

            valid_loss += loss_fn(pred, y_dev).item()

            # 將結果移回 CPU 並轉為 NumPy 方便後續統計與繪圖
            preds = pred.argmax(1).cpu().numpy()
            trues = y.numpy()

            all_preds.extend(preds)
            all_trues.extend(trues)
            all_images.extend(X.numpy())

    avg_loss = valid_loss / num_batches
    return avg_loss, np.array(all_preds), np.array(all_trues), np.array(all_images)

# 7. 選擇 Loss function 與 Optimizer
loss_fn = nn.CrossEntropyLoss()
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
torch.save(model.state_dict(), "model_CNN_MNIST.pth")
print("Saved PyTorch Model State to model_CNN_MNIST.pth")

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


