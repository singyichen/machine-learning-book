import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
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

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 20

# Fashion-MNIST 類別標籤文字
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# 2. 取得資料集
training_data = datasets.FashionMNIST(
    root="FashionMNIST",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

test_data = datasets.FashionMNIST(
    root="FashionMNIST",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# 3. 創建 DataLoader
train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True
)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 4. 定義 NN Model
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)  # 將模型移至指定硬體

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

# 6. 定義 Test Loop
def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    all_preds = []
    all_trues = []
    all_images = []

    with torch.no_grad():
        for X, y in dataloader:
            X_dev, y_dev = X.to(device), y.to(device)
            pred = model(X_dev)

            test_loss += loss_fn(pred, y_dev).item()

            # 將結果移回 CPU 並轉為 NumPy 方便後續統計與繪圖
            preds = pred.argmax(1).cpu().numpy()
            trues = y.numpy()

            all_preds.extend(preds)
            all_trues.extend(trues)
            all_images.extend(X.numpy())

    avg_loss = test_loss / num_batches
    return avg_loss, np.array(all_preds), np.array(all_trues), np.array(all_images)

# 7. 選擇 Loss function 與 Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# 8. 訓練與測試模型
train_loss_history = []
test_loss_history = []

print("Start Training...")
for t in range(epochs):
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, preds, trues, images = test_loop(test_dataloader, model, loss_fn)

    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)

    print(f"Epoch {t+1:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
print("Done!\n")

# 儲存模型
torch.save(model.state_dict(), "model_MLP_Fashion-MNIST.pth")
print("Saved PyTorch Model State to model_MLP_Fashion-MNIST.pth")

# ==========================================
# 視覺化 (Visualization)
# ==========================================
# 繪製 Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_loss_history, label="Train Loss", marker="o")
plt.plot(range(1, epochs + 1), test_loss_history, label="Test Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
