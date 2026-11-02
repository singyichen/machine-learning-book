from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 1. 載入 Digits 手寫數字資料集 (64維，低解析度 8x8 影像)
digits = load_digits()
X = digits.data
y = digits.target
target_names = [str(i) for i in range(10)]

# 2. 初始化 t-SNE 
# 將 64 維降至 2 維以利視覺化，設定 random_state 確保結果可重複
tsne = TSNE(n_components=2, init='pca', random_state=42)

# 3. 對資料進行擬合與轉換
X_tsne = tsne.fit_transform(X)

# 4. 輸出降維前後的結果
print(f"原本的資料形狀: {X.shape}")
print(f"降維後的資料形狀: {X_tsne.shape}\n")

# 5. 將降維後的二維結果視覺化
plt.figure(figsize=(12, 10))

# 使用 matplotlib 內建的離散色彩地圖 (tab10 剛好符合 0-9 共 10 個數字)
scatter = plt.scatter(
    X_tsne[:, 0], 
    X_tsne[:, 1], 
    c=y, 
    cmap='tab10', 
    alpha=0.8, 
    edgecolors='none'
)

# 在各數字群聚的中心 (中間值) 加上數字標籤
# 僅為視覺化效果，t-SNE 是非監督式學習的降維技術
for i in range(10):
    # 計算該數字所有點的中心位置
    center = np.median(X_tsne[y == i], axis=0)
    plt.text(
        center[0], center[1], str(i),
        fontsize=18,
        weight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
    )

plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.title("t-SNE visualization of Digits dataset")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
