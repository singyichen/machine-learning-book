from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 載入 WDBC 資料集
wdbc = load_breast_cancer()
X = wdbc.data
y = wdbc.target
target_names = wdbc.target_names

# 2. 黃金法則：進行任何 Processing 之前，先切分資料集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 資料標準化（PCA 對資料尺度非常敏感，降維前必須先標準化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

print(f"原本的資料形狀: {X_scaled.shape}\n")
print(f"{'目標維度 (k)':<12}{'總解釋變異量 (Total Explained Variance)':<30}")
print("-" * 50)

# 4. 迴圈測試 k=2 到 10，彙整總解釋變異量並輸出
for k in range(2, 11):
    # 初始化 PCA 並指定降維至 k 維
    pca = PCA(n_components=k)
    
    # 對標準化後的資料進行擬合
    pca.fit(X_scaled)
    
    # 計算總解釋變異量
    total_variance = pca.explained_variance_ratio_.sum()
    
    # 輸出結果
    print(f"{k:<20}{total_variance:.2%}")

# 5. 對標準化後的資料進行擬合與轉換
X_pca = pca.fit_transform(X_scaled)

# 6. 輸出 PCA 重要的分析結果
print(f"原本的資料形狀: {X_scaled.shape}")
print(f"降維後的資料形狀: {X_pca.shape}\n")

# 解釋變異量比率 (Explained Variance Ratio)
# 這代表每個主要成分（Principal Component）各別保留了多少原始資料的變異資訊
print(f"各主成分的解釋變異量比率: {pca.explained_variance_ratio_}")
print(f"總共保留的資訊量 (總解釋變異量): {pca.explained_variance_ratio_.sum():.2%}")

# 7. 將降維後的結果視覺化
plt.figure(figsize=(8, 6))
colors = ["navy", "turquoise"]

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(
        X_pca[y_train == i, 0],
        X_pca[y_train == i, 1],
        color=color,
        alpha=0.8,
        lw=2,
        label=target_name,
    )

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of WDBC dataset")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.grid(True)
plt.show()
