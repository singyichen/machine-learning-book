from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# 1. 載入 WDBC 資料集
wdbc = load_breast_cancer()
X = wdbc.data
y = wdbc.target
target_names = wdbc.target_names

# 2. 黃金法則：進行任何 Processing 之前，先切分資料集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 4. 初始化 LDA 
# 注意：因為 WDBC 只有 2 個類別 (M/B)，n_components 最大只能設定為 1 (類別數 - 1)
lda = LDA(n_components=1)

# 5. 對標準化後的資料進行擬合與轉換
# 監督式學習：fit_transform 必須同時傳入 X_scaled 與 y_train
X_lda = lda.fit_transform(X_scaled, y_train)

# 6. 輸出 LDA 重要的分析結果
print(f"原本的資料形狀: {X_scaled.shape}")
print(f"降維後的資料形狀: {X_lda.shape}\n")

# 解釋變異量比率 (Explained Variance Ratio)
# 在 LDA 中，這代表該判別軸區分類別的能力佔總區分能力的比例
print(f"各成分的解釋變異量比率: {lda.explained_variance_ratio_}")
print(f"總共保留的資訊量 (總解釋變異量): {lda.explained_variance_ratio_.sum():.2%}\n")

# 7. 將降維後的結果視覺化
plt.figure(figsize=(8, 6))
colors = ["navy", "turquoise"]

# 因為 LDA 降維後只有 1 維 (X_lda[:, 0])，為了在二維圖上視覺化，
# 將 y 軸設為 0（加上一點隨機抖動 jitter 避免點全部重疊），以便觀察一維上的類別分離度
for color, i, target_name in zip(colors, [0, 1], target_names):
    # 篩選出該類別的資料
    X_class = X_lda[y_train == i, 0]
    # 建立與資料數量相同的 y 軸值（全部為 0）並加上微小隨機值
    y_jitter = np.zeros_like(X_class) + np.random.normal(0, 0.02, size=len(X_class))
    
    plt.scatter(
        X_class,
        y_jitter,
        color=color,
        alpha=0.8,
        lw=2,
        label=target_name,
    )

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of WDBC dataset")
plt.xlabel(f"LD1 ({lda.explained_variance_ratio_[0]:.2%})")
plt.ylabel("Jittered Density") # 修改 y 軸標籤
plt.ylim(-0.2, 0.2) # 限制 y 軸範圍
plt.grid(True)
plt.show()
