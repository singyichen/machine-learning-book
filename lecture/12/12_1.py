import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

# 1. 資料準備與降維
digits = load_digits()
X, y = digits.data, digits.target

# 使用 t-SNE 將 64 維降至 2 維 (便於視覺化與分群比較)
tsne = TSNE(n_components=2, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X)

# 2. 實作三種分群演算法 (基於 t-SNE 降維後的 2D 數據)
# K-Means (設定 10 個群集)
# 原始寫法為 n_init='auto'，那是 sklearn 1.2+ 才支援的值；本機為 1.0.2，
# 改寫成等效的 n_init=10（k-means++ 初始化下 'auto' 即為 10）
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_tsne)

# Agglomerative Clustering (設定 10 個群集)
agg = AgglomerativeClustering(n_clusters=10, linkage="ward")
y_agg = agg.fit_predict(X_tsne)

# DBSCAN (調整參數以適應 t-SNE 的密度分佈)
# t-SNE 後的數據間距較大，這裡將 eps 設為大約 4.5 到 5.5 之間
dbscan = DBSCAN(eps=4.8, min_samples=15)
y_dbscan = dbscan.fit_predict(X_tsne)

# 3. 計算評估指標 (Adjusted Rand Index, ARI)
ari_kmeans = adjusted_rand_score(y, y_kmeans)
ari_agg = adjusted_rand_score(y, y_agg)
ari_dbscan = adjusted_rand_score(y, y_dbscan)

# 4. 數據結果列印
print("\n--- ARI 比較 (越接近 1 越完美) ---")
print(f"K-Means ARI              : {ari_kmeans:.4f}")
print(f"Agglomerative (Ward) ARI : {ari_agg:.4f}")
print(f"DBSCAN ARI               : {ari_dbscan:.4f}")

# 5. 繪圖視覺化比較
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
cmap_discrete = plt.get_cmap("tab10", 10)

# 子圖 1: Ground Truth (真實標籤)
scatter0 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap_discrete, alpha=0.6, s=15)
axes[0].set_title("Ground Truth (0-9 Digits)", fontsize=14)
fig.colorbar(scatter0, ax=axes[0], ticks=range(10), label="Digit")

# 子圖 2: K-Means
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, cmap=cmap_discrete, alpha=0.6, s=15)
axes[1].set_title(f"K-Means (ARI: {ari_kmeans:.2f})", fontsize=14)

# 子圖 3: Agglomerative
axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_agg, cmap=cmap_discrete, alpha=0.6, s=15)
axes[2].set_title(f"Agglomerative (ARI: {ari_agg:.2f})", fontsize=14)

# 子圖 4: DBSCAN (雜訊會被標記為 -1，在圖中會呈現不同的顏色)
axes[3].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_dbscan, cmap="rainbow", alpha=0.6, s=15)
n_clusters_db_ = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
axes[3].set_title(f"DBSCAN (ARI: {ari_dbscan:.2f})\nClusters found: {n_clusters_db_}", fontsize=14)

plt.tight_layout()
plt.show()

