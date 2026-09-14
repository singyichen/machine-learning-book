import numpy as np
from scipy import linalg

# 1. 定義向量 (Vector)
# 一維陣列常被視為向量或矩陣向量的基礎
vec = np.array([1, 2, 3])
print("向量內容:", vec)
print("向量維度 (Dimension):", vec.ndim) # 輸出 1
print("向量形狀 (Shape):", vec.shape) # 輸出 (3,)

# 2. 定義矩陣 (Matrix)
# 使用巢狀的列表建立 2x3 矩陣
mat = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("\n2x3 矩陣:\n", mat)
print("矩陣維度 (Dimension):", mat.ndim) # 輸出 2
print("矩陣形狀 (Shape):", mat.shape) # 輸出 (2, 3)

# 3. 特殊矩陣建立
zeros_mat = np.zeros((2, 2))     # 2x2 全零矩陣
ones_mat = np.ones((3, 2))       # 3x2 全一矩陣
identity_mat = np.eye(3)         # 3x3 單位矩陣 (對角線為 1)
print("\n單位矩陣:\n", identity_mat)

# 4. 指定資料型別 (Data Types)
# 預設通常是浮點數，根據輸入內容而定，也可以手動指定
float_vec = np.array([1, 2, 3], dtype='float64')
print("\n浮點數向量型別:", float_vec.dtype) # 輸出 float64

# 5. --- SciPy Note ---
# SciPy functions usually take NumPy arrays as input.
# For example, checking the norm (magnitude) of a vector:
v_norm = linalg.norm(vec)
print(f"\nNorm of the vector: {v_norm:.2f}") # 輸出 3.74
