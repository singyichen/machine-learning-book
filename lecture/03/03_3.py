import numpy as np
A = np.arange(6).reshape(3,2)
B = np.arange(10).reshape(5,2)
# (3, 1, 2) - (5, 2) -> (3, 5, 2)
dist_matrix = np.linalg.norm(A[:, np.newaxis, :] - B, axis=2)

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"Distance Matrix of Shape {dist_matrix.shape}:\n{dist_matrix}")
