import numpy as np

# Define two 2x2 matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# --- 1. Matrix Addition ---
# Condition: Dimensions must be the same
# OR compatible for Broadcasting.
add_result = A + B
print("Matrix Addition (A + B):\n", add_result)

# --- 2. Scalar Multiplication ---
# Multiplies every element by the constant.
scalar = 10
scalar_result = A * scalar
print(f"\nScalar Multiplication ({scalar} * A):\n", scalar_result)

# --- 3. Element-wise Multiplication (Caution!) ---
# This is NOT standard matrix multiplication. It multiplies position by position.
element_wise = A * B
print("\nElement-wise Multiplication (A * B):\n", element_wise)

# --- 4. Standard Matrix Multiplication ---
# This follows the linear algebra rule (rows of A times columns of B).
# Condition: Number of columns in A must equal number of rows in B.
C = np.array([[1, 0, 1], [0, 1, 0]]) # 2x3 Matrix
matrix_mult = A @ C  # Preferred modern syntax
# Alternative: np.matmul(A, C) or np.dot(A, C)
print("\nMatrix Multiplication (A @ C, results in 2x3):\n", matrix_mult)

# --- 5. Vector Dot Product ---
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
dot_product = np.dot(v, w) # 1*4 + 2*5 + 3*6 = 32
print(f"\nVector Dot Product (v · w): {dot_product}")
