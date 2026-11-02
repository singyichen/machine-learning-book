import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# 1. Generate synthetic non-linear data
np.random.seed(42)
X = np.linspace(-3, 3, 500).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 2. Split and Scale features (MLPs are sensitive to feature scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Define and train the MLP Regressor
# We use two hidden layers (64 and 32 neurons) 
# with ReLU activation
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=1000, 
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 4. Predict
predictions = mlp.predict(X_test_scaled)

# 5. Evaluation
r2score = r2_score(y_test, predictions)
print(f"Test Score (R²): {r2score:.4f}")

# plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label='Test Data', s=15)
X_all_scaled = scaler.transform(X)
y_all_pred = mlp.predict(X_all_scaled)
plt.plot(X, y_all_pred, color='red', linewidth=2.5, label='MLP Regression Fit')
plt.title('Nonlinear Regression (scikit-learn MLP)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()