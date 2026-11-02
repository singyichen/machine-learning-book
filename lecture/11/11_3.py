import numpy as np
import matplotlib.pyplot as plt
# 原始寫法為 plt.style.use('seaborn-v0_8-poster')，那是 matplotlib 3.6+ 的樣式名稱；
# 本機為 3.5.3，改用舊名稱 'seaborn-poster'（同一個樣式）
plt.style.use('seaborn-poster')

# generate x and y
x = np.linspace(0, 1, 100)
y = 1 + x + x * np.random.random(len(x))
# assemble matrix A
A = np.vstack([np.ones(len(x)), x]).T
# turn y into a column vector
y = y[:, np.newaxis]
print(y.shape)

# Least square regression
alpha = np.linalg.lstsq(A, y, rcond=None)[0]
print(alpha)

# plot the results
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha[1]*x + alpha[0], 'r')
plt.title('LLS Regression (Numpy lstsq())')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

