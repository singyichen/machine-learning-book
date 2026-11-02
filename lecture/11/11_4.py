import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
# 原始寫法為 plt.style.use('seaborn-v0_8-poster')，那是 matplotlib 3.6+ 的樣式名稱；
# 本機為 3.5.3，改用舊名稱 'seaborn-poster'（同一個樣式）
plt.style.use('seaborn-poster')

# generate x and y
x = np.linspace(0, 1, 100)
y = 1 + x + x * np.random.random(len(x))

# Curve Fit by a function
def func(x, a0, a1):
    y = a1*x + a0
    return y

alpha = optimize.curve_fit(func, xdata = x, ydata = y)[0]
print(alpha)

# plot the results
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha[1]*x + alpha[0], 'r')
plt.title('LLS Regression (SciPy curve_fit())')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

