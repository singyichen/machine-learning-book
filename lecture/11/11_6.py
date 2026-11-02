import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
# 原始寫法為 plt.style.use('seaborn-v0_8-poster')，那是 matplotlib 3.6+ 的樣式名稱；
# 本機為 3.5.3，改用舊名稱 'seaborn-poster'（同一個樣式）
plt.style.use('seaborn-poster')

# generate x and y, and add some noise into y
x = np.linspace(0, 10, 100)
y = 0.1*np.exp(0.3*x) + 0.1*(np.random.random(len(x)) - 0.5)
# assemble matrix A
A = np.vstack([np.ones(len(x)), x]).T

# Curve Fit by a function
def func(x, a, b):
    y = a*np.exp(b*x)
    return y

alpha, beta = optimize.curve_fit(func, xdata = x, ydata = y)[0]
print(f'alpha={alpha}, beta={beta}')

# plot the results
plt.figure(figsize = (10,8))
plt.plot(x, y, 'b.')
plt.plot(x, alpha*np.exp(beta*x), 'r')
plt.title('Nonlinear Regression (SciPy curve_fit())')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

