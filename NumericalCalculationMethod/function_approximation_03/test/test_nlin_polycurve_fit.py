# -*- coding: UTF-8 -*-
"""
@file:test_nlin_polycurve_fit.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.least_square_curve_fitting import LeastSquarePolynomialCurveFitting


x = np.linspace(-1, 3, 15, endpoint=True)
np.random.seed(0)
y = 3 * np.exp(0.5 * x) + np.random.randn(15) / 2
ln_y = np.log(y)  # 线性转换
lspcf = LeastSquarePolynomialCurveFitting(x, ln_y, k=1)
lspcf.fit_ls_curve()
a, b = np.exp(lspcf.poly_coefficient[0]), lspcf.poly_coefficient[1]
print("系数a和b分别为", a, b)

xi = np.linspace(min(x), max(x), 100, endpoint=True)
plt.figure(figsize=(14, 5))
plt.subplot(121)
yi = a * np.exp(b * xi)
y_pred = a * np.exp(b * x)  # 拟合曲线对应离散点的预测值
mse = np.mean((y - y_pred) ** 2)
plt.plot(xi, yi, "k-", lw=1.5, label="$y=%.3fe^{%.3fx}$" % (a, b) )
plt.plot(x, y, "ro", label="$(x_k, y_k)$")
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title(r"非线性曲线拟合：含噪声$\epsilon \sim N(0, 1), MSE = %.3e$" % mse, fontdict={"fontsize": 18})

plt.subplot(122)
y = 3 * np.exp(0.5 * x)  # 不添加噪声
ln_y = np.log(y)  # 线性转换
lspcf = LeastSquarePolynomialCurveFitting(x, ln_y, k=1)
lspcf.fit_ls_curve()
a, b = np.exp(lspcf.poly_coefficient[0]), lspcf.poly_coefficient[1]
print("系数a和b分别为", a, b)
yi = a * np.exp(b * xi)
y_pred = a * np.exp(b * x)  # 拟合曲线对应离散点的预测值
mse = np.mean((y - y_pred) ** 2)
plt.plot(xi, yi, "k-", lw=1.5, label="$y=%.3fe^{%.3fx}$" % (a, b) )
plt.plot(x, y, "ro", label="$(x_k, y_k)$")
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title(r"非线性曲线拟合：不含噪声：$MSE = %.3e$" % mse, fontdict={"fontsize": 18})

plt.show()

