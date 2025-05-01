# -*- coding: UTF-8 -*-
"""
@file:test_orth_poly_ls.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from function_approximation_03.orthogonal_polynomial_ls_fitting import OrthogonalPolynomialLSFitting
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.linspace(0, 8, 20, endpoint=True)
y = np.exp(-x) * np.sin(x) + 0.01 * np.random.randn(20)
orders = np.arange(1, 19, 1)
xi = np.linspace(min(x), max(x), 150, endpoint=True)
yi = np.exp(-xi) * np.sin(xi)  # 真值
plt.figure(figsize=(14, 5))
plt.subplot(121)
line_style = [":", "-.", "--", "-"]
train_mse, test_mse = np.zeros(len(orders)), np.zeros(len(orders))  # 存储均方误差
for i, order in enumerate(orders):
    oplsf = OrthogonalPolynomialLSFitting(x, y, k=order)
    oplsf.fit_orthogonal_poly()
    y_hat = oplsf.predict_x0(xi)
    train_mse[i] = oplsf.mse
    test_mse[i] = np.sqrt(np.mean((yi - y_hat) ** 2))
    if order in [4, 5, 6, 7]:
        plt.plot(xi, y_hat, line_style[order-4], lw=2, label="$k: %d, \  MSE: %.2e$" % (order, oplsf.mse))
        print("order = %d：" % order, oplsf.poly_coefficient)

plt.plot(x, y, "ko", label="$(x_k, y_k)$")
plt.grid(ls=":")
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("不同阶次下的正交多项式最小二乘拟合曲线", fontdict={"fontsize": 18})
plt.subplot(122)
plt.semilogy(orders, train_mse, "ro-", label="$Train \ MSE$")
plt.semilogy(orders, test_mse, "ks-", label="$Test \ MSE$")
plt.grid(ls=":")
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$Orders(k)$", fontdict={"fontsize": 18})
plt.ylabel("$MSE$", fontdict={"fontsize": 18})
plt.title("不同阶次下拟合曲线的训练$MSE$和测试$MSE$", fontdict={"fontsize": 18})
plt.xticks(np.linspace(2, 18, 9))
plt.show()