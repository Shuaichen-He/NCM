# -*- coding: UTF-8 -*-
"""
@file:test_polycurve_fit.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.least_square_curve_fitting import LeastSquarePolynomialCurveFitting

x = np.linspace(1, 5, 5, endpoint=True)
y = np.array([4, 4.5, 6, 8, 8.5])
w = np.array([2, 1, 3, 1, 1])
orders = np.arange(1, 4, 1)
xi = np.linspace(min(x), max(x), 100, endpoint=True)
plt.figure(figsize=(7, 5))
line_style = ["-", "--", "-."]
for i, order in enumerate(orders):
    lspcf = LeastSquarePolynomialCurveFitting(x, y, k=order, w=w)
    lspcf.fit_ls_curve()
    print("order = %d：" % order, lspcf.poly_coefficient, lspcf.mse)
    yi = lspcf.predict_x0(xi)
    plt.plot(xi, yi, line_style[i], lw=2, label="$k = %d, \ MSE = %.2e$" % (order, lspcf.mse))
plt.plot(x, y, "ko", label="$(x_k, y_k)$")
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$p(x)$", fontdict={"fontsize": 18})
plt.title("不同阶次的最小二乘多项式曲线拟合", fontdict={"fontsize": 18})
plt.show()

# 欠拟合与过拟合测试代码
fun = lambda x: 0.5 * x ** 2 + x + 2  # 假设目标函数

np.random.seed(42)  # 设置随机种子
n = 30  # 样本量
raw_x = np.sort(6 * np.random.rand(n) - 3)  # 采样数据：[-3, 3]区间均匀分布随机数
raw_y = fun(raw_x) + 0.5 * np.random.randn(n)  # 采样目标数据 + 噪声
degree = [1, 2, 5, 10, 15, 20]  # 拟合阶次
xi = np.linspace(-3, 3, 150, endpoint=True)  # 测试数据
yi = fun(xi)  # 真值
plt.figure(figsize=(16, 8))
for i, d in enumerate(degree):
    lspcf = LeastSquarePolynomialCurveFitting(raw_x, raw_y, k=d)
    lspcf.fit_ls_curve()
    y_pred = lspcf.predict_x0(xi)  # 预测曲线
    plt.subplot(231 + i)
    plt.scatter(raw_x, raw_y, edgecolors="k", s=18, label="$(x_k, y_k)$")
    plt.plot(xi, yi, "k-", lw=1.5, label="$f(x)$")
    plt.plot(xi, y_pred, "r--", lw=2, label="$p(x): k=%d$" % d)
    plt.legend(frameon=False, fontsize=16, loc=2, ncol=2)
    # plt.grid(ls=":")
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.xlabel("$x$", fontdict={"fontsize": 18})
    plt.ylabel("$f(x) \ / \ p(x)$", fontdict={"fontsize": 18})
    test_ess = (y_pred - yi) ** 2  # 测试误差平方和
    test_mse, test_std = np.mean(test_ess), np.std(test_ess)  # 均值和方差
    train_ess = (lspcf.predict_x0(raw_x) - raw_y) ** 2
    train_mse, train_std = np.mean(train_ess), np.std(train_ess)  # 均值和方差
    plt.title("$Train_{mse}=%.2e(\pm %.2e)$" % (train_mse, train_std), fontdict={"fontsize": 18})
    plt.text(-2.8, 5.5, "$Test_{mse}=%.2e(\pm %.2e)$" % (test_mse, test_std), fontdict={"fontsize": 16})
    plt.axis([-3, 3, 0, 9])
plt.show()
