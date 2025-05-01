# -*- coding: UTF-8 -*-
"""
@file: test_lagrange_poly2.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from interpolation_02.lagrange_interpolation import LagrangeInterpolation
from util_font import *

fh1 = lambda x: 0.5 * x ** 5 - x ** 4 + 1.8 * x ** 3 - 3.6 * x ** 2 + 4.5 * x + 2  # 模拟函数1
fh2 = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 模拟函数2

# 拉格朗日插值多项式的误差随着阶次的增高而变化情况
xi = np.linspace(-1, 3, 200)
nums = np.arange(5, 21)  # 则阶次k = n - 1
mse_1_list, mse_2_list = [], []
max_err_1_list, max_err_2_list = [], []
for i, n in enumerate(nums):
    print("阶次：", n - 1)
    x = np.linspace(-1, 3, n, endpoint=True)
    y = fh1(x)  # 取值模拟，多项式函数
    lag = LagrangeInterpolation(x, y)
    lag.fit_interp()
    yi = lag.predict_x0(xi)  # 预测
    if n == 6:
        print("5阶多项式系数：\n", lag.poly_coefficient)
    mse_1_list.append(np.mean((fh1(xi) - yi) ** 2))  # MSE
    max_err_1_list.append(np.max((fh1(xi) - yi) ** 2))  # 最大绝对值平方误差
    y = fh2(x)  # 取值模拟，三角函数
    lag = LagrangeInterpolation(x, y)
    lag.fit_interp()
    yi = lag.predict_x0(xi)  # 预测
    mse_2_list.append(np.mean((fh2(xi) - yi) ** 2))
    max_err_2_list.append(np.max((fh2(xi) - yi) ** 2))

plt.figure(figsize=(14, 5))
plt.subplot(121)
idx = int(np.argmin(mse_1_list))
plt.semilogy(nums - 1, mse_1_list, "o-", label="$MSE$")  # 阶次与MSE
plt.semilogy(nums - 1, max_err_1_list, "*-", label="$MAX \ SE$")  # 阶次与最大平方误差
plt.semilogy(nums[idx] - 1, mse_1_list[idx], "rD", label="$k=%d,\ MSE=%.3e$" % (nums[idx] - 1, mse_1_list[idx]))
plt.xlabel("$Order(k)$", fontdict={"fontsize": 18})  # 阶次
plt.ylabel("$MSE$", fontdict={"fontsize": 18})  # 精度
plt.title("$Lagrange$插值不同阶次下的误差度量：$f_1(x)$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=18)  # 刻度字体大小18
plt.xticks(np.linspace(4, 20, 9))
plt.grid(ls=":")
plt.subplot(122)
idx = int(np.argmin(mse_2_list))
plt.semilogy(nums - 1, mse_2_list, "o-", label="$MSE$")  # 阶次与MSE
plt.semilogy(nums - 1, max_err_2_list, "*-", label="$MAX \ SE$")  # 阶次与最大平方误差
plt.semilogy(nums[idx] - 1, mse_2_list[idx], "rD", label="$k=%d,\ MSE=%.3e$" % (nums[idx] - 1, mse_2_list[idx]))
plt.xlabel("$Order(k)$", fontdict={"fontsize": 18})  # 阶次
plt.ylabel("$MSE$", fontdict={"fontsize": 18})  # 精度
plt.title("$Lagrange$插值不同阶次下的误差度量：$f_2(x)$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=18)  # 刻度字体大小18
plt.xticks(np.linspace(4, 20, 9))
plt.grid(ls=":")
plt.show()
