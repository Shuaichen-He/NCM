# -*- coding: UTF-8 -*-
"""
@file:test_chebyshev.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from function_approximation_03.chebyshev_series_approximation import ChebyshevSeriesApproximation
from util_font import *

# 例3测试代码
t = sympy.Symbol("t")
fun = sympy.exp(t)  # 定义符号函数
csa = ChebyshevSeriesApproximation(fun, x_span=[-1, 1], k=3)  # 实例化对象
csa.fit_approximation()  # 求解切比雪夫级数逼近
print("切比雪夫级数逼近系数及阶次：")
print(csa.poly_coefficient, "\n", csa.polynomial_orders)
print("切比雪夫级数逼近各递推项和系数")
print(csa.T_coefficient[0], "\n", csa.T_coefficient[1])
print("切比雪夫级数逼近最大绝对误差：", csa.max_abs_error)
print("=" * 100)
plt.figure(figsize=(14, 5))
plt.subplot(121)
csa.plt_approximate(is_show=False, is_fh_marker=True)  # 可视化
plt.subplot(122)
csa10 = ChebyshevSeriesApproximation(fun, x_span=[-1, 1], k=10)  # 实例化对象
csa10.fit_approximation()  # 求解切比雪夫级数逼近
print("切比雪夫级数逼近系数及阶次：")
print(csa10.poly_coefficient, "\n", csa10.polynomial_orders)
print("切比雪夫级数逼近各递推项和系数")
print(csa10.T_coefficient[0], "\n", csa10.T_coefficient[1])
print("切比雪夫级数逼近最大绝对误差：", csa10.max_abs_error)
csa10.plt_approximate(is_show=False, is_fh_marker=True)  # 可视化
plt.show()

# 例4测试代码
t = sympy.Symbol("t")
fun = sympy.sin(t) * sympy.exp(-t)  # 符号函数定义
plt.figure(figsize=(14, 5))
orders = np.arange(2, 20, 1)
mae = np.zeros(len(orders))  # 存储逼近均值
csa_obj = None
for i, k in enumerate(orders):
    csa = ChebyshevSeriesApproximation(fun, x_span=[0, 5], k=k)
    csa.fit_approximation()
    mae[i] = csa.mae
    print(k, mae[i])
    if k == 17:
        csa_obj = csa
    # csa.plt_approximate(is_show=False)
plt.subplot(121)
plt.semilogy(orders, mae, "o-")
idx = np.argmin(mae)
plt.semilogy(orders[idx], mae[idx], "s", label="$k=%d, MAE_{10}=%.2e$" % (orders[idx], mae[idx]))
plt.semilogy(orders[-2], mae[-2], "D", label="$k=%d, MAE_{10}=%.2e$" % (orders[-2], mae[-2]))
# plt.semilogy(orders[-1], mae[-1], "D", label="$k=%d, MAE_{10}=%.2e$" % (orders[-1], mae[-1]))
plt.legend(frameon=False, fontsize=18)
plt.xlabel(r"$orders(k)$", fontdict={"fontsize": 18})
plt.ylabel(r"$MAE_{10}$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xticks(np.arange(2, 20, 2))
plt.grid(ls=":")
plt.title("不同阶次下逼近的绝对误差均值", fontdict={"fontsize": 18})
plt.subplot(122)
csa_obj.plt_approximate(is_show=False, is_fh_marker=True)
plt.show()
