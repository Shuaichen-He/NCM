# -*- coding: UTF-8 -*-
"""
@file:test_pade.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.pade_rational_fraction_approximation import PadeRationalFractionApproximation

import matplotlib as mpl

# 设置数学模式下的字体格式和中文显示
rc = {"font.family": "serif", "mathtext.fontset": "cm"}
plt.rcParams.update(rc)
mpl.rcParams["font.family"] = "FangSong"  # 中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决坐标轴负数的负号显示问题

t = sympy.Symbol("t")
# fun = sympy.exp(-t)
# fun = 1 / (1 - t)
fun = sympy.log(1 + t)
fun_expr = sympy.lambdify(t, fun, "numpy")
# fun = sympy.sin(3 * t) * sympy.log(1 + 0.5 * t)

prfa = PadeRationalFractionApproximation(fun, order=3)
prfa.fit_rational_fraction()
print(prfa.rational_fraction)

plt.figure(figsize=(14, 5))
plt.subplot(121)
prfa = PadeRationalFractionApproximation(fun, order=3)
prfa.fit_rational_fraction()
print(prfa.rational_fraction)
prfa.plt_approximate(0, 10, is_show=False)
plt.subplot(122)
orders = np.linspace(2, 20, 19)
xi = np.linspace(0, 10, 200)
mae_k = []
for i, k in enumerate(orders):
    prfa = PadeRationalFractionApproximation(fun, order=int(k))
    prfa.fit_rational_fraction()
    yi = prfa.cal_x0(xi)  # 预测值
    error = np.abs(yi - fun_expr(xi))
    mae_k.append(np.mean(error))
    if k == 20:
        poly = prfa.rational_fraction

plt.semilogy(orders, mae_k, "o-", label="$MAE$")
idx = np.argmin(mae_k)
plt.semilogy(orders[idx], mae_k[idx], "D",
             label="$R_{%d,%d}(x), \ MAE=%0.5e$" % (orders[idx], orders[idx], mae_k[idx]))
plt.xlabel("$n,m \ (n = m)$", fontsize=18)
plt.ylabel("$MAE$", fontsize=18)
plt.legend(frameon=False, fontsize=18)
plt.tick_params(labelsize=18)
plt.title("帕德逼近$R_{n,m}(x)$平均绝对值误差曲线", fontsize=18)
plt.grid(ls=":")
plt.xticks(np.linspace(2, 20, 10))
plt.show()

mae_list = []
poly_ = sympy.lambdify(t, poly, "numpy")
for i in range(200):
    idx = np.random.randint(0, 199, 5)
    xi_e = np.copy(xi)
    xi_e[idx] = xi[idx] + 0.0001 * np.random.randn(5)
    yi_e = poly_(xi_e)
    error = np.abs(yi_e - fun_expr(xi))
    mae_20 = np.mean(error)
    # print(mae_k[-1], mae_20, np.abs(mae_20 - mae_k[-1]) / mae_k[-1])
    mae_list.append(mae_20)
print(mae_k[-1])
print("平均：", np.mean(mae_list))

plt.figure(figsize=(7, 5))
plt.semilogy(np.arange(1, 201), mae_list, "--", lw=1.5, label="扰动后$MAE_k$")
plt.semilogy(np.arange(1, 201), np.mean(mae_list) * np.ones(200), "-.", lw=2, label="${MAE_k}$的均值")
plt.semilogy(np.arange(1, 201), mae_k[-1] * np.ones(200), "-", lw=2, label="未添加扰动的$MAE$")
plt.xlabel("试验次数", fontsize=18)
plt.ylabel("$MAE_k$", fontsize=18)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.tick_params(labelsize=16)
plt.title("$R_{20,20}(x)$对$x$的微小扰动，函数值的扰动变化", fontsize=18)
plt.grid(ls=":")
plt.ylim([1e-7, 6e-6])
plt.show()