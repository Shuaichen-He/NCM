# -*- coding: UTF-8 -*-
"""
@file_name:test_uniform_square.py
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.best_square_approximation import BestSquarePolynomiaApproximation
from function_approximation_03.best_uniform_approximation import BestUniformApproximation

t = sympy.Symbol("t")
# fun = 1 / (1 + t ** 2)
fun = 11 * sympy.sin(t) + 7 * sympy.cos(5 * t)
orders = np.arange(3, 21, 1)
mae_u, mae_s = np.zeros(len(orders)), np.zeros(len(orders))
for i, order in enumerate(orders):
    bua = BestUniformApproximation(fun, k=order, interval=[-np.pi, np.pi])
    bua.fit_approximation()
    mae_u[i] = bua.mae
    bspa = BestSquarePolynomiaApproximation(fun, k=order, interval=[-np.pi, np.pi])
    bspa.fit_approximation()
    mae_s[i] = bspa.mae
    print(order, bua.mae, bspa.mae)
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(orders, mae_u, "ro-", label="最佳一致逼近")
plt.plot(orders, mae_s, "k*-", label="最佳平方逼近")
plt.xlabel("$Orders(k)$", fontdict={"fontsize": 18})
plt.ylabel("$MAE_{10}$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title("不同阶次下最佳一致和最佳平方逼近的$MAE_{10}$曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.grid(ls=":")
plt.xticks(np.linspace(2, 20, 10))
plt.subplot(122)
bspa.plt_approximate(is_show=False, is_fh_marker=True)
plt.show()
