# -*- coding: UTF-8 -*-
"""
@file:test_best_square_appr.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from util_font import *
from function_approximation_03.best_square_approximation import BestSquarePolynomiaApproximation
from function_approximation_03.best_uniform_approximation import BestUniformApproximation


t = sympy.Symbol("t")
# fun = sympy.sin(t) * sympy.exp(-t)
# fun = 11 * sympy.sin(t) + 7 * sympy.cos(5 * t)
fun = sympy.sqrt(1 + t ** 2)
plt.figure(figsize=(16, 4))
orders = [1, 3, 7]
# orders = [6, 10, 25, 30]
for i, order in enumerate(orders):
    plt.subplot(131 + i)
    # bspa = BestSquarePolynomiaApproximation(fun, k=order, interval=[-np.pi, np.pi])
    bspa = BestSquarePolynomiaApproximation(fun, k=order, interval=[0, 1])
    bspa.fit_approximation()
    if order == 1:
        bspa.plt_approximate(is_show=False)
    else:
        bspa.plt_approximate(is_show=False, is_fh_marker=True)
    print("order=%d" % order, "，最佳平方逼近多项式的绝对误差均值为：", bspa.mae)
    print("order=%d" % order, "，最佳平方逼近多项式的最大绝对误差为：", bspa.max_abs_error)
    print(bspa.poly_coefficient)
plt.show()
