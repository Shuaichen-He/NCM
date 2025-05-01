# -*- coding: UTF-8 -*-
"""
@file:test_best_unappr.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from util_font import *
from function_approximation_03.best_uniform_approximation import BestUniformApproximation

t = sympy.Symbol("t")
fun = 11 * sympy.sin(t) + 7 * sympy.cos(5 * t)
# fun = sympy.tan(sympy.cos((np.sqrt(3) + sympy.sin(2 * t)) / (3 + 4 * t ** 2)))  # 被逼近函数
plt.figure(figsize=(14, 9))
orders = [6, 10, 25, 30]
for i, order in enumerate(orders):
    plt.subplot(221 + i)
    bua = BestUniformApproximation(fun, k=order, interval=[-np.pi, np.pi], eps=1e-10)
    bua.fit_approximation()
    if order in [25, 30]:
        bua.plt_approximate(is_show=False, is_fh_marker=True)
    else:
        bua.plt_approximate(is_show=False)
    print("order=%d" % order, "，最佳一致逼近多项式的逼近误差精度为：", bua.abs_error)
    print("order=%d" % order, "，最佳一致逼近多项式的最大绝对误差为：", bua.max_abs_error)
    print("交错点组：", bua.cross_point_group)
plt.show()