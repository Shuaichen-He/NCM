# -*- coding: UTF-8 -*-
"""
@file_name: test_composite_quad_formula_2.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import time
from numerical_integration_04.composite_quadrature_formula import CompositeQuadratureFormula
from util_font import *

t = sympy.Symbol("t")
fun1 = sympy.exp(t ** 2)  # 分段函数1
fun2 = 80 / (4 - sympy.sin(16 * np.pi * t))  # 分段函数2
fun = sympy.Piecewise((fun1, t <= 2), (fun2, t <= 4))  # 定义分段函数
fun_expr = sympy.lambdify(t, fun)
plt.figure(figsize=(14, 5))
plt.subplot(121)
xi = np.linspace(0, 4, 1000)
yi = fun_expr(xi)
plt.plot(xi, yi, "k-")
plt.fill_between(xi, yi, color="c", alpha=0.5)
plt.xlabel(r"$x$", fontdict={"fontsize": 20})
plt.ylabel(r"$f(x)$", fontdict={"fontsize": 20})
plt.title("带奇点分段震荡函数积分区域可视化", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16

plt.subplot(122)
interval_n = np.arange(1000, 21001, 2000)
err_time = np.zeros((len(interval_n), 2))
for i, n in enumerate(interval_n):
    cqf = CompositeQuadratureFormula(fun, [0, 4], n, int_type="cotes")
    start = time.time()
    cqf.fit_int()
    end = time.time()
    err_time[i, 0] = np.abs(cqf.int_value - 57.764450125048512)
    err_time[i, 1] = end - start
    print("划分区间数：%d，积分值：%.15f， 误差：%.10e，运行消耗时间：%.10fs" %
          (n, cqf.int_value, cqf.int_value - 57.764450125048512, end - start))
plt.semilogy(interval_n, err_time[:, 0][::-1], "s--", lw=1.5, label="$Time(s)$")
plt.semilogy(interval_n, err_time[:, 1][::-1], "o-", lw=1.5, label=r"$\epsilon=\vert I - I^* \vert$")
plt.xlabel("区间数 $n$", fontdict={"fontsize": 18})
plt.ylabel("绝对误差$\epsilon$,执行时间$Time$", fontdict={"fontsize": 18})
plt.legend(fontsize=16, frameon=False)
plt.title(r"复合科特斯公式积分误差和时间消耗变化曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.ylim([1e-4, 1e-1])
plt.show()