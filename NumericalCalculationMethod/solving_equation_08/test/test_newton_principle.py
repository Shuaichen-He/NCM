# -*- coding: UTF-8 -*-
"""
@file_name: test_newton_principle.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *

tol, xk = 1e-2, 3.0  # 精度及初值
fx, dfx = lambda x: np.exp(x) - 2, lambda x: np.exp(x)  # 方程，一阶导
plt.figure(figsize=(7, 5))  # 可视化牛顿迭代法根的搜索过程
xi = np.linspace(-1, 3.1, 200)  # 等分离散值
plt.plot(xi, fx(xi), "-", lw=2)  # 方程图形
plt.axhline(0, ls=':', color='k')  # x=0轴水平线
n = 0  # 迭代次数
while abs(fx(xk)) > tol:  # 如下重复迭代，直到满足精度要求，即收敛精度
    xk_new = xk - fx(xk) / dfx(xk)  # 牛顿迭代公式
    plt.plot([xk, xk], [0, fx(xk)], color="k", ls=':')  # 垂直线
    plt.plot(xk, fx(xk), 'ko')  # 描点，迭代过程的近似根
    plt.text(xk, -1, r"$x_%d$" % n, ha="center", fontsize=18)  # 添加文本标注
    plt.plot([xk, xk_new], [fx(xk), 0], "r--")  # 切线
    xk = xk_new  # 值更新
    n += 1
plt.text(-0.8, 16, "$x_{k+1} = x_k - \dfrac{f(x_k)}{f^{\prime}(x_k)},"
                   "\ k = 0,1,2,\cdots$", fontsize=18)
plt.annotate("$x^* = %.8f$" % xk, fontsize=18, family="serif", xy=(xk, fx(xk)),
             xycoords="data", xytext=(-100, +50), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.5"))
plt.title(r"牛顿迭代法的几何意义:$f(x) = e^x - 2, \ \vert f(x^*) \vert \leq 0.01$",
          fontdict={"fontsize": 18})
plt.xlabel("$x(x_0=3)$", fontdict={"fontsize": 18})
plt.ylabel("$f(x)$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.show()
