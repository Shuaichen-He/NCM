# -*- coding: UTF-8 -*-
"""
@file_name: test_newton_flight.py
@time: 2021-11-28
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from solving_equation_08.newton_root_method import NewtonRootMethod
from util_font import *

t = sympy.Symbol("t")
ft = 4800 * (1 - sympy.exp(-t / 10)) - 320 * t
rt = lambda t: 1600 * (1 - np.exp(-t / 10))

plt.figure(figsize=(14, 5))
plt.subplot(121)
axes = plt.gca()  # 获取坐标轴对象
axes.spines["right"].set_color("none")  # 去掉右边框框颜色
axes.spines["top"].set_color("none")  # 去掉上边框颜色
axes.set_xlim(0, 10)  # 设置x坐标轴范围
axes.set_ylim(-120, 320)  # 设置y坐标轴范围
# 移动左边框和下边框到中间(set_position的参数是元组)
axes.spines["left"].set_position(("data", 0))  # 移动左边框到中间(set_position的参数是元组)
axes.spines["bottom"].set_position(("data", 0))  # 移动下边框到中间
xi = np.linspace(0, 10, 200)  # 区间等分200个数据点
yi = sympy.lambdify(t, ft)(xi)  # 对应y值
plt.plot(xi, yi, "k-", lw=1.5)  # 绘制函数
plt.text(9.5, 20, "$t$", fontdict={"fontsize": 22})  # 标记x轴
plt.text(0.2, 300, "$f(t)$", fontdict={"fontsize": 22})  # 标记y轴
plt.title(r"$f(t)=4800 \left(1-e^{-0.1t} \right)-320t \quad t \in [0, 10]$", fontsize=20)
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.grid(ls=":")

plt.subplot(122)
line_style = ["*--", "o-"]
method_vector = ["newton", "halley"]
for style, method in zip(line_style, method_vector):
    newton = NewtonRootMethod(ft, x0=8.5, eps=1e-16, method=method)
    newton.fit_root()
    iter_root_precision = np.asarray(newton.root_precision_info, np.float)
    plt.semilogy(iter_root_precision[:, 0], iter_root_precision[:, 2], style, lw=2,
                 label="$%s(k=%d, \ \epsilon=%.2e)$" % (method, len(iter_root_precision), iter_root_precision[-1, 2]))
    print("飞行时间为：", newton.root)
    print("飞行水平距离为：", rt(newton.root))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"牛顿法与哈利法$(x_0=8.5)$：$x_k$的$\epsilon=\vert f(x_k)\vert$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
# t_root = NewtonRootMethod(ft, 8, eps=1e-16, method="halley")
# t_root.fit_root()
# print("飞行时间为：", t_root.root)
# print("飞行水平距离为：", rt(t_root.root))
