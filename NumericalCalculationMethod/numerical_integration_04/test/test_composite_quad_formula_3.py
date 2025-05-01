# -*- coding: UTF-8 -*-
"""
@file_name: test_composite_quad_formula_3.py
@time: 2021-09-22
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_integration_04.composite_quadrature_formula import CompositeQuadratureFormula
from util_font import *

x = sympy.Symbol("x")  # 定义符号变量
fun = x / (4 + x ** 2)  # 被积函数，符号函数
for type in ["trapezoid", "simpson", "cotes"]:
    cqf = CompositeQuadratureFormula(fun, [0, 1], interval_num=16, int_type=type, is_remainder=True)
    cqf.fit_int()
    print("积分方法：%s，积分近似值：%.15f，余项：%.15e" % (type, cqf.int_value, cqf.int_remainder))
    print("误差：%.15e" % (0.5 * np.log(1.25) - cqf.int_value))


interval_n = np.arange(5, 41, 1)
int_type=["trapezoid", "simpson", "cotes"]
exact_val = np.log(5 / 4) / 2
int_error = np.zeros((len(interval_n), 3))
for i in range(3):
    for j, n in enumerate(interval_n):
        cqf = CompositeQuadratureFormula(fun, [0, 1], n, int_type=int_type[i])
        cqf.fit_int()
        int_error[j, i] = np.abs(exact_val - cqf.int_value)
        print(i, n, int_error[j, i])
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(interval_n, int_error[:, 0], "*--", lw=1.5, label="$Trapezoid$")
plt.plot(interval_n, int_error[:, 1], "o-", lw=1.5, label="$Simpson$")
plt.plot(interval_n, int_error[:, 2], "s-", lw=1.5, label="$Cotes$")
plt.xlabel("区间数 $n$", fontdict={"fontsize": 18})
plt.ylabel("绝对误差 $\epsilon$", fontdict={"fontsize": 18})
plt.legend(fontsize=16, frameon=False, ncol=3)
plt.title(r"三种复合求积公式积分误差$\epsilon=\vert I - I^* \vert$变化曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
plt.semilogy(interval_n, int_error[:, 0], "*--", lw=1.5, label="$Trapezoid$")
plt.semilogy(interval_n, int_error[:, 1], "o-", lw=1.5, label="$Simpson$")
plt.semilogy(interval_n, int_error[:, 2], "s-", lw=1.5, label="$Cotes$")
plt.xlabel("区间数 $n$", fontdict={"fontsize": 18})
plt.ylabel("绝对误差 $\epsilon$", fontdict={"fontsize": 18})
plt.legend(fontsize=16, frameon=False, ncol=3)
plt.title(r"三种复合求积公式积分误差$\epsilon=\vert I - I^* \vert$变化曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.ylim([1e-16, 1e+0])
plt.grid(ls=":")
plt.show()

# 双y轴坐标绘制
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# # 由于复合辛普森精确值与积分值之差为负数，为便于可视化比较，转换为正值，不影响误差大小
# line_1 = ax.plot(interval_n, -int_error[:, 0], "r*--", lw=1.5, label="$Simpson \ (1e-7)$")
# ax.set_ylabel("复合辛普生", fontdict={"fontsize": 18})
# ax.set_xlabel("划分区间数", fontdict={"fontsize": 18})
# ax.tick_params(labelsize=16)  # 刻度字体大小16
# ax2 = ax.twinx()  # 添加第二个y轴
# line_2 = ax2.plot(interval_n, int_error[:, 1], "b+-", lw=1.5, label="$Cotes \ (1e-11)$")
# ax2.set_ylabel("复合科特斯", fontdict={"fontsize": 18})
# # 合并图例显示
# lines = line_1 + line_2
# labs = [line.get_label() for line in lines]
# ax.legend(lines, labs, loc="center right", fontsize=18, frameon=False)
# plt.title("复合辛普生和复合科特斯积分误差变化曲线", fontdict={"fontsize": 18})
# ax2.tick_params(labelsize=16)  # 刻度字体大小16
# plt.show()
