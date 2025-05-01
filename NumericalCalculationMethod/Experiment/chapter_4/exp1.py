# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_integration_04.newton_cotes_integration import NewtonCotesIntegration
from numerical_integration_04.composite_quadrature_formula import CompositeQuadratureFormula
from numerical_integration_04.romberg_acceleration_quad import RombergAccelerationQuadrature
from Experiment.util_font import *  # 导入字体文件

int_fun = lambda x: x ** 5 * np.exp(-x) * np.sin(x)
I_f = -np.exp(-5) * (3660 * np.cos(5) + 975 * np.sin(5) / 2) - 15  # 积分精确值

# (1) 牛顿—科特斯积分
num = np.linspace(2, 8, 7)  # 划分区间数
int_res = np.zeros(len(num))  # 存储各划分区间数下的积分值
for i, n in enumerate(num):
    nci = NewtonCotesIntegration(int_fun, int_interval=[0, 5], interval_num=n)  # 实例化对象
    nci.fit_cotes_int()  # 求解积分
    print("n = %d，科特斯系数：\n" % n, nci.cotes_coefficient)  # 打印科特斯系数
    int_res[i] = nci.int_value  # 存储积分值
print("=" * 80)
print("积分精确值：%.8f" % I_f)
print("各划分区间数下的积分近似值：\n", int_res)  # 误差计算
print("各划分区间数下的积分绝对值误差：\n", abs(I_f - int_res))  # 误差计算
print("=" * 80)

# (2) 复合积分公式
x = sympy.Symbol("x")  # 定义符号变量
fun = x ** 5 * sympy.exp(-x) * sympy.sin(x)  # 被积函数，符号函数
for type in ["trapezoid", "simpson", "cotes"]:
    cqf = CompositeQuadratureFormula(fun, [0, 5], interval_num=16, int_type=type, is_remainder=False)
    cqf.fit_int()
    print("积分方法：%s，积分近似值：%.8f，绝对值误差：%.8e" % (type, cqf.int_value, abs(I_f - cqf.int_value)))
print("=" * 80)

interval_n = np.arange(5, 41, 1)
int_type = ["trapezoid", "simpson", "cotes"]
int_error = np.zeros((len(interval_n), 3))
for i in range(3):
    for j, n in enumerate(interval_n):
        cqf = CompositeQuadratureFormula(fun, [0, 5], n, int_type=int_type[i])
        cqf.fit_int()
        int_error[j, i] = np.abs(I_f - cqf.int_value)
plt.figure(figsize=(7, 5))
plt.semilogy(interval_n, int_error[:, 0], "*--", lw=1.5, label="$Trapezoid$")
plt.semilogy(interval_n, int_error[:, 1], "o-", lw=1.5, label="$Simpson$")
plt.semilogy(interval_n, int_error[:, 2], "s-", lw=1.5, label="$Cotes$")
plt.xlabel("区间数 $n$", fontdict={"fontsize": 18})
plt.ylabel("绝对误差 $\epsilon$", fontdict={"fontsize": 18})
plt.legend(fontsize=16, frameon=False, ncol=3)
plt.title(r"三种复合求积公式积分误差$\epsilon=\vert I - I^* \vert$变化曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()

# 龙贝格求积公式
accelerate_num = np.arange(6, 11, 1)
for an in accelerate_num:
    raq = RombergAccelerationQuadrature(int_fun, int_interval=[0, 5], accelerate_num=an)
    raq.fit_int()  # 计算龙贝格积分
    if an == 6:
        print(raq.Romberg_acc_table)  # 打印外推计算过程表
    print("%d: 积分近似值：%.15f, 绝对值误差：%.15e" % (an, raq.int_value, abs(I_f - raq.int_value)))
