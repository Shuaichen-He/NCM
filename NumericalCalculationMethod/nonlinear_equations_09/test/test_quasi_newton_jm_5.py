# -*- coding: UTF-8 -*-
"""
@file:test_quasi_newton_5.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from nonlinear_equations_09.rank2_quasi_newton_jm import Rank2QuasiNewton
from nonlinear_equations_09.rank1_quasi_newton_jm import Rank1QuasiNewton
import time
from util_font import *

n = 100  # 方程个数


def fun_nlinq(x):
    """
    定义非线性方程组
    :param x: 迭代解向量
    :return:
    """
    A = np.diag(2 * np.ones(n)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
    f_x = np.arctan(x) - 1
    y = np.dot(A, x) + 1 / (n + 1) ** 2 * f_x
    return np.asarray(y, dtype=np.float64)


x = sympy.symbols("x_1:101")
Fx_sym = sympy.zeros(1, n)
for i in range(n):
    if i == 0:
        Fx_sym[i] = 2 * x[i] - x[i + 1] + 1 / (n + 1) ** 2 * (sympy.atan(x[i]) - 1)
    elif i == n - 1:
        Fx_sym[i] = -1 * x[i - 1] + 2 * x[i] + 1 / (n + 1) ** 2 * (sympy.atan(x[i]) - 1)
    else:
        Fx_sym[i] = -1 * x[i - 1] + 2 * x[i] - 1 * x[i + 1] + 1 / (n + 1) ** 2 * (sympy.atan(x[i]) - 1)

x0 = np.linspace(1, n, n)  # 迭代初值
plt.figure(figsize=(14, 5))
line_style = ["--o", ":s", "-.*", "-p"]  # 线型与点型
class_ = ["Rank1QuasiNewton", "Rank2QuasiNewton"]  # 类名，秩1和秩2算法
for k in range(2):
    plt.subplot(121 + k)  # 绘制子图
    if k == 0:
        method_list = ["Broyden", "Broyden2Th", "InvBroyden", "InvBroyden2Th"]  # 秩1算法
    else:
        method_list = ["DFP", "BFS", "BFGS", "InvBFGS"]  # 秩2算法
    for i, method in enumerate(method_list):
        print(method)
        start = time.time()  # 标记开始时间
        msr = eval(class_[k])(Fx_sym, x, x0, max_iter=1000, eps=1e-15, method=method, is_plt=False)
        msr.fit_nlin_roots()  # 求解近似解
        end = time.time()  # 标记算法终止时间
        print("时间：", end - start)
        print("=" * 80)
        irp = [rt[-1] for rt in msr.iter_roots_precision]  # 获取其迭代精度
        label_txt = method + ": k=" + str(len(irp)) + ", \ time: %.5f" % (end - start)
        plt.semilogy(range(1, len(irp) + 1), irp, line_style[i], label="$%s$" % label_txt)
    plt.xlabel(r"$Iterations(k)$", fontdict={"fontsize": 18})
    plt.ylabel(r"$\epsilon-Precision$", fontdict={"fontsize": 18})
    if k == 0:
        plt.title("不同秩1拟牛顿法的解向量 $x^*$ 精度收敛性", fontdict={"fontsize": 18})
    else:
        plt.title("不同秩2拟牛顿法的解向量 $x^*$ 精度收敛性", fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16)
    plt.tick_params(labelsize=18)  # 刻度字体大小16
    plt.grid(ls=":")
plt.show()
