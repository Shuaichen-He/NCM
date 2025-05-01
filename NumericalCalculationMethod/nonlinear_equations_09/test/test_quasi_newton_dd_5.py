# -*- coding: UTF-8 -*-
"""
@file:test_quasi_newton_dd_5.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from nonlinear_equations_09.rank2_quasi_newton_dd import Rank2QuasiNewton
from nonlinear_equations_09.rank1_quasi_newton_dd import Rank1QuasiNewton
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


x0 = np.linspace(1, n, n)  # 迭代初值
# x0 = np.zeros(n)  # 迭代初值
# x0 = 100 * np.ones(n)  # 迭代初值
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
        msr = eval(class_[k])(fun_nlinq, x0, max_iter=1000, eps=1e-15, method=method, is_plt=False, diff_mat_h=0.01)
        msr.fit_nlin_roots()  # 求解近似解
        end = time.time()  # 标记算法终止时间
        print("时间：", end - start)
        print("解向量的精度范数：%.10e" % np.linalg.norm(msr.fxs_precision))
        print("解向量的收敛精度：%.10e" % msr.iter_roots_precision[-1][-1])
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
