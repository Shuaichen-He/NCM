# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import time
from Experiment.util_font import *
from nonlinear_equations_09.rank2_quasi_newton_dd import Rank2QuasiNewton
from nonlinear_equations_09.rank1_quasi_newton_dd import Rank1QuasiNewton


plt.figure(figsize=(7, 5))
x, y = sympy.symbols("x, y")
nlin_equs = [x ** 2 * sympy.exp(-x * y ** 2 / 2) + sympy.exp(-x / 2) * sympy.sin(x * y),
            y ** 2 * sympy.cos(x + y ** 2) + x ** 2 * sympy.exp(x + y)]
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r")
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="c")
p0.extend(p1)
p0.show()

def nlin_funs(x):
    nlinequs = [x[0] ** 2 * np.exp(-x[0] * x[1] ** 2 / 2) + np.exp(-x[0] / 2) * np.sin(x[0] * x[1]),
                x[1] ** 2 * np.cos(x[0] + x[1] ** 2) + x[0] ** 2 * np.exp(x[0] + x[1])]
    return np.asarray(nlinequs, dtype=np.float64)


x_init = np.array([[-0.8, 1.5], [1, -1], [-0.5, 2.2]])  # 迭代初值
for x0 in x_init:
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
            start = time.time()  # 标记开始时间
            msr = eval(class_[k])(nlin_funs, x0, max_iter=1000, eps=1e-15, method=method,
                                  is_plt=False, diff_mat_h=0.01)
            msr.fit_nlin_roots()  # 求解近似解
            end = time.time()  # 标记算法终止时间
            print("时间：", end - start)
            print(method, ":", x0, msr.iter_roots_precision[-1])
            print("=" * 80)
            error_2 = np.linalg.norm(nlin_funs(msr.roots))  # 解的精度
            irp = [rt[-1] for rt in msr.iter_roots_precision]  # 获取其迭代精度
            label_txt = method + ": k=" + str(len(irp)) + ",\ \epsilon_{F_2}=%.3e" % error_2
            plt.semilogy(range(1, len(irp) + 1), irp, line_style[i], label="$%s$" % label_txt)
        plt.xlabel(r"$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel(r"$\epsilon-Precision$", fontdict={"fontsize": 18})
        if k == 0:
            plt.title("不同秩1拟牛顿法的解向量 $x^*$ 精度收敛性", fontdict={"fontsize": 18})
        else:
            plt.title("不同秩2拟牛顿法的解向量 $x^*$ 精度收敛性", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=18)  # 刻度字体大小18
        plt.grid(ls=":")
    plt.show()
