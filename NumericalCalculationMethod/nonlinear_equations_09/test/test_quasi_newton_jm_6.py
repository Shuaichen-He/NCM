# -*- coding: UTF-8 -*-
"""
@file_name: test_quasi_newton_jm_5.py
@time: 2023-02-18
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import time
from nonlinear_equations_09.rank1_quasi_newton_jm import Rank1QuasiNewton
from nonlinear_equations_09.rank2_quasi_newton_jm import Rank2QuasiNewton
from util_font import *

x, y, z = sympy.symbols("x, y, z")
# 非线性方程组(1)
nlin_equs_sym = [sympy.sin(x) + y ** 2 + sympy.log(z) - 7,
                 3 * x + 2 ** y - z ** 3 + 1,
                 x + y + z - 5]


def nlin_funs_expr(x):
    y = [np.sin(x[0]) + x[1] ** 2 + np.log(x[2]) - 7,
         3 * x[0] + 2 ** x[1] - x[2] ** 3 + 1,
         x[0] + x[1] + x[2] - 5]
    return np.asarray(y, dtype=np.float64)


# 非线性方程组(2)
# nlin_equs_sym = [3 * x - sympy.cos(y * z) - 0.5,
#                  x ** 2 - 81 * (y + 0.1) ** 2 + sympy.sin(z) + 1.06,
#                  sympy.exp(-x * y) + 20 * z + 10 / 3 * sympy.pi - 1]
#
#
# def nlin_funs_expr(x):
#     y = [3 * x[0] - np.cos(x[1] * x[2]) - 0.5,
#          x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + np.sin(x[2]) + 1.06,
#          np.exp(-x[0] * x[1]) + 20 * x[2] + 10 / 3 * np.pi - 1]
#     return np.asarray(y, dtype=np.float)


# x0 = np.array([1, 1.8, 1.5])  # 方程组(1)
x0 = np.array([0.5, 1.8, 2])  # 方程组(1)
# x0 = np.array([0.1, 0.1, -0.1])  # 方程组(2)
# x0 = np.array([0, 0, 0])  # 方程组(2)
# x0 = np.array([2, 0, -5])  # 方程组(2)

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
        msr = eval(class_[k])(nlin_equs_sym, (x, y, z), x0, max_iter=1000, eps=1e-15,
                              method=method, is_plt=False)
        msr.fit_nlin_roots()  # 求解近似解
        end = time.time()  # 标记算法终止时间
        print("%s：" % method, "时间：", end - start, "收敛精度：%.10e" % msr.iter_roots_precision[-1][-1])
        for i in range(3):
            print("%.20f, %.15e" % (msr.roots[i], msr.fxs_precision[i]))
        print("=" * 80)
        error_2 = np.linalg.norm(nlin_funs_expr(msr.roots))  # 解的精度
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
