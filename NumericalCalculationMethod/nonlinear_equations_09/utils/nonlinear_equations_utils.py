# -*- coding: UTF-8 -*-
"""
@file_name: nonlinear_equations_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class NonLinearEquationsUtils:
    """
    求解非线性方程组工具类。
    """

    def __init__(self, nlin_Fxs, x0, max_iter=200, eps=1e-15, is_plt=False):
        self.nlin_Fxs = nlin_Fxs  # 非线性方程组定义
        self.x0 = np.asarray(x0, dtype=np.float64).reshape(-1, 1)  # 迭代初始值，列向量形式
        self.max_iter, self.eps = max_iter, eps  # 最大迭代次数和解的精度要求
        self.is_plt = is_plt  # 是否可视化is_plt
        self.iter_roots_precision = []  # 存储迭代过程中的信息
        self.roots = None  # 满足精度或迭代要求的最终解

    def plt_precision_convergence_curve(self, is_show=True, title=""):
        """
        可视化解的精度迭代收敛曲线
        """
        rp = [rs[-1] for rs in self.iter_roots_precision]  # 精度列表
        n = len(self.iter_roots_precision)  # 迭代次数
        if is_show:
            plt.figure(figsize=(7, 5))
        # 修改label值：牛顿法为$\epsilon=\Vert F(x^{(k)}) \Vert _2$；
        # 不动点迭代法为$\epsilon=\Vert \Phi(x^{(k)}) - x^{(k)} \Vert _2$
        # 其他方法的label值默认如下
        plt.semilogy(range(1, n + 1), rp, "*--",
                     label="$\epsilon=\Vert x_{k+1} - x_{k} \Vert _2$")
        # plt.semilogy(range(1, n + 1), rp, "*--", label="$\epsilon=\Vert F(x^{(k)}) \Vert _2$")  # 牛顿法
        # plt.semilogy(range(1, n + 1), rp, "*--", label="$\epsilon=\Vert \Phi(x^{(k)}) - x^{(k)} \Vert _2$")  # 不动点迭代法
        plt.semilogy(n, rp[-1], "D", label="$\epsilon=%.5e, \ k=%d$" %
                                           (rp[-1], len(rp)))  # 最终精度
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})  # 迭代次数
        plt.ylabel("$\epsilon-Precision$", fontdict={"fontsize": 18})  # 精度
        plt.title("$%s$：解向量精度 $\epsilon$ 收敛曲线" % title,
                  fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=18)
        plt.tick_params(labelsize=18)  # 刻度字体大小18
        plt.grid(ls=":")
        if is_show:
            plt.show()

    def plt_roots_convergence_curve(self, is_show=True, title=""):
        """
        可视化非线性方程组解的收敛曲线
        """
        rp = np.asarray([list(rs[1]) for rs in self.iter_roots_precision], dtype=np.float64)
        n, m = rp.shape  # 迭代次数和解向量个数
        if is_show:
            plt.figure(figsize=(7, 5))
        p_type = ["*", "+", "x", "o", "v", "^", "<", ">", "p", "s", "h", "d"]  # 点的类型
        for i in range(m):
        # for i in range(5):  # 针对解特别多的情况下，仅显示前6个解的收敛性
            plt.plot(range(1, n + 1), rp[:, i], p_type[i] + "-",
                     label=r"$x_{%d}=%.10f$" % (i + 1, rp[-1, i]))
        # 略去图形修饰，参考方法plt_precision_convergence_curve()
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$Sol \ x^*$", fontdict={"fontsize": 18})
        plt.title("$%s$：解向量 $x^*$ 收敛性" % title, fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=18)  # 刻度字体大小18
        plt.grid(ls=":")
        if is_show:
            plt.show()
