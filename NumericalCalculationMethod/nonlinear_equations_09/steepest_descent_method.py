# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:steepest_descent_method.py
@time:2021/08/22
"""
import numpy as np
import matplotlib.pyplot as plt


class SteepestDescentIteration:
    """
    最速下降迭代法求解非线性方程组的解
    """

    def __init__(self, nonlinear_equations, x0, h, max_iter=200, eps=1e-10, display="iter", is_plt=False):
        self.nonlinear_equations = nonlinear_equations  # 非线性方程组
        self.x0 = np.asarray(x0, dtype=np.float).reshape(-1, 1)  # 迭代初始值，向量形式
        self.n = len(x0)  # 解的个数
        self.h = np.asarray(h, dtype=np.float)  # 数值微分增量
        self.max_iter = max_iter  # 最大迭代次数
        self.eps = eps  # 解的精度要求
        self.display = display  # 是否显示过程
        self.is_plt = is_plt  # 是否绘制收敛曲线
        self.iter_roots_precision = []  # 存储迭代过程种的信息
        self.root = None  # 满足精度或迭代要求的最终的解

    def diff_mat(self, x_b, sol_xb):
        """
        求解差商形式的雅可比矩阵
        :param x_b: 迭代的上一次值
        :param sol_xb: x_b的方程组的值
        :return:
        """
        J = np.zeros((self.n, self.n))
        for i in range(self.n):
            xd = np.copy(x_b)
            xd[i] += self.h[i]
            J[:, i] = ((self.nonlinear_equations(xd) - sol_xb) / self.h[i]).reshape(-1)
        return J

    def fit_nlinequs_roots(self):
        """
        最速下降迭代法求解非线性方程组的解，核心算法
        :return:
        """
        iter_, sol_tol, x_n = 0, np.infty, self.x0
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            iter_ += 1
            x_b = np.copy(x_n)
            sol_xb = self.nonlinear_equations(x_b)
            J = self.diff_mat(x_b, sol_xb)
            lambda_ = sol_xb / np.sum(np.diag(np.dot(J.T, J)))
            x_n = x_b - np.dot(J, lambda_)
            sol_tol = np.linalg.norm(x_n - x_b)
            self.iter_roots_precision.append([iter_, x_n, sol_tol])
            # 前期步长大，后期步长小
            # if np.mod(iter_, 10) == 0:
            #     self.h /= 2

        # 求解过程的显示控制
        self.display_iter_process()

        # 是否可视化图像
        if self.is_plt:
            self.plt_precision_convergence_curve()
            self.plt_roots_convergence_curve()

    def display_iter_process(self):  # 参考离散牛顿法
        """
        显示迭代过程信息
        :return:
        """
        if self.display == "iter":
            for val in self.iter_roots_precision:
                print("Iter：%d  ApproximateRoots：" % val[0], end="")
                for r in val[1]:
                    print("%.15f" % r, end=" ")
                print("  Precision：%.15e" % val[2])
        elif self.display == "final":
            print("Iter：%d, Precision：%.10e"
                  % (self.iter_roots_precision[-1][0], self.iter_roots_precision[-1][2]))
        else:
            raise ValueError("参数display的值仅能为final或iter!")
        self.root = self.iter_roots_precision[-1][1]  # 满足精度的根
        precision = self.nonlinear_equations(self.root)
        print("The root satisfying the precision is")
        for i, (val, p) in enumerate(zip(self.root, precision)):
            print("x_%d = %.20f，precision = %.15e" % (i + 1, val, p))

    def plt_precision_convergence_curve(self):  # 参考非线性方程组的不动点方法
        """
        可视化解的精度迭代收敛曲线
        :return:
        """
        precision = [roots[-1] for roots in self.iter_roots_precision]
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(precision) + 1), precision, "k*-", label="%.10e" % precision[-1])
        plt.xlabel("Number of Iteration", fontdict={"fontsize": 12})
        plt.ylabel(r"Root precision", fontdict={"fontsize": 12})
        plt.title("The precision of nonlinear equations with quasi newton iteration",
                  fontdict={"fontsize": 14})
        plt.legend(title="Final precision  norm($x_{k+1} - x_{k}$)")
        plt.show()

    def plt_roots_convergence_curve(self):  # 参考非线性方程组的不动点方法
        """
        可视化非线性方程组解的收敛曲线
        :return:
        """
        roots = [list(roots[1]) for roots in self.iter_roots_precision]
        roots = np.asarray(roots, np.float)
        plt.figure(figsize=(8, 6))
        points_type = ["*", "+", "x", "o", "v", "^", "<", ">", "p", "s", "h", "d"]
        for i in range(roots.shape[1]):
            plt.plot(range(1, roots.shape[0] + 1), roots[:, i], points_type[i] + "-",
                     label=r"$x_{%d}$" % (i + 1))
        plt.xlabel("Number of Iteration", fontdict={"fontsize": 12})
        plt.ylabel("Root Values", fontdict={"fontsize": 12})
        plt.title("The Roots convergence curve of nonlinear equations with quasi newton", fontdict={"fontsize": 14})
        plt.legend()
        plt.grid(ls=":")
        plt.show()

