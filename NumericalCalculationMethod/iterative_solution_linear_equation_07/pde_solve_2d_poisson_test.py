# -*- coding: UTF-8 -*-
"""
@file_name: pde_2d_poisson_test.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import scipy.sparse as sp  # 用于构造稀疏矩阵
# from iterative_solution_linear_equation_07.poisson_model import PoissonModel  # 方程模型
from Experiment.chapter_7.poisson_model import PoissonModel # 方程模型
# 以下为求解大型稀疏矩阵的方法
from iterative_solution_linear_equation_07.pre_conjugate_gradient \
    import PreConjugateGradient  # 预处理共轭梯度法
from iterative_solution_linear_equation_07.conjugate_gradient_method \
    import ConjugateGradientMethod  # 共轭梯度法
from iterative_solution_linear_equation_07.steepest_descent_method \
    import SteepestDescentMethod  # 最速下降法
from iterative_solution_linear_equation_07.jacobi_gauss_seidel_iterative \
    import JacobiGSlIterativeMethod  # 雅可比与G-S迭代法
from iterative_solution_linear_equation_07.SOR_iterative import SORIteration
from util_font import *


class PDESolvePoisson2dModel:
    """
    二维泊松方程模型，五点差分格式求解，采用共轭梯度法求解
    """

    def __init__(self, x_span, y_span, n_x, n_y, is_show=False, is_exact_fun=False):
        self.x_span = np.asarray(x_span, np.float64)  # x方向求解区间
        self.y_span = np.asarray(y_span, np.float64)  # y方向求解区间
        self.n_x, self.n_y = n_x, n_y  # x方向和y方向划分区间数
        self.h_x, self.h_y, self.xi, self.yi = self._space_grid()
        self.is_show = is_show  # 是否可视化泊松方程解的图像
        self.is_exact_fun = is_exact_fun  # 是否存在精确解，用于可视化误差

    def _space_grid(self):
        """
        划分二维平面网格
        :return:
        """
        xi = np.linspace(self.x_span[0], self.x_span[1], self.n_x + 1)  # 等分x
        yi = np.linspace(self.y_span[0], self.y_span[1], self.n_y + 1)  # 等分y
        h_x = (self.x_span[1] - self.x_span[0]) / self.n_x  # x区间步长
        h_y = (self.y_span[1] - self.y_span[0]) / self.n_y  # y区间步长
        return h_x, h_y, xi, yi

    def fit_pde(self):
        """
        核心算法：二维泊松方程求解，五点差分格式
        :return:
        """
        ym, xm = np.meshgrid(self.yi, self.xi)  # 生成二维网格点
        u_xy = np.zeros((self.n_x + 1, self.n_y + 1))  # 泊松方程的数值解
        # 解的边界情况处理
        u_xy[0, :] = PoissonModel.left_boundary(self.yi)  # 左边界
        u_xy[-1, :] = PoissonModel.right_boundary(self.yi)  # 右边界
        u_xy[:, 0] = PoissonModel.lower_boundary(self.xi)  # 底部
        u_xy[:, -1] = PoissonModel.upper_boundary(self.xi)  # 顶部
        # 按照稀疏矩阵形式构造，即构造块三角矩阵
        c_identity = np.ones(self.n_x - 1)  # 单位向量，构成一块C中的对角线元素
        # 主对角线上下次对角线，即C，主对角线块
        c_diag = sp.diags([2 * c_identity, -c_identity, -c_identity], [0, -1, 1],
                          format='csc', dtype=np.float64)
        identity_mat = sp.eye(self.n_y - 1, format='csc')  # 稀疏单位阵，按此构造快
        # sp.kron(A, B)为A和B的克罗内克积，CSC(压缩的列)格式
        # 单位阵中的1以d_diag为矩阵块张量成对角块矩阵，即三对角块矩阵中的对角块C
        C = sp.kron(identity_mat, c_diag, format='csc') / self.h_x ** 2
        # d_diag中的非零元素按照单位阵的构造，张量成对角块矩阵，包括C的主对角线元素以及三对角块矩阵中的D
        D = sp.kron(c_diag, identity_mat, format='csc') / self.h_y ** 2
        difference_matrix = C + D  # 如此形成的三对角块矩阵，主对角线系数为2倍。
        # 构造右端向量，构成右端fi.pde右端方程，内部节点(n-1)*(m-1)
        fi = PoissonModel.fun_xy(xm[1: -1, 1: -1], ym[1: -1, 1: -1])
        fi[0, :] = fi[0, :] + u_xy[0, 1:-1] / self.h_x ** 2  # 第一行单独处理，φ(x0,yj)
        fi[-1, :] = fi[-1, :] + u_xy[-1, 1:-1] / self.h_x ** 2  # 最后一行单独处理，φ(xm,yj)
        fi = fi.T.flatten()  # 展平成向量，默认按行
        d_diag = np.diag(-c_identity / self.h_y ** 2)  # 单个块D对角矩阵
        fi[:self.n_x - 1] = fi[: self.n_x - 1] - np.dot(d_diag, u_xy[1:-1, 0])  # f_1-D*u_0
        fi[1 - self.n_x:] = fi[1 - self.n_x:] - np.dot(d_diag, u_xy[1:-1, -1])  # f_(n-1) - D*u_n
        # 采用各种迭代法求解大型稀疏矩阵
        sol = self._solve_sparse_matrix_method_(difference_matrix.toarray(), fi)
        u_xy[1:-1, 1:-1] = sol.reshape(self.n_y - 1, self.n_x - 1).T
        if self.is_show:  # 可视化泊松方程数值解(解析解)图像
            self.plt_2d_poisson(xm, ym, u_xy, self.is_exact_fun)
        return xm, ym, u_xy

    @staticmethod
    def _solve_sparse_matrix_method_(sp_mat, b):
        """
        求解大型稀疏矩阵，采用预处理共轭梯度法
        :return:
        """
        x0 = np.zeros(len(b))  # 初始解向量
        pcg = PreConjugateGradient(sp_mat, b, x0, 1e-5, omega=1.5, max_iter=200, is_out_info=True)
        sol_x = pcg.fit_solve()  # 求解
        # pcg.plt_convergence_x()  # 可视化
        pcg_iter = pcg.iterative_info["Iteration_number"]

        # 如下代码可实现四种方法的可视化
        cg = ConjugateGradientMethod(sp_mat, b, x0, 1e-5, max_iter=200, is_out_info=True)  # 共轭梯度法
        cg.fit_solve()
        cg_iter = cg.iterative_info["Iteration_number"]
        sdm = SteepestDescentMethod(sp_mat, b, x0, 1e-5, is_out_info=True)  # 最速下降法
        sdm.fit_solve()
        gs = JacobiGSlIterativeMethod(sp_mat, b, x0, eps=1e-5, method="g-s", is_out_info=True)  # G-S迭代法
        gs.fit_solve()

        sor = SORIteration(sp_mat, b, x0, eps=1e-5, omega=1.5, is_out_info=True)  # G-S迭代法
        sor.fit_solve()
        plt.figure(figsize=(7, 5))
        plt.semilogy(range(1, pcg_iter + 1), pcg.precision, "o-", lw=2,
                     label="$PCG: \epsilon=%.3e, \ k=%d$" % (pcg.precision[-1], pcg_iter))
        plt.semilogy(range(1, cg_iter + 1), cg.precision, "s--", lw=2,
                     label="$CG: \epsilon=%.3e, \ k=%d$" % (cg.precision[-1], cg_iter))
        # plt.semilogy(range(1, 201), sdm.precision, "+-.", lw=2,
        #              label="$SD: \epsilon=%.3e, \ k=%d$" % (sdm.precision[-1], len(sdm.precision)))
        plt.semilogy(range(1, len(sor.precision) + 1), sor.precision, "*:", lw=2,
                     label="$SOR: \epsilon=%.3e, \ k=%d$" % (sor.precision[-1], len(sor.precision)))
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
        plt.title("各迭代法求解泊松方程的解向量$x^*$收敛曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
        return sol_x  # 预处理共轭梯度法的解

    @staticmethod
    def plt_2d_poisson(xm, ym, uh, is_error=True):
        """
        可视化泊松方程的解
        :return:
        """
        if is_error:
            plt.figure(figsize=(14, 5))
            ax = plt.subplot(121, projection='3d')
        else:
            plt.figure(figsize=(7, 5))
            ax = plt.subplot(111, projection='3d')
        ax.plot_surface(xm, ym, uh, cmap="rainbow")
        ax.set_xlabel("$x$", fontdict={"fontsize": 20})
        ax.set_ylabel("$y$", fontdict={"fontsize": 20})
        ax.set_zlabel("$z$", fontdict={"fontsize": 20})
        plt.title("二维泊松方程的解曲面", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if is_error:
            error = uh - PoissonModel.analytic_sol(xm, ym)  # 与精确解的误差
            ax = plt.subplot(122, projection='3d')
            ax.plot_surface(xm, ym, error, cmap="rainbow")
            ax.set_xlabel("$x$", fontdict={"fontsize": 20})
            ax.set_ylabel("$y$", fontdict={"fontsize": 20})
            ax.set_zlabel(r"$\varepsilon$", fontdict={"fontsize": 20})
            mae = np.mean(np.abs(error))  # 平均绝对误差能更好地反映预测值误差的实际情况.
            plt.title("误差曲面：$MAE=%0.3e$" % mae, fontdict={"fontsize": 18})
            plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.show()
