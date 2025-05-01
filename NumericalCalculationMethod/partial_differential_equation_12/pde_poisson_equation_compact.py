# -*- coding: UTF-8 -*-
"""
@file_name: pde_poisson_equation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import scipy.sparse as sp  # 用于构造稀疏矩阵
import matplotlib.pyplot as plt
from iterative_solution_linear_equation_07.pre_conjugate_gradient import PreConjugateGradient


class PDEPoissonEquationCompact:
    """
    二维泊松方程模型，9点差分格式求解，采用共轭梯度法求解
    """

    def __init__(self, fxy_fun, left_boundary, right_boundary, lower_boundary, upper_boundary,
                 x_span, y_span, n_x, n_y, is_show=False, pde_model=None):
        self.fxy_fun = fxy_fun  # 泊松方程的右端函数f(x, y)，关于自变量x和y的函数
        # 边界函数，对应u(0, y)和u(a, y)，关于y的函数
        self.left_boundary, self.right_boundary = left_boundary, right_boundary
        # 边界函数，对应u(x, 0)和u(x, b)，关于x的函数
        self.lower_boundary, self.upper_boundary = lower_boundary, upper_boundary
        self.x_span, self.y_span = np.asarray(x_span, np.float), np.asarray(y_span, np.float)
        self.n_x, self.n_y = n_x, n_y  # 划分区间数
        # 等分区间点和区间步长
        self.h_x, self.h_y, self.xi, self.yi = self._space_grid_()
        self.is_show = is_show  # 是否可视化泊松方程解的图像
        self.pde_model = pde_model  # 是否存在解析解，用于误差分析，可视化误差
        self.u_xy = None  # 存储pde数值解

    def _space_grid_(self):
        """
        划分二维平面网格
        :return:
        """
        xi = np.linspace(self.x_span[0], self.x_span[1], self.n_x + 1)  # 等分x
        yi = np.linspace(self.y_span[0], self.y_span[1], self.n_y + 1)  # 等分y
        h_x = (self.x_span[1] - self.x_span[0]) / self.n_x  # x区间步长
        h_y = (self.y_span[1] - self.y_span[0]) / self.n_y  # y区间步长
        return h_x, h_y, xi, yi

    def solve_pde(self):
        """
        二维泊松方程求解，五点差分格式
        :return:
        """
        ym, xm = np.meshgrid(self.yi, self.xi)  # 生成二维网格点
        self.u_xy = np.zeros((self.n_x + 1, self.n_y + 1))  # 泊松方程的数值解
        # 解的边界情况处理
        self.u_xy[0, :] = self.left_boundary(self.yi)  # 左边界
        self.u_xy[-1, :] = self.right_boundary(self.yi)  # 右边界
        self.u_xy[:, 0] = self.lower_boundary(self.xi)  # 底部
        self.u_xy[:, -1] = self.upper_boundary(self.xi)  # 顶部
        # 按照稀疏矩阵形式构造，即构造块三角矩阵
        c1 = 5 / 3 * (1 / self.h_x ** 2 + 1 / self.h_y ** 2)
        c2 = -1 / 6 * (5 / self.h_x ** 2 - 1 / self.h_y ** 2)
        c3 = -1 / 6 * (5 / self.h_y ** 2 - 1 / self.h_x ** 2)
        c4 = -1 / 12 * (1 / self.h_x ** 2 + 1 / self.h_y ** 2)
        c_identity = np.ones(self.n_x - 1)  # 单位向量，构成一块C中的对角线元素
        # 主对角线上下次对角线，即C，主对角线块
        c_diag_1 = sp.diags([c1 * c_identity, c2 * c_identity, c2 * c_identity], [0, -1, 1],
                            format='csc', dtype=np.float64)
        c_diag_2 = sp.diags([c3 * c_identity, c4 * c_identity, c4 * c_identity], [0, -1, 1],
                            format='csc', dtype=np.float64)
        identity_mat = sp.eye(self.n_y - 1, format='csc')  # 稀疏单位阵，按此构造快
        # sp.kron(A, B)为A和B的克罗内克积，CSC(压缩的列)格式
        # 单位阵中的1以d_diag为矩阵块张量成对角块矩阵，即三对角块矩阵中的对角块C
        C = sp.kron(identity_mat, c_diag_1, format='csc') / self.h_x ** 2
        # d_diag中的非零元素按照单位阵的构造，张量成对角块矩阵，包括C的主对角线元素以及三对角块矩阵中的D
        D = sp.kron(c_diag_2, identity_mat, format='csc') / self.h_y ** 2
        difference_matrix = C + D  # 如此形成的三对角块矩阵，主对角线系数为2倍。
        # 构造右端向量，构成右端fi
        fi = self.fxy_fun(xm[1: -1, 1: -1], ym[1: -1, 1: -1])  # pde右端方程，内部节点(n-1)*(m-1)
        fi[0, :] = fi[0, :] + self.u_xy[0, 1:-1] / self.h_x ** 2  # 第一行单独处理，φ(x0,yj)
        fi[-1, :] = fi[-1, :] + self.u_xy[-1, 1:-1] / self.h_x ** 2  # 最后一行单独处理，φ(xm,yj)
        fi = fi.T.flatten()  # 展平成向量，默认按行
        d_diag = np.diag(-c_identity / self.h_y ** 2)  # 单个块D对角矩阵
        fi[:self.n_x - 1] = fi[: self.n_x - 1] - np.dot(d_diag, self.u_xy[1:-1, 0])  # f_1-D*u_0
        fi[1 - self.n_x:] = fi[1 - self.n_x:] - np.dot(d_diag, self.u_xy[1:-1, -1])  # f_(n-1) - D*u_n
        # 采用预处理共轭梯度法求解大型稀疏矩阵
        pcg = PreConjugateGradient(difference_matrix.toarray(), fi, np.zeros(len(fi)),
                                   eps=1e-15, omega=1.5, is_out_info=False)
        sol = pcg.fit_solve()
        self.u_xy[1:-1, 1:-1] = sol.reshape(self.n_y - 1, self.n_x - 1).T
        return self.u_xy

    def plt_pde_poisson_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi = np.linspace(self.x_span[0], self.x_span[1], self.n_x + 1)
        yi = np.linspace(self.y_span[0], self.y_span[1], self.n_y + 1)
        y, x = np.meshgrid(yi, xi)
        if self.pde_model:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, self.u_xy, cmap='rainbow')
        ax.set_xlabel("x", fontdict={"fontsize": 14})
        ax.set_ylabel("y", fontdict={"fontsize": 14})
        ax.set_zlabel("U", fontdict={"fontsize": 14})
        plt.title("poisson equation numerical solution", fontdict={"fontsize": 16})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, y)
            error_ = analytical_sol - self.u_xy  # 误差
            print("均方误差：", np.sqrt(np.mean(error_ ** 2)))
            ax.plot_surface(x, y, error_, cmap='rainbow')
            z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            ax.zaxis.set_major_formatter(z_format)
            ax.set_xlabel("x", fontdict={"fontsize": 14})
            ax.set_ylabel("y", fontdict={"fontsize": 14})
            ax.set_zlabel("Error", fontdict={"fontsize": 14})
            plt.title("The error of poisson numerical solution", fontdict={"fontsize": 16})
            # fig.tight_layout()
        plt.show()

    def plt_pde_poisson_curve_contourf(self):  # 参考一维波动方程
        """
        可视化某些时刻的数值解，以及等值线图
        :return:
        """
        # 1、不同时刻的波的传播随空间坐标的变化
        xi = np.linspace(self.x_span[0], self.x_span[1], self.n_x + 1)
        yi = np.linspace(self.y_span[0], self.y_span[1], self.n_y + 1)
        idx = np.array([1, len(yi) / 4, len(yi) / 2, 3 * len(yi) / 4, len(yi)], np.int) - 1
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        for i in idx:
            plt.plot(xi, self.u_xy[:, i], label='y=%.5f' % yi[i], lw=1.5)
        plt.ylabel('U(x,y)', fontdict={"fontsize": 12})
        plt.xlabel('x', fontdict={"fontsize": 12})
        plt.legend(loc='upper right')
        plt.title("Numerical solution of poisson at some y values", fontdict={"fontsize": 14})
        # 2、带有填充区域的等值线图
        plt.subplot(122)
        extent = [self.y_span[0], self.y_span[1] + self.h_y, self.x_span[0], self.x_span[1] + self.h_x]
        plt.contourf(self.u_xy.T, levels=20, origin='lower', extent=extent, cmap=plt.get_cmap("jet"))
        # plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
        plt.colorbar()  # 颜色bar
        plt.ylabel('y', fontdict={"fontsize": 12})
        plt.xlabel('x', fontdict={"fontsize": 12})
        plt.title("The contour face of poisson equation", fontdict={"fontsize": 14})
        plt.show()

    # def _cal_error_(self, xm, ym, u_xy):
    #     """
    #     误差图像
    #     :return:
    #     """
    #     error = u_xy - self.pde_model(xm, ym)  # 与精确解的误差
    #     plt.figure(figsize=(8, 6))
    #     ax = plt.subplot(111, projection='3d')
    #     ax.plot_surface(xm, ym, error, cmap="rainbow")
    #     ax.set_xlabel("X", fontdict={"fontsize": 12})
    #     ax.set_ylabel("Y", fontdict={"fontsize": 12})
    #     ax.set_zlabel("Error", fontdict={"fontsize": 12})
    #     plt.title("Surface of Error of two-dimensional Poisson equation", fontdict={"fontsize": 14})
    #     plt.show()
    #
    # @staticmethod
    # def plt_2d_poisson(xm, ym, u_xy):
    #     """
    #     可视化泊松方程的解
    #     :return:
    #     """
    #     plt.figure(figsize=(8, 6))
    #     ax = plt.subplot(111, projection='3d')
    #     ax.plot_surface(xm, ym, u_xy, cmap="rainbow")
    #     ax.set_xlabel("X", fontdict={"fontsize": 12})
    #     ax.set_ylabel("Y", fontdict={"fontsize": 12})
    #     ax.set_zlabel("Solution", fontdict={"fontsize": 12})
    #     plt.title("Surface of solution of two-dimensional Poisson equation", fontdict={"fontsize": 14})
    #     plt.show()
