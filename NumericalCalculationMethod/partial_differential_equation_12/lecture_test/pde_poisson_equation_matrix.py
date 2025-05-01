# -*- coding: UTF-8 -*-
"""
@file_name: pde_poisson_trib_matrix.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import scipy.sparse as sp  # 用于构造稀疏矩阵
from iterative_solution_linear_equation_07.jacobi_gauss_seidel_iterative import JacobiGSlIterativeMethod
from partial_differential_equation_12.lecture_test.block_iterative_method import BlockIterative
from util_font import *


class PDEPoissonEquationTriBMatrix:
    """
    二维泊松方程模型，五点差分格式求解，采用共轭梯度法求解
    """

    def __init__(self, fxy_fun, f_ux0, f_uxb, f_u0y, f_uay,
                 x_span, y_span, n_x, n_y, is_show=False, pde_model=None):
        self.fxy_fun = fxy_fun  # 泊松方程的右端函数f(x, y)，关于自变量x和y的函数
        self.f_ux0, self.f_uxb = f_ux0, f_uxb  # 边界函数，对应u(x, 0)和u(x, b)，关于x的函数
        self.f_u0y, self.f_uay = f_u0y, f_uay  # 边界函数，对应u(0, y)和u(a, y)，关于y的函数
        self.x_span, self.y_span = np.asarray(x_span, np.float), np.asarray(y_span, np.float)
        self.n_x, self.n_y = n_x, n_y  # 划分区间数
        # 等分区间点和区间步长
        self.h_x, self.h_y, self.xi, self.yi = self._space_grid_()
        self.is_show = is_show  # 是否可视化泊松方程解的图像
        self.pde_model = pde_model  # 是否存在解析解，用于误差分析，可视化误差
        self.u_xy = None  # 存储pde数值解
        self.objs = []

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
        self.u_xy[0, :] = self.f_u0y(self.yi)  # 左边界
        self.u_xy[-1, :] = self.f_uay(self.yi)  # 右边界
        self.u_xy[:, 0] = self.f_ux0(self.xi)  # 底部
        self.u_xy[:, -1] = self.f_uxb(self.xi)  # 顶部
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
        # 构造右端向量，构成右端fi
        fi = self.fxy_fun(xm[1: -1, 1: -1], ym[1: -1, 1: -1])  # pde右端方程，内部节点(n-1)*(m-1)
        fi[0, :] = fi[0, :] + self.u_xy[0, 1:-1] / self.h_x ** 2  # 第一行单独处理，φ(x0,yj)
        fi[-1, :] = fi[-1, :] + self.u_xy[-1, 1:-1] / self.h_x ** 2  # 最后一行单独处理，φ(xm,yj)
        fi = fi.T.flatten()  # 展平成向量，默认按行
        d_diag = np.diag(-c_identity / self.h_y ** 2)  # 单个块D对角矩阵
        fi[:self.n_x - 1] = fi[: self.n_x - 1] - np.dot(d_diag, self.u_xy[1:-1, 0])  # f_1-D*u_0
        fi[1 - self.n_x:] = fi[1 - self.n_x:] - np.dot(d_diag, self.u_xy[1:-1, -1])  # f_(n-1) - D*u_n
        # 采用预处理共轭梯度法求解大型稀疏矩阵
        # print(difference_matrix.toarray() / self.n_x ** 2)  # 打印系数矩阵
        # print(fi / self.n_x ** 2)  # 打印右端向量
        methods = ["Jacobi", "G-S"]
        for method in methods:
            jgs = JacobiGSlIterativeMethod(difference_matrix.toarray() / self.n_x ** 2,  # 系数矩阵
                                           fi / self.n_x ** 2, np.zeros(len(fi)),  # 右端向量和迭代初值
                                           eps=1e-10, method=method, max_iter=10000, is_out_info=False)
            sol = jgs.fit_solve()
            # jgs.plt_convergence(is_show=False)
            self.objs.append(jgs)

            block = 15 * np.ones(15)
            print(difference_matrix.toarray().shape)
            bi = BlockIterative(A=difference_matrix.toarray() / self.n_x ** 2,  # 系数矩阵
                                b=fi / self.n_x ** 2, x0=np.zeros(len(fi)),  # 右端向量和迭代初值
                                block=block, eps=1e-10, max_iter=1000,
                                method=method, is_out_info=False)
            sol = bi.fit_solve()
            # bi.plt_convergence(is_show=False)
            self.objs.append(bi)

        self.u_xy[1:-1, 1:-1] = sol.reshape(self.n_y - 1, self.n_x - 1).T
        return self.u_xy

    def plt_convergence_precision(self, is_show=True, method="", style="-"):
        """
        可视化迭代解的精度曲线，is_show便于子图绘制，若为子图，则值为False，method为迭代方法，用于title
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        labels = ["J", "BJ", "GS", "BGS"]
        for i in range(4):
            obj = self.objs[i]
            iter_num = obj.iterative_info["Iteration_number"]  # 获取迭代次数
            iter_num = np.linspace(1, iter_num, iter_num)  # 等距取值，作为x轴绘图数据
            plt.semilogy(iter_num, obj.precision, "%s" % style, lw=2,
                         label="$%s: \ \epsilon=%.3e, \ k=%d$" % (labels[i], obj.precision[-1], iter_num[-1])) # 对数坐标
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
        plt.title("$Poisson$方程$\epsilon=\Vert b - Ax^* \Vert _2$的收敛曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()

    def plt_pde_poisson_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi = np.linspace(self.x_span[0], self.x_span[1], self.n_x + 1)
        yi = np.linspace(self.y_span[0], self.y_span[1], self.n_y + 1)
        y, x = np.meshgrid(yi, xi)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, self.u_xy, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("$Poisson$方程($Dirichlet$)数值解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, y)
            error_ = analytical_sol - self.u_xy  # 误差
            ax.plot_surface(x, y, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 平均绝对值误差
            print("平均绝对值误差：%.10e" % mae)
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$y$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            ax.zaxis.set_major_formatter(z_format)
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("$\epsilon=U(x,y) - \hat U(x,y),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()
