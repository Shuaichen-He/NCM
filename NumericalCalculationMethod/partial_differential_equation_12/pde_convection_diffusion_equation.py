# -*- coding: UTF-8 -*-
"""
@file_name: pde_convection_diffusion_equation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
# 追赶法求解隐格式，三对角矩阵线性方程组
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix import ChasingMethodTridiagonalMatrix
from util_font import *


class PDEConvectionDiffusionEquation:
    """
    求解对流扩散方程
    """

    def __init__(self, f_ut0, alpha_fun, beta_fun, a_const, b_const, x_span, t_T,
                 x_h, t_h, pde_model=None, diff_type="center"):
        self.f_ut0 = f_ut0  # 初始函数
        self.alpha_fun, self.beta_fun = alpha_fun, beta_fun
        self.a, self.b = a_const, b_const  # 对流扩散方程的系数
        self.x_a, self.x_b = x_span[0], x_span[1]  # 空间求解区间
        self.t_T = t_T  # 时间求解区间，默认左端点为0
        self.x_h, self.t_h = x_h, t_h  # 分别表示自变量x和t的求解步长
        self.x_n = int((self.x_b - self.x_a) / self.x_h) + 1  # 空间网格区间点数
        self.t_m = int(self.t_T / self.t_h) + 1  # 时间划分数
        self.pde_model = pde_model  # 若存在解析解，则分析误差
        self.diff_type = diff_type  # 差分格式
        self.u_xt = None  # 存储pde数值解

    def solve_pde(self):
        """
        求解对流扩散方程
        :return:
        """
        xi = np.linspace(self.x_a, self.x_b, self.x_n)
        ti =  np.linspace(0, self.t_T, self.t_m)
        self.u_xt = np.zeros((self.x_n, self.t_m))  # 波动方程的数值解
        self.u_xt[:, 0] = self.f_ut0(xi)  # 初值问题
        ti = np.linspace(0, self.t_T, self.t_m)
        self.u_xt[[0, -1], :] = self.alpha_fun(ti), self.beta_fun(ti)  # 边界条件
        if self.diff_type.lower() == "center":  # 中心差分格式
            self._solve_pde_central_()
        elif self.diff_type.lower() == "exp":  # 指数差分格式
            self._solve_pde_exponential_()
        elif self.diff_type.lower() == "samarskii":  # 萨马尔斯基（Samarskii）格式
            self._solve_pde_samarskii_()
        elif self.diff_type.lower() == "crank-nicolson":  # crank_nicolson隐格式
            self._solve_pde_crank_nicolson_()
        else:
            raise ValueError("差分格式有误，仅支持center、exp、samarskii和crank-nicolson.")
        return self.u_xt

    def _solve_pde_central_(self):
        """
        中心差分格式求解
        :return:
        """
        if self.t_h > 2 * self.b / self.a ** 2 or \
                self.t_h > self.x_h ** 2 / (2 * self.b):
            raise ValueError("非稳定格式，重新划分步长.")
        else:
            c1 = self.a * self.t_h / (2 * self.x_h)
            c2 = (2 * self.b * self.t_h + self.t_h ** 2 * self.a ** 2) / (2 * self.x_h ** 2)
            # c2 = self.b * self.t_h / self.x_h ** 2
            for k in range(1, self.t_m):
                u1 = self.u_xt[2:, k - 1] - self.u_xt[:-2, k - 1]  # 子项
                self.u_xt[1:-1, k] = self.u_xt[1:-1, k - 1] - c1 * u1 + c2 * \
                                     (self.u_xt[2:, k - 1] - 2 * self.u_xt[1:-1, k - 1] +
                                      self.u_xt[:-2, k - 1])
        return self.u_xt

    def _solve_pde_exponential_(self):
        """
        指数型差分格式求解
        :return:
        """
        c1 = self.a * self.t_h / (2 * self.x_h)
        c2 = c1 * np.cosh(self.a * self.x_h / 2 / self.b) / \
             np.sinh(self.a * self.x_h / 2 / self.b)
        for j in range(1, self.t_m):
            u1 = self.u_xt[2:, j - 1] - self.u_xt[:-2, j - 1]  # 子项
            self.u_xt[1:-1, j] = self.u_xt[1:-1, j - 1] - c1 * u1 + c2 * \
                                 (self.u_xt[2:, j - 1] - 2 * self.u_xt[1:-1, j - 1] +
                                  self.u_xt[:-2, j - 1])
        return self.u_xt

    def _solve_pde_samarskii_(self):
        """
        萨马尔斯基（Samarskii）格式
        :return:
        """
        c1 = self.a * self.t_h / self.x_h
        c2 = self.b * self.t_h / (1 + self.a * self.x_h / (2 * self.b)) / self.x_h ** 2
        for j in range(1, self.t_m):
            u1 = self.u_xt[1:-1, j - 1] - self.u_xt[:-2, j - 1]  # 子项
            self.u_xt[1:-1, j] = self.u_xt[1:-1, j - 1] - c1 * u1 + c2 * \
                                 (self.u_xt[2:, j - 1] - 2 * self.u_xt[1:-1, j - 1] +
                                  self.u_xt[:-2, j - 1])
        return self.u_xt

    def _solve_pde_crank_nicolson_(self):
        """
        Crank-nicolson隐格式求解
        :return:
        """
        miu, r = self.b * self.t_h / self.x_h ** 2, self.a * self.t_h / self.x_h
        a_diag = (1 + miu) * np.ones(self.x_n - 2)  # 主对角线
        b_diag = -(r / 4 + miu / 2) * np.ones(self.x_n - 3)  # 主对角线以下
        c_diag = (r / 4 - miu / 2) * np.ones(self.x_n - 3)  # 主对角线以上
        # 等号右端三对角矩阵
        b_mat = np.diag((1 - miu) * np.ones(self.x_n - 2)) + \
                np.diag((r / 4 + miu / 2) * np.ones(self.x_n - 3), -1) + \
                np.diag((miu / 2 - r / 4) * np.ones(self.x_n - 3), 1)
        F = np.zeros(self.x_n - 2)
        for j in range(1, self.t_m):
            F[0] = (r + 2 * miu) * (self.u_xt[0, j] + self.u_xt[0, j - 1]) / 4
            F[-1] = (-r + 2 * miu) * (self.u_xt[-1, j] + self.u_xt[-1, j - 1]) / 4
            d_vector = np.dot(b_mat, self.u_xt[1:-1, j - 1]) + F
            # 追赶法求解
            cmtm = ChasingMethodTridiagonalMatrix(b_diag, a_diag, c_diag, d_vector)
            self.u_xt[1:-1, j] = cmtm.fit_solve()

        return self.u_xt

    def plt_pde_heat_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi, ti = np.linspace(self.x_a, self.x_b, self.x_n), np.linspace(0, self.t_T, self.t_m)
        x, t = np.meshgrid(xi, ti)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, t, self.u_xt.T, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$t$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("对流扩散方程数值解曲面$(%s)$" % self.diff_type, fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, t)
            error_ = analytical_sol - self.u_xt.T  # 误差
            ax.plot_surface(x, t, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 平均绝对值误差
            print("平均绝对值误差：%.10e" % mae)
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$t$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            # z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            # ax.zaxis.set_major_formatter(z_format)
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("$\epsilon=U(x,t) - \hat U(x,t),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
        plt.show()

    def plt_pde_heat_curve_contourf(self):  # 参考一维波动方程
        """
        可视化某些时刻的数值解，以及等值线图
        :return:
        """
        # 1、不同时刻的波的传播随空间坐标的变化
        xi, ti = np.linspace(self.x_a, self.x_b, self.x_n), np.linspace(0, self.t_T, self.t_m)
        idx = np.array([1, len(ti) / 4, len(ti) / 2, 3 * len(ti) / 4, len(ti)], np.int) - 1
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        ls_ = ["-", "--", "-.", ":", "-"]
        for k, i in enumerate(idx):
            plt.plot(xi, self.u_xt[:, i], ls_[k], label='$t=%.4f$' % ti[i], lw=1.5)
        plt.ylabel('$U(x,t)$', fontdict={"fontsize": 18})
        plt.xlabel('$x$', fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc='upper right')
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.title("一维热传导方程在某些时刻的数值解曲线", fontdict={"fontsize": 18})
        # 2、带有填充区域的等值线图
        plt.subplot(122)
        extent = [0, self.t_T + self.t_h, 0, self.x_a + self.x_h]  # 时间和空间的取值范围
        plt.contourf(self.u_xt, levels=10, origin='lower', extent=extent, cmap=plt.get_cmap("jet"))
        # plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
        plt.colorbar()  # 颜色bar
        plt.ylabel('$x$', fontdict={"fontsize": 18})
        plt.xlabel('$t$', fontdict={"fontsize": 18})
        plt.title("一维热传导方程在时刻$t = %.1f$的等值线图" % self.t_T, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.show()
