# -*- coding: UTF-8 -*-
"""
@file_name: pde_heat_conduction_equ_nonhomogeneous.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
# 追赶法求解三对角矩阵
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix import ChasingMethodTridiagonalMatrix
from util_font import *


class PDEHeatConductionEquationNonhomogeneous:
    """
    求解抛物线偏微分方程：非齐次热传导方程的定解问题
    """

    def __init__(self, f_fun, a, init_f, alpha_f, beta_f, x_a, t_T, x_h, t_h,
                 pde_model=None, diff_type="forward"):
        self.f_fun, self.init_f = f_fun, init_f  # 方程右端函数以及初始条件函数
        self.alpha_f, self.beta_f = alpha_f, beta_f  # 边界函数
        self.x_a, self.t_T = x_a, t_T  # 分别表示自变量x和t的求解区域右端点，左端点默认为0
        self.a = a  # 一维热传导方程的常数项
        self.x_h, self.t_h = x_h, t_h  # 分别表示自变量x和t的求解步长
        self.x_n, self.t_m = int(self.x_a / self.x_h) + 1, int(self.t_T / self.t_h) + 1  # 划分网格区间点数
        self.u_xt = None  # 存储pde数值解
        self.pde_model = pde_model  # 解析解存在的情况下，可进行误差分析
        self.diff_type = diff_type  # 差分格式

    def solve_pde(self):
        """
        求解非齐次热传导方程的定解问题，差分格式
        :return:
        """
        r = self.a * self.t_h / self.x_h ** 2  # 步长比
        xi, ti = np.linspace(0, self.x_a, self.x_n), np.linspace(0, self.t_T, self.t_m)
        self.u_xt = np.zeros((self.x_n, self.t_m))  # 波动方程的数值解
        self.u_xt[:, 0] = self.init_f(xi)
        self.u_xt[0, :] = self.alpha_f(ti)
        self.u_xt[-1, :] = self.beta_f(ti)
        if self.diff_type.lower() == "forward":  # 向前Euler差分格式
            self._solve_pde_forward_(r, xi, ti)
        elif self.diff_type.lower() == "backward":  # 向后Euler差分格式
            self._solve_pde_backward_(r, xi, ti)
        elif self.diff_type.lower() == "crank-nicolson":  # Crank-Nicolson差分格式
            self._solve_pde_crank_nicolson_(r, xi, ti)
        elif self.diff_type.lower() == "compact":  # 紧差分格式
            self._solve_pde_compact_(r, xi, ti)
        else:
            raise ValueError("仅支持forward、backward、crank_nicolson和compact.")

    def _solve_pde_forward_(self, r, xi, ti):
        """
        向前Euler差分格式
        :return:
        """
        if r > 0.5:
            raise ValueError("r = %.5f，非稳定格式，重新划分步长." % r)
        print("r = %.5f，稳定格式求解一维热传导方程的数值解." % r)
        for j in range(1, self.t_m):
            self.u_xt[1:-1, j] = (1 - 2 * r) * self.u_xt[1:-1, j - 1] + \
                                 r * (self.u_xt[:-2, j - 1] +
                                      self.u_xt[2:, j - 1]) + \
                                 self.t_h * self.f_fun(xi[1:-1], ti[j - 1])
        return self.u_xt

    def _solve_pde_backward_(self, r, xi, ti):
        """
        向后Euler差分格式
        :return:
        """

        diag_b = (1 + 2 * r) * np.ones(self.x_n - 2)  # 主对角线
        diag_c = -r * np.ones(self.x_n - 3)  # 次对角线
        for j in range(1, self.t_m):
            fi = self.u_xt[1:-1, j - 1] + self.t_h * self.f_fun(xi[1:-1], ti[j])  # 右端向量
            fi[0], fi[-1] = fi[0] + r * self.u_xt[0, j], fi[-1] + r * self.u_xt[-1, j]
            cmtm = ChasingMethodTridiagonalMatrix(diag_c, diag_b, diag_c, fi)
            self.u_xt[1:-1, j] = cmtm.fit_solve()
        return self.u_xt

    def _solve_pde_crank_nicolson_(self, r, xi, ti):
        """
        Crank-Nicolson差分格式
        :return:
        """
        diag_b = (1 + r) * np.ones(self.x_n - 2)  # 主对角线
        diag_c = -r / 2 * np.ones(self.x_n - 3)  # 次对角线
        # 构造三对角矩阵B
        B = np.diag((1 - r) * np.ones(self.x_n - 2)) + \
            np.diag(r / 2 * np.ones(self.x_n - 3), 1) + \
            np.diag(r / 2 * np.ones(self.x_n - 3), -1)
        for j in range(1, self.t_m):
            fi = self.t_h * self.f_fun(xi[1:-1], 0.5 * (ti[j - 1] + ti[j]))  # 右端向量
            fi[0] = fi[0] + r / 2 * (self.u_xt[0, j - 1] + self.u_xt[0, j])
            fi[-1] = fi[-1] + r / 2 * (self.u_xt[-1, j - 1] + self.u_xt[-1, j])
            fi = fi + np.dot(B, self.u_xt[1:-1, j - 1])
            cmtm = ChasingMethodTridiagonalMatrix(diag_c, diag_b, diag_c, fi)
            self.u_xt[1:-1, j] = cmtm.fit_solve()
        return self.u_xt

    def _solve_pde_compact_(self, r, xi, ti):
        """
        紧差分格式
        :return:
        """
        diag_b = (5 / 6 + r) * np.ones(self.x_n - 2)  # 主对角线
        diag_c = (1 / 12 - r / 2) * np.ones(self.x_n - 3)  # 次对角线
        # 构造三对角矩阵B
        B = np.diag((5 / 6 - r) * np.ones(self.x_n - 2)) + \
            np.diag((1 / 12 + r / 2) * np.ones(self.x_n - 3), 1) + \
            np.diag((1 / 12 + r / 2) * np.ones(self.x_n - 3), -1)
        for j in range(1, self.t_m):
            t_ = 0.5 * (ti[j - 1] + ti[j])
            fi = self.t_h * (self.f_fun(xi[:-2], t_) + 10 * self.f_fun(xi[1:-1], t_) +
                             self.f_fun(xi[2:], t_)) / 12
            fi[0] = fi[0] + (1 / 12 + r / 2) * self.u_xt[0, j - 1] - \
                    (1 / 12 - r / 2) * self.u_xt[0, j]
            fi[-1] = fi[-1] + (1 / 12 + r / 2) * self.u_xt[-1, j - 1] - \
                     (1 / 12 - r / 2) * self.u_xt[-1, j]
            fi = fi + np.dot(B, self.u_xt[1:-1, j - 1])
            cmtm = ChasingMethodTridiagonalMatrix(diag_c, diag_b, diag_c, fi)
            self.u_xt[1:-1, j] = cmtm.fit_solve()
        return self.u_xt

    def plt_pde_heat_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi, ti = np.linspace(0, self.x_a, self.x_n), np.linspace(0, self.t_T, self.t_m)
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
        plt.title("一维热传导方程数值解曲面$(%s)$" % self.diff_type, fontdict={"fontsize": 18})
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
        xi = np.linspace(0, self.x_a, self.x_n)
        ti = np.linspace(0, self.t_T, self.t_m)
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
