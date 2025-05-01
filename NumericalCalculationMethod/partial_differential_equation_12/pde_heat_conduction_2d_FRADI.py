# -*- coding: UTF-8 -*-
"""
@file_name: pde_heat_conduction_equ_2d_FRADI.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix import ChasingMethodTridiagonalMatrix
from util_font import *


class PDEHeatConductionEquation_2D_FRADI:
    """
    二维热传导方程，交替方向隐格式
    """

    def __init__(self, a_const, f_xyt, f_ut0, f_u0yt, f_u1yt, f_u0xt, f_u1xt,
                 x_span, y_span, t_T, tau, h, pde_model=None):
        self.a_const, self.f_xyt = a_const, f_xyt  # 二维热传导方程系数，以及方程右端向量函数
        self.f_ut0 = f_ut0  # 初始化函数
        # 边界条件，分别对应u(0,y,t)、u(1,y,t)、u(x,0,t)和u(x,1,t)
        self.f_u0yt, self.f_u1yt = f_u0yt, f_u1yt  # 边界条件，对应u(0, y, t)和u(1, y, t)
        self.f_u0xt, self.f_u1xt = f_u0xt, f_u1xt  # 边界条件，对应u(x, 0, t)和u(x, 1, t)
        self.x_a, self.x_b = x_span[0], x_span[1]  # x的求解区间左右端点
        self.y_a, self.y_b = y_span[0], y_span[1]  # y的求解区间左右端点
        self.tau, self.h, self.t_T = tau, h, t_T  # 时间与空间步长
        self.x_n = int((self.x_b - self.x_a) / h) + 1  # 空间网格
        self.y_n = int((self.y_b - self.y_a) / h) + 1  # # 空间网格
        self.t_m = int(self.t_T / tau) + 1  # 时间区间数
        self.pde_model = pde_model  # 存在解析解，则分析误差
        self.u_xyt = None  # 存储二维热传导方程的数值解

    def solve_pde(self):
        """
        求解二维热传导方程
        :return:
        """
        xi = np.linspace(self.x_a, self.x_b, self.x_n)  # 等分网格点
        yi = np.linspace(self.y_a, self.y_b, self.y_n)  # 等分网格点
        ti = np.linspace(0, self.t_T, self.t_m)  # 时间网格点
        self.u_xyt = np.zeros((self.t_m, self.x_n, self.y_n))  # 数值解
        t_0, x_0, y_0 = np.meshgrid(ti, xi, yi)  # 三维网格
        # 保持维度与解维度一致
        t_0, x_0, y_0 = t_0.swapaxes(1, 0), x_0.swapaxes(1, 0), y_0.swapaxes(1, 0)
        self.u_xyt[0, :, :] = self.f_ut0(x_0[0, :, :], y_0[0, :, :])  # 初始化，对应u(x,y,0)
        self.u_xyt[:, 0, :] = self.f_u0yt(y_0[:, 0, :], t_0[:, 0, :])  # 边界值
        self.u_xyt[:, -1, :] = self.f_u1yt(y_0[:, -1, :], t_0[:, -1, :])  # 边界值
        self.u_xyt[:, :, 0] = self.f_u0xt(x_0[:, :, 0], t_0[:, :, 0])  # 边界值
        self.u_xyt[:, :, -1] = self.f_u1xt(x_0[:, :, -1], t_0[:, :, -1])  # 边界值
        r = self.tau * self.a_const / self.h ** 2  # 网格比
        # 三对角方程组的三条对角线元素
        a_diag_x = (1 + r) * np.ones(self.x_n - 2)
        b_diag_x = -r / 2 * np.ones(self.x_n - 3)
        a_diag_y = (1 + r) * np.ones(self.y_n - 2)
        b_diag_y = -r / 2 * np.ones(self.y_n - 3)
        uc = np.zeros((self.t_m, self.x_n, self.y_n))  # 中间层
        for k in range(1, self.t_m):  # 对时间层递推
            for j in range(1, self.y_n - 1):  # y方向计算
                a1 = r * (self.u_xyt[k, 0, j - 1] - 2 * self.u_xyt[k, 0, j] +
                          self.u_xyt[k, 0, j + 1])
                b1 = r * (self.u_xyt[k - 1, 0, j - 1] - 2 * self.u_xyt[k - 1, 0, j] +
                          self.u_xyt[k - 1, 0, j + 1])
                am = r * (self.u_xyt[k, -1, j - 1] - 2 * self.u_xyt[k, -1, j] +
                          self.u_xyt[k, -1, j + 1])
                bm = r * (self.u_xyt[k - 1, -1, j - 1] - 2 * self.u_xyt[k - 1, -1, j] +
                          self.u_xyt[k - 1, -1, j + 1])
                uc[k, 0, j] = 0.5 * (self.u_xyt[k - 1, 0, j] +
                                     self.u_xyt[k, 0, j]) - 0.25 * (a1 - b1)
                uc[k, -1, j] = 0.5 * (self.u_xyt[k - 1, -1, j] +
                                      self.u_xyt[k, -1, j]) - 0.25 * (am - bm)
                # 方程组右端向量
                f_val = self.f_xyt(x_0[k - 1, 1:-1, j], y_0[k - 1, 1:-1, j],
                                   t_0[k - 1, 1:-1, j] + self.tau / 2)
                f1 = r / 2 * self.u_xyt[k - 1, 1:-1, j - 1] + \
                     (1 - r) * self.u_xyt[k - 1, 1:-1, j] + \
                     r / 2 * self.u_xyt[k - 1, 1:-1, j + 1] + self.tau / 2 * f_val
                f1[0] = f1[0] + r / 2 * uc[k, 0, j]
                f1[-1] = f1[-1] + r / 2 * uc[k, -1, j]
                cmtm = ChasingMethodTridiagonalMatrix(b_diag_y, a_diag_y, b_diag_y, f1)  # 追赶法
                uc[k, 1:-1, j] = cmtm.fit_solve()
            for i in range(1, self.x_n - 1):  # x方向计算
                f_val = self.f_xyt(x_0[k - 1, i, 1:- 1], y_0[k - 1, i, 1:-1],
                                   t_0[k - 1, i, 1:-1] + self.tau / 2)
                # 右端向量
                f2 = r / 2 * uc[k, i - 1, 1:-1] + (1 - r) * uc[k, i, 1:-1] + \
                     r / 2 * uc[k, i + 1, 1:-1] + self.tau / 2 * f_val
                f2[0] = f2[0] + r / 2 * self.u_xyt[k, i, 0]
                f2[-1] = f2[-1] + r / 2 * self.u_xyt[k, i, -1]
                cmtm = ChasingMethodTridiagonalMatrix(b_diag_x, a_diag_x, b_diag_x, f2)  # 追赶法
                self.u_xyt[k, i, 1:-1] = cmtm.fit_solve()

    def plt_pde_heat_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi = np.linspace(self.x_a, self.x_b, self.x_n)
        yi = np.linspace(self.y_a, self.y_b, self.y_n)
        ti = np.linspace(0, self.t_T, self.t_m)
        x, y = np.meshgrid(xi, yi)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, self.u_xyt[-1, :, :], cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("二维热传导方程$(FR-ADI)$数值解曲面", fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            x_, y_, t_ = np.meshgrid(xi, yi, ti)
            analytical_sol = self.pde_model(x_, y_, t_)
            error_ = analytical_sol[:, :, -1] - self.u_xyt[-1, :, :]  # 误差
            ax.plot_surface(x, y, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 平均绝对值误差
            print("平均绝对值误差：%.10e" % mae)
            print("最大绝对值误差：%.10e" % np.max(np.abs(error_)))
            ax.set_xlabel("$x$", fontdict={"fontsize": 18})
            ax.set_ylabel("$y$", fontdict={"fontsize": 18})
            ax.set_zlabel("$\epsilon$", fontdict={"fontsize": 18})
            # z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
            # ax.zaxis.set_major_formatter(z_format)
            plt.tick_params(labelsize=16)  # 刻度字体大小16
            plt.title("$\epsilon=U(x,y,t) - \hat U(x,y,t),\ MAE=%.3e$" % mae, fontdict={"fontsize": 18})
            fig.tight_layout()
        plt.show()
