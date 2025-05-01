# -*- coding: UTF-8 -*-
"""
@file_name: pde_heat_conduction_equ_2d.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEHeatConductionEquation_2D:
    """
    二维热传导方程，显式差分格式
    """

    def __init__(self, a_const, f_xyt, f_u0yt, f_u1yt, f_u0xt, f_u1xt, f_ut0,
                 x_span, y_span, t_T, xy_h, t_h, pde_model=None, diff_type="Du-Fort-Frankel"):
        self.a_const = a_const  # 方程系数
        self.f_xyt = f_xyt  # 二维热传导方程右端函数f(x,y,t)
        self.f_u0yt, self.f_u1yt = f_u0yt, f_u1yt  # 对应u(0, y, t)和u(1, y, t)
        self.f_u0xt, self.f_u1xt = f_u0xt, f_u1xt  # 对应u(x, 0, t)和u(x, 1, t)
        self.f_ut0 = f_ut0  # 对应u(x,y,0)
        self.x_a, self.x_b = x_span[0], x_span[1]  # x的求解区间左右端点
        self.y_a, self.y_b = y_span[0], y_span[1]  # y的求解区间左右端点
        self.t_T = t_T  # 时间的右端点，默认左端点为0时刻
        self.xy_h, self.t_h = xy_h, t_h  # 空间和时间步长
        self.xy_n = int((self.x_b - self.x_a) / self.xy_h) + 1  # 空间网格区间点数
        self.t_m = int(self.t_T / self.t_h) + 1  # 时间划分数
        self.pde_model = pde_model  # 存在解析解，则分析误差
        self.diff_type = diff_type
        self.u_xyt = None  # 存储二维热传导方程的数值解

    def solve_pde(self):
        """
        求解二维热传导方程
        :return:
        """
        r = self.t_h * self.a_const / self.xy_h ** 2  # 网格比
        print("二维热传导方程的网格比：%.6f" % r)
        xi = np.linspace(self.x_a, self.x_b, self.xy_n)  # 空间离散
        yi = np.linspace(self.y_a, self.y_b, self.xy_n)  # 空间离散
        ti = np.linspace(0, self.t_T, self.t_m)
        self.u_xyt = np.zeros((self.xy_n, self.xy_n)) # 数值解
        x_, y_ = np.meshgrid(xi, yi)
        self.u_xyt = self.f_ut0(x_, y_)  # 初始化， 第一层结点计算
        self._cal_boundary_condition_(xi, yi, ti[0])  # 计算边界条件
        if self.diff_type.lower() == "du-fort-frankel":
            self._solve_pde_du_fort_frankel_(r, xi, yi, ti)
        elif self.diff_type.lower() == "classical":  # 古典显式格式
            self._solve_pde_classical_(r, xi, yi, ti)
        else:
            raise ValueError("仅支持Du_Fort_Frankel，Classical两种显式格式")
        return self.u_xyt

    def _cal_boundary_condition_(self, xi, yi, tk):
        """
        计算边界条件
        :return:
        """
        self.u_xyt[:, 0] = self.f_u0yt(yi, tk)
        self.u_xyt[:, -1] = self.f_u1yt(yi, tk)
        self.u_xyt[0, :] = self.f_u0xt(xi, tk)
        self.u_xyt[-1, :] = self.f_u1xt(xi, tk)

    def _solve_pde_classical_(self, r, xi, yi, ti):
        """
        二维richardson显式格式，稳定条件为网格比小于等于0.25
        :return:
        """
        if r > 0.25:
            raise ValueError("二维古典显式格式，非稳定。r = %.3f" % r)
        else:
            print("二维古典显式格式，稳定计算。r = %.3f" % r)
        x_, y_ = np.meshgrid(xi, yi)
        for k in range(1, self.t_m):
            u_xyt = np.copy(self.u_xyt)
            f_val = self.f_xyt(x_, y_, ti[k])
            term = u_xyt[2:, 1:-1] + u_xyt[:-2, 1:-1] + u_xyt[1:-1, 2:] + \
                   u_xyt[1:-1, :-2] - 4 * u_xyt[1:-1, 1:-1]
            self.u_xyt[1:-1, 1:-1] = u_xyt[1:-1, 1:-1] + r * term + \
                                     self.t_h * f_val[1:-1, 1:-1]
            self._cal_boundary_condition_(xi, yi, ti[k])
        return self.u_xyt

    def _solve_pde_du_fort_frankel_(self, r, xi, yi, ti):
        """
        无条件稳定格式，Du Fort-Frankel显式格式
        :return:
        """
        x_, y_ = np.meshgrid(xi, yi)
        # 第二层结点计算以及边界条件
        f_val = self.f_xyt(x_, y_, ti[1])
        u_xyt_1 = np.copy(self.u_xyt)  # 表示u_(k-1)，self.u_xyt表示u_(k+1), u_xyt表示u_(k)
        term = self.u_xyt[2:, 1:-1] + self.u_xyt[:-2, 1:-1] + \
               self.u_xyt[1:-1, 2:] + self.u_xyt[1:-1, :-2]
        self.u_xyt[1:-1, 1:-1] = term / 4 + self.t_h * f_val[1:-1, 1:-1]
        self._cal_boundary_condition_(xi, yi, ti[1])
        # 差分方程的系数
        c1, c2, c3 = 2 * r / (1 + 4 * r), (1 - 4 * r) / (1 + 4 * r), 2 * self.t_h / (1 + 4 * r)
        print("Du Fort-Frankel显式格式的系数：%.6f, %.6f。" % (c1, c2))
        for k in range(2, self.t_m):
            u_xyt_0 = np.copy(self.u_xyt)
            f_val = self.f_xyt(x_, y_, ti[k])
            term = u_xyt_0[2:, 1:-1] + u_xyt_0[:-2, 1:-1] + \
                   u_xyt_0[1:-1, 2:] + u_xyt_0[1:-1, :-2]
            self.u_xyt[1:-1, 1:-1] = c1 * term + c2 * u_xyt_1[1:-1, 1:-1] + \
                                     c3 * f_val[1:-1, 1:-1]
            self._cal_boundary_condition_(xi, yi, ti[k])  # 边界条件计算
            u_xyt_1 = np.copy(u_xyt_0)
        return self.u_xyt

    def plt_pde_heat_surface(self):
        """
        可视化数值解
        :return:
        """
        xi, yi = np.linspace(self.x_a, self.x_b, self.xy_n), np.linspace(self.y_a, self.y_b, self.xy_n)
        ti = np.linspace(0, self.t_T, self.t_m)
        x, y = np.meshgrid(xi, yi)
        if self.pde_model:
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(121, projection='3d')
        else:
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca(projection='3d')
        ax.plot_surface(x, y, self.u_xyt, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("二维热传导方程数值解曲面$(%s)$" % self.diff_type, fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            x, y, t = np.meshgrid(xi, yi, ti)
            analytical_sol = self.pde_model(x, y, t)
            error_ = analytical_sol[:, :, -1] - self.u_xyt  # 误差
            x, y = np.meshgrid(xi, yi)
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
        plt.show()

