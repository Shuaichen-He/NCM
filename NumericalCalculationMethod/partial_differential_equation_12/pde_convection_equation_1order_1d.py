# -*- coding: UTF-8 -*-
"""
@file_name: pde_convection_equation_1order_1d.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PDEConvectionEquationFirstOrder1D:
    """
    双曲型偏微分方程，一阶一维常系数对流方程，五种差分格式实现
    """

    def __init__(self, a_const, f_u0, x_span, t_T, x_j, t_n, diff_type="lax-wendroff"):
        self.a = np.asarray(a_const, np.float64)  # 一维对流方程的常系数
        self.f_u0 = f_u0  # 边界条件函数，u_0(x,0)
        self.x_a, self.x_b, self.t_T = x_span[0], x_span[1], t_T  # 求解区间，时间默认左端点为0
        self.x_j, self.t_n = x_j, t_n  # 求解空间和时间步长
        # ["upwind", "leapfrog", "lax_friedrichs", "lax_wendroff", "beam_warming"]
        self.diff_type = diff_type  # 求解的差分格式
        self.u_xt = None  # 存储pde数值解

    def solve_pde(self):
        """
        求解一阶常系数对流方程
        :return:
        """
        xi = np.linspace(self.x_a, self.x_b, self.x_j + 1)  # 空间划分
        ti = np.linspace(0, self.t_T, self.t_n + 1)  # 时间划分
        x_h, t_h = xi[1] - xi[0], ti[1] - ti[0]  # 步长
        r = abs(self.a) * t_h / x_h  # 差分格式稳定性条件，有限差分格式中的常数系数
        self.u_xt = np.zeros((len(ti), len(xi)))  # 一维对流方程的数值解，行为时间格式，列为空间递推
        self.u_xt[0, :] = self.f_u0(xi)  # 初始化，即初始时刻的值，第一行t = 0
        if r >= 1:
            raise ValueError("r = %.5f，非稳定格式，重新划分步长." % r)
        elif self.diff_type.lower() == "beam-warming" and r > 2:
            raise ValueError("r = %.5f，beam-warming非稳定格式，重新划分步长." % r)
        elif self.diff_type.lower() in ["upwind", "leapfrog", "lax-friedrichs",
                                        "lax-wendroff", "beam-warming"]:
            print("r = %.5f，稳定格式(%s)求解一阶常系数对流方程的数值解." %
                  (r, self.diff_type))
            if self.diff_type.lower() == "upwind":  # 迎风格式
                for n in range(self.t_n):  # 对每个时间格式，递推空间格式
                    self.u_xt[n + 1, 0] = self.f_u0(self.x_a)  # 每个空间的第一个为初始值
                    self.u_xt[n + 1, 1:] = (1 - r) * self.u_xt[n, 1:] + \
                                           r * self.u_xt[n, :-1]
            elif self.diff_type.lower() == "leapfrog":  # 蛙跳格式
                self.u_xt[1, :] = self.f_u0(xi)  # 由于需要两层信息才能递推第三层，故第二层初始化
                for n in range(1, self.t_n):
                    self.u_xt[n + 1, 1:-1] = self.u_xt[n - 1, 1:-1] - \
                                             r * (self.u_xt[n, 2:] - self.u_xt[n, :-2])
                    # 第1个值为求解区间的起点，最后一个值为求解区间的终点
                    self.u_xt[n + 1, [0, -1]] = [self.f_u0(self.x_a),
                                                 self.f_u0(self.x_b)]
            elif self.diff_type.lower() == "lax-wendroff":
                for n in range(self.t_n):
                    self.u_xt[n + 1, [0, -1]] = [self.f_u0(self.x_a),
                                                 self.f_u0(self.x_b)]
                    self.u_xt[n + 1, 1:-1] = 0.5 * r * (r - 1) * self.u_xt[n, 2:] + \
                                             (1 - r ** 2) * self.u_xt[n, 1:-1] + \
                                             0.5 * r * (r + 1) * self.u_xt[n, :-2]
            elif self.diff_type.lower() == "lax-friedrichs":
                for n in range(self.t_n):
                    self.u_xt[n + 1, [0, -1]] = [self.f_u0(self.x_a),
                                                 self.f_u0(self.x_b)]
                    self.u_xt[n + 1, 1:-1] = 0.5 * (1 - r) * self.u_xt[n, 2:] + \
                                             0.5 * (1 + r) * self.u_xt[n, :-2]
            elif self.diff_type.lower() == "beam-warming":
                for n in range(self.t_n):
                    # 递推每一层，需要两个时间起点值，故第二个为起始点+时间步长
                    self.u_xt[n + 1, [0, 1]] = [self.f_u0(self.x_a),
                                                self.f_u0(self.x_a + t_h)]
                    c1, c2 = 1 - 1.5 * r + 0.5 * r ** 2, 2 * r - r ** 2
                    self.u_xt[n + 1, 2:] = c1 * self.u_xt[n, 2:] + \
                                           c2 * self.u_xt[n, 1:-1] + \
                                           0.5 * (r ** 2 - r) * self.u_xt[n, :-2]
        else:
            self.diff_type = "unstable"  # 不稳定格式
            for n in range(self.t_n):
                self.u_xt[n + 1, [0, -1]] = [self.f_u0(self.x_a), self.f_u0(self.x_b)]
                self.u_xt[n + 1, 1:-1] = 0.5 * self.u_xt[n, :-2] + \
                                         self.u_xt[n, 1:-1] - 0.5 * self.u_xt[n, 2:]
            print("完全不稳定格式，可重新选择差分格式。")
        return self.u_xt

    def plt_convection_curve(self, is_show=True):
        """
        可视化一维对流方程曲线
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        xi = np.linspace(self.x_a, self.x_b, self.x_j + 1)
        ti = np.linspace(0, self.t_T, self.t_n + 1)  # 时刻
        idx = np.array([1, len(ti) / 4, len(ti) / 2, 3 * len(ti) / 4, len(ti)], np.int64) - 1
        ls_ = ["-", "-.", ":", "-.", ":"]
        for k, i in enumerate(idx):
            plt.plot(xi, self.u_xt[i, :], ls=ls_[k], label="$t=%.3f$" % ti[i], lw=2)
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$u(x,t)$", fontdict={"fontsize": 18})
        plt.title("一阶常系数对流方程的数值解($%s$)" % self.diff_type, fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc="upper right")
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
        return ti[idx]

    def plt_convection_surf(self):
        """
        可视化数值解曲面
        :return:
        """
        xi = np.linspace(self.x_a, self.x_b, self.x_j + 1)
        ti = np.linspace(0, self.t_T, self.t_n + 1)
        x, t = np.meshgrid(xi, ti)
        plt.figure(figsize=(8, 6))
        ax = plt.gca(projection='3d')
        ax.plot_surface(x, t, self.u_xt, cmap='rainbow')
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$t$", fontdict={"fontsize": 18})
        ax.set_zlabel("$U$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.title("一阶常系数对流方程的数值解曲面($%s$)" % self.diff_type, fontdict={"fontsize": 18})
        plt.show()
