# -*- coding: UTF-8 -*-
"""
@file_name: pde_heat_conduction_equation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import pandas as pd
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix import ChasingMethodTridiagonalMatrix
from util_font import *


class PDEHeatConductionEquation:
    """
    一维热传导方程求解
    """

    def __init__(self, f_fun, c1, c2, x_a, t_b, const, x_h, t_h, pde_model=None, pde_method="explicit"):
        self.f_fun, self.c1, self.c2 = f_fun, c1, c2  # 初始边界条件函数
        self.x_a, self.t_b = x_a, t_b  # 分别表示自变量x和t的求解区域右端点
        self.const = const  # 一维热传导方程的常数项
        self.x_h, self.t_h = x_h, t_h  # 分别表示自变量x和t的求解步长
        self.n, self.m = int(self.x_a / self.x_h) + 1, int(self.t_b / self.t_h) + 1  # 划分网格区间点数
        self.u_xt = None  # 存储pde数值解
        self.pde_model = pde_model  # 解析解存在的情况下，可进行误差分析
        self.pde_method = pde_method  # 显式Explicit方法或隐式Implicit方法

    def cal_pde(self):
        """
        显式前向差分格式和克兰克—尼科尔森隐式格式求解
        :return:
        """
        self.u_xt = np.zeros((self.n, self.m))  # 波动方程的数值解
        if self.pde_method.lower() == "explicit":
            return self._cal_pde_explicit_()
        elif self.pde_method.lower() == "implicit":
            return self._cal_pde_implicit_()

    def _cal_pde_explicit_(self):
        """
        显式前向差分格式求解
        :return:
        """
        r = self.const ** 2 * self.t_h / self.x_h ** 2  # 差分格式系数常量
        if r > 0.5:
            raise ValueError("r = %.5f，非稳定格式，重新划分步长." % r)
        print("r = %.5f，稳定格式求解一维热传导方程的数值解." % r)
        cf = 1 - 2 * r  # 差分格式的系数
        # 初始条件，第一列和最后一列数值，数值解按例存储
        self.u_xt[:, [0, -1]] = self.c1, self.c2
        # 计算第1行
        i = np.arange(1, self.n - 1)
        self.u_xt[1:-1, 0] = self.f_fun(i * self.x_h)
        # 计算剩余的列
        for j in range(1, self.m):
            self.u_xt[1:-1, j] = cf * self.u_xt[1:-1, j - 1] + \
                                 r * (self.u_xt[:-2, j - 1] +
                                      self.u_xt[2:, j - 1])
        return self.u_xt.T

    def _cal_pde_implicit_(self):
        """
        克兰克—尼科尔森隐式格式求解
        :return:
        """
        r = self.const ** 2 * self.t_h / self.x_h ** 2  # 差分格式系数常量
        cf_1, cf_2 = 2 + 2 / r, 2 / r - 2  # 差分格式的系数
        # 初始条件，第一列和最后一列数值，数值解按例存储
        self.u_xt[:, [0, -1]] = self.c1, self.c2
        # 计算第1行
        i = np.arange(1, self.n - 1)
        self.u_xt[1:-1, 0] = self.f_fun(i * self.x_h)
        # 隐式格式求解其他列
        vd = cf_1 * np.ones(self.n)  # 主对角线元素
        vd[0], vd[-1] = 1, 1
        va, vc = -np.ones(self.n - 1), -np.ones(self.n - 1)  # 次对角线元素
        va[-1], vc[0] = 0, 0
        vb = np.zeros(self.n)  # 右端向量
        vb[0], vb[-1] = self.c1, self.c2
        for j in range(1, self.m):
            vb[1:-1] = self.u_xt[:-2, j - 1] + self.u_xt[2:, j - 1] + \
                       cf_2 * self.u_xt[1:-1, j - 1]
            # 追赶法求解三对角矩阵形式的方程组的解
            cmtm = ChasingMethodTridiagonalMatrix(va, vd, vc, vb)
            self.u_xt[:, j] = cmtm.fit_solve()  # 存储
        return self.u_xt.T

    def plt_pde_heat_surface(self):  # 参考一维波动方程
        """
        可视化数值解
        :return:
        """
        xi, ti = np.linspace(0, self.x_a, self.n), np.linspace(0, self.t_b, self.m)
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
        plt.title("一维热传导方程数值解曲面$(%s)$" % self.pde_method, fontdict={"fontsize": 18})
        if self.pde_model:
            ax = fig.add_subplot(122, projection='3d')
            analytical_sol = self.pde_model(x, t)
            error_ = analytical_sol - self.u_xt.T  # 误差
            ax.plot_surface(x, t, error_, cmap='rainbow')
            mae = np.mean(np.abs(error_))  # 均方根误差
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
        xi = np.linspace(0, self.x_a, self.n)
        ti = np.linspace(0, self.t_b, self.m)
        idx = np.array([1, len(ti) / 4, len(ti) / 2, 3 * len(ti) / 4, len(ti)], np.int) - 1
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        ls_ = ["-", "--", "-.", ":", "-"]
        for k, i in enumerate(idx):
            if k == 0:
                plt.plot(xi, self.u_xt[:, i], ls_[k], label='$t=%.4f$' % ti[i], lw=2.5)
            else:
                plt.plot(xi, self.u_xt[:, i], ls_[k], label='$t=%.4f$' % ti[i], lw=1.5)
        plt.ylabel('$U(x,t)$', fontdict={"fontsize": 18})
        plt.xlabel('$x$', fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc='upper right')
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.title("一维热传导方程在某些时刻的数值解曲线", fontdict={"fontsize": 18})
        # 2、带有填充区域的等值线图
        plt.subplot(122)
        extent = [0, self.t_b + self.t_h, 0, self.x_a + self.x_h]  # 时间和空间的取值范围
        plt.contourf(self.u_xt, levels=10, origin='lower', extent=extent, cmap=plt.get_cmap("jet"))
        # plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
        plt.colorbar()  # 颜色bar
        plt.ylabel('$x$', fontdict={"fontsize": 18})
        plt.xlabel('$t$', fontdict={"fontsize": 18})
        plt.title("一维热传导方程在时刻$t = %.1f$的等值线图" % self.t_b, fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.show()
