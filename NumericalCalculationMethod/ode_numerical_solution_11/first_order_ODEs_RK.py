# -*- coding: UTF-8 -*-
"""
@file_name: first_order_ODEs_RK.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class FirstOrderODEsRK:
    """
    一阶常微分方程组，龙格库塔方法求解
    """

    def __init__(self, ode_funs, x0, y0, x_final, h=0.1):
        self.ode_funs = ode_funs  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值，y0是向量
        self.n = len(self.y0)  # 方程个数
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.ode_sol = None  # 求解的微分数值解

    def fit_odes(self):
        """
        龙格库塔法求解一阶常微分方程组算法
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), self.n + 1))  # ode的数值解
        self.ode_sol[:, 0] = x_array  # 第一列存储x
        self.ode_sol[0, 1:] = self.y0  # 每一次递推值按一行存储，即一列代表一个微分方程数值解
        for idx, _ in enumerate(x_array[1:]):
            K1 = self.ode_funs(x_array[idx], self.ode_sol[idx, 1:])
            K2 = self.ode_funs(x_array[idx] + self.h / 2,
                               self.ode_sol[idx, 1:] + self.h / 2 * K1)
            K3 = self.ode_funs(x_array[idx] + self.h / 2,
                               self.ode_sol[idx, 1:] + self.h / 2 * K2)
            K4 = self.ode_funs(x_array[idx] + self.h,
                               self.ode_sol[idx, 1:] + self.h * K3)
            self.ode_sol[idx + 1, 1:] = \
                self.ode_sol[idx, 1:] + self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return self.ode_sol

    def plt_odes_rk(self, is_show=True):
        """
        可视化数值解
        :return:
        """
        if is_show:
            plt.figure(figsize=(8, 6))
        line_style = ["-", "--", "-.", ":"]
        for i in range(self.n):
            if self.n > 4:
                plt.plot(self.ode_sol[:, 0], self.ode_sol[:, i + 1], label="$\hat y_{%d}(x)$" % (i + 1))
            else:
                plt.plot(self.ode_sol[:, 0], self.ode_sol[:, i + 1], line_style[i],
                         lw=1.5, label="$\hat y_{%d}(x)$" % (i + 1))
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$\hat y_i(x)$", fontdict={"fontsize": 18})
        plt.title("龙格库塔法求解一阶$ODEs$数值解曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
