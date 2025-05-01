# -*- coding: UTF-8 -*-
"""
@file_name: variable_step_runge_kutta.py
@time: 2021-11-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class VariableStepRungeKutta:
    """
    变步长龙格—库塔法求解一阶常微分方程。
    """

    def __init__(self, ode_fun, x0, y0, x_final, h=0.1, eps=1e-10, is_plt=False):
        self.ode_fun = ode_fun  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.eps = eps  # 变步长后的偏差精度
        self.is_plt = is_plt  # 是否可视化数值解
        self.ode_sol = None  # 求解的微分数值解
        self.adaptive_sol_x, self.adaptive_sol_y = [], []  # 变步长计算得到的离散值及其数值解

    def fit_ode(self):
        """
        变步长龙格—库塔法求解一阶常微分方程。
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), 2))  # ode的数值解
        self.ode_sol[:, 0], self.ode_sol[0, 1] = x_array, self.y0
        for idx, _ in enumerate(x_array[1:]):
            v_h, n = self.h / 2, 2  # v_h为变步长，n为折半后计算的跨度数
            y_n = self._standard_runge_kutta_(x_array[idx],
                                              self.ode_sol[idx, 1], self.h)  # 以步长h求下一个近似值
            # 折半跨两次计算
            y_halve_tmp = self._standard_runge_kutta_(x_array[idx],
                                                      self.ode_sol[idx, 1], v_h)
            y_halve = self._standard_runge_kutta_(x_array[idx] + v_h, y_halve_tmp, v_h)
            if abs(y_halve - y_n) > self.eps:  # 区间长度折半，细分区间
                self._halve_step_cal(x_array, y_n, y_halve, v_h, n, idx)
            else:  # 区间长度加倍，合并区间
                self.ode_sol[idx + 1, 1] = y_halve
        if self.is_plt:
            self.plt_histogram_dist()
        return self.ode_sol

    def _standard_runge_kutta_(self, x_b, y_b, v_h):
        """
        标准的4级4阶龙格—库塔公式求解每一步的近似数值解
        :param x_b: 某个离散数据值
        :param y_b: 某个数值解
        :param v_h: 变步长
        :return:
        """
        K1 = self.ode_fun(x_b, y_b)
        K2 = self.ode_fun(x_b + v_h / 2, y_b + v_h / 2 * K1)
        K3 = self.ode_fun(x_b + v_h / 2, y_b + v_h / 2 * K2)
        K4 = self.ode_fun(x_b + v_h, y_b + v_h * K3)
        return y_b + v_h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

    def _halve_step_cal(self, x_array, y_n, y_halve, v_h, n, idx):
        """
        区间折半计算
        :return:
        """
        ada_x, ada_y = None, None  # 存储折半过程中的数值解
        while abs(y_halve - y_n) > self.eps:
            ada_x, ada_y = [], []  # 存储折半过程中的数值解
            y_n = y_halve
            v_h /= 2  # 反复折半
            n *= 2  # 跨度计算次数
            y_halve = self.ode_sol[idx, 1]  # 反复计算到下一个y(i+1)
            for i in range(n):
                y_halve = self._standard_runge_kutta_(x_array[idx] + i * v_h,
                                                      y_halve, v_h)
                ada_y.append(y_halve)
                ada_x.append(x_array[idx] + i * v_h)
        self.ode_sol[idx + 1, 1] = y_halve
        self.adaptive_sol_x.extend(ada_x)
        self.adaptive_sol_y.extend(ada_y)

    def plt_histogram_dist(self, is_show=True):
        """
        绘制变步长节点分布情况的直方图
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        bins = np.linspace(self.x0, self.x_final, 21)
        n = plt.hist(self.adaptive_sol_x, bins=bins, rwidth=0.8, color="c", alpha=0.5)
        print(n)
        plt.plot((n[1][:-1] + n[1][1:]) / 2, n[0], "ko--", lw=2, markerfacecolor="r", markeredgecolor="r")
        plt.title("变步长龙格库塔法的节点划分数量的分布直方图", fontdict={"fontsize": 18})
        plt.ylabel("$Frequency$", fontdict={"fontsize": 18})
        plt.xlabel("$Bins$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
