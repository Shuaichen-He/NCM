# -*- coding: UTF-8 -*-
"""
@file_name: stiff_ODES_rk_pcs.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import math
from util_font import *


class StiffODEsRKPCS:
    """
    3阶显式龙格—库塔公式 + 3级6阶隐式龙格—库塔公式构成预测校正系统，求解刚性微分方程组
    """

    def __init__(self, ode_funs, x0, y0, x_final, h=0.1):
        self.ode_funs = ode_funs  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值， y0是向量
        self.n = len(self.y0)  # 方程个数
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        v15 = math.sqrt(15)
        # 3级6阶隐式龙格—库塔公式的y方向系数矩阵
        self.GLFIRK_mat = np.array([[5 / 36, 2 / 9 - v15 / 15, 5 / 36 - v15 / 30],
                                    [5 / 36 + v15 / 24, 2 / 9, 5 / 36 - v15 / 24],
                                    [5 / 36 + v15 / 30, 2 / 9 + v15 / 15, 5 / 36]])
        self.ode_sol = None  # 求解的微分数值解

    def fit_odes(self):
        """
        3阶显式 + 3级6阶隐式龙格库塔法求解刚性微分方程组算法
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), self.n + 1))  # ode的数值解
        self.ode_sol[:, 0] = x_array  # 第一列存储x
        self.ode_sol[0, 1:] = self.y0  # 每一次递推值按一行存储，即一列代表一个微分方程数值解
        # 变量x的递增向量
        x_k = self.h * np.array([(5 - math.sqrt(15)) / 10, 1 / 2,
                                 (5 + math.sqrt(15)) / 10])
        for idx, _ in enumerate(x_array[1:]):
            # 1. 3阶显式龙格—库塔法预测，得到k1, k2, k3和方程组的预测值向量y_predict
            y_predict, k1, k2, k3 = \
                self._rk_3_order_explict_(x_array[idx], self.ode_sol[idx, 1:])
            k_mat = self.h * np.array([k1, k2, k3])  # 由k1, k2, k3构成矩阵
            # 2. 3级6阶隐式龙格—库塔公式校正一次
            self._rk_3_order_implicit_(x_array[idx], x_k, idx, y_predict, k_mat)

    def _rk_3_order_explict_(self, x, y):
        """
        三阶龙格库塔公式，显式方法
        :return:
        """
        k1 = self.ode_funs(x, y)
        k2 = self.ode_funs(x + self.h / 2, y + self.h / 2 * k1)
        k3 = self.ode_funs(x + self.h, y - self.h * k1 + 2 * self.h * k2)
        sol = y + self.h / 6 * (k1 + 4 * k2 + k3)
        return sol, k1, k2, k3

    def _rk_3_order_implicit_(self, x, x_k, idx, y_predict, k_mat):
        """
        三阶龙格库塔公式，隐式方法
        :param x: 待递推下个值的起点
        :param x_k: 变量x的递增量
        :param idx: 待递推下个值的起点索引
        :param y_predict: 3阶显式龙格—库塔公式预测值，向量
        :param k_mat: 3阶显式龙格—库塔公式k1，k2，k3构成的矩阵
        :return:
        """
        k1 = self.ode_funs(x + x_k[0], y_predict + np.dot(self.GLFIRK_mat[0, :], k_mat))
        k2 = self.ode_funs(x + x_k[1], y_predict + np.dot(self.GLFIRK_mat[1, :], k_mat))
        k3 = self.ode_funs(x + x_k[2], y_predict + np.dot(self.GLFIRK_mat[2, :], k_mat))
        self.ode_sol[idx + 1, 1:] = y_predict + self.h * (5 * k1 + k2 + 5 * k3) / 18

    def plt_odes_rk_pcs(self, is_show=True):
        """
        可视化数值解
        """
        if is_show:
            plt.figure(figsize=(8, 6))
        line_style = ["-", "--", "-.", ":"]
        for i in range(self.n):
            if self.n > 4:
                plt.plot(self.ode_sol[:, 0], self.ode_sol[:, i + 1], label="$y_{%d}(x)$" % (i + 1))
            else:
                plt.plot(self.ode_sol[:, 0], self.ode_sol[:, i + 1], line_style[i],
                         lw=1.5, label="$\hat y_{%d}(x)$" % (i + 1))
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$\hat y_i(x)$", fontdict={"fontsize": 18})
        plt.title("$RKPCS$求解一阶刚性微分方程组数值解曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
