# -*- coding: UTF-8 -*-
"""
@file_name: ode_utils.py
@time: 2022-11-15
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from interpolation_02.cubic_spline_interpolation import CubicSplineInterpolation
from util_font import *


class ODESolUtils:
    """
    常微分方程初值问题数值解的工具类，可通过插值求解任意点的值，以及可视化数值解
    """

    def __init__(self, ode_obj, analytic_ft=None):
        self.ode_obj = ode_obj  # 求解ODE问题的方法对象
        self.analytic_ft = analytic_ft  # 解析解，可用于分析精度

    def predict_x0(self, x):
        """
        求解区间内任意时刻的微分方程数值解，采用三次样条插值法
        :param x: 任意值向量列表
        :return:
        """
        if self.ode_obj.ode_sol is None:
            self.ode_obj.fit_ode()
        if isinstance(x, np.int64) or isinstance(x, np.float64):
            x = [x]  # 对于单个值的特殊处理
        x, y = np.asarray(x, np.float64), np.zeros(len(x))
        sol_x, sol_y = self.ode_obj.ode_sol[:, 0], self.ode_obj.ode_sol[:, 1]
        idx = 0  # 所在区间索引，默认第一个区间
        for j, xi in enumerate(x):
            if xi <= self.ode_obj.x0 or xi >= self.ode_obj.x_final:
                print("所求值%f不在求解区间." % xi)
                exit(0)
            # 查找所求值所在区间索引
            flag = False  # 判断所求值的解是否已存在
            for i in range(1, len(sol_x)):
                if xi == sol_x[i]:
                    y[j], flag = sol_y[i], True
                    break
                if sol_x[i] <= xi < sol_x[i + 1]:
                    idx = i
                    break
            if flag is False:
                # 相邻区间取三个值，进行三次样条插值
                if idx <= 1:
                    x_list, y_list = sol_x[:idx + 3], sol_y[:idx + 3]  # 取最初的3个值
                elif idx >= len(sol_x) - 2:
                    x_list, y_list = sol_x[-3:], sol_y[-3:]  # 取最终的3个值
                else:
                    x_list, y_list = sol_x[idx - 1:idx + 2], sol_y[idx - 1:idx + 2]  # 取相邻区间的3个值
                dy = self.ode_obj.ode_fun(x_list[[0, -1]], y_list[[0, -1]])  # 边界处的一阶导数值
                # 三次样条插值, 第一边界条件
                csi = CubicSplineInterpolation(x_list, y_list, dy=dy,
                                               boundary_cond="complete")
                csi.fit_interp()  # 三次样条插值求解
                y[j] = csi.predict_x0([xi])
        return y

    def plt_ode_numerical_sol(self, is_show=True, label_txt=""):
        # 略去可视化ode数值解曲线的具体代码.
        """
        可视化ode数值解曲线
        :return:
        """
        if is_show:
            if self.analytic_ft is not None:
                plt.figure(figsize=(14, 5))
                plt.subplot(121)
            else:
                plt.figure(figsize=(7, 5))
        plt.plot(self.ode_obj.ode_sol[:, 0], self.ode_obj.ode_sol[:, 1], lw=1.5, label="%s" % label_txt)
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$\hat y(x)$", fontdict={"fontsize": 18})
        plt.title("求解$ODE$初值问题数值解曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if self.analytic_ft is not None:
            plt.subplot(122)
            xi = np.arange(0, 1 + self.ode_obj.h, self.ode_obj.h)
            yi = self.analytic_ft(xi)  # 精确解
            err = yi - self.ode_obj.ode_sol[:, 1]
            precision = np.linalg.norm(err)  # 精度误差
            plt.plot(xi, err, "-", lw=1.5, label="$\epsilon=%.10e$" % precision)
            plt.xlabel("$x$", fontdict={"fontsize": 18})
            plt.ylabel("$\epsilon = y_k - \hat y_k$", fontdict={"fontsize": 18})
            plt.title("$ODE$初值问题数值解的误差曲线 $\epsilon = \Vert y - \hat{y} \Vert$", fontdict={"fontsize": 18})
            plt.legend(frameon=False, fontsize=16, loc="upper left")
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
