# -*- coding: UTF-8 -*-
"""
@file_name: ode_runge_kuta_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import math


class ODERungeKuttaMethod:
    """
    龙龙格库塔法求解一阶常微分方程，包括龙格-库塔法和龙格-库塔-芬尔格法
    """

    def __init__(self, ode_fun, x0, y0, x_final, h=0.1, rk_type="RK", is_plt=False):
        # 略去必要的实例属性初始化具体代码
        self.ode_fun = ode_fun  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.rk_type = rk_type  # 求解的龙格库塔方法，默认为经典的龙格库塔法
        self.is_plt = is_plt  # 是否可视化数值解
        self.ode_sol = None  # 求解的微分数值解

    def fit_ode(self):
        """
        根据参数ode_method，采用不同的龙格库塔法求解一阶微分方程数值解
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), 2))  # ode的数值解
        self.ode_sol[:, 0] = x_array  # 第1列为离散的xi
        if self.rk_type.lower() == "rk":  # 4级4阶龙格-库塔公式
            self.ode_sol[:, 1] = self._standard_runge_kutta_(x_array)
        elif self.rk_type.lower() == "rkf":  # 龙格-库塔-芬尔格公式
            self.ode_sol[:, 1] = self._runge_kutta_fehlberg(x_array)
        return self.ode_sol

    def _standard_runge_kutta_(self, xi):
        """
        标准的4级4阶龙格—库塔公式求解
        :param xi: 离散数据值
        :return:
        """
        sol = np.zeros(len(xi))  # 数值解
        sol[0] = self.y0  # 初值
        for idx, _ in enumerate(xi[1:]):  # 逐步递推
            K1 = self.ode_fun(xi[idx], sol[idx])
            K2 = self.ode_fun(xi[idx] + self.h / 2, sol[idx] + self.h / 2 * K1)
            K3 = self.ode_fun(xi[idx] + self.h / 2, sol[idx] + self.h / 2 * K2)
            K4 = self.ode_fun(xi[idx] + self.h, sol[idx] + self.h * K3)
            sol[idx + 1] = sol[idx] + self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return sol

    def _runge_kutta_fehlberg(self, xi):
        """
        龙格—库塔—芬尔格公式求解
        :param xi: 离散数据值
        :return:
        """
        sol = np.zeros(len(xi))  # 数值解
        sol[0] = self.y0  # 初值
        for idx, _ in enumerate(xi[1:]):  # 逐步递推
            K1 = self.h * self.ode_fun(xi[idx], sol[idx])
            K2 = self.h * self.ode_fun(xi[idx] + self.h / 4,
                                       sol[idx] + self.h / 4 * K1)
            K3 = self.h * self.ode_fun(xi[idx] + 3 / 8 * self.h,
                                       sol[idx] + 3 / 32 * K1 + 9 / 32 * K2)
            K4 = self.h * self.ode_fun(xi[idx] + 12 / 13 * self.h, sol[idx] +
                                       (1932 * K1 - 7200 * K2 + 7296 * K3) / 2197)
            K5 = self.h * self.ode_fun(xi[idx] + self.h, sol[idx] +
                                       439 / 216 * K1 - 8 * K2 +
                                       3680 / 513 * K3 - 845 / 4104 * K4)
            K6 = self.h * self.ode_fun(xi[idx] + self.h / 2, sol[idx] -
                                       8 / 27 * K1 + 2 * K2 - 3544 / 2565 * K3 +
                                       1859 / 4104 * K4 - 11 / 40 * K5)
            sol[idx + 1] = sol[idx] + 16 / 135 * K1 + 6656 / 12825 * K3 + \
                           28561 / 56430 * K4 - 9 / 50 * K5 + 2 / 55 * K6
        # c4 = np.array([1932 / 2197, -7200 / 2197, 7296 / 2197], dtype=np.float64)
        # c5 = np.array([439 / 216, -8, 3680 / 513, -845 / 4104], dtype=np.float64)
        # c6 = np.array([-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40], dtype=np.float64)
        # c_sol = np.array([16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55], dtype=np.float64)
        # for idx, _ in enumerate(xi[1:]):  # 逐步递推
        #     K1 = self.h * self.ode_fun(xi[idx], sol[idx])
        #     K2 = self.h * self.ode_fun(xi[idx] + self.h / 4, sol[idx] + self.h / 4 * K1)
        #     K3 = self.h * self.ode_fun(xi[idx] + 3 / 8 * self.h, sol[idx] + 3 / 32 * K1 + 9 / 32 * K2)
        #     K4 = self.h * self.ode_fun(xi[idx] + 12 / 13 * self.h, sol[idx] + np.dot(c4, np.array([K1, K2, K3])))
        #     K5 = self.h * self.ode_fun(xi[idx] + self.h, sol[idx] + np.dot(c5, np.array([K1, K2, K3, K4])))
        #     K6 = self.h * self.ode_fun(xi[idx] + self.h / 2, sol[idx] + np.dot(c6, np.array([K1, K2, K3, K4, K5])))
        #     sol[idx + 1] = sol[idx] + np.dot(c_sol, np.array([K1, K3, K4, K5, K6]))
        return sol
