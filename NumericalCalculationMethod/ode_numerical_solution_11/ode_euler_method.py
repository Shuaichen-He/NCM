# -*- coding: UTF-8 -*-
"""
@file_name: ode_euler_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class ODEEulerMethod:
    """
    欧拉法求解一阶微分方程，包括五种公式：显式欧拉法、隐式欧拉法、梯形公式、
    中点公式和改进的欧拉法
    """

    def __init__(self, ode_fun, x0, y0, x_final, h=0.1, ode_method="PC"):
        self.ode_fun = ode_fun  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.ode_method = ode_method  # 求解的欧拉方法，默认为改进的欧拉法
        self.ode_sol = None  # 求解的微分数值解

    def fit_ode(self):
        """
        根据参数ode_method，采用不同的欧拉法求解一阶微分方程数值解
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), 2))  # ode的数值解
        self.ode_sol[:, 0] = x_array  # 第1列为离散的xi
        if self.ode_method.lower() == "explicit":
            self.ode_sol[:, 1] = self._explicit_euler_(x_array)  # 显式欧拉法
        elif self.ode_method.lower() == "implicit":
            self.ode_sol[:, 1] = self._implicit_euler_(x_array)  # 隐式欧拉法
        elif self.ode_method.lower() == "trapezoid":
            self.ode_sol[:, 1] = self._trapezoid_euler_(x_array)  # 梯形公式法
        elif self.ode_method.lower() == "middle":
            self.ode_sol[:, 1] = self._middle_euler_(x_array)  # 中点欧拉法
        elif self.ode_method.lower() == "pc":
            self.ode_sol[:, 1] = self._predictive_correction_euler_(x_array)  # 预测校正系统
        else:
            print("仅支持explicit、implicit、trapezoid、middle和PC.")
            exit(0)
        return self.ode_sol

    def _explicit_euler_(self, xi):
        """
        显式欧拉法求解ode数值解
        :return:
        """
        sol = np.zeros(len(xi))  # 数值解
        sol[0] = self.y0  # 第一个值为初值
        for idx, _ in enumerate(xi[1:]):  # 逐步递推
            sol[idx + 1] = sol[idx] + self.h * self.ode_fun(xi[idx], sol[idx])
        return sol

    def _implicit_euler_(self, xi):
        """
        隐式欧拉法求解ode数值解
        :return:
        """
        y_explicit = self._explicit_euler_(xi)  # 显式欧拉法求解
        sol = np.zeros(len(xi))  # 隐式欧拉法数值解
        sol[0] = self.y0  # 第一个值为初值
        for idx, _ in enumerate(xi[1:]):  # 逐步递推
            # 初始迭代yk^(0)
            x_n = sol[idx] + self.h * self.ode_fun(xi[idx + 1],
                                                   y_explicit[idx + 1])
            x_b = np.infty  # 上一次迭代值
            # 反复迭代，直到收敛，及前后两次的值小于给定精度
            while abs(x_n - x_b) > 1e-12:
                x_b = x_n  # 迭代值更新
                # 不断用隐式欧拉公式逼近精度
                x_n = sol[idx] + self.h * self.ode_fun(xi[idx + 1], x_b)
            sol[idx + 1] = x_n
        return sol

    def _trapezoid_euler_(self, xi):
        """
        梯形公式法求解ode数值解
        :return:
        """
        y_explicit = self._explicit_euler_(xi)  # 显式欧拉法求解
        sol = np.zeros(len(xi))  # 梯形欧拉法数值解
        sol[0] = self.y0  # 第一个值为初值
        for idx, _ in enumerate(xi[1:]):  # 逐步递推
            # 初始迭代yk^(0)
            item_1 = self.ode_fun(xi[idx], sol[idx])  # f(x_{k-1}, y_{k-1})
            item_2 = self.ode_fun(xi[idx + 1], y_explicit[idx + 1])  # f(x_k, y*_k)
            x_n = sol[idx] + self.h / 2 * (item_1 + item_2)
            x_b = np.infty  # 上一次迭代值
            # 反复迭代，直到收敛，及前后两次的值小于给定精度
            while abs(x_n - x_b) > 1e-12:
                x_b = x_n  # 迭代值更新
                # 不断用梯形欧拉公式逼近精度
                x_n = sol[idx] + self.h / 2 * (item_1 + self.ode_fun(xi[idx + 1], x_b))
            sol[idx + 1] = x_n
        return sol

    def _middle_euler_(self, xi):
        """
        中点欧拉法求解ode数值解
        :return:
        """
        y1 = self.y0 + self.h * self.ode_fun(self.x0, self.y0)  # 起始的第2个值
        sol = np.zeros(len(xi))
        sol[0], sol[1] = self.y0, y1  # 起始的2个值
        for idx, _ in enumerate(xi[2:]):  # 逐步递推
            sol[idx + 2] = sol[idx] + 2 * self.h * \
                           self.ode_fun(xi[idx + 1], sol[idx + 1])
        return sol

    def _predictive_correction_euler_(self, xi):
        """
        改进的欧拉法（预测校正）求解ode数值解
        :return:
        """
        sol = np.zeros(len(xi))  # 改进的欧拉法数值解
        sol[0] = self.y0  # 第一个值为初值
        for idx, _ in enumerate(xi[1:]):  # 逐步递推
            # 1. 显式欧拉法预测
            y_predict = sol[idx] + self.h * self.ode_fun(xi[idx], sol[idx])
            # 2. 梯形公式法校正
            val_term = self.ode_fun(xi[idx], sol[idx]) + \
                       self.ode_fun(xi[idx + 1], y_predict)
            sol[idx + 1] = sol[idx] + self.h / 2 * val_term
        return sol
