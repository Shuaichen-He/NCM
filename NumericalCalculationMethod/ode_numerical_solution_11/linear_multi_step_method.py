# -*- coding: UTF-8 -*-
"""
@file_name: linear_multi_step_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class LinearMultiStepMethod:
    """
    五种线性多步法求解带有初值问题的ODE
    """

    def __init__(self, ode_fun, x0, y0, x_final, h=0.1, ode_method="e_admas"):
        self.ode_fun = ode_fun  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.ode_method = ode_method  # 线性多步法的五种方法
        self.ode_sol = None  # 求解的微分数值解

    def fit_ode(self):
        """
        核心算法：根据参数ode_method，采用不同的线性多步法
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), 2))  # ode的数值解
        self.ode_sol[:, 0] = x_array  # 第一列存储离散待递推数值
        self.ode_sol[:4, 1] = self._rk_start_value(x_array[:4])  # 采用龙格库塔方法计算启动的前四个值
        if self.ode_method.lower() == "e_admas":
            self._explicit_adams_(x_array)  # 显式
        elif self.ode_method.lower() == "i_admas":
            self._implicit_adams_(x_array)  # 隐式
        elif self.ode_method.lower() == "milne":
            self._explicit_milne_(x_array)  # 显式米尔尼方法
        elif self.ode_method.lower() == "simpson":
            self._implicit_simpson_(x_array)  # 隐式辛普森方法
        elif self.ode_method.lower() == "hamming":
            self._implicit_hamming_(x_array)  # 隐式汉明方法
        else:
            print("仅支持e_admas、i_admas、milne、simpson或hanming.")
            exit(0)
        return self.ode_sol

    def _explicit_adams_(self, xi):
        """
        显式四阶Adams方法
        :return:
        """
        for idx in range(3, xi.shape[0] - 1):
            # 计算fn, f_{n-1}, f_{n-2}, f_{n-3}
            x_val = np.array([xi[idx - 3], xi[idx - 2], xi[idx - 1], xi[idx]])
            y_val = self.ode_fun(x_val, self.ode_sol[idx - 3:idx + 1, 1])
            coefficient = np.array([-9, 37, -59, 55])  # adams系数
            self.ode_sol[idx + 1, 1] = self.ode_sol[idx, 1] + \
                                       self.h / 24 * np.dot(coefficient, y_val)

    def _implicit_adams_(self, xi):
        """
        隐式四阶Adams方法
        :return:
        """
        for idx in range(2, xi.shape[0] - 1):
            x_val = np.array([xi[idx - 2], xi[idx - 1], xi[idx]])
            y_val = self.ode_fun(x_val, self.ode_sol[idx - 2:idx + 1, 1])  # 计算fn, fn-1, fn-2, fn-3
            coefficient = np.array([1, -5, 19])  # 隐式adams后三个系数，
            x_dot_y = np.dot(coefficient, y_val)  # 该值重复在以下迭代过程中不变
            f_n1 = self.ode_fun(xi[idx + 1], self.ode_sol[idx + 1, 1])  # 初始采用R-K第4个值启动
            x_n = self.ode_sol[idx, 1] + self.h / 24 * (x_dot_y + 9 * f_n1)  # 隐式adams第一个值单独计算
            x_b = np.infty  # 上一次迭代值
            # 反复迭代，直到收敛，及前后两次的值小于给定精度
            while abs(x_n - x_b) > 1e-12:
                x_b = x_n  # 迭代值更新
                f_n1 = self.ode_fun(xi[idx + 1], x_n)  # 初始采用R-K第4个值启动
                # 不断用隐式公式逼近精度
                x_n = self.ode_sol[idx, 1] + self.h / 24 * (x_dot_y + 9 * f_n1)
            self.ode_sol[idx + 1, 1] = x_n

    def _explicit_milne_(self, xi):
        """
        显式4步4阶米尔尼方法
        :return:
        """
        for idx in range(3, xi.shape[0] - 1):
            x_val = np.array([xi[idx - 2], xi[idx - 1], xi[idx]])
            y_val = self.ode_fun(x_val, self.ode_sol[idx - 2:idx + 1, 1])  # 计算fn, fn-1, fn-2
            coefficient = np.array([2, -1, 2])  # 米尔尼系数
            self.ode_sol[idx + 1, 1] = self.ode_sol[idx - 3, 1] + \
                                       4 * self.h / 3 * np.dot(coefficient, y_val)

    def _implicit_simpson_(self, xi):
        """
        隐式2步4阶辛普森方法
        :return:
        """
        for idx in range(1, xi.shape[0] - 1):
            x_val = np.array([xi[idx - 1], xi[idx]])
            y_val = self.ode_fun(x_val, self.ode_sol[idx - 1:idx + 1, 1])  # 计算fn, f_{n+1}
            coefficient = np.array([1, 4])  # 隐式辛普森前两个系数，
            x_dot_y = np.dot(coefficient, y_val)  # 该值重复在以下迭代过程中不变
            f_n1 = self.ode_fun(xi[idx + 1], self.ode_sol[idx + 1, 1])  # 初始采用R-K第3个值启动
            x_n = self.ode_sol[idx - 1, 1] + self.h / 3 * (x_dot_y + f_n1)  # 隐式f_{n+1}值单独计算
            x_b = np.infty  # 上一次迭代值
            # 反复迭代，直到收敛，及前后两次的值小于给定精度
            while abs(x_n - x_b) > 1e-12:
                x_b = x_n  # 迭代值更新
                f_n1 = self.ode_fun(xi[idx + 1], x_n)  # 迭代计算fn+1
                # 不断用隐式公式逼近精度
                x_n = self.ode_sol[idx - 1, 1] + self.h / 3 * (x_dot_y + f_n1)
            self.ode_sol[idx + 1, 1] = x_n

    def _implicit_hamming_(self, xi):
        """
        3步隐式4阶汉明算法
        :return:
        """
        for idx in range(2, xi.shape[0] - 1):
            x_val = np.array([xi[idx - 1], xi[idx]])
            y_val = self.ode_fun(x_val, self.ode_sol[idx - 1:idx + 1, 1])  # 计算fn, f_{n-1}
            coefficient = np.array([-1, 2])  # 隐式汉明后2个系数，
            x_dot_y = np.dot(coefficient, y_val)  # 该值重复在以下迭代过程中不变
            f_n1 = self.ode_fun(xi[idx + 1], self.ode_sol[idx + 1, 1])  # 初始采用R-K第4个值启动
            x_n = (9 * self.ode_sol[idx, 1] - self.ode_sol[idx - 2, 1]) / 8 + \
                  3 * self.h / 8 * (x_dot_y + f_n1)  # 汉明公式， f_n1为隐式解
            x_b = np.infty  # 上一次迭代值
            # 反复迭代，直到收敛，及前后两次的值小于给定精度
            while abs(x_n - x_b) > 1e-12:
                x_b = x_n  # 迭代值更新
                f_n1 = self.ode_fun(xi[idx + 1], x_n)
                # 不断用隐式公式逼近精度
                x_n = (9 * self.ode_sol[idx, 1] - self.ode_sol[idx - 2, 1]) / 8 + \
                      3 * self.h / 8 * (x_dot_y + f_n1)  # 汉明公式， f_n1为隐式解，迭代逼近
            self.ode_sol[idx + 1, 1] = x_n

    def _rk_start_value(self, xi):
        """
        标准的4级4阶龙格—库塔公式启动三个值y1, y2, y3，param xi: 离散数据值
        :return:
        """
        sol = np.zeros(len(xi))
        sol[0] = self.y0
        for idx, _ in enumerate(xi[1:]):
            K1 = self.ode_fun(xi[idx], sol[idx])
            K2 = self.ode_fun(xi[idx] + self.h / 2, sol[idx] + self.h / 2 * K1)
            K3 = self.ode_fun(xi[idx] + self.h / 2, sol[idx] + self.h / 2 * K2)
            K4 = self.ode_fun(xi[idx] + self.h, sol[idx] + self.h * K3)
            sol[idx + 1] = sol[idx] + self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return sol
