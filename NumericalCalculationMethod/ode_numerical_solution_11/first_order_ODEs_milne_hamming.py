# -*- coding: UTF-8 -*-
"""
@file_name: first_order_ODEs_milne_hamming.py
@time: 2021-11-13
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np



class FirstOrderODEsMilneHamming:
    """
    修正米尔尼—汉明预测—校正格式，求解一阶微分方程组
    """
    def __init__(self, ode_funs, x0, y0, x_final, h=0.1, is_plt=False):
        self.ode_funs = ode_funs  # 待求解的微分方程组
        self.x0, self.y0 = x0, y0  # 初值
        self.n = len(self.y0)  # 方程个数
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.is_plt = is_plt  # 是否可视化数值解
        self.ode_sol = None  # 求解的微分数值解

    def fit_odes(self):
        """
        修正米尔尼—汉明预测—校正格式求解刚性微分方程数值解
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), self.n + 1))  # ode的数值解
        self.ode_sol[:, 0] = x_array
        self.ode_sol[:4, 1:] = self._rk_start_value(x_array[:4])  # 采用龙格库塔方法计算启动的前四个值

        y_predict = self._milne_predictor_(3, x_array)
        self.ode_sol[4, 1:] = self._hamming_corrector_(3, x_array, y_predict)
        pn, cn = np.copy(y_predict), np.copy(self.ode_sol[4, 1:])  # 修正部分变量
        for idx in range(4, x_array.shape[0] - 1):
            # 1. 米尔尼显式预测部分
            y_predict = self._milne_predictor_(idx, x_array)
            # 2. 显式修正部分
            m_part = y_predict + 112 / 121 * (cn - pn)
            pn = np.copy(y_predict)  # 更新预测值
            # 3. 隐式校正部分
            y_correct = self._hamming_corrector_(idx, x_array, m_part)
            cn = np.copy(y_correct)
            # 4. 隐式修正部分
            self.ode_sol[idx + 1, 1:] = y_correct - 9 / 121 * (y_correct - y_predict)


    def _milne_predictor_(self, idx, xi):
        """
        米尔尼显式预测部分
        :return:
        """
        # 计算fn, fn-1, fn-2
        x_val = np.array([xi[idx - 2], xi[idx - 1], xi[idx]])
        f_val = self.ode_funs(x_val, self.ode_sol[idx - 2:idx + 1, 1:])
        coefficient = np.array([2, -1, 2])  # 米尔尼系数
        y_predict = self.ode_sol[idx - 3, 1] + 4 * self.h / 3 * np.dot(coefficient, f_val)
        return y_predict

    def _hamming_corrector_(self, idx, xi, m_part):
        """
        汉明隐式校正部分
        :return:
        """
        x_val = np.array([xi[idx - 1], xi[idx]])
        f_val = self.ode_funs(x_val, self.ode_sol[idx - 1:idx + 1, 1:])
        coefficient = np.array([-1, 2])  # 隐式汉明后2个系数，
        f_n1 = self.ode_funs(xi[idx + 1], m_part)
        y_correct = (9 * self.ode_sol[idx, 1:] - self.ode_sol[idx - 2, 1:]) / 8 + \
                    3 * self.h / 8 * (np.dot(coefficient, f_val) + f_n1)  # 汉明公式， f_n1为隐式解
        return y_correct

    def _rk_start_value(self, xi):
        """
        标准的4级4阶龙格—库塔公式启动三个值y1, y2, y3
        :param xi: 离散数据值
        :return:
        """
        sol = np.zeros((len(xi), self.n))
        sol[0, :] = self.y0
        for idx, _ in enumerate(xi[1:]):
            K1 = self.ode_funs(xi[idx], sol[idx, :])
            K2 = self.ode_funs(xi[idx] + self.h / 2, sol[idx, :] + self.h / 2 * K1)
            K3 = self.ode_funs(xi[idx] + self.h / 2, sol[idx, :] + self.h / 2 * K2)
            K4 = self.ode_funs(xi[idx] + self.h, sol[idx, :] + self.h * K3)
            sol[idx + 1, :] = sol[idx, :] + self.h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return sol
