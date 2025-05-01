# -*- coding: UTF-8 -*-
"""
@file_name: predictive_correction_system.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class PredictiveCorrectionSystem:
    """
    预测—校正系统方法：
    包括Adams 预测-校正、修正预测—校正格式和修正米尔尼—汉明预测—校正
    """

    def __init__(self, ode_fun, x0, y0, x_final, h=0.1, pc_form="PECE"):
        self.ode_fun = ode_fun  # 待求解的微分方程
        self.x0, self.y0 = x0, y0  # 初值
        self.x_final = x_final  # 求解区间的终点
        self.h = h  # 求解步长
        self.pc_form = pc_form  # 预测—校正系统的方法选择
        self.ode_sol = None  # 求解的微分数值解

    def fit_ode(self):
        """
        根据参数pc_form，采用不同的预测—校正系统方法求解一阶微分方程数值解
        :return:
        """
        x_array = np.arange(self.x0, self.x_final + self.h, self.h)  # 待求解ode区间的离散数值
        self.ode_sol = np.zeros((len(x_array), 2))  # ode的数值解
        self.ode_sol[:, 0] = x_array  # 第一列存储离散待递推数值
        self.ode_sol[:4, 1] = self._rk_start_value(x_array[:4])  # 采用龙格库塔方法计算启动的前四个值
        if self.pc_form.upper() == "PECE":
            self._pc_adams_explicit_implicit_(x_array)  # Adams 预测-校正
        elif self.pc_form.upper() == "PMECME_A":  # 修正的Adams预测—校正系统
            self._mpc_adams_explict_implicit_(x_array)
        elif self.pc_form.upper() == "PMECME_MH":
            self._mpc_milne_hamming_(x_array)  # 修正米尔尼—汉明的预测—校正格式
        else:
            print("仅支持PECM, PMECME_A和PMECME_MH.")
            exit(0)
        return self.ode_sol

    def _adams_predictor_(self, idx, xi):
        """
        预测部分，显式公式计算
        :param idx: 当前时刻索引
        :return:
        """
        # 计算f_n, f_{n-1}, f_{n-2}, f_{n-3}
        x_val = np.array([xi[idx - 3], xi[idx - 2], xi[idx - 1], xi[idx]])
        f_val = self.ode_fun(x_val, self.ode_sol[idx - 3:idx + 1, 1])
        coefficient = np.array([-9, 37, -59, 55])  # adams系数
        y_predict = self.ode_sol[idx, 1] + self.h / 24 * np.dot(coefficient, f_val)  # 显式Adams
        return y_predict, f_val

    def _adams_corrector_(self, idx, x_next, m_part, f_val):
        """
        校正部分， 隐式公式计算
        :param x_next: 下一时刻的值
        :param m_part: 修正的预测值
        :param f_val: fn, fn-1, fn-2
        :return:
        """
        coefficient = np.array([1, -5, 19])  # 隐式adams后三个系数，
        fn_1 = self.ode_fun(x_next, m_part)  # 校正值单独计算
        return self.ode_sol[idx, 1] + \
               self.h / 24 * (9 * fn_1 + np.dot(coefficient, f_val))

    def _pc_adams_explicit_implicit_(self, xi):
        """
        Adams 预测-校正
        :param xi: 求解区间以步长h离散化的值
        :return:
        """
        for idx in range(3, xi.shape[0] - 1):
            # 1. 显式预测部分
            y_predict, f_val = self._adams_predictor_(idx, xi)
            # 2. 隐式校正部分。
            self.ode_sol[idx + 1, 1] = \
                self._adams_corrector_(idx, xi[idx + 1], y_predict, f_val[1:])

    def _mpc_adams_explict_implicit_(self, xi):
        """
        修正的Adams预测—校正系统
        :return:
        """
        # 第五个值采用预测—校正系统，以便后期递推值的偏差修正
        y_predict, f_val = self._adams_predictor_(3, xi)  # 显式预测部分
        self.ode_sol[4, 1] = self._adams_corrector_(3, xi[4], y_predict, f_val[1:])  # 隐式校正部分
        pn, cn = y_predict, self.ode_sol[4, 1]  # 修正部分变量
        for idx in range(4, xi.shape[0] - 1):
            # 1. 显式预测部分，预测下一课时刻的值
            y_predict, f_val = self._adams_predictor_(idx, xi)
            # 2. 显式修正部分
            m_part = y_predict + 251 / 270 * (cn - pn)
            pn = y_predict  # 更新预测值
            # 3. 隐式校正部分
            y_correct = self._adams_corrector_(idx, xi[idx + 1], m_part, f_val[1:])
            cn = y_correct
            # 4. 隐式修正部分
            self.ode_sol[idx + 1, 1] = y_correct - 19 / 270 * (y_correct - y_predict)

    def _milne_predictor_(self, idx, xi):
        """
        米尔尼显式预测部分
        :return:
        """
        # 计算f_n, f_{n-1}, f_{n-2}
        x_val = np.array([xi[idx - 2], xi[idx - 1], xi[idx]])
        f_val = self.ode_fun(x_val, self.ode_sol[idx - 2:idx + 1, 1])
        coefficient = np.array([2, -1, 2])  # 米尔尼系数
        y_predict = self.ode_sol[idx - 3, 1] + \
                    4 * self.h / 3 * np.dot(coefficient, f_val)
        return y_predict

    def _hamming_corrector_(self, idx, xi, m_part):
        """
        汉明隐式校正部分
        :return:
        """
        x_val = np.array([xi[idx - 1], xi[idx]])
        f_val = self.ode_fun(x_val, self.ode_sol[idx - 1:idx + 1, 1])
        coefficient = np.array([-1, 2])  # 隐式汉明后2个系数，
        f_n1 = self.ode_fun(xi[idx + 1], m_part)
        y_correct = (9 * self.ode_sol[idx, 1] - self.ode_sol[idx - 2, 1]) / 8 + \
                    3 * self.h / 8 * (np.dot(coefficient, f_val) + f_n1)  # 汉明公式， f_n1为隐式解
        return y_correct

    def _mpc_milne_hamming_(self, xi):
        """
        修正米尔尼—汉明的预测—校正格式
        :param xi:
        :return:
        """
        y_predict = self._milne_predictor_(3, xi)
        self.ode_sol[4, 1] = self._hamming_corrector_(3, xi, y_predict)
        pn, cn = y_predict, self.ode_sol[4, 1]  # 修正部分变量
        for idx in range(4, xi.shape[0] - 1):
            # 1. 米尔尼显式预测部分
            y_predict = self._milne_predictor_(idx, xi)
            # 2. 显式修正部分
            m_part = y_predict + 112 / 121 * (cn - pn)
            pn = y_predict  # 更新预测值
            # 3. 隐式校正部分
            y_correct = self._hamming_corrector_(idx, xi, m_part)
            cn = y_correct
            # 4. 隐式修正部分
            self.ode_sol[idx + 1, 1] = y_correct - 9 / 121 * (y_correct - y_predict)

    def _rk_start_value(self, xi):
        """
        标准的4级4阶龙格—库塔公式启动四个值y1, y2, y3，y4
        :param xi: 离散数据值
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
