# -*- coding: UTF-8 -*-
"""
@file_name: steepest_descent_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from __future__ import division
import numpy as np
from iterative_solution_linear_equation_07.utils.Iterative_linear_equs_utils import IterativeLinearEquationsUtils


class SteepestDescentMethod(IterativeLinearEquationsUtils):
    """
    最速下降法，继承IterativeLinearEquationsUtils
    """

    def __init__(self, A, b, x0, eps=1e-8, max_iter=200, is_out_info=False):
        IterativeLinearEquationsUtils.__init__(self, A, b, x0, eps, max_iter, is_out_info)
        self.iterative_info = {}  # 组合存储迭代信息
        self.precision = []  # 存储每次迭代误差精度

    def fit_solve(self):
        """
        最速下降法求解
        :return:
        """
        x_next = np.copy(self.x0)  # x_next表示x(k+1)， x_before表示x(k)
        cond_num = np.linalg.cond(self.A)  # 系数矩阵的条件数
        iteration = 0  # 迭代变量
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 解的迭代
            r_k = self.b - np.dot(self.A, x_before)  # 残差向量，负梯度方向
            ak = np.dot(np.dot(self.A, r_k), r_k)
            if ak <= 1e-50:  # 终止条件
                print("SteepestDescent_IterativeStopCond：(A*rk, rk): %.10e" % ak)
                break
            alpha_k = np.dot(r_k, r_k) / ak  # 搜索步长
            x_next = x_before + alpha_k * r_k  # 最速下降法公式
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "最速下降法，最速下降法已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合最速下降法迭代法的结果信息
            self.iterative_info["Success_Info"] = "最速下降法，迭代终止，收敛到近似解"
        self.iterative_info["Condition_number"] = cond_num
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_next
        self.iterative_info["Precision"] = self.precision[-1]
        if self.is_out_info:
            for key in self.iterative_info.keys():
                print(key + ":", self.iterative_info[key])
        return x_next

    def plt_convergence_x(self, is_show=True, style="o-"):  # 参考超松弛SOR迭代法
        """
        可视化迭代解的精度曲线
        """
        IterativeLinearEquationsUtils._plt_convergence_precision(self, is_show, "Steepest \ Descent", style)
