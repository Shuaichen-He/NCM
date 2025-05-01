# -*- coding: UTF-8 -*-
"""
@file_name: conjugate_gradient_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from iterative_solution_linear_equation_07.utils.Iterative_linear_equs_utils import IterativeLinearEquationsUtils

class ConjugateGradientMethod(IterativeLinearEquationsUtils):
    """
    共轭梯度法，对于对称正定矩阵，迭代速度非常快, 继承IterativeLinearEquationsUtils
    """

    def __init__(self, A, b, x0, eps=1e-8, max_iter=200, is_out_info=False):
        IterativeLinearEquationsUtils.__init__(self, A, b, x0, eps, max_iter, is_out_info)
        self.iterative_info = {}  # 组合存储迭代信息
        self.precision = []  # 存储每次迭代误差精度

    def fit_solve(self):
        """
        核心算法：共轭梯度法求解
        :return:
        """
        if self._symmetric_positive_definite_() == "no_symmetric":
            print("非对称矩阵，不适宜共轭梯度法.")
            self.is_out_info = False
            return
        elif self._symmetric_positive_definite_() == "no_positive":
            print("非正定矩阵，不适宜共轭梯度法.")
            self.is_out_info = False
            return
        cond_num = np.linalg.cond(self.A)  # 系数矩阵的条件数
        iteration, iter_process = 0, []  # 迭代变量和迭代过程所求的解x
        rk_next = self.b - np.dot(self.A, self.x0)  # 残差向量，负梯度方向
        x_next, pk_next = np.copy(self.x0), np.copy(rk_next)  # 初始化
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 解的迭代
            rk_before, pk_before = np.copy(rk_next), np.copy(pk_next)  # 共轭梯度参数的更新
            epsilon = np.dot(pk_before, np.dot(self.A, pk_before))
            if epsilon <= 1e-50:  # 终止条件
                print("CG_IterativeStopCond：(pk, A*pk): %.10e" % epsilon)
                break
            # 共轭梯度公式求解过程
            alpha_k = np.dot(rk_before, rk_before) / epsilon  # 搜索步长
            x_next = x_before + alpha_k * pk_before  # 解向量的更新
            rk_next = rk_before - alpha_k * np.dot(self.A, pk_before)  # 更新梯度向量
            if max(np.abs(rk_next)) < 1e-50:  # 终止条件
                print("CG_IterativeStopCond：max(abs(rk)): %.10e" % max(np.abs(rk_next)))
                break
            beta_k = np.dot(rk_next, rk_next) / np.dot(rk_before, rk_before)  # 组合系数
            pk_next = rk_next + beta_k * pk_before  # 新的共轭方向
            iter_process.append(x_next)  # 存储解的迭代信息
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "共轭梯度法已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合共轭梯度法的结果信息
            self.iterative_info["Success_Info"] = "共轭梯度法，迭代终止，收敛到近似解"
        self.iterative_info["Condition_number"] = cond_num
        self.iterative_info["Iteration_number"] = len(self.precision)
        self.iterative_info["Solution_X"] = x_next
        if self.precision:
            self.iterative_info["Precision"] = self.precision[-1]
        if self.is_out_info:  # 是否输出迭代结果信息
            for key in self.iterative_info.keys():
                print(key + ":", self.iterative_info[key])
        return x_next

    def _symmetric_positive_definite_(self):
        """
        判断系数矩阵是否是对称正定矩阵
        :return:
        """
        if np.array_equal(self.A, self.A.T):
            try:
                np.linalg.cholesky(self.A)  # 采用库函数
                return True
            except np.linalg.LinAlgError:
                return "no_positive"
        else:
            return "no_symmetric"

    def plt_convergence_x(self, is_show=True):
        """
        可视化迭代解的精度曲线
        """
        IterativeLinearEquationsUtils._plt_convergence_precision(self, is_show, "CG")
