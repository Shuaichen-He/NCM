# -*- coding: UTF-8 -*-
"""
@file_name: SOR_iterative.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from iterative_solution_linear_equation_07.utils.Iterative_linear_equs_utils import IterativeLinearEquationsUtils


class SORIteration(IterativeLinearEquationsUtils):
    """
    逐次超松弛迭代法，继承IterativeLinearEqusEntityUtils
    """

    def __init__(self, A, b, x0, eps=1e-8, max_iter=200, omega=1.5, is_out_info=False):
        IterativeLinearEquationsUtils.__init__(self, A, b, x0, eps, max_iter,
                                               is_out_info)
        self.omega = omega  # 松弛因子
        self.iterative_info = {}  # 组合存储迭代信息
        self.precision = []  # 存储每次迭代误差精度
        self.max_lambda = np.infty  # 迭代矩阵的谱半径

    def fit_solve(self):
        """
        雅可比迭代法和高斯-赛德尔迭代法求解
        :return:
        """
        if self._is_convergence_():  # 判断迭代矩阵是否收敛
            self.x = self._solve_sor_()
        else:
            raise ValueError("SOR迭代法不收敛.")
        if self.is_out_info:  # 是否输出迭代过程信息
            for key in self.iterative_info.keys():
                print(key + ":", self.iterative_info[key])

    def _solve_sor_(self):
        """
        核心算法：超松弛高斯—赛德尔迭代法求解
        :return:
        """
        x_next = self.x0  # x_next表示x(k+1)
        iteration = 0  # 迭代变量
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 迭代序列更新 x_b表示x(k)第k次迭代向量。
            for j in range(self.n):
                # 第一步：高斯—赛德尔公式
                sum_g = np.dot(self.A[j, :j], x_next[:j]) + \
                        np.dot(self.A[j, j + 1:], x_before[j + 1:])
                x_next[j] = (self.b[j] - sum_g) / self.A[j, j]  # 迭代公式
                # 第二步：超松弛迭代公式
                x_next[j] = x_before[j] + self.omega * (x_next[j] - x_before[j])
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "SOR迭代法已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合超松弛迭代法的结果信息
            self.iterative_info["Success_Info"] = "SOR迭代，优化终止，收敛到近似解"
        self.iterative_info["Omega"] = "Omega %.5f" % self.omega
        self.iterative_info["Convergence"] = "Spectral radius %.5f" % self.max_lambda
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_next
        self.iterative_info["Precision"] = self.precision[-1]
        return x_next

    def _is_convergence_(self):  # 参考G-S迭代法
        """
        判断迭代矩阵是否收敛
        :return: 收敛，最大谱半径；不收敛，False
        """
        if np.linalg.det(self.A) != 0:  # 非奇异
            D = np.diag(self.A)  # 以方阵A对角线元素构成的一个向量
            if not np.any(D == 0):  # 对角线元素全部不为零
                D = np.diag(D)  # 构造一个方阵D，取A对角线元素
                L, U = np.tril(self.A, -1), np.triu(self.A, 1)
                inv_D_ = np.linalg.inv(D - self.omega * L)
                B = np.dot(inv_D_, (1 - self.omega) * D + self.omega * U)  # 块G-S迭代矩阵
                eigenvalues = np.linalg.eig(B)[0]  # 求特征值，即索引为0的元素
                max_lambda = np.max(np.abs(eigenvalues))  # 取特征值的绝对值最大的
                if max_lambda >= 1:  # 不收敛
                    print("谱半径：%s，迭代矩阵不收敛." % max_lambda)
                    return False
                else:  # 收敛
                    self.max_lambda = max_lambda
                    return True
            else:
                print("矩阵对角线元素包含零元素，不宜用迭代法求解.")
                return False
        else:
            print("奇异矩阵，不能用迭代法求解.")
            return False

    def plt_convergence(self, is_show=True):
        """
        可视化迭代解的精度曲线
        """
        IterativeLinearEquationsUtils._plt_convergence_precision(self, is_show, "SOR")
