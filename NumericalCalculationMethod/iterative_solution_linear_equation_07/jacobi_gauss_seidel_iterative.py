# -*- coding: UTF-8 -*-
"""
@file_name: jacobi_gauss_seidel_iterative.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.lecture_test.Iterative_linear_equs_utils import IterativeLinearEquationsUtils


class JacobiGSlIterativeMethod(IterativeLinearEquationsUtils):
    """
    雅可比迭代法和高斯-赛德尔迭代法求解线性方程组的解，继承IterativeLinearEquationsUtils
    """

    def __init__(self, A, b, x0, eps=1e-8, max_iter=200, method="jacobi", is_out_info=False):
        IterativeLinearEquationsUtils.__init__(self, A, b, x0, eps, max_iter,
                                               is_out_info)
        self.method = method  # 迭代方法，默认雅可比，包括高斯-赛德尔gs
        self.max_lambda = np.infty  # 迭代矩阵的谱半径

    def fit_solve(self):
        """
        雅可比迭代法和高斯-赛德尔迭代法求解
        :return:
        """
        if self._is_convergence_():  # 判断迭代矩阵是否收敛
            if self.method.lower() == "jacobi":
                self.x = self._solve_jacobi_()  # 雅可比迭代法
            elif self.method.lower() == "g-s":
                self.x = self._solve_gauss_seidel_()  # 高斯—塞德尔迭代法
            else:
                raise ValueError("仅支持Jacobi迭代法和Gauss-Seidel迭代法.")
        else:
            raise ValueError("Jacobi或G-S迭代法不收敛.")
        if self.is_out_info:  # 是否输出迭代过程信息
            for key in self.iterative_info.keys():
                print(key + ":", self.iterative_info[key])
        return self.x

    def _solve_jacobi_(self):
        """
        核心算法：雅可比迭代法求解
        :return:
        """
        x_next = np.copy(self.x0)  # x_next表示x(k+1)， x_before表示x(k)
        iteration = 0  # 迭代变量
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 迭代序列更新 x_b表示x(k)第k次迭代向量。
            for i in range(self.n):
                sum_j = np.dot(self.A[i, :i], x_before[:i]) + \
                        np.dot(self.A[i, i + 1:], x_before[i + 1:])
                x_next[i] = (self.b[i] - sum_j) / self.A[i, i]  # 迭代公式
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "雅可比迭代法已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合雅可比迭代法的结果信息
            self.iterative_info["Success_Info"] = "Jacobi迭代，优化终止，收敛到近似解"
        self.iterative_info["Convergence"] = "Spectral radius %.5f" % self.max_lambda
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_next
        self.iterative_info["Precision"] = self.precision[-1]
        return x_next

    def _solve_gauss_seidel_(self):
        """
        核心算法：高斯—赛德尔迭代法求解
        :return:
        """
        x_iter = self.x0  # x_next表示x(k+1)
        iteration = 0  # 迭代变量
        for iteration in range(self.max_iter):
            for j in range(self.n):
                # G-S不同于Jacobi迭代法的地方，即np.dot(self.A[j, :j], x_iter[:j])
                # 每次采用新迭代值
                sum_g = np.dot(self.A[j, :j], x_iter[:j]) + \
                        np.dot(self.A[j, j + 1:], x_iter[j + 1:])
                x_iter[j] = (self.b[j] - sum_g) / self.A[j, j]  # 迭代公式
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_iter)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "高斯-赛德尔迭代法已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合雅可比迭代法的结果信息
            self.iterative_info["Success_Info"] = "G-S迭代，优化终止，收敛到近似解"
        self.iterative_info["Convergence"] = "Spectral radius %.5f" % self.max_lambda
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_iter
        self.iterative_info["Precision"] = self.precision[-1]
        return x_iter

    def _is_convergence_(self):
        """
        判断迭代矩阵是否收敛
        :return: 收敛，最大谱半径；不收敛，False
        """
        if np.linalg.det(self.A) != 0:  # 非奇异
            D = np.diag(self.A)  # 以方阵A对角线元素构成的一个向量
            if not np.any(D == 0):  # 对角线元素全部不为零
                D, B = np.diag(D), np.eye(self.n)  # 构造一个方阵D，取A对角线元素
                L, U = -np.tril(self.A, -1), -np.triu(self.A, 1)
                if self.method.lower() == "jacobi":
                    B = np.dot(np.linalg.inv(D), (L + U))  # 雅可比迭代矩阵
                elif self.method.lower() == "g-s":
                    B = np.dot(np.linalg.inv(D - L), U)  # G-S迭代矩阵
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
        IterativeLinearEquationsUtils._plt_convergence_precision(self, is_show, self.method)
