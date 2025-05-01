# -*- coding: UTF-8 -*-
"""
@file_name: block_iterative_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from partial_differential_equation_12.lecture_test.Iterative_linear_equs_utils import IterativeLinearEquationsUtils


class BlockIterative(IterativeLinearEquationsUtils):
    """
    块迭代法，分别实现块Jacobi迭代法、块G-S迭代法和块SOR迭代法，继承IterativeLinearEqusEntityUtils
    """

    def __init__(self, A, b, x0, block, eps=1e-8, max_iter=1000, omega=1.5,
                 method="jacobi", is_out_info=False):
        IterativeLinearEquationsUtils.__init__(self, A, b, x0, eps, max_iter, is_out_info)
        self.block = np.asarray(block, dtype=np.int)  # 分块向量
        if sum(self.block) != self.n:
            raise ValueError("分块向量和的维度与系数矩阵不匹配.")
        self.method = method  # 块迭代方法
        self.omega = omega  # 松弛因子
        self.iterative_info = {}  # 组合存储迭代信息
        self.is_out_info = is_out_info  # 是否输出迭代信息
        self.precision = []  # 存储每次迭代误差精度
        self.max_lambda = np.infty  # 迭代矩阵的谱半径

    def fit_solve(self):
        """
        块迭代法求解，首先分块，然后采用相应迭代法求解
        :return:
        """
        # 分析块索引，即每块的起始索引
        n_b = len(self.block)
        block_start_idx = np.zeros(n_b, dtype=np.int)
        block_start_idx[1:] = np.cumsum(self.block)[:-1]
        # 矩阵A进行分块，即A = D - L - U
        D, inv_D = np.zeros((self.n, self.n)), np.zeros((self.n, self.n))
        for i in range(n_b):
            s_idx = block_start_idx[i]  # 块起始索引
            e_idx = block_start_idx[i] + self.block[i]  # 块终止索引
            D[s_idx:e_idx, s_idx:e_idx] = self.A[s_idx:e_idx, s_idx:e_idx]  # 分块对角矩阵A^ii
            # A对角分块矩阵的逆矩阵
            inv_D[s_idx:e_idx, s_idx:e_idx] = np.linalg.inv(self.A[s_idx:e_idx, s_idx:e_idx])
        # 求系数矩阵A的下三角分块阵和上三角分块阵
        block_low_mat, block_up_mat = -np.tril(self.A - D), -np.triu(self.A - D)
        if self._is_convergence_(D, inv_D, block_low_mat, block_up_mat):
            if self.method.lower() == "jacobi":
                self.x = self._solve_block_jacobi_(D, inv_D)
            elif self.method.lower() == "g-s":
                self.x = self._solve_block_gauss_seidel_(D, block_low_mat, block_up_mat)
            elif self.method.lower() == "sor":
                self.x = self._solve_block_sor_(D, block_low_mat, block_up_mat)
            else:
                raise ValueError("仅支持块Jacobi、块Gauss-Seidel和块SOR迭代法.")
        else:
            raise ValueError("块迭代法不收敛.")
        if self.is_out_info:
            for key in self.iterative_info.keys():
                print(key + ":", self.iterative_info[key])
        return self.x

    def _solve_block_jacobi_(self, D, inv_D):
        """
        核心算法：块雅可比迭代法
        :return:
        """
        x_next = np.copy(self.x0)  # x_next表示x(k+1)， x_before表示x(k)
        iteration = 0  # 迭代变量
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 迭代序列更新。必须为copy，不能赋值
            # 块Jacobi矩阵形式迭代公式
            x_next = np.dot(np.dot(inv_D, D - self.A), x_before) + np.dot(inv_D, self.b)
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            tol = np.linalg.norm(x_next - x_before)
            if self.precision[-1] <= self.eps or tol <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "块Jacobi迭代已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合块迭代法的结果信息
            self.iterative_info["Success_Info"] = "块Jacobi迭代，优化终止，收敛到近似解"
        self.iterative_info["Convergence"] = "Spectral radius %.5f" % self.max_lambda
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_next
        self.iterative_info["Precision"] = self.precision[-1]
        return x_next

    def _solve_block_gauss_seidel_(self, D, block_low_mat, block_up_mat):
        """
        块高斯-赛德尔迭代法
        :return:
        """
        x_next = np.copy(self.x0)  # x_next表示x(k+1)， x_before表示x(k)
        iteration = 0  # 迭代变量
        inv_DL = np.linalg.inv(D - block_low_mat)
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 迭代序列更新
            # 块G-S矩阵形式迭代公式
            x_next = np.dot(np.dot(inv_DL, block_up_mat), x_before) + np.dot(inv_DL, self.b)
            tol = np.linalg.norm(x_next - x_before)
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps or tol <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "块G-S迭代法已达最大迭代次数"
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合块G-S迭代法的结果信息
            self.iterative_info["Success_Info"] = "块G-S迭代，优化终止，收敛到近似解"
        self.iterative_info["Convergence"] = "Spectral radius %.5f" % self.max_lambda
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_next
        self.iterative_info["Precision"] = self.precision[-1]
        return x_next

    def _solve_block_sor_(self, D, block_low_mat, block_up_mat):
        """
        块超松弛迭代法
        :return:
        """
        x_next = np.copy(self.x0)  # x_next表示x(k+1)， x_before表示x(k)
        iteration = 0  # 迭代变量
        inv_DL = np.linalg.inv(D - self.omega * block_low_mat)  # 带松弛因子
        omega_ = 1 - self.omega
        for iteration in range(self.max_iter):
            x_before = np.copy(x_next)  # 迭代序列更新
            x_next = np.dot(np.dot(inv_DL, omega_ * D + self.omega * block_up_mat), x_before) + \
                     self.omega * np.dot(inv_DL, self.b)  # 块超松弛矩阵形式迭代公式
            tol = np.linalg.norm(x_next - x_before)
            self.precision.append(np.linalg.norm(self.b - np.dot(self.A, x_next)))  # 每次迭代的误差
            if self.precision[-1] <= self.eps or tol <= self.eps:  # 满足精度要求，迭代终止
                break
        if iteration >= self.max_iter - 1:
            self.iterative_info["Success_Info"] = "块SOR迭代法已达最大迭代次数."
        else:
            # 在最大迭代次数内收敛到精度要求，用字典组合块SOR迭代法的结果信息
            self.iterative_info["Success_Info"] = "块SOR迭代，优化终止，收敛到近似解"
        self.iterative_info["Omega"] = "Omega %.5f" % self.omega
        self.iterative_info["Convergence"] = "Spectral radius %.5f" % self.max_lambda
        self.iterative_info["Iteration_number"] = iteration + 1
        self.iterative_info["Solution_X"] = x_next
        self.iterative_info["Precision"] = self.precision[-1]
        return x_next

    def _is_convergence_(self, D, inv_D, block_low_mat, block_up_mat):
        """
        判断迭代矩阵是否收敛
        :return: 收敛，最大谱半径；不收敛，False
        """
        if np.linalg.det(self.A) != 0:  # 非奇异
            B = np.eye(self.n)
            if self.method.lower() == "jacobi":
                B = np.dot(inv_D, (block_low_mat + block_up_mat))  # 块Jacobi迭代矩阵
            elif self.method.lower() == "g-s":
                B = np.dot(np.linalg.inv(D - block_low_mat), block_up_mat)  # 块G-S迭代矩阵
            elif self.method.lower() == "sor":
                inv_D_ = np.linalg.inv(D - self.omega * block_low_mat)
                B = np.dot(inv_D_, (1 - self.omega) * D + self.omega * block_up_mat)  # 块G-S迭代矩阵
            eigenvalues = np.linalg.eig(B)[0]  # 求特征值，即索引为0的元素
            max_lambda = np.max(np.abs(eigenvalues))  # 取特征值的绝对值最大的
            if max_lambda >= 1:  # 不收敛
                print("谱半径：%s，块迭代法不收敛." % max_lambda)
                return False
            else:  # 收敛
                self.max_lambda = max_lambda
                return True
        else:
            print("奇异矩阵，不能用迭代法求解.")
            return False

    def plt_convergence(self, is_show=True):
        """
        可视化迭代解的精度曲线
        :return:
        """
        IterativeLinearEquationsUtils._plt_convergence_precision(self, is_show, "Block \ " + self.method)
