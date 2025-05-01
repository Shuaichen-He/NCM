# -*- coding: UTF-8 -*-
"""
@file_name: jacobi_matrix.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from copy import deepcopy
import numpy as np
import sympy


class JacobiMatrix:
    """
    求解符号方程定义的雅可比矩阵，雅可比矩阵函数值以及非线性方程的函数值
    """

    def __init__(self, sym_NLFx, sym_vars):
        self.sym_NLFx = sym_NLFx  # 符号定义方程组
        self.sym_vars = sym_vars  # 符号变量
        self.n = len(self.sym_vars)  # 变量的个数

    def cal_fx_values(self, x):
        """
        方程组求值， 列向量x的shape=(n, 1)
        :return:
        """
        nonlinear_fx = deepcopy(self.sym_NLFx)  # 拷贝一份，以便变量替换为值
        for i in range(self.n):  # 针对每个方程
            for j, var in enumerate(self.sym_vars):  # 针对每个符号变量
                nonlinear_fx[i] = nonlinear_fx[i].subs(var, x[j][0])  # 对应方程各变量符号替换为数值
        fx_values = np.asarray(nonlinear_fx, np.float64).reshape(-1, 1)  # 转换为数值
        return fx_values

    def solve_jacobi_mat(self):
        """
        求解雅可比矩阵
        :return:
        """
        jacobi_mat = sympy.zeros(self.n, self.n)  # 初始化雅可比矩阵，符号矩阵
        for i in range(self.n):  # 针对每个方程
            for j, var in enumerate(self.sym_vars):  # 针对每个符号变量
                jacobi_mat[i, j] = self.sym_NLFx[i].diff(var, 1)  # 一阶导函数
        return jacobi_mat

    def cal_jacobi_mat_values(self, jacobi_mat, x):
        """
        求解雅可比矩阵的值，分别对雅可比矩阵的每个元素求值
        :return:
        """
        jacobi_matrix = deepcopy(jacobi_mat)
        for i in range(self.n):
            for j in range(self.n):
                for k, var in enumerate(self.sym_vars):  # 雅可比矩阵每个元素，每个变量替换
                    jacobi_matrix[i, j] = jacobi_matrix[i, j].subs(var, float(x[k]))
        jacobi_val = np.asarray(jacobi_matrix, np.float64)  # 转换为数值
        return jacobi_val
