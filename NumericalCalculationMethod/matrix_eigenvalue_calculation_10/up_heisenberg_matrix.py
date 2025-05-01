# -*- coding: UTF-8 -*-
"""
@file_name: up_heisenberg_matrix.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import math


class UPHeisenbergMatrix:
    """
    用正交相似变换约化一般矩阵为上海森伯格矩阵
    """
    def __init__(self, A):
        self.A = np.asarray(A, np.float64)
        if self.A.shape[0] != self.A.shape[1]:
            print("非方阵，不能化为上海森伯格矩阵.")
            exit(0)
        self.n = self.A.shape[0]  # 维度

    def cal_heisenberg_mat(self):
        """
        用正交相似变换约化一般矩阵为上海森伯格矩阵，具体算法实现
        :return:
        """
        heisenberg_mat = np.copy(self.A)
        for i in range(self.n - 2):
            max_val = max(np.abs(heisenberg_mat[1 + i:, i]))
            c = heisenberg_mat[1 + i:, i] / max_val  # 规范化
            sigma = np.sign(c[0]) * math.sqrt(np.sum(c ** 2))
            u = (c + sigma * np.eye(self.n - 1 - i)[0, :]).reshape(-1, 1)
            beta = sigma * (sigma + c[0])
            R = np.eye(self.n - 1 - i) - u.T * u / beta
            U = np.eye(self.n)
            U[1 + i:, 1 + i:] = R
            heisenberg_mat = np.dot(np.dot(U, heisenberg_mat), U)
        return heisenberg_mat
